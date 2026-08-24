"""Placing orders: brokerage-facing, and deliberately ignorant of algorithms.

Everything here takes intents, prices and sizing knobs and reports what the broker did with
them. Nothing imports the algorithm registry, which is what lets ``BaseAlgorithm`` import this
module outright rather than reaching for it from inside a method body. Composing an algorithm
with its data and its account is ``src/core/runner.py``'s job, one layer up.
"""

from __future__ import annotations

import logging
from typing import Any

from src.brokerages.registry import get_brokerage_class
from src.core.config import get_account_broker_type
from src.core.interfaces import (
    MODE_INCREMENTAL,
    MODE_TARGET,
    Brokerage,
    Intent,
    PortfolioSnapshot,
    weights_from_intents,
)
from src.core.orders import (
    FUNDING_GREEDY,
    FUNDING_PRO_RATA,
    SHARE_EPSILON,
    fund_planned_orders,
    plan_share_orders,
    resolve_target_shares,
    submit_planned_orders,
)

logger = logging.getLogger(__name__)


class UnknownBrokerageError(Exception):
    """Raised when the account's configured brokerage has no registered implementation."""

    def __init__(self, broker_type: str) -> None:
        self.broker_type = broker_type
        super().__init__(f"Unknown brokerage: {broker_type}")


def sizing_equity(config, account_equity: float) -> float:
    cap = max(float(getattr(config, "algorithm_equity_cap", 0.0) or 0), 0.0)
    return min(account_equity, cap) if cap > 0 else account_equity


def resolve_brokerage(config) -> Brokerage:
    """Instantiate the brokerage registered for the account, or raise ``UnknownBrokerageError``."""
    broker_type = get_account_broker_type(config.account_id)
    try:
        return get_brokerage_class(broker_type)(config)
    except KeyError:
        raise UnknownBrokerageError(broker_type) from None


def read_snapshot(config, brokerage: Brokerage) -> PortfolioSnapshot:
    """Read holdings and sizing equity from the brokerage."""
    account_state = brokerage.get_account_state()
    account_equity = float(account_state.get("equity", 0.0))
    equity = sizing_equity(config, account_equity)
    if equity < account_equity:
        logger.info("Algorithm sizing equity capped at %.2f from account equity %.2f", equity, account_equity)
    return PortfolioSnapshot(
        positions=brokerage.get_positions(),
        equity=equity,
        cash=float(account_state.get("cash", 0.0) or 0.0),
        # Absent is not the same as zero. A brokerage that reports no buying power tells us
        # nothing about what it will fund, and reading that silence as "no money" would trim
        # every buy to nothing; fall back to the widest figure it did report instead, which
        # leaves such a brokerage behaving as it did before funding existed.
        buying_power=float(
            account_state.get("buying_power", account_state.get("cash", account_equity)) or 0.0
        ),
    )


# --------------------------------------------------------------------------------------
# Running an algorithm: assemble its context, ask it for a plan. Reads the account but never
# writes to it, so the dashboard, the MCP agent and the scheduler all make this same call.
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# Placing orders: brokerage only. No algorithm, no data sources -- prices ride on the plan.
# --------------------------------------------------------------------------------------


def weight_diff(
    current_weights: dict[str, float], final_weights: dict[str, float]
) -> list[dict[str, Any]]:
    """Per-symbol before/after weights, sorted by the size of the change."""
    rows = [
        {
            "symbol": symbol,
            "current_weight": round(current_weights.get(symbol, 0.0), 6),
            "final_weight": round(final_weights.get(symbol, 0.0), 6),
            "change": round(final_weights.get(symbol, 0.0) - current_weights.get(symbol, 0.0), 6),
        }
        for symbol in sorted(set(current_weights) | set(final_weights))
    ]
    for row in rows:
        row["action"] = "add" if row["change"] > 0 else "trim" if row["change"] < 0 else "hold"
    return sorted(rows, key=lambda row: -abs(row["change"]))


def place_orders(
    intents: list[Intent],
    config,
    brokerage: Brokerage,
    *,
    latest_prices: dict[str, float],
    signals: dict[str, dict[str, Any]] | None = None,
    mode: str = MODE_TARGET,
    min_trade_dollars: float = 0.0,
    rebalance_threshold: float = 0.0,
) -> dict[str, Any]:
    """Size ``intents`` against what the account holds and submit the resulting orders.

    Knows nothing about algorithms -- it takes intents, prices and two sizing knobs, and
    reports what the broker did with them. Everything strategy-specific was hoisted out: what
    a fill means for an accrued budget is a question only the algorithm can answer, and it
    answers it in ``BaseAlgorithm.execute``.

    Does no market-data fetching, so a symbol must already be priced in ``latest_prices``.
    """
    signals = signals or {}
    snapshot = read_snapshot(config, brokerage)

    # ``shares`` intents carry their own quantity; everything else has to be priced to size.
    unpriced = sorted(
        {intent.symbol for intent in intents if intent.kind != "shares" and latest_prices.get(intent.symbol, 0.0) <= 0}
    )
    if unpriced:
        raise ValueError(
            f"No price available for {', '.join(unpriced)}. Placing orders does not fetch "
            "market data, so a symbol must have been priced by the plan it came from."
        )

    fractional = getattr(brokerage, "supports_fractional_shares", False)
    sized_shares = resolve_target_shares(
        intents,
        mode,
        snapshot.positions,
        latest_prices,
        snapshot.equity,
        supports_fractional_shares=fractional,
    )

    planned_orders = plan_share_orders(
        latest_prices,
        snapshot.positions,
        sized_shares,
        snapshot.equity,
        min_trade_dollars=min_trade_dollars,
        rebalance_threshold=rebalance_threshold,
        supports_fractional_shares=fractional,
        target_weights=weights_from_intents(intents),
    )

    fundable, unfunded, funding = fund_planned_orders(
        planned_orders,
        buying_power=snapshot.buying_power,
        reserve=snapshot.equity * max(0.0, min(1.0, float(getattr(config, "cash_buffer", 0.0) or 0.0))),
        cash_equivalents=getattr(brokerage, "get_cash_equivalents", dict)(),
        min_trade_dollars=min_trade_dollars,
        supports_fractional_shares=fractional,
        policy=FUNDING_GREEDY if mode == MODE_INCREMENTAL else FUNDING_PRO_RATA,
    )

    share_order_results = submit_planned_orders(brokerage, fundable)
    share_order_results.extend(
        {**order, "action": "skip", "quantity": 0, "status": "unfunded"} for order in unfunded
    )

    order_results = share_order_results

    # A brokerage that keeps its own book (the local paper one) has no price feed, so its
    # equity would stay marked at the last fill until someone traded again.
    if hasattr(brokerage, "mark_prices"):
        brokerage.mark_prices(latest_prices)

    rejected = [order for order in order_results if order.get("status") == "rejected"]
    submitted = [order for order in order_results if order.get("status") == "submitted"]
    resulting_weights = PortfolioSnapshot(
        positions=_resulting_positions(snapshot.positions, submitted), equity=snapshot.equity
    ).weights(latest_prices)
    return {
        "mode": mode,
        "status": _batch_status(order_results, submitted, rejected, unfunded, funding),
        "equity": snapshot.equity,
        "final_weights": resulting_weights,
        "final_intents": [
            {"symbol": intent.symbol, "kind": intent.kind, "value": intent.value}
            for intent in intents
        ],
        "diff": weight_diff(snapshot.weights(latest_prices), resulting_weights),
        "planned_orders": planned_orders,
        "order_results": order_results,
        "funding": funding,
        "unfunded": [
            {"symbol": order["symbol"], "action": order["action"], "reason": order["reason"]}
            for order in unfunded
        ],
        "rejected": [
            {"symbol": order["symbol"], "action": order["action"], "reason": order["reason"]}
            for order in rejected
        ],
    }


def _resulting_positions(
    positions: dict[str, float], submitted: list[dict[str, Any]]
) -> dict[str, float]:
    """Apply the submitted legs to the held book, at the sizes that were actually sent."""
    resulting = dict(positions)
    for order in submitted:
        quantity = float(order.get("quantity") or 0.0)
        signed = quantity if order.get("action") == "buy" else -quantity
        symbol = str(order["symbol"])
        resulting[symbol] = float(resulting.get(symbol, 0.0)) + signed
    return {symbol: shares for symbol, shares in resulting.items() if abs(shares) > SHARE_EPSILON}


#: Every leg went out at the size the plan asked for.
STATUS_SUBMITTED = "submitted"
#: The batch was deliberately fitted to available funds. A success, not a failure: an agent
#: that reads this as a rejection and retries would be re-submitting orders that were trimmed
#: on purpose, which is exactly what the old ``partial`` reading caused.
STATUS_SUBMITTED_REDUCED = "submitted_reduced"


def _batch_status(order_results, submitted, rejected, unfunded=(), funding=None) -> str:
    if not order_results:
        return "no_orders"
    if rejected:
        return "partial" if submitted else "rejected"
    if unfunded or (funding or {}).get("reduced"):
        return STATUS_SUBMITTED_REDUCED if submitted else "unfunded"
    return STATUS_SUBMITTED
