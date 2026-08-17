from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from src.algorithms.registry import get_algorithm_class
from src.brokerages import BROKERAGE_REGISTRY
from src.core.config import get_account_broker_type
from src.core.interfaces import (
    MODE_INCREMENTAL,
    MODE_TARGET,
    AlgorithmDecision,
    AlgorithmResult,
    Brokerage,
    PortfolioSnapshot,
    intents_from_weights,
    weights_from_intents,
)
from src.core.market_context import build_algorithm_context
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

#: How stale a step-1 result may be before step 2 refuses to trade on its prices.
DEFAULT_MAX_RESULT_AGE_SECONDS = 900


class UnknownBrokerageError(Exception):
    """Raised when the account's configured brokerage has no registered implementation."""

    def __init__(self, broker_type: str) -> None:
        self.broker_type = broker_type
        super().__init__(f"Unknown brokerage: {broker_type}")


class StaleResultError(Exception):
    """Raised when a step-1 result is too old to size orders against."""


def sizing_equity(config, account_equity: float) -> float:
    cap = max(float(getattr(config, "algorithm_equity_cap", 0.0) or 0), 0.0)
    return min(account_equity, cap) if cap > 0 else account_equity


def resolve_brokerage(config) -> Brokerage:
    """Instantiate the brokerage registered for the account, or raise ``UnknownBrokerageError``."""
    broker_type = get_account_broker_type(config.account_id)
    broker_cls = BROKERAGE_REGISTRY.get(broker_type)
    if broker_cls is None:
        raise UnknownBrokerageError(broker_type)
    return broker_cls(config)


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
# Step 1: algorithm + data sources. No brokerage, no positions, no equity.
# --------------------------------------------------------------------------------------


def run_algorithm(strategy: str, config, *, data_client: Any = None) -> AlgorithmResult:
    """Run ``strategy`` against market data and return the portfolio it proposes.

    Needs no brokerage and no account, so it works before a broker is configured and is the
    same call the dashboard, the scheduled runner, and the MCP agent all use.
    """
    algorithm = get_algorithm_class(strategy).from_config(config)
    requirements = algorithm.requirements(config, {})
    context = build_algorithm_context(config, requirements, data_client=data_client)
    latest_prices = context.latest_prices

    decision: AlgorithmDecision = algorithm.analyze(context)

    intents = [
        intent for intent in decision.resolved_intents() if intent.kind != "weight" or intent.value
    ]
    return AlgorithmResult(
        strategy=strategy,
        intents=intents,
        signals=decision.signals,
        latest_prices=latest_prices,
        metadata={**decision.metadata, "requirements": requirements},
        mode=decision.mode,
        # The moment the context described, which live is now and in a replay is the signal
        # date. Taking it from the context rather than defaulting to the wall clock is what
        # lets step 2 measure elapsed time without a clock of its own.
        as_of=context.timestamp,
    )


# --------------------------------------------------------------------------------------
# Step 2: algorithm + brokerage. No data sources -- prices ride along on the step-1 result.
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
    result: AlgorithmResult,
    config,
    brokerage: Brokerage,
    *,
    target_weights: dict[str, float] | None = None,
    require_approval: bool = False,
    approval_timeout_seconds: int = 300,
    approval_poll_seconds: int = 5,
    max_result_age_seconds: int = DEFAULT_MAX_RESULT_AGE_SECONDS,
    algorithm: Any = None,
) -> dict[str, Any]:
    """Turn a step-1 result into submitted orders, given what the account currently holds.

    ``target_weights`` overrides the result's own intents, which is how a reviewing agent
    submits an edited portfolio; the set is the complete intended portfolio, so a held symbol
    left out of it is exited. Does no market-data fetching -- prices come from ``result``.
    """
    _assert_fresh(result, max_result_age_seconds)

    # Re-resolving from the registry is only a convenience for callers holding a bare result.
    # A caller that already has the instance -- the replay steps the same one through every
    # date -- passes it, which also lets an unregistered algorithm be driven through step 2.
    algorithm = algorithm or get_algorithm_class(result.strategy).from_config(config)
    snapshot = read_snapshot(config, brokerage)
    latest_prices = result.latest_prices

    mode = result.mode
    if target_weights is None:
        proposed = list(result.intents)
    else:
        # An edited weight set is by definition the whole intended portfolio.
        proposed, mode = intents_from_weights(target_weights), MODE_TARGET

    # ``shares`` intents carry their own quantity; everything else has to be priced to size.
    unpriced = sorted(
        {intent.symbol for intent in proposed if intent.kind != "shares" and latest_prices.get(intent.symbol, 0.0) <= 0}
    )
    if unpriced:
        raise ValueError(
            f"No price available for {', '.join(unpriced)}. Step 2 does not fetch market data, "
            "so a symbol must have been priced by the algorithm run it came from."
        )

    sizing = algorithm.sizing(config)
    final_intents = algorithm.refine(
        proposed, result.signals, snapshot, latest_prices, config, result.as_of
    )

    fractional = getattr(brokerage, "supports_fractional_shares", False)
    # Sized against the whole book. What the account can afford is a separate question, asked
    # once below against buying power rather than smuggled in here as a haircut on targets.
    sized_shares = resolve_target_shares(
        final_intents,
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
        min_trade_dollars=sizing["min_trade_dollars"],
        rebalance_threshold=sizing["rebalance_threshold"],
        supports_fractional_shares=fractional,
        target_weights=weights_from_intents(final_intents),
    )

    fundable, unfunded, funding = fund_planned_orders(
        planned_orders,
        buying_power=snapshot.buying_power,
        # A fraction of the book, not of buying power: on a margin account the latter is an
        # inflated number and a percentage of it would reserve an arbitrary amount.
        reserve=snapshot.equity * max(0.0, min(1.0, float(getattr(config, "cash_buffer", 0.0) or 0.0))),
        # Optional on the brokerage, like ``mark_prices``: one that cannot value its holdings
        # simply funds from buying power alone.
        cash_equivalents=getattr(brokerage, "get_cash_equivalents", dict)(),
        min_trade_dollars=sizing["min_trade_dollars"],
        supports_fractional_shares=fractional,
        # An incremental batch is a set of independent commitments; a target batch is one
        # portfolio. See the policy constants in ``orders``.
        policy=FUNDING_GREEDY if mode == MODE_INCREMENTAL else FUNDING_PRO_RATA,
    )

    order_results = submit_planned_orders(
        brokerage,
        fundable,
        require_approval=require_approval,
        approval_timeout_seconds=approval_timeout_seconds,
        approval_poll_seconds=approval_poll_seconds,
    )
    # Recorded as results rather than dropped, so a caller can tell a leg that was deliberately
    # left out for want of funds from one that never existed.
    order_results.extend(
        {**order, "action": "skip", "quantity": 0, "status": "unfunded"} for order in unfunded
    )

    algorithm.settle(config, order_results, final_intents)

    # A brokerage that keeps its own book (the local paper one) has no price feed, so its
    # equity would stay marked at the last fill until someone traded again.
    if hasattr(brokerage, "mark_prices"):
        brokerage.mark_prices(latest_prices)

    rejected = [order for order in order_results if order.get("status") == "rejected"]
    submitted = [order for order in order_results if order.get("status") == "submitted"]
    # Built from what was actually submitted rather than from the pre-funding targets, so a
    # trimmed, unfunded, or broker-rejected leg is reflected instead of being reported as
    # though it had filled at full size.
    resulting_weights = PortfolioSnapshot(
        positions=_resulting_positions(snapshot.positions, submitted), equity=snapshot.equity
    ).weights(latest_prices)
    return {
        "strategy": result.strategy,
        "mode": mode,
        "status": _batch_status(order_results, submitted, rejected, unfunded, funding),
        "equity": snapshot.equity,
        "proposed_weights": weights_from_intents(proposed),
        "final_weights": resulting_weights,
        "final_intents": [
            {"symbol": intent.symbol, "kind": intent.kind, "value": intent.value}
            for intent in final_intents
        ],
        "diff": weight_diff(snapshot.weights(latest_prices), resulting_weights),
        "planned_orders": planned_orders,
        "order_results": order_results,
        # How the batch was paid for, so a caller can explain a trim rather than just report
        # one -- and tell a deliberate downsizing apart from a refusal it should react to.
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


def _assert_fresh(result: AlgorithmResult, max_age_seconds: int) -> None:
    if max_age_seconds <= 0:
        return
    age = datetime.now(timezone.utc) - result.as_of
    if age > timedelta(seconds=max_age_seconds):
        raise StaleResultError(
            f"Algorithm result is {int(age.total_seconds())}s old (limit {max_age_seconds}s). "
            "Its prices are too stale to size orders; run the algorithm again."
        )


#: Every leg went out at the size the plan asked for.
STATUS_SUBMITTED = "submitted"
#: The batch was deliberately fitted to available funds. A success, not a failure: an agent
#: that reads this as a rejection and retries would be re-submitting orders that were trimmed
#: on purpose, which is exactly what the old ``partial`` reading caused.
STATUS_SUBMITTED_REDUCED = "submitted_reduced"


def _batch_status(order_results, submitted, rejected, unfunded=(), funding=None) -> str:
    if not order_results:
        return "no_orders"
    # Checked before anything else: a denied approval means nothing was attempted, so reading
    # the batch for funding or refusals afterwards would describe a submission that never
    # happened -- which is what reporting it as "submitted" used to do.
    if any(order.get("approval_status") == "not_approved" for order in order_results):
        return "not_approved"
    if rejected:
        return "partial" if submitted else "rejected"
    if unfunded or (funding or {}).get("reduced"):
        return STATUS_SUBMITTED_REDUCED if submitted else "unfunded"
    return STATUS_SUBMITTED
