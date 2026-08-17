"""DCA as an ordinary algorithm.

Before the unified intent model this had to be a separate submission path, because algorithms
spoke *target portfolio* ("hold 44% BBC") while DCA speaks *increment* ("buy $200 of VTI").
It now emits ``notional`` intents in ``incremental`` mode and shares step 2 with everything
else -- the same refine/size/submit path, the same order plumbing.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ...core.interfaces import (
    DAILY_AT_OPEN,
    MODE_INCREMENTAL,
    AlgorithmContext,
    AlgorithmDecision,
    AlgorithmRequirements,
    Intent,
    PortfolioSnapshot,
    SignalView,
    describe_schedule,
)
from ..base import BaseAlgorithm
from . import BUCKETS, load_dca_plan, unknown_plan_symbols
from .accrual import (
    SymbolState,
    accrue,
    load_accrual_state,
    min_executable,
    save_accrual_state,
)

logger = logging.getLogger(__name__)


def plan_budgets(plan: dict[str, Any]) -> dict[str, float]:
    """Monthly dollar budget per symbol, signed: buy items positive, sell items negative."""
    budgets: dict[str, float] = {}
    for bucket in BUCKETS:
        sign = 1.0 if bucket == "buy" else -1.0
        for item in plan.get(bucket, {}).get("items", []) or []:
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            try:
                amount = float(item.get("amount") or 0.0)
            except (TypeError, ValueError):
                continue
            budgets[symbol] = budgets.get(symbol, 0.0) + (sign * abs(amount))
    return budgets


#: Trading days in a month, for restating a monthly budget at a human scale.
TRADING_DAYS_IN_MONTH = 21.75


def budget_line(monthly_budget: float, equity: float = 0.0) -> str:
    """Restate a monthly budget in the units someone reviewing the plan actually thinks in."""
    budget = abs(float(monthly_budget))
    if budget <= 0:
        return ""
    line = f"${budget:,.0f}/month ~ ${budget / TRADING_DAYS_IN_MONTH:,.2f}/trading day"
    if equity > 0:
        line += f" ~ {budget / equity:.1%} of equity"
    return line


def broker_supports_fractional_shares(account_id: str) -> bool:
    """Read the account's brokerage capability without instantiating (and authenticating) it."""
    from ...brokerages import BROKERAGE_REGISTRY
    from ...core.config import get_account_broker_type

    broker_cls = BROKERAGE_REGISTRY.get(get_account_broker_type(account_id))
    return bool(getattr(broker_cls, "supports_fractional_shares", False))


class DCAAlgorithm(BaseAlgorithm):
    """Steady DCA: deploy the accrued budget as soon as it can clear a broker minimum."""

    algorithm_id = "dca"

    #: Mondays at the open. Accrual is wall-clock, so a rarer cadence spends exactly as much
    #: as a frequent one -- it just deploys in fewer, larger trades, which is what clears a
    #: broker minimum sooner and costs fewer market data calls.
    schedule = replace(DAILY_AT_OPEN, weekdays=(0,))

    # ----------------------------------------------------------------------------------
    # Plan and configuration
    # ----------------------------------------------------------------------------------

    def config_fingerprint(self, config: Any) -> dict[str, Any]:
        """The plan is this algorithm's real configuration, so it belongs in the fingerprint.

        Editing a bucket weight changes every future decision, and a cached backtest computed
        under the old plan describes a strategy that no longer exists.
        """
        return {**super().config_fingerprint(config), "plan": self.plan(config)}

    def plan(self, config: Any) -> dict[str, Any]:
        from ...api.api_payloads import universe_payload

        # The plan is this account's config, so it must be read for the account being traded.
        return load_dca_plan(universe_payload()["rows"], account_id=str(getattr(config, "account_id", "") or ""))

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        return AlgorithmRequirements(price_symbols=sorted(plan_budgets(self.plan(config))))

    def sizing(self, config: Any) -> dict[str, float]:
        """No drift thresholds.

        ``rebalance_threshold`` is a target-drift concept: applied here it would wrongly
        suppress a small DCA buy that is exactly what the plan asked for. Accrual already
        enforces a per-trade floor through ``min_executable``.
        """
        return {"min_trade_dollars": 0.0, "rebalance_threshold": 0.0}

    # ----------------------------------------------------------------------------------
    # Step 1
    # ----------------------------------------------------------------------------------

    def generate_signals(self, context: AlgorithmContext) -> list[dict[str, Any]]:
        return []

    def analyze(self, context: AlgorithmContext) -> AlgorithmDecision:
        """Accrue each symbol's budget and emit an intent for whatever can trade now.

        Accrual is persisted here rather than at submission time because it is a pure function
        of elapsed time: running this twice in quick succession accrues the same total as
        running it once, so a dashboard preview cannot inflate or lose budget.
        """
        config = context.config
        account_id = getattr(config, "account_id", "") or ""
        plan = self.plan(config)
        budgets = plan_budgets(plan)
        # A naive timestamp is read as UTC rather than replaced by the wall clock. Substituting
        # "now" for a historical bar is how a replay ends up accruing a whole backtest's budget
        # against the moment it was run.
        now = context.timestamp if context.timestamp.tzinfo else context.timestamp.replace(tzinfo=timezone.utc)
        fractional = broker_supports_fractional_shares(account_id)
        min_trade_dollars = float(getattr(config, "min_trade_dollars", 0.0) or 0.0)

        state = load_accrual_state(self.algorithm_id, account_id)
        intents: list[Intent] = []
        signals: dict[str, dict[str, Any]] = {}

        for symbol, monthly_budget in sorted(budgets.items()):
            symbol_state = accrue(state.get(symbol, SymbolState()), abs(monthly_budget), now)
            state[symbol] = symbol_state

            price = float(context.latest_prices.get(symbol, 0.0) or 0.0)
            floor_dollars = min_executable(price, min_trade_dollars, fractional)
            # How many months of budget are sitting undeployed. A variant that waits for a
            # dip needs this to know when waiting has stopped being a strategy.
            months_behind = symbol_state.accrued / abs(monthly_budget) if monthly_budget else 0.0
            trigger = self.trigger(symbol, context, plan, months_behind)
            ready = symbol_state.accrued >= floor_dollars

            notional = 0.0
            if ready and trigger["fires"]:
                notional = symbol_state.accrued if monthly_budget >= 0 else -symbol_state.accrued
                intents.append(Intent(symbol=symbol, kind="notional", value=notional))

            signals[symbol] = {
                "signal": 1 if monthly_budget >= 0 else -1,
                "score": symbol_state.accrued,
                "close": price or None,
                "monthly_budget": monthly_budget,
                "accrued": round(symbol_state.accrued, 2),
                "deployed_this_month": round(symbol_state.deployed_this_month, 2),
                "min_executable": round(floor_dollars, 2),
                "notional": round(notional, 2),
                "reason": self.reason(symbol_state, floor_dollars, trigger, ready),
                "warning": self.whole_share_warning(symbol, monthly_budget, price, fractional),
                "budget_line": budget_line(monthly_budget),
                **trigger.get("detail", {}),
            }

        save_accrual_state(self.algorithm_id, account_id, state)
        return AlgorithmDecision(
            intents=intents,
            mode=MODE_INCREMENTAL,
            signals=signals,
            metadata={
                "allocation_mode": "DCA",
                "monthly_total": sum(abs(value) for value in budgets.values()),
                "schedule": describe_schedule(self.schedule),
            },
        )

    def trigger(
        self,
        symbol: str,
        context: AlgorithmContext,
        plan: dict[str, Any],
        months_behind: float = 0.0,
    ) -> dict[str, Any]:
        """Steady DCA has no timing condition: accrual alone decides."""
        return {"fires": True, "detail": {}}

    def signal_view(self, config: Any, *, data_client: Any = None) -> SignalView:
        """Render every configured bucket, whether or not the bot is running.

        The algorithm bot's switch controls whether orders are placed, not whether the plan
        exists -- a view that goes blank when the bot is off gives no way to check what it
        would do before turning it on.
        """
        from ...core.market_context import build_algorithm_context

        context = build_algorithm_context(config, self.requirements(config, {}), data_client=data_client)
        decision = self.analyze(context)

        leaders = [
            {
                **values,
                "symbol": symbol,
                "signal": "LONG" if float(values.get("monthly_budget") or 0.0) >= 0 else "SHORT",
                "side": "LONG" if float(values.get("monthly_budget") or 0.0) >= 0 else "SHORT",
                "target_weight": None,
            }
            for symbol, values in decision.signals.items()
        ]
        leaders.sort(key=lambda row: (row["reason"] != "Ready", row["symbol"]))
        monthly_total = float(decision.metadata.get("monthly_total") or 0.0)
        summary = [
            {"label": "Mode", "value": str(decision.metadata.get("allocation_mode") or "DCA")},
            {"label": "Schedule", "value": str(decision.metadata.get("schedule") or "--")},
            # The configured monthly total, not what happens to be deployable this minute.
            {"label": "Planned", "value": f"${monthly_total:.0f}/month"},
            {"label": "Symbols", "value": str(len(leaders))},
        ]
        # A typo in a bucket is otherwise dropped by sanitisation and shows up as nothing at all.
        unknown = unknown_plan_symbols()
        if unknown:
            summary.append({"label": "Not tradable", "value": ", ".join(unknown)})
        return SignalView(leaders=leaders, summary=summary)

    def whole_share_warning(self, symbol: str, monthly_budget: float, price: float, fractional: bool) -> str:
        """Warn when a budget is too small to clear one share for months on this brokerage."""
        if fractional or price <= 0 or monthly_budget <= 0 or price <= monthly_budget:
            return ""
        months = price / monthly_budget
        return f"${monthly_budget:.0f}/month accrues ~{months:.0f} months before it can trade on this brokerage."

    def reason(
        self,
        state: SymbolState,
        floor_dollars: float,
        trigger: dict[str, Any],
        ready: bool,
    ) -> str:
        if state.accrued < 0:
            # ``settle`` subtracts the filled notional with no floor, so a trade sized above
            # what had accrued -- which is what Bursty DCA's value averaging does when it is
            # catching up to the path -- leaves a debt. That is the mechanism that keeps the
            # long-run spend rate honest, but rendered raw it read as "Accruing ($-450 of $1)".
            return f"Ahead of plan (repaying ${abs(state.accrued):.0f})"
        if not ready:
            return f"Accruing (${state.accrued:.0f} of ${floor_dollars:.0f})"
        if not trigger["fires"]:
            return str(trigger.get("reason") or "Waiting")
        return "Ready"

    # ----------------------------------------------------------------------------------
    # Step 2
    # ----------------------------------------------------------------------------------

    def refine(
        self,
        intents: list[Intent],
        signals: dict[str, dict[str, Any]],
        snapshot: PortfolioSnapshot,
        latest_prices: dict[str, float],
        config: Any,
        as_of: datetime,
    ) -> list[Intent]:
        """Never sell more than is held: a DCA sell trims a position, it does not open a short."""
        refined: list[Intent] = []
        for intent in intents:
            value = intent.value
            price = float(latest_prices.get(intent.symbol, 0.0) or 0.0)
            if value < 0 and price > 0:
                held_dollars = float(snapshot.positions.get(intent.symbol, 0.0)) * price
                value = -min(abs(value), held_dollars)
            if abs(value) > 0:
                refined.append(replace(intent, value=value))
        return refined

    def settle(
        self,
        config: Any,
        order_results: list[dict[str, Any]],
        intents: list[Intent],
    ) -> None:
        """Draw down the accrued budget by what actually reached the market.

        Deducting the *filled* notional rather than the intent keeps the remainder -- the part
        that rounded away below one share -- accrued for the next run instead of silently
        vanishing, which is what makes a whole-share brokerage eventually trade.
        """
        account_id = getattr(config, "account_id", "") or ""
        state = load_accrual_state(self.algorithm_id, account_id)
        changed = False
        for order in order_results:
            if order.get("status") != "submitted":
                continue
            symbol = str(order.get("symbol", "")).upper()
            deployed = abs(float(order.get("quantity") or 0.0)) * float(order.get("latest_price") or 0.0)
            if symbol not in state or deployed <= 0:
                continue
            symbol_state = state[symbol]
            state[symbol] = replace(
                symbol_state,
                accrued=symbol_state.accrued - deployed,
                deployed_this_month=symbol_state.deployed_this_month + deployed,
            )
            changed = True
        if changed:
            save_accrual_state(self.algorithm_id, account_id, state)


#: Kept so existing configs and saved state that name ``DCABot`` keep resolving.
DCABot = DCAAlgorithm
