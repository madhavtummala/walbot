"""Bursty DCA: accrue a monthly budget per symbol, deploy it sized by two multiplied factors.

  z    = (moving_average - price) / stdev        signed: positive is cheap, negative is dear
  size = budget x conviction(z) x willingness(backlog)

``conviction`` is the valuation half -- how good is this price, signed against the direction
the plan wants to trade. ``willingness`` is the budget half -- a symbol ahead of its plan
resists spending more, one sitting on unspent budget pushes to deploy it. They multiply rather
than gate each other, so an exceptional dislocation against heavy resistance still deploys
most of a month's budget, while a mediocre signal at the same backlog deploys almost nothing.

``spending_allowance`` bounds both together: a run may deploy at most the balance the symbol has
actually banked, plus an overdraft ``conviction`` earns. Without it, ``willingness`` (a
multiplier on a *monthly* budget applied once per *run*, never reaching zero) would let cadence
rather than the plan set the spend rate.

Every amount in a plan means dollars per month, per symbol, accrued against elapsed wall-clock
time rather than divided by run count -- so cadence controls only the *opportunity* to act, and
a missed run self-corrects on the next one's longer interval.

Emits ``notional`` intents in ``incremental`` mode. Its memory depends on what actually
*filled*, which is why it's the only algorithm overriding ``state_after``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ...common.config_utils import as_float
from ...common.timeutils import parse_iso_utc
from ...core.orders import FRACTIONAL_SHARE_PRECISION
from ...core.interfaces import (
    MODE_INCREMENTAL,
    AlgorithmContext,
    AlgorithmPlan,
    AlgorithmRequirements,
    Intent,
    SignalView,
)
from ..base import BaseAlgorithm
from ...common.config_utils import raw_plan
from .config import BurstyConfig, plan_budgets, sanitize_plan
from .signals import signal_rows, signal_view

logger = logging.getLogger(__name__)

#: Average hours in a calendar month (365.25 * 24 / 12).
HOURS_IN_MONTH = 730.5

#: Ceiling on how much budget one gap between runs may accrue. Without it, a bot that was off
#: for half a year would come back and try to deploy half a year of budget in one session.
MAX_CATCHUP_MONTHS = 1.0

#: Standard deviations of dislocation past which the valuation factor stops growing. A move
#: this far from the mean is far more often a bad split adjustment or a stale bar than a real
#: opportunity, and without a clamp ``scaling_factor`` would happily size against it.
MAX_SIGMA = 3.0

#: Fewest closes that will produce a mean and a deviation worth sizing against, when a symbol
#: has less history than ``regime_ma_days``. Demanding the whole window instead would refuse to
#: trade a recent listing indefinitely, and would silently refuse every backtest short enough
#: that its replay window is smaller than the average it is measuring against.
MIN_VALUATION_BARS = 30


# --------------------------------------------------------------------------------------
# Accrual: the memory this algorithm carries between runs.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolState:
    """One symbol's accrued budget, carried on the context and returned on the plan."""

    accrued: float = 0.0
    last_run_at: str = ""
    deployed_this_month: float = 0.0
    month: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accrued": round(self.accrued, 6),
            "last_run_at": self.last_run_at,
            "deployed_this_month": round(self.deployed_this_month, 6),
            "month": self.month,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> SymbolState:
        if not isinstance(raw, dict):
            return cls()
        return cls(
            accrued=as_float(raw.get("accrued")),
            last_run_at=str(raw.get("last_run_at") or ""),
            deployed_this_month=as_float(raw.get("deployed_this_month")),
            month=str(raw.get("month") or ""),
        )


def accrual_from_state(state: Any) -> dict[str, SymbolState]:
    """Read the whole book of per-symbol accrual off ``AlgorithmContext.state``."""
    if not isinstance(state, dict):
        return {}
    return {str(symbol).upper(): SymbolState.from_dict(value) for symbol, value in state.items()}


def accrual_to_state(accrual: dict[str, SymbolState]) -> dict[str, Any]:
    """The inverse, for the plan to carry back. Nothing here touches a store."""
    return {symbol: symbol_state.as_dict() for symbol, symbol_state in accrual.items()}


def accrue(state: SymbolState, monthly_budget: float, now: datetime) -> SymbolState:
    """Advance ``state`` to ``now``, adding the budget earned over the elapsed wall-clock time.

    The first run only seeds the clock -- budget is never accrued retroactively. A month
    boundary resets the cumulative deployment cap.
    """
    month = now.strftime("%Y-%m")
    if state.month and state.month != month:
        state = replace(state, deployed_this_month=0.0)
    state = replace(state, month=month)

    previous = parse_iso_utc(state.last_run_at)
    if previous is None:
        return replace(state, last_run_at=now.isoformat())

    elapsed_hours = max((now - previous).total_seconds() / 3600.0, 0.0)
    elapsed_months = min(elapsed_hours / HOURS_IN_MONTH, MAX_CATCHUP_MONTHS)
    return replace(
        state,
        accrued=state.accrued + (max(monthly_budget, 0.0) * elapsed_months),
        last_run_at=now.isoformat(),
    )


def min_executable(price: float, supports_fractional_shares: bool) -> float:
    """Smallest trade that survives share rounding for this symbol on this brokerage.

    Arithmetic, not policy: derived from the rounding rule that would otherwise truncate a small
    order to nothing. On a whole-share brokerage that's one share.
    """
    price = max(float(price), 0.0)
    return price / 10**FRACTIONAL_SHARE_PRECISION if supports_fractional_shares else price


# --------------------------------------------------------------------------------------
# Sizing and timing.
# --------------------------------------------------------------------------------------


def evaluate_valuation(bars: pd.DataFrame | None, settings: BurstyConfig) -> dict[str, Any]:
    """How dislocated the last close is from its moving average, in standard deviations.

    Signed and two-sided, so the model can scale a buy up on a dip and down on a melt-up, and a
    sell isn't limited to firing at an exact high-water mark. Dividing by the deviation rather
    than the mean is what makes ``scaling_factor`` transfer across symbols of different vol.
    """
    closes = pd.to_numeric(bars["close"], errors="coerce").dropna() if _has_closes(bars) else pd.Series(dtype=float)
    window = max(int(settings.regime_ma_days), 2)
    # Less history than the window: measured against everything available, down to
    # MIN_VALUATION_BARS, rather than refused outright (which would accrue forever).
    minimum = min(window, MIN_VALUATION_BARS)
    if len(closes) < minimum:
        return {"ok": False, "reason": "No price history", "detail": {}}

    close = float(closes.iloc[-1])
    moving_average = float(closes.rolling(window, min_periods=minimum).mean().iloc[-1])
    sigma = float(closes.rolling(window, min_periods=minimum).std().iloc[-1])
    if not math.isfinite(moving_average) or not math.isfinite(sigma):
        return {"ok": False, "reason": "No price history", "detail": {}}

    # No deviation to divide by is a neutral reading (size at the plan rate), not a refusal --
    # a price that hasn't moved isn't dislocated.
    z = 0.0 if sigma <= 0 else max(-MAX_SIGMA, min(MAX_SIGMA, (moving_average - close) / sigma))
    return {
        "ok": True,
        "reason": _valuation_reason(z),
        "detail": {
            "close": close,
            "moving_average": round(moving_average, 4),
            "sigma": round(sigma, 4),
            "z": round(z, 3),
            # The percent form too, for sanity-checking against a chart.
            "distance": round((moving_average - close) / moving_average, 4) if moving_average else 0.0,
        },
    }


def _valuation_reason(z: float) -> str:
    if z > 0:
        return f"{z:.1f}σ below its average"
    if z < 0:
        return f"{abs(z):.1f}σ above its average"
    return "At its average"


def _has_closes(bars: pd.DataFrame | None) -> bool:
    return bars is not None and not bars.empty and "close" in bars


def conviction(z: float, buying: bool, settings: BurstyConfig) -> float:
    """The valuation factor, oriented to the direction the plan wants to trade.

    A buy wants price below the average (``z`` positive), a sell wants it above, so the sell
    side reads the same number with the sign flipped. Floored at zero rather than allowed to go
    negative: a buy bucket that looks badly priced should buy *nothing*, never sell.
    """
    favourable = float(z) if buying else -float(z)
    return max(0.0, 1.0 + settings.scaling_factor * favourable)


def willingness(backlog_months: float, settings: BurstyConfig) -> float:
    """The backlog factor: how entitled this symbol is to spend right now.

    ``backlog_months`` is accrued budget over monthly budget -- positive when sitting on
    unspent money, negative when repaying an overdraft. ``tanh`` is near-linear around zero and
    saturates smoothly, so the factor stays bounded however extreme the backlog.
    """
    width = max(float(settings.relax_months), 1e-9)
    return 1.0 + settings.relax_depth * math.tanh(float(backlog_months) / width)


def spending_allowance(
    monthly_budget: float, backlog_months: float, conviction_factor: float, settings: BurstyConfig
) -> float:
    """The most this symbol may deploy on one run: everything banked, plus an earned overdraft.

    Bounding against the accrued balance is what keeps ``willingness`` (a per-run multiplier
    that never reaches zero) from letting cadence set the spend rate -- over any horizon a
    symbol cannot spend faster than it earns, since the balance is drawn down by what filled.

    The overdraft is scaled by ``conviction`` rather than fixed, deliberately: how far a symbol
    may run ahead of plan should be bought by the quality of the price, so an exceptional
    dislocation can borrow several months forward while an ordinary one at the same backlog
    borrows nothing. It stays a constant ceiling since ``conviction`` is bounded by ``MAX_SIGMA``.
    """
    budget = abs(float(monthly_budget))
    overdraft = max(settings.relax_months, 0.0) * max(float(conviction_factor), 0.0)
    return max(budget * (float(backlog_months) + overdraft), 0.0)


def monthly_cap(monthly_budget: float, settings: BurstyConfig) -> float:
    """Most this symbol may deploy this month, in dollars -- a flat multiple of the budget."""
    return abs(float(monthly_budget)) * settings.max_monthly_multiple


def planned_order_size(
    monthly_budget: float,
    z: float,
    buying: bool,
    backlog_months: float,
    deployed_this_month: float,
    settings: BurstyConfig,
    *,
    accrued: float = 0.0,
    floor_dollars: float = 0.0,
) -> float:
    """Dollar size for one symbol, before position and broker clamps.

    ``size = |budget| x conviction x willingness``, lifted to the smallest sendable order when
    the accrued balance covers one, then clamped to the month's remaining cap room. Shared by
    ``plan`` and the signal rows, so the dashboard preview matches exactly what a run would order.
    """
    budget = abs(float(monthly_budget))
    if budget <= 0:
        return 0.0
    conviction_factor = conviction(z, buying, settings)
    desired = budget * conviction_factor * willingness(backlog_months, settings)
    remaining = max(monthly_cap(budget, settings) - max(float(deployed_this_month), 0.0), 0.0)
    allowance = spending_allowance(budget, backlog_months, conviction_factor, settings)
    sized = min(desired, remaining, allowance)

    # A budget smaller than one share can never size its way to a sendable order, since
    # ``willingness`` saturates below the factor a small budget would need. Once the accrued
    # balance covers a share, lift to exactly one rather than accruing forever. Applied
    # *after* the monthly cap deliberately -- a cap in multiples of budget can't express a
    # position whose smallest tradable unit is several months of it, and letting this exceed
    # the cap is safe because ``accrued`` is the real governor: it falls by the whole fill.
    # Skipped when ``conviction`` zeroed the order -- that bucket wants none of its budget.
    if 0.0 < sized < float(floor_dollars) <= float(accrued):
        return float(floor_dollars)
    return sized


def broker_supports_fractional_shares(account_id: str) -> bool:
    """Read the account's capability off the class, without instantiating (and authenticating) it.

    ``BaseBrokerage`` declares the attribute, so an unknown account falls back to whole shares
    -- the conservative reading -- rather than to a missing attribute.
    """
    from ...brokerages.base import BaseBrokerage
    from ...brokerages.registry import get_brokerage_class
    from ...core.config import get_account_broker_type

    try:
        broker_cls = get_brokerage_class(get_account_broker_type(account_id))
    except KeyError:
        return BaseBrokerage.supports_fractional_shares
    return broker_cls.supports_fractional_shares


class BurstyDCAAlgorithm(BaseAlgorithm):
    """Accrue a monthly budget per symbol and deploy it sized by valuation and by backlog."""

    algorithm_id = "bursty_dca"
    tuning_class = BurstyConfig

    #: The budgets are this algorithm's real configuration -- a list of dollar amounts per
    #: symbol is not a parameter form.
    tune_editor = "budgets"

    #: No floors of any kind. ``rebalance_threshold`` is a target-drift concept that would
    #: suppress a small DCA buy exactly as planned; a dollar minimum is no better, since the
    #: only reason an order can't send is share rounding, which :func:`min_executable` derives
    #: directly rather than from a number someone has to pick.
    min_trade_dollars = 0.0
    rebalance_threshold = 0.0

    #: One run a day at 11:00 market time, where spreads are tightest. Both directions trade on
    #: whichever runs the cron fires -- no separate buy/sell schedule underneath it.
    cron = "0 11 * * 1-5"

    def budget_plan(self, config: Any) -> dict[str, Any]:
        """The per-symbol monthly budgets, filtered to what is actually tradable."""
        from ...data.universe import tradable_symbols

        return sanitize_plan(raw_plan(config, self.algorithm_id), tradable_symbols(config))

    def config_fingerprint(self, config: Any) -> dict[str, Any]:
        """The plan is this algorithm's real configuration, so it belongs in the fingerprint --
        editing a bucket amount changes every future decision."""
        return {**super().config_fingerprint(config), "plan": self.budget_plan(config)}

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        settings = self.tuning(config)
        return AlgorithmRequirements(
            price_symbols=sorted(plan_budgets(self.budget_plan(config))),
            daily_lookback_days=settings.required_daily_bars,
            daily_ma_days=settings.regime_ma_days,
            needs_state=True,  # The accrued budget per symbol.
        )

    def plan(self, context: AlgorithmContext) -> AlgorithmPlan:
        """Accrue each symbol's budget and emit an intent for whatever can trade now.

        Accrual is a pure function of elapsed time, so running this twice in quick succession
        accrues the same total as running it once. Sizing rides along in the same pass, clamped
        for a sell to what is actually held so it trims a position and never shorts.
        """
        config = context.config
        settings = self.tuning(config)
        budgets = plan_budgets(self.budget_plan(config))
        fractional = broker_supports_fractional_shares(getattr(config, "account_id", "") or "")

        # A naive timestamp is read as UTC, not the wall clock -- else a replay accrues a whole
        # backtest's budget against the moment it was run.
        now = context.timestamp if context.timestamp.tzinfo else context.timestamp.replace(tzinfo=timezone.utc)

        accrual = accrual_from_state(context.state)
        intents: list[Intent] = []
        rows: list[dict[str, Any]] = []

        for symbol, monthly_budget in sorted(budgets.items()):
            symbol_state = accrue(accrual.get(symbol, SymbolState()), abs(monthly_budget), now)
            accrual[symbol] = symbol_state

            buying = monthly_budget >= 0
            price = float(context.latest_prices.get(symbol, 0.0) or 0.0)
            floor_dollars = min_executable(price, fractional)
            # Months of budget banked, signed: positive is unspent money waiting, negative is
            # repaying an overdraft. Input to ``willingness``.
            backlog_months = symbol_state.accrued / abs(monthly_budget) if monthly_budget else 0.0
            valuation = evaluate_valuation(context.daily_bars_by_symbol.get(symbol), settings)
            z = float(valuation.get("detail", {}).get("z") or 0.0)

            # What a run would order on its own terms, before the gates -- computed once here
            # so the dashboard's "upcoming" figure matches exactly.
            size = planned_order_size(
                monthly_budget,
                z,
                buying,
                backlog_months,
                symbol_state.deployed_this_month,
                settings,
                accrued=symbol_state.accrued,
                floor_dollars=floor_dollars,
            )
            if not buying:
                size = min(size, float(context.positions.get(symbol, 0.0)) * price)

            # Hard facts only: is there data to size against, does the order clear rounding.
            # The preferences all live in ``size``.
            deploys = bool(valuation["ok"]) and price > 0 and size >= floor_dollars
            if deploys:
                intents.append(Intent(symbol=symbol, kind="notional", value=size if buying else -size))

            rows.append({
                "symbol": symbol,
                "buying": buying,
                "held": float(context.positions.get(symbol, 0.0)),
                "price": price,
                "monthly_budget": monthly_budget,
                "state": symbol_state,
                "floor_dollars": floor_dollars,
                "valuation": valuation,
                "backlog_months": backlog_months,
                "conviction": conviction(z, buying, settings),
                "willingness": willingness(backlog_months, settings),
                "size": size,
                "deployed": deploys,
                "fractional": fractional,
                "monthly_cap": monthly_cap(monthly_budget, settings),
            })

        return AlgorithmPlan(
            intents=intents,
            mode=MODE_INCREMENTAL,
            signals=signal_rows(rows),
            metadata={
                "allocation_mode": "DCA",
                "monthly_total": sum(abs(value) for value in budgets.values()),
                "scaling_factor": settings.scaling_factor,
                "max_monthly_multiple": settings.max_monthly_multiple,
                "relax_months": settings.relax_months,
                "regime_ma_days": settings.regime_ma_days,
            },
            state=accrual_to_state(accrual),
        )

    def signal_view(self, plan: AlgorithmPlan) -> SignalView:
        return signal_view(plan, unknown=self._unknown_symbols())

    def _unknown_symbols(self) -> list[str]:
        """Plan symbols sanitisation dropped -- otherwise a typo shows up as nothing at all."""
        from ...data.universe import tradable_symbols
        from .config import unknown_plan_symbols

        return unknown_plan_symbols(raw_plan(self.config, self.algorithm_id), tradable_symbols(self.config))

    def state_after(self, plan: AlgorithmPlan, outcome: dict[str, Any]) -> dict[str, Any]:
        """Draw down the accrued budget by what actually reached the market.

        Deducting the *filled* notional rather than the intent keeps the remainder (the part
        that rounded away below one share) accrued for the next run rather than vanishing. A
        run that proposes but never submits therefore keeps its budget.
        """
        accrual = accrual_from_state(plan.state)
        for order in outcome.get("order_results") or []:
            if order.get("status") != "submitted":
                continue
            symbol = str(order.get("symbol", "")).upper()
            deployed = abs(as_float(order.get("quantity"))) * as_float(order.get("latest_price"))
            if symbol not in accrual or deployed <= 0:
                continue
            symbol_state = accrual[symbol]
            accrual[symbol] = replace(
                symbol_state,
                accrued=symbol_state.accrued - deployed,
                deployed_this_month=symbol_state.deployed_this_month + deployed,
            )
        return accrual_to_state(accrual)
