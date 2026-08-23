"""Bursty DCA: accrue a monthly budget per symbol, deploy it sized by two multiplied factors.

  z    = (moving_average - price) / stdev        signed: positive is cheap, negative is dear
  size = budget x conviction(z) x willingness(backlog)

``conviction`` is the valuation half -- how good is this price, in standard deviations of
dislocation from the moving average, signed against the direction the plan wants to trade.
``willingness`` is the budget half -- a symbol that has already spent months ahead of its plan
resists spending more, and one sitting on unspent budget pushes to deploy it.

They multiply rather than gate each other. That is the whole design: a three-sigma dislocation
against heavy resistance still deploys about nine-tenths of a month's budget, while a mediocre
signal at the same backlog deploys almost nothing -- and neither case needed a rule of its own.

One bound sits over the top of both. ``spending_allowance`` caps a run at the balance the
symbol has actually banked, plus an overdraft that ``conviction`` earns. Without it the model
is not merely loose but wrong in kind: ``willingness`` is a multiplier on a *monthly* budget
applied once per *run* and it never reaches zero, so an hourly cadence would spend its floor
value ~141 times a month against a daily one's ~31, and cadence -- not the plan -- would set
the spend rate. ``max_monthly_multiple`` cannot stand in for it, because a ceiling that resets
every month holds any cadence to three times the plan rate rather than to the plan rate.

When it runs is the binding's cron expression, and nothing here reads a clock to second-guess
it: whichever runs fire, both directions trade on all of them.

Every amount in a plan means **dollars per month, per symbol**. The budget is deliberately
*not* divided by run count: at an hourly cadence that is ~141 runs a month, so a $100 budget
would yield $0.71 per run -- below every broker minimum -- and the divisor would change whenever
the schedule was edited. Accruing against elapsed wall-clock time instead leaves cadence
controlling only the *opportunity* to act, never the spend rate, and lets missed runs
self-correct because the next run observes a longer interval.

It emits ``notional`` intents in ``incremental`` mode and shares the order path with every other
algorithm. One thing is still peculiar to it: its memory depends on what actually *filled*,
which is why it is the only algorithm that overrides ``state_after``.
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
from .config import BurstyConfig, plan_budgets, raw_plan, sanitize_plan
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

    The first run only seeds the clock: budget is never accrued retroactively for time before
    the plan was known. A month boundary resets the cumulative deployment cap.
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

    Arithmetic, not policy. There is no configured trade minimum to consult: the only reason a
    small order cannot be sent is that ``round_shares`` would truncate it to nothing, so the
    floor is derived from the rounding rule that would do the truncating. On a whole-share
    brokerage that is one share, which is why a $100/month budget against a $500 ETF still has
    to accrue for months before it can trade at all.
    """
    price = max(float(price), 0.0)
    return price / 10**FRACTIONAL_SHARE_PRECISION if supports_fractional_shares else price


# --------------------------------------------------------------------------------------
# Sizing and timing.
# --------------------------------------------------------------------------------------


def evaluate_valuation(bars: pd.DataFrame | None, settings: BurstyConfig) -> dict[str, Any]:
    """How dislocated the last close is from its moving average, in standard deviations.

    Signed and two-sided on purpose. The predecessor measured drawdown from a running
    ``cummax``, which could only ever report "cheap" or "neutral": it had no way to say a
    symbol was expensive, so the model could scale a buy up on a dip but never down on a
    melt-up, and a sell fired only at an exact high-water mark -- a handful of days a year.

    Dividing by the deviation rather than by the mean is what makes ``scaling_factor`` a single
    number that transfers across a plan holding both a bond ETF and a single name.
    """
    closes = pd.to_numeric(bars["close"], errors="coerce").dropna() if _has_closes(bars) else pd.Series(dtype=float)
    window = max(int(settings.regime_ma_days), 2)
    # A symbol with less history than the window is measured against everything it has, down to
    # ``MIN_VALUATION_BARS``. The average is then shorter than configured, which is the honest
    # answer for a recent listing -- refusing outright would leave it accruing forever.
    minimum = min(window, MIN_VALUATION_BARS)
    if len(closes) < minimum:
        return {"ok": False, "reason": "No price history", "detail": {}}

    close = float(closes.iloc[-1])
    moving_average = float(closes.rolling(window, min_periods=minimum).mean().iloc[-1])
    sigma = float(closes.rolling(window, min_periods=minimum).std().iloc[-1])
    if not math.isfinite(moving_average) or not math.isfinite(sigma):
        return {"ok": False, "reason": "No price history", "detail": {}}

    # A series with no deviation has nothing to divide by. That is a neutral reading rather
    # than a refusal: a price that has not moved is not dislocated, so the honest answer is to
    # size at the plan rate -- which is exactly what straight DCA would have done. Refusing
    # instead would leave the symbol accruing forever over a data quirk, and a genuinely dead
    # feed is already caught upstream by quote staleness.
    z = 0.0 if sigma <= 0 else max(-MAX_SIGMA, min(MAX_SIGMA, (moving_average - close) / sigma))
    return {
        "ok": True,
        "reason": _valuation_reason(z),
        "detail": {
            "close": close,
            "moving_average": round(moving_average, 4),
            "sigma": round(sigma, 4),
            "z": round(z, 3),
            # The percent form too: sigma is the honest unit for sizing, but "8% below its
            # average" is the one a reader can sanity-check against a chart.
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

    ``backlog_months`` is accrued budget over monthly budget -- positive when the symbol is
    sitting on money it has not deployed, negative when previous runs deployed more than had
    accrued and it is repaying. ``tanh`` gives a curve that is near-linear around zero, where
    almost all the time is spent, and saturates smoothly rather than at a threshold, so nothing
    changes character in one step and the factor stays bounded however extreme the backlog.
    """
    width = max(float(settings.relax_months), 1e-9)
    return 1.0 + settings.relax_depth * math.tanh(float(backlog_months) / width)


def spending_allowance(
    monthly_budget: float, backlog_months: float, conviction_factor: float, settings: BurstyConfig
) -> float:
    """The most this symbol may deploy on one run: everything banked, plus an earned overdraft.

    A bound of *some* kind here is not optional. ``willingness`` shapes how eagerly the budget
    is spent, but it is a multiplier on a monthly budget applied once *per run* and it never
    reaches zero -- so on its own an hourly cadence spends its floor value ~141 times a month
    against a daily one's ~31, and the two disagree by more than 4x. Cadence would then control
    the spend rate, which is the single thing the whole accrual design exists to prevent.

    Bounding against the accrued balance closes that: over any long horizon a symbol cannot
    spend faster than it earns, whatever the cadence, because the balance is drawn down by what
    actually filled.

    The overdraft is what keeps this a resistance rather than the hard ``accrued >= floor`` gate
    it replaced, and it is scaled by ``conviction`` rather than fixed. That is the deliberate
    part: how far a symbol may run ahead of its plan should be bought by the quality of the
    price, so an exceptional dislocation can borrow several months forward while an ordinary
    one at the same backlog can borrow nothing. A fixed overdraft cannot express that -- it
    clamps both to the same number and erases the distinction precisely where it matters most.
    Long-run conservation survives because ``conviction`` is itself bounded by ``MAX_SIGMA``,
    so the overdraft is a constant ceiling rather than a growing one.
    """
    budget = abs(float(monthly_budget))
    overdraft = max(settings.relax_months, 0.0) * max(float(conviction_factor), 0.0)
    return max(budget * (float(backlog_months) + overdraft), 0.0)


def monthly_cap(monthly_budget: float, settings: BurstyConfig) -> float:
    """Most this symbol may deploy this month, in dollars.

    A flat multiple of the budget now. It used to scale with drawdown via ``cap_boost``, which
    was a second, coarser way of saying what ``conviction`` already says per order -- and the
    two could disagree, since one read drawdown from peak and the other from the same peak but
    at a different moment in the run.
    """
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
    ``plan`` and the signal rows, so the dashboard previews exactly what a run would order
    rather than a parallel approximation of it.
    """
    budget = abs(float(monthly_budget))
    if budget <= 0:
        return 0.0
    conviction_factor = conviction(z, buying, settings)
    desired = budget * conviction_factor * willingness(backlog_months, settings)
    remaining = max(monthly_cap(budget, settings) - max(float(deployed_this_month), 0.0), 0.0)
    allowance = spending_allowance(budget, backlog_months, conviction_factor, settings)
    sized = min(desired, remaining, allowance)

    # A budget smaller than one share can never *size* its way to a sendable order: $100/month
    # against a $500 share would need the factors to reach 5x, and ``willingness`` saturates
    # below 2x by design. Accrual is what closes that gap -- once the balance the symbol has
    # actually banked covers a share, send exactly one rather than accruing forever while the
    # deck reports an order about to happen.
    #
    # Deliberately applied *after* the monthly cap, and therefore able to exceed it. A cap
    # stated in multiples of the budget cannot express a position whose smallest tradable unit
    # is five months of that budget: clamping here would forbid the only order the symbol will
    # ever be able to place, which is how this reintroduces the accrues-forever bug through a
    # second door. Letting it through is safe because ``accrued`` is the real governor -- the
    # balance falls by the whole fill, so the next lift cannot come until it is re-earned at
    # exactly the plan rate.
    #
    # Skipped when ``conviction`` zeroed the order, because a bucket that wants none of its
    # budget wants none of it at any size, however much has accrued.
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

    #: The budgets are this algorithm's real configuration, and a list of dollar amounts per
    #: symbol is not a parameter form. Declaring the editor here rather than listing the
    #: algorithm on the dashboard is what keeps the frontend from carrying its own idea of
    #: which algorithms are "DCA ones".
    tune_editor = "budgets"

    #: No floors of any kind, and nothing here reads the account's. ``rebalance_threshold`` is
    #: a target-drift concept: applied here it would suppress a small DCA buy that is exactly
    #: what the plan asked for. A dollar minimum is no better -- the only reason an order cannot
    #: be sent is that it rounds away to no shares, which :func:`min_executable` derives from the
    #: rounding rule itself rather than from a number someone has to pick.
    min_trade_dollars = 0.0
    rebalance_threshold = 0.0

    #: One run a day at 11:00 market time, which is where spreads are tightest.
    #:
    #: Both directions trade on whichever runs the cron fires. This used to split them -- buys
    #: on an 11:00 bucket, sells on a 15:00 one, decided inside ``plan`` by reading the hour off
    #: the timestamp. That was a second schedule living underneath the first, and once a binding
    #: states its own cron the two disagree silently: a cron naming any single time made one
    #: side of the plan permanently unreachable, with nothing on the deck to say so.
    cron = "0 11 * * 1-5"

    def budget_plan(self, config: Any) -> dict[str, Any]:
        """The per-symbol monthly budgets, filtered to what is actually tradable."""
        from ...data.universe import tradable_symbols

        return sanitize_plan(raw_plan(config, self.algorithm_id), tradable_symbols(config))

    def config_fingerprint(self, config: Any) -> dict[str, Any]:
        """The plan is this algorithm's real configuration, so it belongs in the fingerprint.

        Editing a bucket amount changes every future decision, and a cached backtest computed
        under the old plan describes a strategy that no longer exists.
        """
        return {**super().config_fingerprint(config), "plan": self.budget_plan(config)}

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        settings = self.tuning(config)
        return AlgorithmRequirements(
            price_symbols=sorted(plan_budgets(self.budget_plan(config))),
            daily_lookback_days=settings.required_daily_bars,
            daily_ma_days=settings.regime_ma_days,
            # The accrued budget per symbol: what makes this algorithm's spend rate a statement
            # about elapsed wall-clock time rather than about run count.
            needs_state=True,
        )

    def plan(self, context: AlgorithmContext) -> AlgorithmPlan:
        """Accrue each symbol's budget and emit an intent for whatever can trade now.

        Accrual is a pure function of elapsed time, so running this twice in quick succession
        accrues the same total as running it once -- and because the result is only written once
        orders go out, a dashboard preview can neither inflate nor lose budget.

        Sizing rides along in the same pass: size = budget x conviction x willingness, clamped
        for a sell to what is actually held, so it trims a position and never shorts.
        """
        config = context.config
        settings = self.tuning(config)
        budgets = plan_budgets(self.budget_plan(config))
        fractional = broker_supports_fractional_shares(getattr(config, "account_id", "") or "")

        # A naive timestamp is read as UTC rather than replaced by the wall clock. Substituting
        # "now" for a historical bar is how a replay ends up accruing a whole backtest's budget
        # against the moment it was run.
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
            # a symbol that deployed ahead of its plan and is repaying. The input to
            # ``willingness``, and the number the deck reports as the backlog.
            backlog_months = symbol_state.accrued / abs(monthly_budget) if monthly_budget else 0.0
            valuation = evaluate_valuation(context.daily_bars_by_symbol.get(symbol), settings)
            z = float(valuation.get("detail", {}).get("z") or 0.0)

            # What a run would order for this symbol on its own terms, before the gates. The
            # dashboard shows it as "next order", so it is computed once here rather than
            # re-derived by the view from a different set of numbers.
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

            # Everything left here is a hard fact rather than a preference: whether there is
            # data to size against and whether the order clears share rounding. The preferences
            # all live in ``size``, and *when* to run is the binding's cron, not this method's.
            deploys = bool(valuation["ok"]) and price > 0 and size >= floor_dollars
            if deploys:
                intents.append(Intent(symbol=symbol, kind="notional", value=size if buying else -size))

            rows.append({
                "symbol": symbol,
                "buying": buying,
                # Shares actually held, so the deck can say *why* a sell row cannot act: a
                # sell budget trims a position and never opens a short, which makes this the
                # one condition that decides a sell row on its own.
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
                # Provenance of the prices these rows were sized at, so a stored-bar fallback
                # never masquerades as a live print on the dashboard.
                "price_quotes": dict(context.extra.get("price_quotes") or {}),
                "scaling_factor": settings.scaling_factor,
                "max_monthly_multiple": settings.max_monthly_multiple,
                "relax_months": settings.relax_months,
                # So the deck can label the average by the window it was taken over rather
                # than calling every one of them "Average".
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

        The one thing this algorithm cannot decide until after the fact. Deducting the *filled*
        notional rather than the intent keeps the remainder -- the part that rounded away below
        one share -- accrued for the next run instead of silently vanishing, which is what makes
        a whole-share brokerage eventually trade. A run that proposes but never submits -- a
        rejected order, a batch no funding could cover -- therefore keeps its budget rather than
        spending it on nothing.
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
