"""Bursty DCA: the same monthly budget as DCA, deployed only when the market cooperates.

The two algorithms accrue identically and differ in exactly one predicate:

===========  ================================================
DCA          executes when ``accrued >= min_executable``
Bursty DCA   ...and a signal fires
===========  ================================================

The signals are established rules rather than a bespoke percentile: a 200-day moving-average
regime gate, Bollinger %B or Connors RSI(2) for timing, and value averaging for sizing. The
regime gate is the component that stops the strategy accumulating into a genuine decline,
which is this design's main failure mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import pandas as pd

from ...core.interfaces import (
    DAILY_AT_OPEN,
    AlgorithmContext,
    AlgorithmRequirements,
    Intent,
    PortfolioSnapshot,
)
from ...data.signals.signals import compute_bollinger_percent_b, compute_rsi
from .accrual import SymbolState, load_accrual_state, path_months
from .bot import DCAAlgorithm, plan_budgets

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BurstyConfig:
    """Tuning for the trigger and the two guards. Both guards are required, not optional."""

    regime_ma_days: int = 200
    percent_b_lookback: int = 20
    percent_b_num_std: float = 2.0
    #: %B below zero means the close is under the lower band.
    percent_b_threshold: float = 0.0
    rsi_lookback: int = 2
    rsi_threshold: float = 10.0
    #: Clamp on a single trade, expressed against the **monthly budget**. Expressing it against
    #: the per-run increment instead makes the position fall permanently behind the value path
    #: while erroring nowhere, which is why it is stated in months here.
    max_trade_multiple: float = 3.0
    #: Cap on cumulative deployment per symbol per month, also in multiples of the budget.
    max_monthly_multiple: float = 3.0
    value_averaging: bool = True
    #: Months of undeployed budget at which the oversold test stops being picky at all.
    #:
    #: Waiting for a dip is the strategy; waiting *forever* is not. In a market that only ever
    #: rises, %B never breaks the lower band and RSI(2) never reaches 10, so the budget accrues
    #: and the eventual entry happens at a higher price than any of the ones declined -- the
    #: opposite of the intent. This relaxes the two oversold thresholds in proportion to how far
    #: behind the budget has fallen, reaching "buy on anything" at this many months.
    #:
    #: The regime gate is deliberately *not* relaxed. That is the component stopping the
    #: strategy accumulating into a genuine decline, which is this design's main failure mode;
    #: loosening it would trade a small problem for the large one.
    #:
    #: 0 disables the whole mechanism, which is the behaviour this had before.
    backlog_relax_months: float = 0.0

    @classmethod
    def from_runtime_config(cls, config: Any) -> BurstyConfig:
        section = (getattr(config, "algorithm_configs", {}) or {}).get("bursty_dca") or {}
        settings: dict[str, Any] = {}
        for name, field in cls.__dataclass_fields__.items():
            if name not in section:
                continue
            try:
                settings[name] = type(field.default)(section[name])
            except (TypeError, ValueError):
                logger.warning("Ignoring unusable bursty_dca.%s: %r", name, section[name])
        return cls(**settings)

    @property
    def required_daily_bars(self) -> int:
        return max(self.regime_ma_days, self.percent_b_lookback, self.rsi_lookback) + 30


def _latest(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    value = series.iloc[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def relaxed_thresholds(settings: BurstyConfig, months_behind: float) -> tuple[float, float]:
    """The oversold thresholds to use, given how much budget is waiting to be deployed.

    Each is interpolated toward the value at which it always fires -- %B at 1.0, RSI at 100 --
    so a symbol that is on schedule keeps the configured pickiness and one that is months behind
    stops declining entries. Linear rather than a step, so there is no month where behaviour
    jumps.
    """
    if settings.backlog_relax_months <= 0:
        return settings.percent_b_threshold, settings.rsi_threshold
    fraction = min(max(months_behind, 0.0) / settings.backlog_relax_months, 1.0)
    return (
        settings.percent_b_threshold + fraction * (1.0 - settings.percent_b_threshold),
        settings.rsi_threshold + fraction * (100.0 - settings.rsi_threshold),
    )


def evaluate_trigger(
    bars: pd.DataFrame | None,
    buying: bool,
    settings: BurstyConfig,
    months_behind: float = 0.0,
) -> dict[str, Any]:
    """Decide whether now is a moment to act on ``bars``, and say why in plain words."""
    if bars is None or bars.empty or "close" not in bars:
        return {"fires": False, "reason": "No price history", "detail": {}}

    closes = pd.to_numeric(bars["close"], errors="coerce").dropna()
    if closes.empty:
        return {"fires": False, "reason": "No price history", "detail": {}}

    close = float(closes.iloc[-1])
    moving_average = float(closes.rolling(settings.regime_ma_days, min_periods=1).mean().iloc[-1])
    percent_b = _latest(
        compute_bollinger_percent_b(
            bars, lookback=settings.percent_b_lookback, num_std=settings.percent_b_num_std
        )
    )
    rsi = _latest(compute_rsi(bars, lookback=settings.rsi_lookback))
    percent_b_threshold, rsi_threshold = relaxed_thresholds(settings, months_behind)
    detail = {
        "close": close,
        "ma_200": round(moving_average, 4),
        "percent_b": None if pd.isna(percent_b) else round(percent_b, 4),
        "rsi_2": None if pd.isna(rsi) else round(rsi, 2),
        "months_behind": round(months_behind, 2),
        "rsi_threshold": round(rsi_threshold, 2),
    }

    if buying:
        if close < moving_average:
            return {"fires": False, "reason": f"Below {settings.regime_ma_days}-day MA", "detail": detail}
        oversold = (not pd.isna(percent_b) and percent_b < percent_b_threshold) or (
            not pd.isna(rsi) and rsi < rsi_threshold
        )
        if not oversold:
            return {"fires": False, "reason": "Waiting for valley", "detail": detail}
        behind = f" (relaxed, {months_behind:.1f}m behind)" if months_behind and settings.backlog_relax_months else ""
        return {"fires": True, "reason": f"Valley{behind}", "detail": detail}

    # Sell side is the mirror: trim into a peak rather than accumulate into a decline.
    overbought = (not pd.isna(percent_b) and percent_b > (1.0 - percent_b_threshold)) or (
        not pd.isna(rsi) and rsi > (100.0 - rsi_threshold)
    )
    if not overbought:
        return {"fires": False, "reason": "Waiting for peak", "detail": detail}
    return {"fires": True, "reason": "Peak", "detail": detail}


class BurstyDCAAlgorithm(DCAAlgorithm):
    algorithm_id = "bursty_dca"

    #: Daily rather than DCA's weekly. The budget is identical either way, but a dip that the
    #: trigger would have fired on can open and close inside a week, so a weekly look would
    #: leave the accrued budget sitting through exactly the entries this variant exists for.
    schedule = DAILY_AT_OPEN

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        settings = BurstyConfig.from_runtime_config(config)
        return AlgorithmRequirements(
            price_symbols=sorted(plan_budgets(self.plan(config))),
            daily_lookback_days=settings.required_daily_bars,
            daily_ma_days=settings.regime_ma_days,
        )

    def trigger(
        self,
        symbol: str,
        context: AlgorithmContext,
        plan: dict[str, Any],
        months_behind: float = 0.0,
    ) -> dict[str, Any]:
        settings = BurstyConfig.from_runtime_config(context.config)
        buying = plan_budgets(plan).get(symbol, 0.0) >= 0
        result = evaluate_trigger(context.bars_by_symbol.get(symbol), buying, settings, months_behind)
        # Carried so ``reason`` can tell the dashboard that step 2 still gets a say. See there.
        result.setdefault("detail", {})["value_averaged"] = settings.value_averaging
        return result

    def reason(
        self,
        state: SymbolState,
        floor_dollars: float,
        trigger: dict[str, Any],
        ready: bool,
    ) -> str:
        """Never promise a trade that value averaging can still veto.

        The signal view is built from ``analyze``, which is step 1; the value-averaging sizing
        below happens in ``refine``, which is step 2 and is the first point that can see the
        position. A symbol already at or above its value path is dropped there, so a bare
        "Ready" was telling the reader a trade was coming that would never be placed.
        """
        text = super().reason(state, floor_dollars, trigger, ready)
        if text == "Ready" and trigger.get("detail", {}).get("value_averaged"):
            return "Ready (sized to value path)"
        return text

    def refine(
        self,
        intents: list[Intent],
        signals: dict[str, dict[str, Any]],
        snapshot: PortfolioSnapshot,
        latest_prices: dict[str, float],
        config: Any,
        as_of: datetime,
    ) -> list[Intent]:
        """Size the trade by value averaging, then apply both guards.

        Value averaging needs the position's current value, which step 1 cannot see, so this
        is the first point where the trade size can be known at all.

        How far along the value path a symbol should be is a question about elapsed time, so it
        is measured from ``as_of`` rather than from the wall clock. Reading the clock here put
        the path wherever the machine happened to be -- months out, in a replay.
        """
        settings = BurstyConfig.from_runtime_config(config)
        if not settings.value_averaging:
            return super().refine(intents, signals, snapshot, latest_prices, config, as_of)

        account_id = getattr(config, "account_id", "") or ""
        state = load_accrual_state(self.algorithm_id, account_id)

        refined: list[Intent] = []
        for intent in intents:
            price = float(latest_prices.get(intent.symbol, 0.0) or 0.0)
            if price <= 0:
                continue
            budget = abs(float(signals.get(intent.symbol, {}).get("monthly_budget") or 0.0))
            if budget <= 0:
                continue
            symbol_state = state.get(intent.symbol, SymbolState())

            elapsed_months = path_months(symbol_state, as_of)
            path_value = budget * elapsed_months
            held_value = float(snapshot.positions.get(intent.symbol, 0.0)) * price
            gap = path_value - held_value

            buying = intent.value >= 0
            desired = gap if buying else -gap
            if desired <= 0:
                # Not an error: the position is already at or past where the value path says it
                # should be. Logged because it is otherwise invisible -- step 1 reported the
                # symbol as ready and no order appears, with nothing in between to explain it.
                logger.info(
                    "%s: value path satisfied, no trade. path=$%.2f held=$%.2f months=%.2f",
                    intent.symbol,
                    path_value,
                    held_value,
                    elapsed_months,
                )
                continue

            # Guard 1: clamp a single trade against the monthly budget.
            desired = min(desired, settings.max_trade_multiple * budget)
            # Guard 2: cap cumulative deployment per symbol per month.
            remaining = max((settings.max_monthly_multiple * budget) - symbol_state.deployed_this_month, 0.0)
            desired = min(desired, remaining)
            if not buying:
                desired = min(desired, held_value)
            if desired <= 0:
                continue

            refined.append(replace(intent, value=desired if buying else -desired))

        return refined
