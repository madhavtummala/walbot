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
from datetime import datetime, timezone
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


def evaluate_trigger(bars: pd.DataFrame | None, buying: bool, settings: BurstyConfig) -> dict[str, Any]:
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
    detail = {
        "close": close,
        "ma_200": round(moving_average, 4),
        "percent_b": None if pd.isna(percent_b) else round(percent_b, 4),
        "rsi_2": None if pd.isna(rsi) else round(rsi, 2),
    }

    if buying:
        if close < moving_average:
            return {"fires": False, "reason": f"Below {settings.regime_ma_days}-day MA", "detail": detail}
        oversold = (not pd.isna(percent_b) and percent_b < settings.percent_b_threshold) or (
            not pd.isna(rsi) and rsi < settings.rsi_threshold
        )
        if not oversold:
            return {"fires": False, "reason": "Waiting for valley", "detail": detail}
        return {"fires": True, "reason": "Valley", "detail": detail}

    # Sell side is the mirror: trim into a peak rather than accumulate into a decline.
    overbought = (not pd.isna(percent_b) and percent_b > (1.0 - settings.percent_b_threshold)) or (
        not pd.isna(rsi) and rsi > (100.0 - settings.rsi_threshold)
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

    def trigger(self, symbol: str, context: AlgorithmContext, plan: dict[str, Any]) -> dict[str, Any]:
        settings = BurstyConfig.from_runtime_config(context.config)
        buying = plan_budgets(plan).get(symbol, 0.0) >= 0
        return evaluate_trigger(context.bars_by_symbol.get(symbol), buying, settings)

    def refine(
        self,
        intents: list[Intent],
        signals: dict[str, dict[str, Any]],
        snapshot: PortfolioSnapshot,
        latest_prices: dict[str, float],
        config: Any,
    ) -> list[Intent]:
        """Size the trade by value averaging, then apply both guards.

        Value averaging needs the position's current value, which step 1 cannot see, so this
        is the first point where the trade size can be known at all.
        """
        settings = BurstyConfig.from_runtime_config(config)
        if not settings.value_averaging:
            return super().refine(intents, signals, snapshot, latest_prices, config)

        account_id = getattr(config, "account_id", "") or ""
        state = load_accrual_state(self.algorithm_id, account_id)
        now = datetime.now(timezone.utc)

        refined: list[Intent] = []
        for intent in intents:
            price = float(latest_prices.get(intent.symbol, 0.0) or 0.0)
            if price <= 0:
                continue
            budget = abs(float(signals.get(intent.symbol, {}).get("monthly_budget") or 0.0))
            if budget <= 0:
                continue
            symbol_state = state.get(intent.symbol, SymbolState())

            elapsed_months = path_months(symbol_state, now)
            path_value = budget * elapsed_months
            held_value = float(snapshot.positions.get(intent.symbol, 0.0)) * price
            gap = path_value - held_value

            buying = intent.value >= 0
            desired = gap if buying else -gap
            if desired <= 0:
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
