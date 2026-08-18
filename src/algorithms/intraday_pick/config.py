"""Tuning for the intraday options-pick algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ...common.config_utils import load_tuning, tuning_section


@dataclass(frozen=True)
class IntradayPickConfig:
    # ── macro trend ──────────────────────────────────────────────────
    benchmark_symbol: str = "SPY"
    trend_ma_period: int = 50
    trend_lookback_days: int = 5
    trend_min_return: float = 0.0

    # ── candidate universe ──────────────────────────────────────────
    candidate_symbols: list[str] = field(default_factory=list)  # empty = all tradable universe
    force_option_type: str = ""  # "call" | "put" | "" = auto from macro trend

    # ── option selection ─────────────────────────────────────────────
    option_dte_min: int = 0
    option_dte_max: int = 2
    call_delta_min: float = 0.35
    call_delta_max: float = 0.55
    put_delta_min: float = -0.55
    put_delta_max: float = -0.35

    # ── volatility / range regime ────────────────────────────────────
    vol_short_window: int = 5
    vol_long_window: int = 20
    vol_regime_threshold: float = 1.0
    range_expansion_threshold: float = 1.0
    atr_period: int = 14

    # ── intraday signals ────────────────────────────────────────────
    intraday_bar_minutes: int = 15
    intraday_momentum_bars: int = 4
    intraday_range_budget_pct: float = 0.5
    vwap_lookback_bars: int = 0  # 0 = use session VWAP from bars

    # ── entry / exit pricing ─────────────────────────────────────────
    entry_discount_pct: float = 0.05
    exit_target_pct: float = 0.30
    iv_estimate_window: int = 20

    # ── position sizing ─────────────────────────────────────────────
    contracts_per_signal: int = 1
    max_notional_per_trade: float = 1000.0

    # ── selection ────────────────────────────────────────────────────
    top_n_candidates: int = 1
    min_volume_ratio: float = 1.0
    min_price: float = 10.0
    max_price: float = 500.0

    @classmethod
    def from_runtime_config(cls, config: Any) -> IntradayPickConfig:
        return load_tuning(cls, tuning_section(config, "intraday_pick"))

    @property
    def required_daily_bars(self) -> int:
        return max(self.trend_ma_period, self.vol_long_window, self.atr_period, self.iv_estimate_window) + 10

    @property
    def required_intraday_bars(self) -> int:
        return max(self.intraday_momentum_bars, self.vwap_lookback_bars, 10) + 5
