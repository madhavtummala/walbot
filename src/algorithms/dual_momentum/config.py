"""Tuning for the dual-momentum algorithm.
"""

from __future__ import annotations



import logging
from dataclasses import dataclass, field
from typing import Any


from ..base import minutes_knob

logger = logging.getLogger(__name__)

STATE_KEY = "dual_momentum_runtime"

EPSILON = 1e-9

#: Trading days per year, for annualising a daily volatility estimate.
TRADING_DAYS = 252


@dataclass(frozen=True)
class DualMomentumConfig:
    """Every knob, with the spec's starting values.

    These are research defaults, not recommended live settings: they need converting to your
    bar frequency and evaluating post-cost, walk-forward, before any of them means anything.
    """

    # -- universes ------------------------------------------------------------------------
    risk_on_universe: list[str] = field(default_factory=lambda: ["QQQM", "VTI", "IWM", "IEMG", "XSD"])
    defensive_universe: list[str] = field(default_factory=lambda: ["BIL", "IEF", "AGG", "GLD"])
    benchmark: str = "QQQM"

    # -- cadence --------------------------------------------------------------------------
    signal_refresh_minutes: int = 60
    risk_refresh_minutes: int = 15

    # -- selection score ------------------------------------------------------------------
    #: Preferred bar resolution, in minutes. Fidelity only: the horizons below are stated in
    #: minutes the market is open, so a finer grid resolves them more precisely and a coarser one
    #: still answers them. 0 takes whatever the feed is configured to prefer.
    intraday_bar_minutes: int = 0
    #: Selection horizons, in minutes. Roughly: a quarter-session, a half-session, three
    #: sessions, and twelve. Twelve sessions is longer than a fresh intraday cache reaches,
    #: which is why the store blends daily bars into the far end of the window.
    selection_horizon_nano_minutes: int = 60
    selection_horizon_micro_minutes: int = 240
    selection_horizon_meso_minutes: int = 1200
    selection_horizon_macro_minutes: int = 4800
    w_nano: float = 0.10
    w_micro: float = 0.20
    w_meso: float = 0.35
    w_macro: float = 0.35
    robust_zscore: bool = True
    #: Rank on return-per-unit-of-volatility rather than raw return. Off, the cross-section
    #: rewards amplitude and the highest-volatility theme that happened to rise wins.
    #:
    #: Defaults to off because that is what the measurements said: over 6M/4M replay windows
    #: raw ranking earned ~3.4pp more, at ~1pp more drawdown. The risk-adjusted variant won
    #: the choppy most-recent quarter and always ran lower volatility, so this is a genuine
    #: trade rather than a settled question -- which is why it is a dashboard knob.
    risk_adjusted_score: bool = False
    score_ema_minutes: int = 45

    # -- market regime gate ---------------------------------------------------------------
    benchmark_ma_days: int = 100
    benchmark_return_days: int = 60
    breadth_ma_days: int = 100
    breadth_min: float = 0.50
    regime_confirm_bars: int = 2
    regime_exit_confirm_bars: int = 2

    # -- per-ETF absolute eligibility -----------------------------------------------------
    etf_ma_days: int = 100
    etf_abs_return_days: int = 60
    etf_min_abs_return: float = 0.0
    etf_fast_return_days: int = 20
    etf_min_fast_return: float = -0.02

    # -- ranking and hysteresis -----------------------------------------------------------
    #: 5 rather than the spec's 3: measured across 6M/4M/3M replay windows, five holdings
    #: added roughly 5pp of return with no increase in drawdown. Momentum concentration
    #: sounds decisive but a single wrong leader dominates a 3-name book.
    max_positions: int = 5
    min_base_score: float = 0.25
    entry_rank_max: int = 5
    exit_rank_max: int = 7
    min_score_delta_to_replace: float = 0.35
    cooldown_after_exit: int = 4

    # -- entry timing ---------------------------------------------------------------------
    momentum_change_ema_minutes: int = 45
    momentum_change_enter: float = 0.0
    pullback_macro_z_min: float = 0.0
    pullback_meso_z_min: float = 0.50
    pullback_nano_z_max: float = -0.75
    pullback_micro_return_min: float = 0.0

    # -- sentiment (phased in last; both weights default to off) --------------------------
    sentiment_weight: float = 0.0
    sentiment_size_scale: float = 0.0
    sentiment_clip: float = 2.0
    sentiment_lookback_minutes: int = 120

    # -- sizing and risk ------------------------------------------------------------------
    name_weight_max: float = 0.35
    risk_on_gross_max: float = 1.0
    #: How much volatility should move a position's size, as an exponent: weight is
    #: proportional to score x sigma ** volatility_tilt.
    #:
    #:   -1.0  risk parity -- divide by volatility, so calm names get the big positions
    #:    0.0  score alone -- volatility does not enter sizing at all
    #:   +1.0  lean in -- scale up with volatility, which is what an ungated momentum book
    #:         like Fast Momentum does implicitly by never dividing
    #:
    #: One number rather than a boolean because the useful settings are not binary: the
    #: question is how hard to press, and the answer is a market regime opinion.
    #: +1.0 measured best across 6M/4M/3M at full coverage, and unusually it improved return
    #: *and* drawdown together (19.4% / -8.0% against 17.0% / -10.8% at zero). Portfolio
    #: volatility barely moved, so it is selecting better rather than simply betting bigger.
    #: Expect that to invert in a sharp reversal -- this presses on the wildest names.
    volatility_tilt: float = 1.0
    #: A crash brake, not a governor. The spec's 12% target was written for a diversified
    #: book; against a 23-58% volatility ETF universe it would scale the portfolio to 28-55%
    #: invested *permanently*, which is a large structural drag on a strategy whose whole
    #: job is to be in the market when the market is working. At 0.30 with a 1.3 trigger,
    #: scaling engages only above ~39% ex-ante volatility.
    target_portfolio_vol: float = 0.30
    vol_estimation_days: int = 20
    vol_scale_floor: float = 0.25
    high_vol_trigger: float = 1.3
    intraday_drawdown_limit: float = -0.015
    rebalance_weight_threshold: float = 0.03
    minimum_trade_notional: float = 100.0
    minimum_trade_nav_fraction: float = 0.005
    defensive_max_positions: int = 2

    @classmethod
    def from_runtime_config(cls, config: Any) -> "DualMomentumConfig":
        raw: dict[str, Any] = {}
        if isinstance(getattr(config, "algorithm_configs", None), dict):
            raw = config.algorithm_configs.get("dual_momentum", {}) or {}
        if not isinstance(raw, dict):
            raw = {}

        defaults = cls()

        def number(key: str, default: float) -> float:
            try:
                return float(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        def integer(key: str, default: int) -> int:
            try:
                return int(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        def flag(key: str, default: bool) -> bool:
            value = raw.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        def text(key: str, default: str) -> str:
            value = str(raw.get(key, default) or "").strip().upper()
            return value or default

        def symbols(key: str, default: list[str]) -> list[str]:
            value = raw.get(key, default)
            if isinstance(value, str):
                parsed = [item.strip().upper() for item in value.split(",") if item.strip()]
            elif isinstance(value, list):
                parsed = [str(item).strip().upper() for item in value if str(item).strip()]
            else:
                parsed = []
            return parsed or list(default)

        return cls(
            risk_on_universe=symbols("risk_on_universe", defaults.risk_on_universe),
            defensive_universe=symbols("defensive_universe", defaults.defensive_universe),
            benchmark=text("benchmark", defaults.benchmark),
            signal_refresh_minutes=integer("signal_refresh_minutes", defaults.signal_refresh_minutes),
            risk_refresh_minutes=integer("risk_refresh_minutes", defaults.risk_refresh_minutes),
            intraday_bar_minutes=integer("intraday_bar_minutes", defaults.intraday_bar_minutes),
            selection_horizon_nano_minutes=minutes_knob(
                raw, "selection_horizon_nano_minutes", defaults.selection_horizon_nano_minutes,
                legacy_key="selection_horizon_nano",
            ),
            selection_horizon_micro_minutes=minutes_knob(
                raw, "selection_horizon_micro_minutes", defaults.selection_horizon_micro_minutes,
                legacy_key="selection_horizon_micro",
            ),
            selection_horizon_meso_minutes=minutes_knob(
                raw, "selection_horizon_meso_minutes", defaults.selection_horizon_meso_minutes,
                legacy_key="selection_horizon_meso",
            ),
            selection_horizon_macro_minutes=minutes_knob(
                raw, "selection_horizon_macro_minutes", defaults.selection_horizon_macro_minutes,
                legacy_key="selection_horizon_macro",
            ),
            w_nano=number("w_nano", defaults.w_nano),
            w_micro=number("w_micro", defaults.w_micro),
            w_meso=number("w_meso", defaults.w_meso),
            w_macro=number("w_macro", defaults.w_macro),
            robust_zscore=flag("robust_zscore", defaults.robust_zscore),
            risk_adjusted_score=flag("risk_adjusted_score", defaults.risk_adjusted_score),
            score_ema_minutes=minutes_knob(raw, "score_ema_minutes", defaults.score_ema_minutes),
            benchmark_ma_days=integer("benchmark_ma_days", defaults.benchmark_ma_days),
            benchmark_return_days=integer("benchmark_return_days", defaults.benchmark_return_days),
            breadth_ma_days=integer("breadth_ma_days", defaults.breadth_ma_days),
            breadth_min=number("breadth_min", defaults.breadth_min),
            regime_confirm_bars=integer("regime_confirm_bars", defaults.regime_confirm_bars),
            regime_exit_confirm_bars=integer("regime_exit_confirm_bars", defaults.regime_exit_confirm_bars),
            etf_ma_days=integer("etf_ma_days", defaults.etf_ma_days),
            etf_abs_return_days=integer("etf_abs_return_days", defaults.etf_abs_return_days),
            etf_min_abs_return=number("etf_min_abs_return", defaults.etf_min_abs_return),
            etf_fast_return_days=integer("etf_fast_return_days", defaults.etf_fast_return_days),
            etf_min_fast_return=number("etf_min_fast_return", defaults.etf_min_fast_return),
            max_positions=integer("max_positions", defaults.max_positions),
            min_base_score=number("min_base_score", defaults.min_base_score),
            entry_rank_max=integer("entry_rank_max", defaults.entry_rank_max),
            exit_rank_max=integer("exit_rank_max", defaults.exit_rank_max),
            min_score_delta_to_replace=number("min_score_delta_to_replace", defaults.min_score_delta_to_replace),
            cooldown_after_exit=integer("cooldown_after_exit", defaults.cooldown_after_exit),
            momentum_change_ema_minutes=minutes_knob(
                raw, "momentum_change_ema_minutes", defaults.momentum_change_ema_minutes
            ),
            momentum_change_enter=number("momentum_change_enter", defaults.momentum_change_enter),
            pullback_macro_z_min=number("pullback_macro_z_min", defaults.pullback_macro_z_min),
            pullback_meso_z_min=number("pullback_meso_z_min", defaults.pullback_meso_z_min),
            pullback_nano_z_max=number("pullback_nano_z_max", defaults.pullback_nano_z_max),
            pullback_micro_return_min=number("pullback_micro_return_min", defaults.pullback_micro_return_min),
            sentiment_weight=number("sentiment_weight", defaults.sentiment_weight),
            sentiment_size_scale=number("sentiment_size_scale", defaults.sentiment_size_scale),
            sentiment_clip=number("sentiment_clip", defaults.sentiment_clip),
            sentiment_lookback_minutes=integer("sentiment_lookback_minutes", defaults.sentiment_lookback_minutes),
            name_weight_max=number("name_weight_max", defaults.name_weight_max),
            volatility_tilt=number("volatility_tilt", defaults.volatility_tilt),
            risk_on_gross_max=number("risk_on_gross_max", defaults.risk_on_gross_max),
            target_portfolio_vol=number("target_portfolio_vol", defaults.target_portfolio_vol),
            vol_estimation_days=integer("vol_estimation_days", defaults.vol_estimation_days),
            vol_scale_floor=number("vol_scale_floor", defaults.vol_scale_floor),
            high_vol_trigger=number("high_vol_trigger", defaults.high_vol_trigger),
            intraday_drawdown_limit=number("intraday_drawdown_limit", defaults.intraday_drawdown_limit),
            rebalance_weight_threshold=number("rebalance_weight_threshold", defaults.rebalance_weight_threshold),
            minimum_trade_notional=number("minimum_trade_notional", defaults.minimum_trade_notional),
            minimum_trade_nav_fraction=number("minimum_trade_nav_fraction", defaults.minimum_trade_nav_fraction),
            defensive_max_positions=integer("defensive_max_positions", defaults.defensive_max_positions),
        )

    @property
    def symbols(self) -> list[str]:
        """Everything the algorithm needs priced, benchmark included."""
        return sorted(set(self.risk_on_universe) | set(self.defensive_universe) | {self.benchmark})

    @property
    def uses_sentiment(self) -> bool:
        """Sentiment is opt-in, so a baseline run costs no provider calls at all."""
        return abs(self.sentiment_weight) > 0 or abs(self.sentiment_size_scale) > 0

    @property
    def required_history_minutes(self) -> int:
        # The slowest horizon plus the smoothing tail, since the score is EMA'd over time.
        return max(
            self.selection_horizon_nano_minutes,
            self.selection_horizon_micro_minutes,
            self.selection_horizon_meso_minutes,
            self.selection_horizon_macro_minutes,
        ) + max(self.score_ema_minutes, self.momentum_change_ema_minutes)

    @property
    def required_daily_bars(self) -> int:
        return (
            max(
                self.benchmark_ma_days,
                self.breadth_ma_days,
                self.etf_ma_days,
                self.etf_abs_return_days,
                self.vol_estimation_days,
            )
            + 5
        )


# =========================================================================================
# Pure feature maths
