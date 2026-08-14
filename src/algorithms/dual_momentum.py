"""Dual momentum: relative strength selects, absolute momentum permits.

Forked from :mod:`src.algorithms.fast_momentum`, which ranks a universe cross-sectionally and
holds the top few. That design always owns *something* risk-on -- during a broad drawdown it
buys the least-bad ETF, because a z-score only says "better than its peers", never "good".

Four changes address that:

1. A market regime gate (benchmark trend + breadth, with hysteresis) decides whether risk-on
   exposure is allowed at all. Risk-off allocates to the defensive universe, not to the best
   of a falling one.
2. Per-ETF absolute eligibility: above its own long moving average, positive medium-term
   return, and not collapsing on the short lookback. Names that fail are not ranked at all,
   so fewer than ``max_positions`` qualifying names means holding less, not lowering the bar.
3. Selection is slow (weighted toward meso/macro horizons) and separate from entry timing,
   which is fast. The pullback setup became a timing *flag* rather than a score bonus -- as a
   bonus it could promote a weak ETF into the book on the strength of the dip alone.
4. Weights are score-over-volatility, then scaled by an ex-ante portfolio volatility target
   computed from the full covariance matrix.

Layering, per the spec this implements:

    Layer            Refresh   Where it lives
    Market regime    60m       analyze (raw gate) + refine (hysteresis; needs state)
    Eligibility      60m       analyze
    Ranking          60m       analyze, gated by signal_refresh_minutes
    Entry timing     15m       analyze
    Risk/execution   15m       refine (needs positions and equity)

``analyze`` stays a pure function of its context -- no clock, no state store -- which is what
lets the backtester replay this algorithm rather than reimplement it. Everything stateful
(regime confirmation counters, re-entry cooldowns, the intraday drawdown breaker) lives in
``refine``, and reads run-level facts that ``analyze`` denormalised onto the signal rows.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable

import pandas as pd

from .base import BaseAlgorithm
from ..core.interfaces import AlgorithmContext, AlgorithmDecision, AlgorithmRequirements, Schedule
from ..data.state_store import load_state, save_state

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
    #: The grid every ``selection_horizon_*`` below is counted on. 15 minutes is what those
    #: horizons were fitted against, so changing it rescales all of them in wall-clock terms
    #: (macro 320 is 12.3 sessions at 15m and 4.1 at 5m) -- backtest before moving it.
    intraday_bar_minutes: int = 15
    selection_horizon_nano: int = 4
    selection_horizon_micro: int = 16
    selection_horizon_meso: int = 80
    selection_horizon_macro: int = 320
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
    score_ema_bars: int = 3

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
    momentum_change_ema_bars: int = 3
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
            selection_horizon_nano=integer("selection_horizon_nano", defaults.selection_horizon_nano),
            selection_horizon_micro=integer("selection_horizon_micro", defaults.selection_horizon_micro),
            selection_horizon_meso=integer("selection_horizon_meso", defaults.selection_horizon_meso),
            selection_horizon_macro=integer("selection_horizon_macro", defaults.selection_horizon_macro),
            w_nano=number("w_nano", defaults.w_nano),
            w_micro=number("w_micro", defaults.w_micro),
            w_meso=number("w_meso", defaults.w_meso),
            w_macro=number("w_macro", defaults.w_macro),
            robust_zscore=flag("robust_zscore", defaults.robust_zscore),
            risk_adjusted_score=flag("risk_adjusted_score", defaults.risk_adjusted_score),
            score_ema_bars=integer("score_ema_bars", defaults.score_ema_bars),
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
            momentum_change_ema_bars=integer("momentum_change_ema_bars", defaults.momentum_change_ema_bars),
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
    def required_intraday_bars(self) -> int:
        # The slowest horizon plus the smoothing tail, since the score is EMA'd across bars.
        return (
            max(
                self.selection_horizon_nano,
                self.selection_horizon_micro,
                self.selection_horizon_meso,
                self.selection_horizon_macro,
            )
            + max(self.score_ema_bars, self.momentum_change_ema_bars)
            + 1
        )

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
# =========================================================================================


def _closes(bars: Any) -> pd.Series:
    frame = bars if isinstance(bars, pd.DataFrame) else pd.DataFrame()
    return pd.to_numeric(frame.get("close", pd.Series(dtype=float)), errors="coerce").dropna()


def _return_over(closes: pd.Series, periods: int) -> float:
    """Simple return across ``periods`` observations, 0.0 when the history is too short."""
    if periods <= 0 or len(closes) <= periods:
        return 0.0
    start = float(closes.iloc[-periods - 1])
    end = float(closes.iloc[-1])
    return end / start - 1.0 if start > 0 else 0.0


def _return_series(closes: pd.Series, periods: int, count: int) -> list[float]:
    """The last ``count`` values of the rolling ``periods``-observation return.

    Oldest first. Short history is padded with the oldest available value rather than zero,
    so a thin cache biases the smoothing toward the data that exists instead of toward flat.
    """
    if periods <= 0 or closes.empty:
        return [0.0] * max(count, 1)
    series = closes.pct_change(periods=periods).dropna()
    if series.empty:
        return [0.0] * max(count, 1)
    values = [float(value) for value in series.tail(max(count, 1))]
    while len(values) < max(count, 1):
        values.insert(0, values[0])
    return values


def _ema(values: Iterable[float], span: int) -> float:
    """Last value of an EMA over ``values`` (oldest first)."""
    data = [float(value) for value in values]
    if not data:
        return 0.0
    if span <= 1 or len(data) == 1:
        return data[-1]
    alpha = 2.0 / (span + 1.0)
    result = data[0]
    for value in data[1:]:
        result = (alpha * value) + ((1 - alpha) * result)
    return result


def _rolling_volatility(closes: pd.Series, window: int) -> float:
    """Standard deviation of one-period returns over the trailing ``window``."""
    if window <= 1 or closes.empty:
        return 0.0
    returns = closes.pct_change().dropna().tail(window)
    if returns.empty:
        return 0.0
    value = float(returns.std())
    return 0.0 if math.isnan(value) else value


def compute_features(
    symbol: str,
    intraday_bars: Any,
    daily_bars: Any,
    config: DualMomentumConfig,
) -> dict[str, Any]:
    """Everything about one symbol that the layers below need, computed once.

    Selection horizons are measured in intraday bars; the eligibility and regime gates are
    measured in daily bars, because "above its 100-day average" is a statement about the
    trend, not about the last few hours.
    """
    closes = _closes(intraday_bars)
    daily_closes = _closes(daily_bars)
    smoothing = max(config.score_ema_bars, 1)

    horizons = {
        "nano": config.selection_horizon_nano,
        "micro": config.selection_horizon_micro,
        "meso": config.selection_horizon_meso,
        "macro": config.selection_horizon_macro,
    }
    return_series = {name: _return_series(closes, bars, smoothing) for name, bars in horizons.items()}

    # Volatility-normalised momentum change: fast momentum relative to its own noise, minus
    # the same for the medium horizon. Positive means the near term is accelerating away
    # from the trend, which is what "enter now" should mean.
    nano_vol = _rolling_volatility(closes, max(config.selection_horizon_nano, 2))
    meso_vol = _rolling_volatility(closes, max(config.selection_horizon_meso, 2))
    change_span = max(config.momentum_change_ema_bars, 1)
    nano_path = _return_series(closes, config.selection_horizon_nano, change_span)
    meso_path = _return_series(closes, config.selection_horizon_meso, change_span)
    change_series = [
        (nano / (nano_vol + EPSILON)) - (meso / (meso_vol + EPSILON))
        for nano, meso in zip(nano_path, meso_path)
    ]

    ma_window = max(config.etf_ma_days, 1)
    # A short history would silently turn "the 100-day average" into the average of whatever
    # happened to be cached, and a name below that shorter average looks like a market fact
    # rather than a data gap. Anything under the window is reported as unknown instead.
    enough_history = len(daily_closes) >= ma_window
    moving_average = float(daily_closes.tail(ma_window).mean()) if enough_history else 0.0
    last_daily = float(daily_closes.iloc[-1]) if not daily_closes.empty else 0.0
    daily_vol = _rolling_volatility(daily_closes, max(config.vol_estimation_days, 2))

    return {
        "symbol": symbol.upper(),
        "close": float(closes.iloc[-1]) if not closes.empty else last_daily,
        "nano_return": return_series["nano"][-1],
        "micro_return": return_series["micro"][-1],
        "meso_return": return_series["meso"][-1],
        "macro_return": return_series["macro"][-1],
        "return_series": return_series,
        "momentum_change": _ema(change_series, change_span),
        "moving_average": moving_average,
        "above_moving_average": bool(enough_history and last_daily > moving_average > 0),
        "daily_bars": int(len(daily_closes)),
        "enough_history": bool(enough_history),
        "abs_return": _return_over(daily_closes, config.etf_abs_return_days),
        "fast_return": _return_over(daily_closes, config.etf_fast_return_days),
        # Annualised, because the volatility target is quoted annually.
        "annual_volatility": daily_vol * math.sqrt(TRADING_DAYS),
        "has_daily": not daily_closes.empty,
        "has_intraday": not closes.empty,
    }


def zscores(values: dict[str, float], robust: bool) -> dict[str, float]:
    """Cross-sectional z-scores, robust (median/MAD) by default.

    A thematic ETF can post an event-driven move that drags the mean and inflates the standard
    deviation enough to flatten everyone else's score. Median and MAD do not move, so one
    outlier stops rewriting the whole cross-section. Falls back to the standard deviation when
    the MAD is zero, which happens when most of the universe posts the identical return.
    """
    if not values:
        return {}
    data = list(values.values())
    count = len(data)
    if robust:
        median = float(pd.Series(data).median())
        deviations = [abs(value - median) for value in data]
        mad = float(pd.Series(deviations).median())
        scale = 1.4826 * mad
        if scale > EPSILON:
            return {symbol: (value - median) / scale for symbol, value in values.items()}
    mean = sum(data) / count
    variance = sum((value - mean) ** 2 for value in data) / count
    std = math.sqrt(variance)
    if std <= EPSILON:
        return {symbol: 0.0 for symbol in values}
    return {symbol: (value - mean) / std for symbol, value in values.items()}


def _risk_scales(features_by_symbol: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Per-symbol volatility divisor, with a sane stand-in for anything unmeasurable.

    A symbol with no volatility estimate would otherwise divide by epsilon and top every
    ranking on arithmetic alone, so it borrows the universe median instead.
    """
    measured = [float(row.get("annual_volatility", 0.0)) for row in features_by_symbol.values()]
    positive = sorted(value for value in measured if value > 0)
    fallback = positive[len(positive) // 2] if positive else 1.0
    return {
        symbol: (float(row.get("annual_volatility", 0.0)) or fallback)
        for symbol, row in features_by_symbol.items()
    }


def base_scores(
    features_by_symbol: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
) -> dict[str, dict[str, Any]]:
    """Slow-weighted composite score, smoothed across the last ``score_ema_bars`` bars.

    The smoothing is done here rather than across runs so that ``analyze`` stays pure: the
    score at bar t-1 is recomputed from the bars, not remembered from the previous run, which
    is what lets a backtest reproduce it exactly.

    ``risk_adjusted_score`` decides *what* is being ranked. Raw returns rank a 58%-volatility
    thematic ETF above a 14%-volatility index fund whenever both are up, because its returns
    are simply four times larger -- the ranking becomes a volatility ranking wearing a
    momentum costume. Dividing by each symbol's own volatility first asks the different
    question: whose trend is strongest *per unit of risk taken*. Sizing already divides by
    volatility, but sizing only applies to names that were selected, so it cannot undo a
    selection bias.
    """
    if not features_by_symbol:
        return {}
    smoothing = max(config.score_ema_bars, 1)
    scale = _risk_scales(features_by_symbol) if config.risk_adjusted_score else {}
    weights = {
        "nano": config.w_nano,
        "micro": config.w_micro,
        "meso": config.w_meso,
        "macro": config.w_macro,
    }

    # One cross-section per historical offset, so z-scores stay relative to the same bar.
    per_offset: list[dict[str, float]] = []
    latest_z: dict[str, dict[str, float]] = {symbol: {} for symbol in features_by_symbol}
    for offset in range(smoothing):
        composite: dict[str, float] = {symbol: 0.0 for symbol in features_by_symbol}
        for horizon, weight in weights.items():
            raw = {
                symbol: float(features["return_series"][horizon][offset]) / scale.get(symbol, 1.0)
                for symbol, features in features_by_symbol.items()
            }
            horizon_z = zscores(raw, config.robust_zscore)
            for symbol, value in horizon_z.items():
                composite[symbol] += weight * value
                if offset == smoothing - 1:
                    latest_z[symbol][horizon] = value
        per_offset.append(composite)

    scored: dict[str, dict[str, Any]] = {}
    for symbol, features in features_by_symbol.items():
        path = [snapshot[symbol] for snapshot in per_offset]
        components = {
            horizon: weights[horizon] * latest_z[symbol].get(horizon, 0.0) for horizon in weights
        }
        scored[symbol] = {
            **features,
            "base_score": _ema(path, smoothing),
            "score_unsmoothed": path[-1],
            "z": latest_z[symbol],
            "score_components": components,
        }
    return scored


# =========================================================================================
# Layer 1: market regime
# =========================================================================================


def market_regime(
    scored: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
) -> dict[str, Any]:
    """The raw risk-on gate: benchmark trend, benchmark absolute momentum, and breadth.

    Raw because the hysteresis that turns this into a *state* needs to count consecutive
    observations, and counting requires memory that ``analyze`` is not allowed to have.
    """
    benchmark = scored.get(config.benchmark, {})
    risk_on = [scored[symbol] for symbol in config.risk_on_universe if symbol in scored]
    above = [row for row in risk_on if row.get("above_moving_average")]
    breadth = len(above) / len(risk_on) if risk_on else 0.0

    # No usable benchmark history is not a bearish reading, it is an unusable one. Say so,
    # and stay risk-off: this gate is the algorithm's whole premise, so guessing is worse
    # than declining to act.
    if not benchmark.get("enough_history"):
        return {
            "risk_on": False,
            "trend_ok": False,
            "momentum_ok": False,
            "breadth": breadth,
            "breadth_ok": False,
            "data_ok": False,
            "detail": (
                f"no usable {config.benchmark} history "
                f"({int(benchmark.get('daily_bars', 0))} of {config.benchmark_ma_days} daily bars)"
            ),
        }

    trend_ok = bool(benchmark.get("above_moving_average"))
    momentum_ok = float(benchmark.get("abs_return", 0.0)) > 0.0
    breadth_ok = breadth >= config.breadth_min
    reasons = []
    if not trend_ok:
        reasons.append(f"{config.benchmark} below its {config.benchmark_ma_days}-day average")
    if not momentum_ok:
        reasons.append(f"{config.benchmark} {config.benchmark_return_days}-day return is negative")
    if not breadth_ok:
        reasons.append(f"breadth {breadth:.0%} below {config.breadth_min:.0%}")

    return {
        "risk_on": trend_ok and momentum_ok and breadth_ok,
        "trend_ok": trend_ok,
        "momentum_ok": momentum_ok,
        "breadth": breadth,
        "breadth_ok": breadth_ok,
        "data_ok": True,
        "detail": "; ".join(reasons) or "benchmark trend, momentum and breadth all pass",
    }


# =========================================================================================
# Layer 2: absolute eligibility
# =========================================================================================


def eligibility(row: dict[str, Any], config: DualMomentumConfig) -> tuple[bool, str]:
    """Whether one ETF may be held at all, independent of how it ranks.

    This is the absolute-momentum half of dual momentum. A name failing here is not ranked,
    so a thin qualifying set means holding less rather than lowering the bar.
    """
    if not row.get("has_daily"):
        return False, "No daily history"
    if not row.get("enough_history"):
        return False, f"Only {int(row.get('daily_bars', 0))} of {config.etf_ma_days} daily bars"
    if not row.get("above_moving_average"):
        return False, f"Below its {config.etf_ma_days}-day average"
    if float(row.get("abs_return", 0.0)) <= config.etf_min_abs_return:
        return False, f"{config.etf_abs_return_days}-day return below {config.etf_min_abs_return:+.0%}"
    if float(row.get("fast_return", 0.0)) <= config.etf_min_fast_return:
        return False, f"{config.etf_fast_return_days}-day return below {config.etf_min_fast_return:+.0%}"
    return True, ""


# =========================================================================================
# Layer 3: entry timing
# =========================================================================================


def timing(row: dict[str, Any], config: DualMomentumConfig) -> tuple[bool, str]:
    """Whether *now* is a moment to open or add, given the name already qualifies.

    Two independent ways in: accelerating volatility-normalised momentum, or a pullback
    inside an intact uptrend. Deliberately a flag rather than a score term -- as a bonus, a
    deep enough dip could outvote the trend horizons and promote a weak ETF.
    """
    if float(row.get("momentum_change", 0.0)) > config.momentum_change_enter:
        return True, "Momentum accelerating"

    z = row.get("z", {}) if isinstance(row.get("z"), dict) else {}
    pullback = (
        float(z.get("macro", 0.0)) >= config.pullback_macro_z_min
        and float(z.get("meso", 0.0)) >= config.pullback_meso_z_min
        and float(z.get("nano", 0.0)) <= config.pullback_nano_z_max
        and float(row.get("micro_return", 0.0)) >= config.pullback_micro_return_min
    )
    if pullback:
        return True, "Pullback in uptrend"
    return False, "Waiting for entry timing"


# =========================================================================================
# Layer 4: sizing and portfolio volatility
# =========================================================================================


def covariance_matrix(
    daily_bars_by_symbol: dict[str, Any],
    symbols: list[str],
    config: DualMomentumConfig,
) -> dict[str, dict[str, float]]:
    """Annualised covariance of daily returns over ``vol_estimation_days``.

    Returned as nested dicts rather than a frame because it travels to step 2 inside the
    signal rows, and step 2 only ever receives JSON-shaped signals.
    """
    frame = pd.DataFrame(
        {
            symbol: _closes(daily_bars_by_symbol.get(symbol)).pct_change().dropna().tail(config.vol_estimation_days)
            for symbol in symbols
        }
    ).dropna(how="all")
    if frame.empty or len(frame) < 2:
        return {symbol: {symbol: 0.0} for symbol in symbols}
    covariance = frame.cov() * TRADING_DAYS
    return {
        row: {
            column: float(value) if pd.notna(value) else 0.0
            for column, value in covariance.loc[row].items()
        }
        for row in covariance.index
    }


def portfolio_volatility(weights: dict[str, float], covariance: dict[str, dict[str, float]]) -> float:
    """Ex-ante annualised volatility, sqrt(w' Sigma w)."""
    variance = 0.0
    for left, left_weight in weights.items():
        if not left_weight:
            continue
        row = covariance.get(left, {})
        for right, right_weight in weights.items():
            if not right_weight:
                continue
            variance += left_weight * right_weight * float(row.get(right, 0.0))
    return math.sqrt(variance) if variance > 0 else 0.0


def volatility_scale(
    weights: dict[str, float],
    covariance: dict[str, dict[str, float]],
    config: DualMomentumConfig,
) -> dict[str, Any]:
    """Scale factor that pulls ex-ante volatility back toward the target.

    Scaling only engages once the estimate exceeds ``high_vol_trigger`` times the target, so
    a portfolio already inside its budget is not re-sized on every 15-minute tick -- turnover
    is a cost, and this overlay is a risk-budgeting device rather than a source of return.
    """
    estimate = portfolio_volatility(weights, covariance)
    target = max(config.target_portfolio_vol, 0.0)
    if estimate <= EPSILON or target <= 0:
        return {"scale": 1.0, "portfolio_volatility": estimate, "engaged": False, "below_floor": False}
    if estimate <= target * max(config.high_vol_trigger, 1.0):
        return {"scale": 1.0, "portfolio_volatility": estimate, "engaged": False, "below_floor": False}

    raw = target / (estimate + EPSILON)
    floor = max(config.vol_scale_floor, 0.0)
    return {
        "scale": min(1.0, max(raw, floor)),
        "portfolio_volatility": estimate,
        "engaged": True,
        # Too volatile to hold even at the floor: the honest answer is the defensive sleeve.
        "below_floor": raw < floor,
    }


def score_to_weights(rows: list[dict[str, Any]], config: DualMomentumConfig) -> dict[str, float]:
    """Score-over-volatility weights, capped per name and in total.

    The score enters as its excess over ``min_base_score``, so a name that only just clears
    the quality floor gets a small position rather than an equal one.
    """
    raw: dict[str, float] = {}
    for row in rows:
        excess = max(float(row.get("base_score", 0.0)) - config.min_base_score, 0.0)
        volatility = float(row.get("annual_volatility", 0.0))
        # sigma ** tilt: negative divides (risk parity), zero ignores it, positive leans in.
        scale = (volatility + EPSILON) ** config.volatility_tilt if config.volatility_tilt else 1.0
        raw[str(row["symbol"])] = excess * scale

    total = sum(raw.values())
    if total <= EPSILON:
        # Every candidate sits exactly at the floor: equal-weight rather than divide by zero.
        share = min(config.name_weight_max, config.risk_on_gross_max / len(raw)) if raw else 0.0
        return {symbol: share for symbol in raw}

    gross = max(config.risk_on_gross_max, 0.0)
    weights = {symbol: min(gross * value / total, config.name_weight_max) for symbol, value in raw.items()}
    total_weight = sum(weights.values())
    if total_weight > gross > 0:
        weights = {symbol: weight * gross / total_weight for symbol, weight in weights.items()}
    return weights


def defensive_weights(
    scored: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
) -> dict[str, float]:
    """Where the book sits when risk-on is not permitted.

    Ranked by medium-term absolute return so the defensive sleeve is itself chosen rather
    than fixed, and deliberately *not* subject to ``name_weight_max``: that cap limits
    single-name risk in the risky sleeve, and applying it here would force idle cash for no
    reason when the whole point is to be in T-bills.
    """
    candidates = [scored[symbol] for symbol in config.defensive_universe if symbol in scored]
    if not candidates:
        return {}
    candidates.sort(key=lambda row: float(row.get("abs_return", 0.0)), reverse=True)
    chosen = candidates[: max(config.defensive_max_positions, 1)]
    share = max(config.risk_on_gross_max, 0.0) / len(chosen)
    return {str(row["symbol"]): share for row in chosen}


def sentiment_adjusted(
    weights: dict[str, float],
    sentiment_scores: dict[str, float],
    config: DualMomentumConfig,
) -> dict[str, float]:
    """A bounded size modifier, never a reason to hold something price logic rejected.

    Capped at the +-10% the spec asks for by construction: a clipped sentiment of +-2 times a
    0.05 scale. Sentiment cannot create a position, only nudge one that already qualified.
    """
    if not config.sentiment_size_scale:
        return dict(weights)
    adjusted: dict[str, float] = {}
    for symbol, weight in weights.items():
        clip = max(config.sentiment_clip, 0.0)
        score = max(-clip, min(clip, float(sentiment_scores.get(symbol, 0.0))))
        modifier = 1.0 + (config.sentiment_size_scale * score)
        low = 1.0 - (config.sentiment_size_scale * clip)
        high = 1.0 + (config.sentiment_size_scale * clip)
        adjusted[symbol] = weight * max(low, min(high, modifier))
    return adjusted


# =========================================================================================
# Assembly
# =========================================================================================


def rank_candidates(
    scored: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
) -> list[dict[str, Any]]:
    """Eligible risk-on names, best first. Ineligible names are never ranked."""
    eligible = []
    for symbol in config.risk_on_universe:
        row = scored.get(symbol)
        if row and row.get("eligible"):
            eligible.append(row)
    eligible.sort(key=lambda row: float(row.get("base_score", 0.0)), reverse=True)
    for position, row in enumerate(eligible, start=1):
        row["rank"] = position
    return eligible


def _selection_reason(row: dict[str, Any], weight: float, regime: dict[str, Any], config: DualMomentumConfig) -> str:
    """One line saying why this symbol is or is not held, for the dashboard and the audit."""
    symbol = str(row.get("symbol", ""))
    if weight > 0:
        if symbol in {name.upper() for name in config.defensive_universe}:
            return "Defensive sleeve"
        return f"Rank {int(row.get('rank') or 0)} - {row.get('timing_reason') or 'held'}"
    if symbol == config.benchmark and symbol not in config.risk_on_universe:
        return "Benchmark only"
    if symbol in {name.upper() for name in config.defensive_universe}:
        return "Defensive sleeve idle" if regime.get("risk_on") else "Not the strongest defensive"
    if not regime.get("risk_on"):
        return f"Risk-off: {regime.get('detail', '')}"
    if not row.get("eligible"):
        return str(row.get("eligibility_reason") or "Not eligible")
    if float(row.get("base_score", 0.0)) < config.min_base_score:
        return "Score below quality floor"
    rank = int(row.get("rank") or 0)
    if rank and rank > config.entry_rank_max:
        return f"Rank {rank}, outside entry rank {config.entry_rank_max}"
    if not row.get("timing"):
        return str(row.get("timing_reason") or "Waiting for entry timing")
    return "No slot"


def build_signals(
    scored: dict[str, dict[str, Any]],
    weights: dict[str, float],
    regime: dict[str, Any],
    vol: dict[str, Any],
    covariance: dict[str, dict[str, float]],
    defensive_book: dict[str, float],
    as_of: datetime,
    config: DualMomentumConfig,
) -> dict[str, dict[str, Any]]:
    """Per-symbol rows: the dashboard's view, step 2's input, and the audit record.

    Run-level facts (regime, volatility scale, timestamp) are denormalised onto every row
    because ``refine`` receives only these signals -- decision metadata does not travel with
    them.
    """
    signals: dict[str, dict[str, Any]] = {}
    for symbol, row in scored.items():
        weight = float(weights.get(symbol, 0.0))
        z = row.get("z", {}) if isinstance(row.get("z"), dict) else {}
        signals[symbol] = {
            "signal": 1 if weight > 0 else 0,
            "score": float(row.get("base_score", 0.0)),
            "base_score": float(row.get("base_score", 0.0)),
            "score_components": {key: float(value) for key, value in (row.get("score_components") or {}).items()},
            "rank": int(row.get("rank") or 0),
            "eligible": 1 if row.get("eligible") else 0,
            "eligibility_reason": str(row.get("eligibility_reason") or ""),
            "timing": 1 if row.get("timing") else 0,
            "timing_reason": str(row.get("timing_reason") or ""),
            "momentum_change": float(row.get("momentum_change", 0.0)),
            "nano_return": float(row.get("nano_return", 0.0)),
            "micro_return": float(row.get("micro_return", 0.0)),
            "meso_return": float(row.get("meso_return", 0.0)),
            "macro_return": float(row.get("macro_return", 0.0)),
            "abs_return": float(row.get("abs_return", 0.0)),
            "fast_return": float(row.get("fast_return", 0.0)),
            "nano_z": float(z.get("nano", 0.0)),
            "meso_z": float(z.get("meso", 0.0)),
            "macro_z": float(z.get("macro", 0.0)),
            "annual_volatility": float(row.get("annual_volatility", 0.0)),
            "realized_volatility": float(row.get("annual_volatility", 0.0)),
            "covariance_row": {
                str(other): float(value) for other, value in (covariance.get(symbol) or {}).items()
            },
            "target_weight": weight,
            # What this symbol would be worth in the defensive book, so step 2 can build one
            # without re-reading market data.
            "defensive_weight": float(defensive_book.get(symbol, 0.0)),
            "social_score": float(row.get("sentiment_score", 0.0)),
            "close": float(row.get("close", 0.0)),
            # Consumed by the dashboard row subtitle; without it every row renders "Inactive".
            "trend_ok": 1 if row.get("above_moving_average") else 0,
            "reason": _selection_reason(row, weight, regime, config),
            # -- run-level, repeated per row so refine can read them ---------------------
            "regime_risk_on": 1 if regime.get("risk_on") else 0,
            "regime_detail": str(regime.get("detail", "")),
            "regime_breadth": float(regime.get("breadth", 0.0)),
            "vol_scale": float(vol.get("scale", 1.0)),
            "portfolio_volatility": float(vol.get("portfolio_volatility", 0.0)),
            "vol_below_floor": 1 if vol.get("below_floor") else 0,
            "as_of": as_of.isoformat(),
        }
    return signals


def allocation_mode(weights: dict[str, float], regime: dict[str, Any], config: DualMomentumConfig) -> str:
    """The one-word summary the dashboard prints for this run."""
    defensive = {name.upper() for name in config.defensive_universe}
    held = {symbol for symbol, weight in weights.items() if weight > 0}
    if not held:
        return "Cash"
    if held <= defensive:
        return "Defensive"
    return "Risk-on" if regime.get("risk_on") else "Risk-on (unconfirmed)"


def analyze_universe(context: AlgorithmContext, config: DualMomentumConfig) -> dict[str, Any]:
    """The whole read-only pipeline: features, regime, eligibility, ranking, timing, weights.

    Returned as a dict rather than assembled inline so tests and the dashboard can inspect
    each layer's output without going through ``AlgorithmDecision``.
    """
    features = {
        symbol: compute_features(
            symbol,
            context.intraday_bars_by_symbol.get(symbol, pd.DataFrame()),
            context.bars_by_symbol.get(symbol, pd.DataFrame()),
            config,
        )
        for symbol in config.symbols
    }
    scored = base_scores(features, config)
    for symbol, row in scored.items():
        row["sentiment_score"] = float(context.sentiment_scores.get(symbol, 0.0))
        ok, reason = eligibility(row, config)
        row["eligible"] = ok
        row["eligibility_reason"] = reason
        timing_ok, timing_reason = timing(row, config)
        row["timing"] = timing_ok
        row["timing_reason"] = timing_reason
        if config.sentiment_weight:
            clip = max(config.sentiment_clip, 0.0)
            tilt = max(-clip, min(clip, row["sentiment_score"])) * config.sentiment_weight
            row["base_score"] = float(row["base_score"]) + tilt
            row["score_components"]["sentiment"] = tilt

    regime = market_regime(scored, config)
    ranked = rank_candidates(scored, config)

    entries = [
        row
        for row in ranked
        if int(row.get("rank") or 0) <= config.entry_rank_max
        and float(row.get("base_score", 0.0)) >= config.min_base_score
        and row.get("timing")
    ][: max(config.max_positions, 0)]

    covariance = covariance_matrix(context.bars_by_symbol, config.symbols, config)
    # Always computed, whatever the regime says: step 2 can decide to go defensive for
    # reasons only it can see (a pending regime confirmation, the drawdown breaker), and it
    # cannot derive a defensive book from a risk-on proposal.
    defensive_book = defensive_weights(scored, config)

    if regime.get("risk_on") and entries:
        weights = score_to_weights(entries, config)
        weights = sentiment_adjusted(weights, context.sentiment_scores, config)
        vol = volatility_scale(weights, covariance, config)
        if vol["below_floor"]:
            weights = dict(defensive_book)
            vol = {**vol, "scale": 1.0}
        else:
            weights = {symbol: weight * vol["scale"] for symbol, weight in weights.items()}
    else:
        weights = dict(defensive_book)
        vol = volatility_scale(weights, covariance, config)

    full_weights = {symbol: float(weights.get(symbol, 0.0)) for symbol in config.symbols}
    return {
        "features": features,
        "scored": scored,
        "regime": regime,
        "ranked": ranked,
        "entries": entries,
        "covariance": covariance,
        "volatility": vol,
        "defensive_book": defensive_book,
        "weights": full_weights,
    }


# =========================================================================================
# Step 2 helpers: stateful, position-aware
# =========================================================================================


def _run_facts(signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Read the run-level fields ``analyze`` denormalised onto every row."""
    for row in signals.values():
        return {
            "regime_risk_on": bool(row.get("regime_risk_on")),
            "regime_detail": str(row.get("regime_detail", "")),
            "vol_scale": float(row.get("vol_scale", 1.0) or 1.0),
            "as_of": str(row.get("as_of", "")),
        }
    return {"regime_risk_on": False, "regime_detail": "no signals", "vol_scale": 1.0, "as_of": ""}


def _defensive_book(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    """The defensive allocation ``analyze`` computed, whatever it ended up proposing."""
    return {
        symbol: float(row.get("defensive_weight", 0.0) or 0.0)
        for symbol, row in rows.items()
        if float(row.get("defensive_weight", 0.0) or 0.0) > 0
    }


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _minutes_between(later: datetime | None, earlier: datetime | None) -> float:
    if later is None or earlier is None:
        return float("inf")
    return abs((later - earlier).total_seconds()) / 60.0


def confirm_regime(state: dict[str, Any], raw_risk_on: bool, config: DualMomentumConfig) -> dict[str, Any]:
    """Turn the raw gate into a state, requiring consecutive agreement in both directions.

    Without this the book flips on a single reading that straddles the threshold, and pays
    the spread twice for the privilege.
    """
    confirmed = bool(state.get("regime_risk_on", False))
    agree = int(state.get("regime_agree", 0) or 0)
    disagree = int(state.get("regime_disagree", 0) or 0)

    if raw_risk_on:
        agree, disagree = agree + 1, 0
        if not confirmed and agree >= max(config.regime_confirm_bars, 1):
            confirmed = True
    else:
        disagree, agree = disagree + 1, 0
        if confirmed and disagree >= max(config.regime_exit_confirm_bars, 1):
            confirmed = False

    return {"regime_risk_on": confirmed, "regime_agree": agree, "regime_disagree": disagree}


def intraday_drawdown_breached(
    state: dict[str, Any],
    equity: float,
    config: DualMomentumConfig,
    session: str = "",
) -> bool:
    """Session circuit breaker: once tripped it stays tripped until the next session.

    ``session`` comes from the algorithm's own timestamp rather than the wall clock, because
    in a replay every step would otherwise be "today": the breaker would trip once on the
    first bad day and stay latched for the entire backtest, which is not what it does live.
    """
    today = session or date.today().isoformat()
    if state.get("session") != today:
        state.update({"session": today, "session_start_equity": equity, "halted": False})
    start = float(state.get("session_start_equity") or equity)
    drawdown = (equity / start - 1.0) if start > 0 else 0.0
    if drawdown <= config.intraday_drawdown_limit:
        state["halted"] = True
    state["session_drawdown"] = drawdown
    return bool(state.get("halted"))


def apply_turnover_filters(
    target: dict[str, float],
    current: dict[str, float],
    equity: float,
    config: DualMomentumConfig,
) -> dict[str, float]:
    """Drop trades too small to be worth their costs, keeping the current weight instead."""
    minimum_notional = max(config.minimum_trade_notional, config.minimum_trade_nav_fraction * max(equity, 0.0))
    filtered: dict[str, float] = {}
    for symbol, weight in target.items():
        held = float(current.get(symbol, 0.0))
        move = abs(weight - held)
        if move < max(config.rebalance_weight_threshold, 0.0):
            filtered[symbol] = held
            continue
        if move * max(equity, 0.0) < minimum_notional:
            filtered[symbol] = held
            continue
        filtered[symbol] = weight
    return filtered


class DualMomentumAlgorithm(BaseAlgorithm):
    """Dual momentum with a regime gate, split timing, and a volatility target."""

    algorithm_id = "dual_momentum"

    #: Every 15 minutes: that is the risk and timing cadence. Re-ranking is throttled
    #: separately to ``signal_refresh_minutes`` inside ``refine``, so the expensive decision
    #: (which names to own) stays slow while the cheap ones (scale, stop, time an entry)
    #: stay fast.
    schedule = Schedule(refresh_minutes=15, jitter_minutes=2)

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        strategy_config = DualMomentumConfig.from_runtime_config(config)
        return AlgorithmRequirements(
            price_symbols=sorted(set(strategy_config.symbols) | set(current_positions)),
            daily_lookback_days=strategy_config.required_daily_bars,
            daily_ma_days=strategy_config.etf_ma_days,
            intraday_lookback_bars=strategy_config.required_intraday_bars,
            intraday_bar_minutes=strategy_config.intraday_bar_minutes,
            needs_sentiment=strategy_config.uses_sentiment,
            # Unproven: keep it on paper until walk-forward results say otherwise.
            paper_only=True,
        )

    def sizing(self, config: Any) -> dict[str, float]:
        """No cash buffer: gross exposure is already bounded inside the weights."""
        strategy_config = DualMomentumConfig.from_runtime_config(config)
        return {
            "cash_buffer": 0.0,
            "min_trade_dollars": strategy_config.minimum_trade_notional,
            "rebalance_threshold": strategy_config.rebalance_weight_threshold,
        }

    def analyze(self, context: AlgorithmContext) -> AlgorithmDecision:
        strategy_config = DualMomentumConfig.from_runtime_config(context.config)
        outcome = analyze_universe(context, strategy_config)
        weights = outcome["weights"]
        regime = outcome["regime"]
        return AlgorithmDecision(
            target_weights=weights,
            signals=build_signals(
                outcome["scored"],
                weights,
                regime,
                outcome["volatility"],
                outcome["covariance"],
                outcome["defensive_book"],
                context.timestamp,
                strategy_config,
            ),
            metadata={
                "allocation_mode": allocation_mode(weights, regime, strategy_config),
                "market_sentiment": context.market_sentiment,
                "regime": regime,
                "portfolio_volatility": outcome["volatility"]["portfolio_volatility"],
                "vol_scale": outcome["volatility"]["scale"],
                "eligible_count": len(outcome["ranked"]),
            },
        )

    def refine_weights(
        self,
        target_weights: dict[str, float],
        signals: dict[str, dict[str, Any]],
        snapshot: Any,
        latest_prices: dict[str, float],
        config: Any,
    ) -> dict[str, float]:
        """Position-aware layer: regime hysteresis, hold/exit asymmetry, cooldown, risk stops.

        Everything here needs either what is currently held or memory of previous runs, which
        is exactly what ``analyze`` is forbidden to touch.
        """
        strategy_config = DualMomentumConfig.from_runtime_config(config)
        account = str(getattr(config, "account_id", "") or "")
        state_key = f"{STATE_KEY}:{account}" if account else STATE_KEY
        state = load_state(state_key, {})
        if not isinstance(state, dict):
            state = {}

        facts = _run_facts(signals)
        now = _parse_time(facts["as_of"])
        current = snapshot.weights(latest_prices)
        held = {symbol for symbol, weight in current.items() if weight > 0}
        defensive = {name.upper() for name in strategy_config.defensive_universe}

        state.update(confirm_regime(state, facts["regime_risk_on"], strategy_config))
        risk_on = bool(state["regime_risk_on"])

        rows = {symbol: dict(row, symbol=symbol) for symbol, row in signals.items()}
        book = _defensive_book(rows)

        def settle(weights: dict[str, float]) -> dict[str, float]:
            """Apply the turnover filters, record exits, and persist state."""
            result = apply_turnover_filters(
                {symbol: float(weights.get(symbol, 0.0)) for symbol in target_weights},
                current,
                float(snapshot.equity or 0.0),
                strategy_config,
            )
            _record_exits(state, held, {s for s, weight in result.items() if weight > 0}, facts["as_of"])
            save_state(state_key, state)
            return result

        session = now.date().isoformat() if now else ""
        if intraday_drawdown_breached(state, float(snapshot.equity or 0.0), strategy_config, session):
            logger.warning(
                "Dual Momentum drawdown breaker active (%.2f%%); holding the defensive sleeve only",
                100 * float(state.get("session_drawdown", 0.0)),
            )
            return settle(book)

        if not risk_on:
            # The raw gate may read risk-on while the confirmation is still pending; the
            # risk-on proposal must not be acted on until it is confirmed. Deliberately not
            # stamping last_selection_at: no selection happened, and pretending one did would
            # throttle the first real one.
            return settle(book)

        proposed = {symbol for symbol, weight in target_weights.items() if weight > 0} - defensive

        keep: set[str] = set()
        for symbol in held - defensive:
            row = rows.get(symbol, {})
            rank = int(row.get("rank") or 0)
            if not int(row.get("eligible", 0)):
                logger.info("Dual Momentum exiting %s: %s", symbol, row.get("eligibility_reason"))
                continue
            if rank and rank > strategy_config.exit_rank_max:
                logger.info("Dual Momentum exiting %s: rank %d beyond exit rank", symbol, rank)
                continue
            # An incumbent is held while it stays eligible and ranked, even if the timing flag
            # is false: timing decides when to *enter*, not whether to stay.
            keep.add(symbol)

        # Re-ranking is throttled: between selection refreshes the book keeps its membership
        # and only the risk layer acts, which is what "slow selection, fast timing" means.
        # A free slot is exempt -- the throttle exists to stop churn between comparable names,
        # and sitting in cash while a qualified name waits out the hour is lag, not risk
        # management.
        since_selection = _minutes_between(now, _parse_time(str(state.get("last_selection_at", ""))))
        may_reselect = (
            since_selection >= max(strategy_config.signal_refresh_minutes, 0)
            or len(keep) < max(strategy_config.max_positions, 0)
        )

        if may_reselect:
            entrants = {symbol for symbol in proposed if symbol not in keep}
            entrants = {symbol for symbol in entrants if not _in_cooldown(state, symbol, now, strategy_config)}
            selection = _resolve_replacements(keep, entrants, rows, strategy_config)
            state["last_selection_at"] = facts["as_of"]
        else:
            selection = set(keep)

        chosen_rows = [rows[symbol] for symbol in selection if symbol in rows]
        weights = score_to_weights(chosen_rows, strategy_config) if chosen_rows else {}

        covariance = {symbol: dict(rows[symbol].get("covariance_row") or {}) for symbol in selection if symbol in rows}
        vol = volatility_scale(weights, covariance, strategy_config)
        if vol["below_floor"]:
            logger.info("Dual Momentum de-risking to defensive: ex-ante vol %.1f%%", 100 * vol["portfolio_volatility"])
            weights = dict(book)
        else:
            weights = {symbol: weight * vol["scale"] for symbol, weight in weights.items()}

        if not any(weights.values()):
            # Nothing qualifies: sit in the defensive sleeve rather than in the least-bad name.
            weights = dict(book)

        return settle(weights)


def _resolve_replacements(
    incumbents: set[str],
    entrants: set[str],
    rows: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
) -> set[str]:
    """Fill free slots first, then let a challenger displace the weakest incumbent.

    The challenger has to win by ``min_score_delta_to_replace``; a hair's-breadth improvement
    is noise, and trading on it costs the spread every time.
    """
    def score(symbol: str) -> float:
        return float(rows.get(symbol, {}).get("base_score", 0.0))

    selection = set(incumbents)
    free = max(config.max_positions, 0) - len(selection)
    ordered = sorted(entrants, key=score, reverse=True)

    for symbol in ordered[: max(free, 0)]:
        selection.add(symbol)

    for symbol in ordered[max(free, 0):]:
        if not selection:
            break
        weakest = min(selection, key=score)
        if score(symbol) <= score(weakest) + max(config.min_score_delta_to_replace, 0.0):
            continue
        logger.info(
            "Dual Momentum replacing %s (%.2f) with %s (%.2f)",
            weakest, score(weakest), symbol, score(symbol),
        )
        selection.discard(weakest)
        selection.add(symbol)
    return selection


def _record_exits(state: dict[str, Any], held: set[str], final: set[str], as_of: str) -> None:
    """Remember when a symbol left the book, so re-entry can be held off for a cooldown."""
    exits = state.setdefault("exited_at", {})
    if not isinstance(exits, dict):
        exits = {}
        state["exited_at"] = exits
    for symbol in held - final:
        exits[str(symbol)] = as_of
    for symbol in final:
        exits.pop(str(symbol), None)


def _in_cooldown(
    state: dict[str, Any],
    symbol: str,
    now: datetime | None,
    config: DualMomentumConfig,
) -> bool:
    """Whether ``symbol`` exited too recently to be re-entered.

    Measured in wall-clock minutes derived from the bar count, using the timestamp
    ``analyze`` recorded -- which the backtester sets to the historical bar, so a replay
    applies the same cooldown the live runner would have.
    """
    exits = state.get("exited_at") if isinstance(state.get("exited_at"), dict) else {}
    exited_at = _parse_time(str(exits.get(symbol, "")))
    if exited_at is None:
        return False
    window = max(config.cooldown_after_exit, 0) * max(config.risk_refresh_minutes, 1)
    if _minutes_between(now, exited_at) >= window:
        return False
    logger.info("Dual Momentum holding off %s: inside the %d-minute re-entry cooldown", symbol, window)
    return True
