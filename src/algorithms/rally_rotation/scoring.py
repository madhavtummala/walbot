"""Per-symbol features and the cross-sectional score built from them.
"""

from __future__ import annotations

from .config import EPSILON, TRADING_DAYS, RallyRotationConfig



import logging
import math
from typing import Any

import pandas as pd

from ...data.bars import (
    TRADING_MINUTES_PER_DAY,
    closes_of as _closes,
    ema,
    market_minutes_per_bar,
    return_over_periods as _return_over,
    return_path,
)

logger = logging.getLogger(__name__)


def _rolling_volatility(closes: pd.Series, window: int) -> float:
    """Standard deviation of one-period returns over the trailing ``window``."""
    if window <= 1 or closes.empty:
        return 0.0
    returns = closes.pct_change().dropna().tail(window)
    if returns.empty:
        return 0.0
    value = float(returns.std())
    return 0.0 if math.isnan(value) else value


def _trend_ma_distance(closes: pd.Series, ma_days: int) -> float:
    """Signed distance from the trend MA (e.g. 20-day). Returns 0 if insufficient data."""
    if ma_days <= 0 or len(closes) < ma_days:
        return 0.0
    trend_ma = float(closes.tail(ma_days).mean())
    last = float(closes.iloc[-1])
    if trend_ma <= 0:
        return 0.0
    return last / trend_ma - 1.0


def compute_features(
    symbol: str,
    daily_bars: Any,
    config: RallyRotationConfig,
) -> dict[str, Any]:
    """Everything about one symbol that the layers below need, computed once.

    Daily bars, and only daily bars. The selection horizons used to be measured against a
    separate intraday window; they are now measured against these, at session granularity.
    That window was never reliably intraday anyway -- ``read_history`` blends resolutions, so
    the same backtest scored its early months from ~18 daily closes and its later ones from
    ~1,300 five-minute bars, with the score's EMA silently collapsing to a single sample
    wherever the cache was daily. One source removes both inconsistencies, and lets a replay
    reach back as far as the daily store does rather than as far as the intraday cache does.
    """
    closes = _closes(daily_bars)
    daily_closes = closes
    step = market_minutes_per_bar(daily_bars, TRADING_MINUTES_PER_DAY)
    smoothing = max(config.score_ema_minutes, step)

    horizons = {
        "nano": config.selection_horizon_nano_minutes,
        "micro": config.selection_horizon_micro_minutes,
        "meso": config.selection_horizon_meso_minutes,
        "macro": config.selection_horizon_macro_minutes,
    }
    return_series = {
        name: return_path(daily_bars, minutes, span_minutes=smoothing)
        for name, minutes in horizons.items()
    }

    ma_window = max(config.etf_ma_days, 1)
    # A short history would silently turn "the 100-day average" into the average of whatever
    # happened to be cached, and a name below that shorter average looks like a market fact
    # rather than a data gap. Anything under the window is reported as unknown instead.
    enough_history = len(daily_closes) >= ma_window
    moving_average = float(daily_closes.tail(ma_window).mean()) if enough_history else 0.0
    last_daily = float(daily_closes.iloc[-1]) if not daily_closes.empty else 0.0
    daily_vol = _rolling_volatility(daily_closes, max(config.vol_estimation_days, 2))

    # -- new risk signals for climax detection --
    # Get high, low, volume columns
    bars_df = daily_bars if isinstance(daily_bars, pd.DataFrame) else pd.DataFrame()
    highs = pd.to_numeric(bars_df.get("high", pd.Series()), errors="coerce").dropna() if not bars_df.empty else pd.Series(dtype=float)
    lows = pd.to_numeric(bars_df.get("low", pd.Series()), errors="coerce").dropna() if not bars_df.empty else pd.Series(dtype=float)
    volumes = pd.to_numeric(bars_df.get("volume", pd.Series()), errors="coerce").dropna() if not bars_df.empty else pd.Series(dtype=float)

    # 5-day annualized volatility (short-term vol vs 20d)
    vol_5d = _rolling_volatility(daily_closes, 5) * math.sqrt(TRADING_DAYS)

    # 20-day average intraday range (high-low)/close
    range_20d_avg = 0.0
    if len(highs) >= 20 and len(lows) >= 20:
        recent_range = (highs.tail(20).values - lows.tail(20).values) / highs.tail(20).values
        range_20d_avg = float(recent_range.mean())

    # Current range expansion: today's range vs 20d average
    range_expansion = 0.0
    if len(highs) >= 1 and len(lows) >= 1 and range_20d_avg > 0:
        today_range = float(highs.iloc[-1] - lows.iloc[-1]) / float(highs.iloc[-1]) if float(highs.iloc[-1]) > 0 else 0.0
        range_expansion = today_range / range_20d_avg if range_20d_avg > 0 else 0.0

    # Volume ratio: last volume vs 20-day average
    volume_ratio = 0.0
    if len(volumes) >= 20:
        avg_vol = float(volumes.tail(20).mean())
        if avg_vol > 0:
            volume_ratio = float(volumes.iloc[-1]) / avg_vol

    return {
        "symbol": symbol.upper(),
        "close": float(closes.iloc[-1]) if not closes.empty else last_daily,
        "nano_return": return_series["nano"][-1],
        "micro_return": return_series["micro"][-1],
        "meso_return": return_series["meso"][-1],
        "macro_return": return_series["macro"][-1],
        "return_series": return_series,
        "step_minutes": step,
        "moving_average": moving_average,
        "above_moving_average": bool(enough_history and last_daily > moving_average > 0),
        # Signed distance from the trend line, so a *held* name can be given a wider
        # band to leave through than it had to clear to enter.
        "ma_distance": (last_daily / moving_average - 1.0) if moving_average > 0 else 0.0,
        "daily_bars": int(len(daily_closes)),
        "enough_history": bool(enough_history),
        "abs_return": _return_over(daily_closes, config.etf_abs_return_days),
        "fast_return": _return_over(daily_closes, config.etf_fast_return_days),
        # Trend filter features
        "trend_ma_distance": _trend_ma_distance(daily_closes, config.trend_ma_days),
        "trend_return": _return_over(daily_closes, config.trend_return_days),
        # Annualised, because the volatility target is quoted annually.
        "annual_volatility": daily_vol * math.sqrt(TRADING_DAYS),
        "vol_5d": vol_5d,
        "range_20d_avg": range_20d_avg,
        "range_expansion": range_expansion,
        "volume_ratio": volume_ratio,
        "has_daily": not daily_closes.empty,
        "has_history": not closes.empty,
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
    config: RallyRotationConfig,
) -> dict[str, dict[str, Any]]:
    """Slow-weighted composite score, smoothed over ``score_ema_minutes``.

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
    # How many samples the smoothing window came out to, which follows the feed's resolution
    # rather than a fixed count: 45 minutes is three points on a 15m grid and nine on a 5m
    # one. Read off the paths themselves so every symbol's cross-section lines up.
    first = next(iter(features_by_symbol.values()))
    smoothing = max(len(first.get("return_series", {}).get("nano", [])), 1)
    step = int(first.get("step_minutes") or 1)
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
            "base_score": ema(path, config.score_ema_minutes, step),
            "score_unsmoothed": path[-1],
            "z": latest_z[symbol],
            "score_components": components,
        }
    return scored


# =========================================================================================
# Cross-sectional scoring
