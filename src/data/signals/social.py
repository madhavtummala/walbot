from __future__ import annotations

import math

import pandas as pd


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _latest_or_nan(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return _safe_float(series.iloc[-1])


def _rolling_z_score(series: pd.Series, lookback: int) -> float:
    if series.empty:
        return 0.0
    lookback = max(lookback, 2)
    prior = series.shift(1).tail(lookback).dropna()
    latest = _latest_or_nan(series)
    if len(prior) < 2 or not math.isfinite(latest):
        return 0.0
    std = float(prior.std())
    if std <= 0 or not math.isfinite(std):
        return 0.0
    return (latest - float(prior.mean())) / std


def compute_social_trend_score(social_df: pd.DataFrame | None, social_lookback_days: int) -> dict[str, float]:
    """Score social data from -1 to 1 using attention spikes and optional sentiment."""
    if social_df is None or social_df.empty:
        return {"social_score": 0.0, "mention_z": 0.0, "sentiment": 0.0}

    work_df = social_df.copy().sort_values("timestamp").reset_index(drop=True)
    mentions = pd.to_numeric(work_df.get("mentions", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    sentiment = pd.to_numeric(work_df.get("sentiment", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    vendor_score = pd.to_numeric(work_df.get("social_score", pd.Series(dtype=float)), errors="coerce")

    mention_z = _clip(_rolling_z_score(mentions.map(math.log1p), social_lookback_days), -3.0, 3.0)
    attention_component = mention_z / 3.0
    sentiment_last = _clip(_latest_or_nan(sentiment), -1.0, 1.0)

    social_component = attention_component
    if abs(sentiment_last) > 0.05:
        social_component = attention_component * sentiment_last

    vendor_last = _latest_or_nan(vendor_score.dropna())
    if math.isfinite(vendor_last):
        if -1.0 <= vendor_last <= 1.0:
            vendor_component = vendor_last
        else:
            vendor_z = _clip(_rolling_z_score(vendor_score, social_lookback_days), -3.0, 3.0)
            vendor_component = vendor_z / 3.0
        social_component = 0.65 * vendor_component + 0.35 * social_component

    return {
        "social_score": _clip(social_component, -1.0, 1.0),
        "mention_z": mention_z,
        "sentiment": sentiment_last,
    }


def compute_rsi(df: pd.DataFrame, *, price_column: str = "close", lookback: int = 14) -> pd.Series:
    """Return RSI values from 0 to 100 aligned to the input rows."""
    if df.empty or price_column not in df:
        return pd.Series(dtype=float, name="rsi")
    prices = pd.to_numeric(df[price_column], errors="coerce")
    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    period = max(int(lookback), 1)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + relative_strength))
    return rsi.fillna(50.0).rename("rsi")


def compute_bollinger_percent_b(
    df: pd.DataFrame,
    *,
    price_column: str = "close",
    lookback: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    """Return Bollinger %B: 0 at the lower band, 1 at the upper, negative below the lower band.

    A flat window has no bands to speak of, so it reports 0.5 (mid-band) rather than dividing
    by a zero width and calling every bar an extreme.
    """
    if df.empty or price_column not in df:
        return pd.Series(dtype=float, name="percent_b")
    prices = pd.to_numeric(df[price_column], errors="coerce")
    period = max(int(lookback), 1)
    middle = prices.rolling(period, min_periods=period).mean()
    width = prices.rolling(period, min_periods=period).std(ddof=0) * float(num_std)
    lower = middle - width
    percent_b = (prices - lower) / (2 * width)
    return percent_b.where(width > 0, 0.5).rename("percent_b")


