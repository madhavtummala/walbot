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


def _squash(value: float, scale: float = 3.0) -> float:
    if not math.isfinite(value):
        return 0.0
    return math.tanh(value * scale)


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


def compute_macd(
    df: pd.DataFrame,
    *,
    price_column: str = "close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Return MACD, signal, and histogram columns aligned to the input rows."""
    if df.empty or price_column not in df:
        return pd.DataFrame(columns=["macd", "macd_signal", "macd_histogram"])
    prices = pd.to_numeric(df[price_column], errors="coerce")
    fast_ema = prices.ewm(span=max(int(fast), 1), adjust=False).mean()
    slow_ema = prices.ewm(span=max(int(slow), 1), adjust=False).mean()
    macd = fast_ema - slow_ema
    macd_signal = macd.ewm(span=max(int(signal), 1), adjust=False).mean()
    return pd.DataFrame(
        {
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_histogram": macd - macd_signal,
        },
        index=df.index,
    )


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


def compute_momentum_and_trend(
    df: pd.DataFrame,
    lookback_days: int,
    ma_days: int,
    short_lookback_days: int = 21,
    volume_lookback_days: int = 20,
    social_df: pd.DataFrame | None = None,
    social_lookback_days: int = 30,
    price_momentum_weight: float = 0.55,
    social_momentum_weight: float = 0.30,
    volume_momentum_weight: float = 0.15,
    min_composite_score: float = 0.05,
) -> dict[str, float | int]:
    """Compute a composite long-only momentum/social trend signal for one symbol."""
    if df.empty:
        return {
            "signal": 0,
            "score": 0.0,
            "price_score": 0.0,
            "social_score": 0.0,
            "volume_score": 0.0,
            "ret_N": float("nan"),
            "ret_short": float("nan"),
            "close": float("nan"),
            "sma_long": float("nan"),
            "realized_vol": float("nan"),
            "trend_ok": 0,
        }

    work_df = df.copy().reset_index(drop=True)
    work_df["close"] = pd.to_numeric(work_df["close"], errors="coerce")
    work_df["volume"] = pd.to_numeric(work_df.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    work_df["ret_N"] = work_df["close"] / work_df["close"].shift(lookback_days) - 1
    work_df["ret_short"] = work_df["close"] / work_df["close"].shift(short_lookback_days) - 1
    work_df["sma_long"] = work_df["close"].rolling(ma_days).mean()
    work_df["daily_ret"] = work_df["close"].pct_change()
    work_df["realized_vol"] = work_df["daily_ret"].rolling(volume_lookback_days).std() * math.sqrt(252)
    work_df["trend_gap"] = work_df["close"] / work_df["sma_long"] - 1

    ret_N_last = _latest_or_nan(work_df["ret_N"]) if len(work_df) > lookback_days else float("nan")
    ret_short_last = _latest_or_nan(work_df["ret_short"]) if len(work_df) > short_lookback_days else float("nan")
    close_last = _latest_or_nan(work_df["close"])
    sma_last = _latest_or_nan(work_df["sma_long"]) if len(work_df) >= ma_days else float("nan")
    trend_gap_last = _latest_or_nan(work_df["trend_gap"])
    realized_vol_last = _latest_or_nan(work_df["realized_vol"])

    volume_z = _clip(_rolling_z_score(work_df["volume"].map(math.log1p), volume_lookback_days), -3.0, 3.0)
    volume_direction = 1.0 if ret_short_last >= 0 else -1.0
    volume_score = (volume_z / 3.0) * volume_direction

    social = compute_social_trend_score(social_df, social_lookback_days)
    price_score = _squash((0.70 * ret_N_last) + (0.25 * ret_short_last) + (0.05 * trend_gap_last))

    total_weight = price_momentum_weight + social_momentum_weight + volume_momentum_weight
    if total_weight <= 0:
        total_weight = 1.0
    score = (
        price_momentum_weight * price_score
        + social_momentum_weight * social["social_score"]
        + volume_momentum_weight * volume_score
    ) / total_weight

    trend_ok = int(math.isfinite(ret_N_last) and math.isfinite(sma_last) and ret_N_last > 0 and close_last > sma_last)
    signal = 1 if trend_ok and score >= min_composite_score else 0

    return {
        "signal": signal,
        "score": score,
        "price_score": price_score,
        "social_score": social["social_score"],
        "mention_z": social["mention_z"],
        "sentiment": social["sentiment"],
        "volume_score": volume_score,
        "volume_z": volume_z,
        "ret_N": ret_N_last,
        "ret_short": ret_short_last,
        "close": close_last,
        "sma_long": sma_last,
        "realized_vol": realized_vol_last,
        "trend_ok": trend_ok,
    }


def compute_signals_for_universe(
    bars_by_symbol: dict[str, pd.DataFrame],
    lookback_days: int,
    ma_days: int,
    short_lookback_days: int = 21,
    volume_lookback_days: int = 20,
    social_by_symbol: dict[str, pd.DataFrame] | None = None,
    social_lookback_days: int = 30,
    price_momentum_weight: float = 0.55,
    social_momentum_weight: float = 0.30,
    volume_momentum_weight: float = 0.15,
    min_composite_score: float = 0.05,
) -> dict[str, dict[str, float | int]]:
    """Compute composite momentum/social signals across the symbol universe."""
    results: dict[str, dict[str, float | int]] = {}
    for symbol, df in bars_by_symbol.items():
        results[symbol] = compute_momentum_and_trend(
            df,
            lookback_days,
            ma_days,
            short_lookback_days=short_lookback_days,
            volume_lookback_days=volume_lookback_days,
            social_df=(social_by_symbol or {}).get(symbol),
            social_lookback_days=social_lookback_days,
            price_momentum_weight=price_momentum_weight,
            social_momentum_weight=social_momentum_weight,
            volume_momentum_weight=volume_momentum_weight,
            min_composite_score=min_composite_score,
        )
    return results
