"""Daily-bar feature frames, and the signal rows the options runner ranks underlyings with.

Not an algorithm registry, despite the name and despite once holding a branch per strategy.
The real algorithms live in ``src/algorithms/`` behind ``AlgorithmPlugin``; what survives here
is the daily-bar feature frame the backtest fetcher also uses, and one scoring rule --
``dual_momentum`` in the *rule set* sense, a 126/252-day blend -- which ``options/swing.py``
uses to pick which underlyings to buy contracts on.

The other five branches (``trend_following``, ``mean_reversion``, ``breakout``, ``risk_parity``
and a second ``fast_momentum`` scorer) were reachable only from their own tests, and the
``fast_momentum`` one was a fourth implementation of dual momentum on top of that.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.common.config_utils import as_float
from src.data.bars import signal_price
from src.data.signals.signals import compute_social_trend_score

STRATEGY_LABELS = {
    "dca": "DCA",
    "bursty_dca": "Bursty DCA",
    "fast_momentum": "Fast Momentum",
    "dual_momentum": "Dual Momentum",
    # Named for what it does: classify SPY into GROWING / FLAT / FALLING / CRISIS and rotate
    # between growth, covered-call income, cash, and hedges.
    "spy_rotation": "SPY Rotation",
}


def prepared_strategy_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["open"] = pd.to_numeric(work.get("open", work["close"]), errors="coerce").fillna(work["close"])
    work["volume"] = pd.to_numeric(work.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    # Performance questions are asked of the total-return series, price-level questions of the
    # raw one. ``close`` stays exactly what printed, because the replay fills against it and an
    # order executes at a real price; every "how has this done" feature below reads ``signal``
    # so a dividend payer is not marked down for having paid. The two must not be mixed inside
    # one feature -- a raw close against a total-return average is a comparison of different
    # quantities, which is how ``z_20`` and the trend gate would silently drift.
    signal = signal_price(work)
    work["ret_21"] = signal / signal.shift(21) - 1
    work["ret_63"] = signal / signal.shift(63) - 1
    work["ret_126"] = signal / signal.shift(126) - 1
    work["ret_252"] = signal / signal.shift(252) - 1
    work["sma_20"] = signal.rolling(20).mean()
    work["sma_50"] = signal.rolling(50).mean()
    work["sma_200"] = signal.rolling(200).mean()
    work["daily_ret"] = signal.pct_change()
    work["realized_vol"] = work["daily_ret"].rolling(20).std() * math.sqrt(252)
    work["z_20"] = (signal - work["sma_20"]) / signal.rolling(20).std().replace(0, pd.NA)
    log_volume = work["volume"].map(math.log1p)
    work["volume_z"] = (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std().replace(0, pd.NA)
    return work


def strategy_row_from_prepared(
    strategy: str,
    symbol: str,
    work: pd.DataFrame,
    social_df: pd.DataFrame | None = None,
    *,
    social_lookback_days: int = 30,
    social_weight: float = 0.0,
) -> dict[str, Any]:
    """One symbol's row under the ``dual_momentum`` rule set: 60% of R126 plus 40% of R252.

    ``strategy`` is kept in the signature because the row carries it through to the caller and
    because an unknown name must produce a neutral row rather than an error, which is what a
    caller reading a saved strategy id needs.
    """
    if work.empty:
        return {"symbol": symbol, "side": "FLAT", "signal": 0, "score": 0.0, "target_weight": 0.0}
    row = work.iloc[-1]
    close = as_float(row.get("close"))
    ret_21 = as_float(row.get("ret_21"))
    ret_63 = as_float(row.get("ret_63"))
    ret_126 = as_float(row.get("ret_126"))
    ret_252 = as_float(row.get("ret_252"))
    sma_50 = as_float(row.get("sma_50"))
    sma_200 = as_float(row.get("sma_200"))
    vol = max(as_float(row.get("realized_vol"), 0.2), 0.05)
    z_20 = as_float(row.get("z_20"))
    volume_z = as_float(row.get("volume_z"))

    side = "FLAT"
    score = 0.0
    price_score = None
    reason = "No active setup"
    social = {"social_score": 0.0, "sentiment": 0.0}

    if strategy == "dual_momentum":
        social = compute_social_trend_score(social_df, social_lookback_days)
        price_score = 0.6 * ret_126 + 0.4 * ret_252
        sentiment_tilt = max(0.0, min(float(social_weight or 0.0), 0.5)) * as_float(social.get("social_score"))
        score = price_score + sentiment_tilt
        if score > 0 and ret_126 > 0:
            side, reason = "LONG", "Positive absolute and relative momentum"
        elif score < -0.02 and ret_126 < 0:
            side, reason = "SHORT", "Negative absolute momentum"
        if abs(sentiment_tilt) > 1e-9:
            reason = f"{reason}; sentiment tilt {sentiment_tilt:+.2f}"

    return {
        "symbol": symbol,
        "side": side,
        "signal": 1 if side == "LONG" else -1 if side == "SHORT" else 0,
        "score": score,
        "close": close,
        "price_score": price_score,
        "social_score": as_float(social.get("social_score")),
        "sentiment": as_float(social.get("sentiment")),
        "ret_short": ret_21,
        "ret_N": ret_63,
        "ret_126": ret_126,
        "ret_252": ret_252,
        "realized_vol": vol,
        "sma_50": sma_50,
        "sma_long": sma_200,
        "z_20": z_20,
        "volume_score": volume_z,
        "reason": reason,
    }


def strategy_row(
    strategy: str,
    symbol: str,
    df: pd.DataFrame,
    social_df: pd.DataFrame | None = None,
    *,
    social_lookback_days: int = 30,
    social_weight: float = 0.0,
) -> dict[str, Any]:
    return strategy_row_from_prepared(
        strategy,
        symbol,
        prepared_strategy_frame(df),
        social_df,
        social_lookback_days=social_lookback_days,
        social_weight=social_weight,
    )


def _rank_strategy_rows(strategy: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Top five long, bottom two short, everything else flat -- then ordered for display."""
    rows = [dict(row) for row in rows]
    if strategy == "dual_momentum":
        ranked = sorted(rows, key=lambda item: item["score"], reverse=True)
        selected_symbols = {row["symbol"] for row in ranked[: min(5, len(ranked))] if row["score"] > 0}
        weak_symbols = {row["symbol"] for row in ranked[-2:] if row["score"] < -0.02}
        for row in rows:
            if row["symbol"] in selected_symbols:
                row["side"] = "LONG"
                row["signal"] = 1
            elif row["symbol"] in weak_symbols:
                row["side"] = "SHORT"
                row["signal"] = -1
            else:
                row["side"] = "FLAT"
                row["signal"] = 0
                row["reason"] = "Outside current dual-momentum selection"
    return sorted(rows, key=lambda item: (item["signal"] != 1, item["signal"] == 0, -abs(item.get("score", 0.0))))


def strategy_signal_rows(
    strategy: str,
    bars_by_symbol: dict[str, pd.DataFrame],
    social_by_symbol: dict[str, pd.DataFrame] | None = None,
    *,
    social_lookback_days: int = 30,
    social_weight: float = 0.0,
) -> list[dict[str, Any]]:
    """Rank a set of symbols under ``strategy``. The options runner's underlying selector."""
    rows = [
        strategy_row(
            strategy,
            symbol,
            df,
            (social_by_symbol or {}).get(symbol),
            social_lookback_days=social_lookback_days,
            social_weight=social_weight,
        )
        for symbol, df in bars_by_symbol.items()
    ]
    return _rank_strategy_rows(strategy, rows)
