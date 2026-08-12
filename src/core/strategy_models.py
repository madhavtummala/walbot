from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.data.signals.signals import compute_social_trend_score

STRATEGY_LABELS = {
    "dca": "DCA",
    "bursty_dca": "Bursty DCA",
    "fast_momentum": "Fast Momentum",
    # Named for what it does: classify SPY into GROWING / FLAT / FALLING / CRISIS and rotate
    # between growth, covered-call income, cash, and hedges.
    "spy_rotation": "SPY Rotation",
}

DEFENSIVE_MOMENTUM_DEFENSIVE_SYMBOLS = {"BIL", "SHY", "SPTS", "IEF", "GOVT", "AGG", "BND", "IUSB", "STIP", "TLT", "GLD"}
DEFENSIVE_MOMENTUM_DEFENSIVE_ORDER = ("BIL", "SHY", "SPTS", "IEF", "GOVT", "AGG", "BND", "IUSB", "STIP", "TLT", "GLD")
DEFENSIVE_MOMENTUM_RISK_ON_LIMIT = 5
DEFENSIVE_MOMENTUM_DEFENSIVE_LIMIT = 3


def json_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed) or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _finite(value: Any, default: float = 0.0) -> float:
    parsed = json_number(value)
    return default if parsed is None else parsed


def prepared_strategy_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work["open"] = pd.to_numeric(work.get("open", work["close"]), errors="coerce").fillna(work["close"])
    work["volume"] = pd.to_numeric(work.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    work["ret_21"] = work["close"] / work["close"].shift(21) - 1
    work["ret_63"] = work["close"] / work["close"].shift(63) - 1
    work["ret_126"] = work["close"] / work["close"].shift(126) - 1
    work["ret_252"] = work["close"] / work["close"].shift(252) - 1
    work["sma_20"] = work["close"].rolling(20).mean()
    work["sma_50"] = work["close"].rolling(50).mean()
    work["sma_200"] = work["close"].rolling(200).mean()
    work["daily_ret"] = work["close"].pct_change()
    work["realized_vol"] = work["daily_ret"].rolling(20).std() * math.sqrt(252)
    work["realized_vol_252"] = work["daily_ret"].rolling(252).std() * math.sqrt(252)
    work["z_20"] = (work["close"] - work["sma_20"]) / work["close"].rolling(20).std().replace(0, pd.NA)
    work["high_55"] = pd.to_numeric(work.get("high", work["close"]), errors="coerce").rolling(55).max()
    work["low_55"] = pd.to_numeric(work.get("low", work["close"]), errors="coerce").rolling(55).min()
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
    if work.empty:
        return {"symbol": symbol, "side": "FLAT", "signal": 0, "score": 0.0, "target_weight": 0.0}
    row = work.iloc[-1]
    close = _finite(row.get("close"))
    ret_21 = _finite(row.get("ret_21"))
    ret_63 = _finite(row.get("ret_63"))
    ret_126 = _finite(row.get("ret_126"))
    ret_252 = _finite(row.get("ret_252"))
    sma_50 = _finite(row.get("sma_50"))
    sma_200 = _finite(row.get("sma_200"))
    vol = max(_finite(row.get("realized_vol"), 0.2), 0.05)
    vol_252 = max(_finite(row.get("realized_vol_252"), vol), 0.05)
    z_20 = _finite(row.get("z_20"))
    volume_z = _finite(row.get("volume_z"))
    high_55 = _finite(row.get("high_55"))
    low_55 = _finite(row.get("low_55"))

    side = "FLAT"
    score = 0.0
    momentum_score = None
    risk_adjusted_score = None
    cash_hurdle = None
    absolute_ok = None
    reason = "No active setup"
    social = compute_social_trend_score(social_df, social_lookback_days) if strategy == "dual_momentum" else {
        "social_score": 0.0,
        "sentiment": 0.0,
    }
    if strategy == "trend_following":
        score = (0.55 * ret_63) + (0.30 * ret_126) + (0.15 * ((close / sma_200 - 1) if sma_200 else 0))
        if close > sma_50 > sma_200 and ret_63 > 0:
            side, reason = "LONG", "Price above 50/200 SMA with positive 63D trend"
        elif close < sma_50 < sma_200 and ret_63 < 0:
            side, reason = "SHORT", "Price below 50/200 SMA with negative 63D trend"
    elif strategy == "mean_reversion":
        trend_ok = close > sma_200 if sma_200 else False
        score = -z_20
        if trend_ok and z_20 <= -1.0:
            side, reason = "LONG", "Oversold vs 20D mean while above 200D trend"
        elif not trend_ok and z_20 >= 1.0:
            side, reason = "SHORT", "Overbought bounce inside a weak long trend"
    elif strategy == "breakout":
        range_width = (high_55 - low_55) / close if close else 0.0
        score = ret_63 + max(volume_z, 0.0) * 0.05 - range_width * 0.1
        if high_55 and close >= high_55 * 0.995 and volume_z >= 0:
            side, reason = "LONG", "Near 55D high with confirming volume"
        elif low_55 and close <= low_55 * 1.005 and volume_z >= 0:
            side, reason = "SHORT", "Near 55D low with expanding volume"
    elif strategy == "risk_parity":
        score = 1 / vol
        side, reason = "LONG", "Low volatility sleeve selected for risk-balanced exposure"
    elif strategy == "dual_momentum":
        price_score = 0.6 * ret_126 + 0.4 * ret_252
        sentiment_tilt = max(0.0, min(float(social_weight or 0.0), 0.5)) * _finite(social.get("social_score"))
        score = price_score + sentiment_tilt
        if score > 0 and ret_126 > 0:
            side, reason = "LONG", "Positive absolute and relative momentum"
        elif score < -0.02 and ret_126 < 0:
            side, reason = "SHORT", "Negative absolute momentum"
        if abs(sentiment_tilt) > 1e-9:
            reason = f"{reason}; sentiment tilt {sentiment_tilt:+.2f}"
    elif strategy == "fast_momentum":
        momentum_score = (0.70 * ret_252) + (0.30 * ret_63)
        risk_adjusted_score = momentum_score / vol_252
        score = risk_adjusted_score
        side = "FLAT"
        if symbol in DEFENSIVE_MOMENTUM_DEFENSIVE_SYMBOLS:
            reason = "Defensive candidate for risk-off rotation"
        else:
            reason = "Waiting for absolute and relative momentum confirmation"

    return {
        "symbol": symbol,
        "side": side,
        "signal": 1 if side == "LONG" else -1 if side == "SHORT" else 0,
        "score": score,
        "close": close,
        "price_score": (0.6 * ret_126 + 0.4 * ret_252) if strategy == "dual_momentum" else None,
        "social_score": _finite(social.get("social_score")),
        "sentiment": _finite(social.get("sentiment")),
        "ret_short": ret_21,
        "ret_N": ret_63,
        "ret_126": ret_126,
        "ret_252": ret_252,
        "momentum_score": momentum_score,
        "risk_adjusted_score": risk_adjusted_score,
        "cash_hurdle": cash_hurdle,
        "absolute_ok": absolute_ok,
        "realized_vol": vol_252 if strategy == "fast_momentum" else vol,
        "sma_50": sma_50,
        "sma_long": sma_200,
        "z_20": z_20,
        "volume_score": volume_z,
        "reason": reason,
    }


def _defensive_momentum_flat_reason(row: dict[str, Any], cash_hurdle: float) -> str:
    if str(row["symbol"]) in DEFENSIVE_MOMENTUM_DEFENSIVE_SYMBOLS:
        return "Defensive sleeve inactive while risk-on momentum is healthy"
    if _finite(row.get("ret_252")) <= 0:
        return "12-month return is below zero"
    if _finite(row.get("ret_252")) <= cash_hurdle:
        return "12-month return is below the cash hurdle"
    if _finite(row.get("momentum_score")) <= 0:
        return "Blended 12M/3M momentum is below zero"
    return "Outside current dual-momentum selection"


def _rank_defensive_momentum_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in rows]
    bil_row = next((row for row in rows if str(row["symbol"]) == "BIL"), None)
    cash_hurdle = max(0.0, _finite(bil_row.get("ret_252") if bil_row else None))

    for row in rows:
        row["cash_hurdle"] = cash_hurdle
        row["absolute_ok"] = int(
            str(row["symbol"]) not in DEFENSIVE_MOMENTUM_DEFENSIVE_SYMBOLS
            and _finite(row.get("ret_252")) > cash_hurdle
            and _finite(row.get("momentum_score")) > 0
        )
        row["side"] = "FLAT"
        row["signal"] = 0
        row["reason"] = _defensive_momentum_flat_reason(row, cash_hurdle)

    risk_on = [
        row
        for row in rows
        if int(row.get("absolute_ok", 0)) == 1 and _finite(row.get("risk_adjusted_score")) > 0
    ]
    risk_on.sort(
        key=lambda item: (_finite(item.get("risk_adjusted_score")), _finite(item.get("momentum_score"))),
        reverse=True,
    )

    if risk_on:
        selected_symbols = {row["symbol"] for row in risk_on[:DEFENSIVE_MOMENTUM_RISK_ON_LIMIT]}
        for row in rows:
            if row["symbol"] in selected_symbols:
                row["side"] = "LONG"
                row["signal"] = 1
                row["reason"] = "Risk-on rank"
        return sorted(rows, key=lambda item: (item["signal"] != 1, -_finite(item.get("score"))))

    available_defensive = {
        str(row["symbol"]): row
        for row in rows
        if str(row["symbol"]) in DEFENSIVE_MOMENTUM_DEFENSIVE_SYMBOLS
    }
    selected_defensive = [
        available_defensive[symbol]
        for symbol in DEFENSIVE_MOMENTUM_DEFENSIVE_ORDER
        if symbol in available_defensive
    ][:DEFENSIVE_MOMENTUM_DEFENSIVE_LIMIT]

    for row in selected_defensive:
        row["side"] = "LONG"
        row["signal"] = 1
        row["reason"] = "Risk-on sleeve failed absolute momentum; rotating to defensive assets"

    return sorted(
        rows,
        key=lambda item: (
            item["signal"] != 1,
            DEFENSIVE_MOMENTUM_DEFENSIVE_ORDER.index(item["symbol"])
            if item["symbol"] in DEFENSIVE_MOMENTUM_DEFENSIVE_ORDER
            else 99,
            -_finite(item.get("score")),
        ),
    )


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
    rows = [dict(row) for row in rows]
    if strategy == "risk_parity":
        rows = sorted(rows, key=lambda item: item["score"], reverse=True)
        selected = rows[: min(5, len(rows))]
        selected_symbols = {row["symbol"] for row in selected}
        for row in rows:
            if row["symbol"] not in selected_symbols:
                row["side"] = "FLAT"
                row["signal"] = 0
                row["reason"] = "Outside current risk-balanced sleeve"
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
    if strategy == "fast_momentum":
        return _rank_defensive_momentum_rows(rows)
    return sorted(rows, key=lambda item: (item["signal"] != 1, item["signal"] == 0, -abs(item.get("score", 0.0))))


def strategy_signal_rows(
    strategy: str,
    bars_by_symbol: dict[str, pd.DataFrame],
    social_by_symbol: dict[str, pd.DataFrame] | None = None,
    *,
    social_lookback_days: int = 30,
    social_weight: float = 0.0,
) -> list[dict[str, Any]]:
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
