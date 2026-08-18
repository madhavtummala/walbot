from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import pandas as pd

from src.common.config_utils import as_float


@dataclass(frozen=True)
class CandidateSpec:
    symbol: str
    bucket: str
    group: str
    priority: float


PREFERRED_CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec("SPY", "Core US equity", "sp500", 1.00),
    CandidateSpec("VOO", "Core US equity", "sp500", 0.94),
    CandidateSpec("VTI", "Core US equity", "total_us", 0.96),
    CandidateSpec("ITOT", "Core US equity", "total_us", 0.88),
    CandidateSpec("ACWI", "Global / ex-US equity", "global_equity", 0.90),
    CandidateSpec("VT", "Global / ex-US equity", "global_equity", 0.84),
    CandidateSpec("IEFA", "Global / ex-US equity", "developed_ex_us", 0.86),
    CandidateSpec("ACWX", "Global / ex-US equity", "developed_ex_us", 0.78),
    CandidateSpec("IEMG", "Global / ex-US equity", "emerging_markets", 0.84),
    CandidateSpec("SPEM", "Global / ex-US equity", "emerging_markets", 0.74),
    CandidateSpec("VTV", "Style & dividends", "value", 0.82),
    CandidateSpec("SCHD", "Style & dividends", "dividend_value", 0.80),
    CandidateSpec("VIG", "Style & dividends", "dividend_growth", 0.76),
    CandidateSpec("XSD", "High-beta sectors", "semiconductors", 0.78),
    CandidateSpec("XBI", "High-beta sectors", "biotech", 0.76),
    CandidateSpec("PBD", "High-beta sectors", "clean_energy", 0.72),
    CandidateSpec("XOP", "High-beta sectors", "energy_producers", 0.70),
    CandidateSpec("GLD", "Real assets", "gold", 0.82),
    CandidateSpec("IAU", "Real assets", "gold", 0.74),
    CandidateSpec("SLV", "Real assets", "silver", 0.72),
    CandidateSpec("USO", "Real assets", "oil", 0.68),
    CandidateSpec("USL", "Real assets", "oil", 0.64),
    CandidateSpec("BNO", "Real assets", "oil", 0.62),
    CandidateSpec("CPER", "Real assets", "copper", 0.66),
    CandidateSpec("BIL", "Defensive", "cash_like", 0.92),
    CandidateSpec("SHY", "Defensive", "short_treasury", 0.88),
    CandidateSpec("AGG", "Defensive", "aggregate_bonds", 0.84),
    CandidateSpec("BND", "Defensive", "aggregate_bonds", 0.80),
    CandidateSpec("IUSB", "Defensive", "core_bonds", 0.78),
    CandidateSpec("IEF", "Defensive", "intermediate_treasury", 0.82),
    CandidateSpec("TLT", "Defensive", "long_treasury", 0.76),
    CandidateSpec("QQQ", "Legacy growth", "nasdaq_100", 0.52),
    CandidateSpec("RSP", "Legacy equity", "equal_weight", 0.50),
    CandidateSpec("IWM", "Legacy equity", "small_cap", 0.48),
    CandidateSpec("IBIT", "Speculative satellite", "bitcoin", 0.42),
)

BUCKET_CAPS = {
    "Core US equity": 2,
    "Global / ex-US equity": 3,
    "Style & dividends": 3,
    "High-beta sectors": 4,
    "Real assets": 4,
    "Defensive": 6,
    "Legacy growth": 1,
    "Legacy equity": 1,
    "Speculative satellite": 1,
}

MIN_HISTORY_ROWS = 260
MAX_STALENESS_DAYS = 10
MIN_AVG_DOLLAR_VOLUME = 10_000.0


def preferred_symbols(tradable_symbols: set[str]) -> list[str]:
    return [spec.symbol for spec in PREFERRED_CANDIDATES if spec.symbol in tradable_symbols]


def candidate_specs_by_symbol(tradable_symbols: set[str]) -> dict[str, CandidateSpec]:
    return {spec.symbol: spec for spec in PREFERRED_CANDIDATES if spec.symbol in tradable_symbols}


def _avg_dollar_volume(df: pd.DataFrame) -> float:
    tail = df.tail(20)
    close = pd.to_numeric(tail.get("close"), errors="coerce")
    volume = pd.to_numeric(tail.get("volume"), errors="coerce")
    mean = (close * volume).dropna().mean()
    return 0.0 if pd.isna(mean) else float(mean)


def _return_over(df: pd.DataFrame, days: int) -> float:
    if len(df) <= days:
        return 0.0
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(close) <= days:
        return 0.0
    start = as_float(close.iloc[-days - 1])
    end = as_float(close.iloc[-1])
    return end / start - 1 if start > 0 else 0.0


def _trend_gap(df: pd.DataFrame) -> float:
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    if len(close) < 200:
        return 0.0
    sma_200 = float(close.tail(200).mean())
    latest = as_float(close.iloc[-1])
    return latest / sma_200 - 1 if sma_200 > 0 else 0.0


def _row_for_candidate(
    *,
    spec: CandidateSpec,
    name: str,
    df: pd.DataFrame,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    latest = pd.Timestamp(work["timestamp"].iloc[-1])
    staleness_days = max(int((as_of.normalize() - latest.normalize()).days), 0)
    history_rows = len(work)
    avg_dollar_volume = _avg_dollar_volume(work)
    ret_63 = _return_over(work, 63)
    ret_126 = _return_over(work, 126)
    trend_gap = _trend_gap(work)
    recency_score = max(0.0, 1.0 - (staleness_days / MAX_STALENESS_DAYS))
    history_score = min(history_rows / 756, 1.0)
    liquidity_score = min(math.log10(max(avg_dollar_volume, 1.0)) / 8.0, 1.0)
    trend_score = max(min((0.65 * ret_126) + (0.25 * ret_63) + (0.10 * trend_gap), 1.0), -1.0)
    score = (
        1.15 * spec.priority
        + 1.10 * recency_score
        + 0.70 * history_score
        + 0.55 * liquidity_score
        + 0.25 * trend_score
    )
    return {
        "symbol": spec.symbol,
        "name": name,
        "bucket": spec.bucket,
        "group": spec.group,
        "score": round(float(score), 6),
        "latest_bar": latest.strftime("%Y-%m-%d"),
        "history_rows": int(history_rows),
        "staleness_days": int(staleness_days),
        "avg_dollar_volume": round(avg_dollar_volume, 2),
        "ret_63": round(ret_63, 6),
        "ret_126": round(ret_126, 6),
        "trend_gap": round(trend_gap, 6),
    }


def recommend_universe_rows(
    *,
    tradable_names: dict[str, str],
    bars_by_symbol: dict[str, pd.DataFrame],
    max_symbols: int = 12,
    as_of: pd.Timestamp | None = None,
) -> dict[str, Any]:
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.to_datetime(as_of, utc=True)
    specs = candidate_specs_by_symbol(set(tradable_names))
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for symbol, spec in specs.items():
        df = bars_by_symbol.get(symbol)
        if df is None or df.empty:
            rejected.append({"symbol": symbol, "name": tradable_names[symbol], "bucket": spec.bucket, "reason": "no_data"})
            continue
        row = _row_for_candidate(spec=spec, name=tradable_names[symbol], df=df, as_of=as_of)
        if row["history_rows"] < MIN_HISTORY_ROWS:
            row["reason"] = "short_history"
            rejected.append(row)
            continue
        if row["staleness_days"] > MAX_STALENESS_DAYS:
            row["reason"] = "stale_bars"
            rejected.append(row)
            continue
        if row["avg_dollar_volume"] < MIN_AVG_DOLLAR_VOLUME:
            row["reason"] = "thin_iex_volume"
            rejected.append(row)
            continue
        eligible.append(row)

    eligible.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    selected_groups: set[str] = set()
    bucket_counts: dict[str, int] = {}

    for row in eligible:
        if len(selected) >= max_symbols:
            break
        if row["group"] in selected_groups:
            continue
        bucket = row["bucket"]
        if bucket_counts.get(bucket, 0) >= BUCKET_CAPS.get(bucket, 1):
            continue
        selected.append(row)
        selected_groups.add(str(row["group"]))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    for row in eligible:
        if len(selected) >= max_symbols:
            break
        if row["group"] in selected_groups:
            continue
        selected.append(row)
        selected_groups.add(str(row["group"]))

    selected_symbols = {row["symbol"] for row in selected}
    alternates = [row for row in eligible if row["symbol"] not in selected_symbols]
    return {
        "rows": selected,
        "eligible": eligible,
        "alternates": alternates,
        "rejected": sorted(rejected, key=lambda item: (str(item.get("reason", "")), str(item.get("symbol", "")))),
        "candidate_count": len(specs),
        "eligible_count": len(eligible),
    }
