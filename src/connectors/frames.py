"""Shaping whatever a provider returned into the one frame everything else reads.

Columns ``timestamp, open, high, low, close, volume, adjusted_close``, stamped at bar *end* in
UTC. Bars are what the market printed: nothing here folds a distribution into a price, because
those are recorded separately and booked as cash.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from ..common.config_utils import json_number

logger = logging.getLogger(__name__)

def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def _normalize_quote(provider: str, symbol: str, price: Any, raw: Any, timestamp: Any = None) -> dict[str, Any] | None:
    parsed_price = json_number(price)
    if parsed_price is None or parsed_price <= 0:
        return None
    parsed_timestamp = pd.to_datetime(timestamp, utc=True, errors="coerce") if timestamp is not None else pd.NaT
    if pd.isna(parsed_timestamp):
        parsed_timestamp = pd.Timestamp.now(tz="UTC")
    return {
        "symbol": symbol.upper(),
        "price": parsed_price,
        "timestamp": parsed_timestamp.isoformat(),
        "provider": provider,
        "raw": raw,
    }


def _bars_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records = []
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"])
    for row in work.to_dict(orient="records"):
        records.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "adjusted_close": float(row.get("adjusted_close", row["close"])),
            }
        )
    return records


def _records_to_bars(records: Any) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"])
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"])
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "adjusted_close"):
        if column not in df.columns:
            df[column] = df["close"] if column == "adjusted_close" else 0.0
        df[column] = pd.to_numeric(df.get(column), errors="coerce")
    df["adjusted_close"] = df["adjusted_close"].fillna(df["close"])
    return (
        df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        [["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"]]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _news_record(provider: str, symbol: str, timestamp: Any, title: str, url: str, sentiment: Any, raw: Any) -> dict[str, Any]:
    parsed_timestamp = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if pd.isna(parsed_timestamp):
        parsed_timestamp = pd.Timestamp.now(tz="UTC")
    sentiment_value = json_number(sentiment)
    sentiment_value = 0.0 if sentiment_value is None else max(-1.0, min(1.0, sentiment_value))
    return {
        "timestamp": parsed_timestamp.isoformat(),
        "symbol": symbol.upper(),
        "mentions": 1.0,
        "sentiment": sentiment_value,
        "social_score": sentiment_value,
        "provider": provider,
        "title": title,
        "url": url,
        "raw": raw,
    }




def normalize_intraday_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"])
    work = df.copy()
    if isinstance(work.columns, pd.MultiIndex):
        work.columns = [str(column[0]).lower() for column in work.columns]
    work = work.reset_index()
    rename = {
        "Datetime": "timestamp",
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjusted_close",
        "Volume": "volume",
    }
    work = work.rename(columns={key: value for key, value in rename.items() if key in work.columns})
    work = work.rename(columns={column: str(column).lower() for column in work.columns})
    if "adj close" in work.columns and "adjusted_close" not in work.columns:
        work = work.rename(columns={"adj close": "adjusted_close"})
    if "timestamp" not in work:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"])
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "adjusted_close"):
        if column not in work:
            work[column] = work["close"] if column == "adjusted_close" and "close" in work else 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["adjusted_close"] = work["adjusted_close"].fillna(work["close"])
    return (
        work.dropna(subset=["timestamp", "open", "high", "low", "close"])
        [["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"]]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def filter_bar_range(
    bars: pd.DataFrame,
    start_date: datetime | None,
    end_date: datetime | None,
) -> pd.DataFrame:
    if bars.empty or (start_date is None and end_date is None):
        return bars
    work = bars.copy()
    timestamps = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    if start_date is not None:
        work = work.loc[timestamps >= pd.Timestamp(start_date)]
        timestamps = timestamps.loc[work.index]
    if end_date is not None:
        work = work.loc[timestamps < pd.Timestamp(end_date)]
    return work.reset_index(drop=True)
