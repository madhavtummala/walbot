from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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
