from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


TIMESTAMP_COLUMNS = ("timestamp", "date", "datetime")
MENTION_COLUMNS = ("mentions", "mention_count", "post_count", "volume", "social_volume")
SENTIMENT_COLUMNS = ("sentiment", "sentiment_score", "bullish_sentiment")
SOCIAL_SCORE_COLUMNS = ("social_score", "trend_score")


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lowered = {column.lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_social_trends_csv(path: str, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load per-symbol social trend data from a CSV file.

    Expected columns:
      symbol, timestamp/date, and at least one of mentions, sentiment, or social_score.
    Optional columns are normalized to: timestamp, symbol, mentions, sentiment, social_score.
    """
    if not path:
        return {}

    csv_path = Path(path)
    if not csv_path.exists():
        logger.warning("Social trends CSV does not exist: %s", csv_path)
        return {}

    df = pd.read_csv(csv_path)
    if df.empty:
        return {}

    df.columns = [str(column).strip() for column in df.columns]
    symbol_col = _first_existing_column(df, ("symbol", "ticker"))
    time_col = _first_existing_column(df, TIMESTAMP_COLUMNS)
    mention_col = _first_existing_column(df, MENTION_COLUMNS)
    sentiment_col = _first_existing_column(df, SENTIMENT_COLUMNS)
    social_score_col = _first_existing_column(df, SOCIAL_SCORE_COLUMNS)

    if symbol_col is None or time_col is None:
        raise ValueError("Social trends CSV must include symbol/ticker and timestamp/date columns.")
    if mention_col is None and sentiment_col is None and social_score_col is None:
        raise ValueError("Social trends CSV must include mentions, sentiment, or social_score.")

    normalized = pd.DataFrame()
    normalized["timestamp"] = pd.to_datetime(df[time_col], utc=True)
    normalized["symbol"] = df[symbol_col].astype(str).str.upper().str.strip()
    normalized["mentions"] = pd.to_numeric(df[mention_col], errors="coerce").fillna(0.0) if mention_col else 0.0
    normalized["sentiment"] = pd.to_numeric(df[sentiment_col], errors="coerce").fillna(0.0) if sentiment_col else 0.0
    normalized["social_score"] = (
        pd.to_numeric(df[social_score_col], errors="coerce").fillna(0.0)
        if social_score_col
        else float("nan")
    )
    normalized = normalized.dropna(subset=["timestamp", "symbol"])

    if symbols is not None:
        universe = {symbol.upper() for symbol in symbols}
        normalized = normalized[normalized["symbol"].isin(universe)]

    grouped = (
        normalized.groupby(["symbol", "timestamp"], as_index=False)
        .agg({"mentions": "sum", "sentiment": "mean", "social_score": "mean"})
        .sort_values(["symbol", "timestamp"])
    )

    trends_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol, symbol_df in grouped.groupby("symbol"):
        trends_by_symbol[symbol] = symbol_df.drop(columns=["symbol"]).reset_index(drop=True)
    return trends_by_symbol


def truncate_social_history(
    social_by_symbol: dict[str, pd.DataFrame],
    end_timestamp: pd.Timestamp,
) -> dict[str, pd.DataFrame]:
    """Return social data available at or before end_timestamp."""
    end_timestamp = pd.to_datetime(end_timestamp, utc=True)
    truncated: dict[str, pd.DataFrame] = {}
    for symbol, df in social_by_symbol.items():
        if df.empty:
            truncated[symbol] = df
            continue
        timestamps = pd.to_datetime(df["timestamp"], utc=True)
        truncated[symbol] = df[timestamps <= end_timestamp].reset_index(drop=True)
    return truncated
