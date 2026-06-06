from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.data.universe import resolve_project_path


DUCKDB_STATE_PATH = os.getenv("STATE_DUCKDB_PATH", "data/walbot.duckdb")
MARKET_BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"]


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise RuntimeError("duckdb is required for columnar data storage. Install project requirements first.") from exc
    return duckdb


def _connect(db_path: str = DUCKDB_STATE_PATH):
    resolved = resolve_project_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = _duckdb().connect(str(resolved))
    initialize_schema(connection)
    return connection


def initialize_schema(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS market_bars (
            category VARCHAR NOT NULL,
            provider VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            timeframe VARCHAR NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume DOUBLE NOT NULL,
            adjusted_close DOUBLE,
            raw_json VARCHAR,
            PRIMARY KEY (category, provider, symbol, timeframe, timestamp)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_records (
            provider VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            mentions DOUBLE NOT NULL,
            sentiment DOUBLE NOT NULL,
            social_score DOUBLE NOT NULL,
            title VARCHAR,
            url VARCHAR,
            raw_json VARCHAR,
            PRIMARY KEY (provider, symbol, timestamp, title)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            key VARCHAR PRIMARY KEY,
            value VARCHAR NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS api_cache (
            category VARCHAR NOT NULL,
            provider VARCHAR NOT NULL,
            cache_key VARCHAR NOT NULL,
            payload VARCHAR NOT NULL,
            fetched_at TIMESTAMPTZ NOT NULL,
            expires_at TIMESTAMPTZ NOT NULL,
            PRIMARY KEY (category, provider, cache_key)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS api_provider_state (
            provider VARCHAR PRIMARY KEY,
            category VARCHAR NOT NULL,
            request_count BIGINT NOT NULL DEFAULT 0,
            window_started_at TIMESTAMPTZ NOT NULL,
            limited_until TIMESTAMPTZ,
            last_error VARCHAR,
            updated_at TIMESTAMPTZ NOT NULL
        )
        """
    )
    connection.execute("UPDATE market_bars SET adjusted_close = close WHERE adjusted_close IS NULL")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_bars(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=MARKET_BAR_COLUMNS)
    work = df.copy()
    if "timestamp" not in work:
        return pd.DataFrame(columns=MARKET_BAR_COLUMNS)
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "adjusted_close"):
        if column not in work:
            work[column] = work["close"] if column == "adjusted_close" and "close" in work else 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["adjusted_close"] = work["adjusted_close"].fillna(work["close"])
    return (
        work.dropna(subset=["timestamp", "open", "high", "low", "close"])
        [MARKET_BAR_COLUMNS]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def market_cache_key(symbols: list[str], timeframe: str, lookback_bars: int | None, start: Any, end: Any) -> str:
    symbols_key = ",".join(sorted({str(symbol).upper() for symbol in symbols}))
    return f"{symbols_key}|{timeframe}|{lookback_bars or ''}|{start or ''}|{end or ''}"


def read_market_bars(
    category: str,
    provider: str | None,
    symbol: str,
    timeframe: str,
    *,
    lookback_bars: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> pd.DataFrame:
    clauses = [
        "category = ?",
        "symbol = ?",
        "timeframe = ?",
    ]
    params: list[Any] = [category, symbol.upper(), timeframe]
    if provider:
        clauses.append("provider = ?")
        params.append(provider)

    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(pd.Timestamp(start).to_pydatetime())
    if end is not None:
        clauses.append("timestamp <= ?")
        params.append(pd.Timestamp(end).to_pydatetime())
    limit_sql = f"LIMIT {int(lookback_bars)}" if lookback_bars else ""
    query = f"""
        SELECT timestamp, open, high, low, close, volume, adjusted_close
        FROM (
            SELECT timestamp, open, high, low, close, volume, adjusted_close,
                   ROW_NUMBER() OVER (PARTITION BY timestamp ORDER BY adjusted_close IS NOT NULL DESC, provider) as rnk
            FROM market_bars
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp DESC
            {limit_sql}
        )
        WHERE rnk = 1
        ORDER BY timestamp ASC
    """
    with _connect(db_path) as connection:
        return _normalize_bars(connection.execute(query, params).fetchdf())


def write_market_bars(
    category: str,
    provider: str,
    symbol: str,
    timeframe: str,
    bars: pd.DataFrame,
    *,
    ttl_seconds: int | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> int:
    normalized = _normalize_bars(bars)
    if normalized.empty:
        return 0
    rows = []
    for row in normalized.to_dict(orient="records"):
        rows.append(
            (
                category,
                provider,
                symbol.upper(),
                timeframe,
                pd.Timestamp(row["timestamp"]).to_pydatetime(),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                float(row.get("adjusted_close", row["close"])),
                None,
            )
        )
    with _connect(db_path) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO market_bars
                (category, provider, symbol, timeframe, timestamp, open, high, low, close, volume,
                 adjusted_close, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def clear_market_bars(
    *,
    category: str | None = None,
    provider: str | None = None,
    symbols: list[str] | None = None,
    timeframe: str | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> int:
    clauses: list[str] = []
    params: list[Any] = []
    if category:
        clauses.append("category = ?")
        params.append(category)
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    if symbols:
        wanted = sorted({symbol.upper() for symbol in symbols})
        clauses.append("symbol IN (" + ",".join(["?"] * len(wanted)) + ")")
        params.extend(wanted)
    if timeframe:
        clauses.append("timeframe = ?")
        params.append(timeframe)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(db_path) as connection:
        deleted = int(connection.execute(f"SELECT COUNT(*) FROM market_bars {where}", params).fetchone()[0] or 0)
        connection.execute(f"DELETE FROM market_bars {where}", params)
        return deleted


def market_bars_summary(
    *,
    category: str | None = None,
    provider: str | None = None,
    symbols: list[str] | None = None,
    timeframe: str | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if category:
        clauses.append("category = ?")
        params.append(category)
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    if symbols:
        wanted = sorted({symbol.upper() for symbol in symbols})
        clauses.append("symbol IN (" + ",".join(["?"] * len(wanted)) + ")")
        params.extend(wanted)
    if timeframe:
        clauses.append("timeframe = ?")
        params.append(timeframe)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        SELECT category, provider, symbol, timeframe, COUNT(*) AS rows,
               MIN(timestamp) AS start, MAX(timestamp) AS end
        FROM market_bars
        {where}
        GROUP BY category, provider, symbol, timeframe
        ORDER BY category, provider, symbol, timeframe
    """
    with _connect(db_path) as connection:
        frame = connection.execute(query, params).fetchdf()
    return [
        {
            "category": str(row["category"]),
            "provider": str(row["provider"]),
            "symbol": str(row["symbol"]),
            "timeframe": str(row["timeframe"]),
            "rows": int(row["rows"]),
            "start": pd.Timestamp(row["start"]).isoformat(),
            "end": pd.Timestamp(row["end"]).isoformat(),
        }
        for row in frame.to_dict(orient="records")
    ]


def read_sentiment_records(
    provider: str,
    symbols: list[str],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> list[dict[str, Any]]:
    wanted = [symbol.upper() for symbol in symbols]
    clauses = ["provider = ?", "symbol IN (" + ",".join(["?"] * len(wanted)) + ")"]
    params: list[Any] = [provider, *wanted]
    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(pd.Timestamp(start).to_pydatetime())
    if end is not None:
        clauses.append("timestamp <= ?")
        params.append(pd.Timestamp(end).to_pydatetime())
    query = f"""
        SELECT timestamp, symbol, mentions, sentiment, social_score, provider, title, url, raw_json
        FROM sentiment_records
        WHERE {' AND '.join(clauses)}
        ORDER BY timestamp ASC
    """
    with _connect(db_path) as connection:
        frame = connection.execute(query, params).fetchdf()
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        raw = row.get("raw_json")
        records.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "symbol": str(row["symbol"]).upper(),
                "mentions": float(row["mentions"]),
                "sentiment": float(row["sentiment"]),
                "social_score": float(row["social_score"]),
                "provider": str(row["provider"]),
                "title": str(row.get("title") or ""),
                "url": str(row.get("url") or ""),
                "raw": json.loads(raw) if raw else {},
            }
        )
    return records


def write_sentiment_records(
    provider: str,
    records: list[dict[str, Any]],
    *,
    ttl_seconds: int | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> int:
    if not records:
        return 0
    rows = []
    for record in records:
        timestamp = pd.to_datetime(record.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        rows.append(
            (
                provider,
                str(record.get("symbol") or "").upper(),
                pd.Timestamp(timestamp).to_pydatetime(),
                float(record.get("mentions") or 0.0),
                float(record.get("sentiment") or 0.0),
                float(record.get("social_score") or 0.0),
                str(record.get("title") or ""),
                str(record.get("url") or ""),
                json.dumps(record.get("raw") or {}, sort_keys=True, default=str),
            )
        )
    if not rows:
        return 0
    with _connect(db_path) as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO sentiment_records
                (provider, symbol, timestamp, mentions, sentiment, social_score, title, url, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def compact_storage(db_path: str = DUCKDB_STATE_PATH) -> None:
    with _connect(db_path) as connection:
        connection.execute("CHECKPOINT")


class DuckDBStore:
    def __init__(self, db_path: str = DUCKDB_STATE_PATH):
        self.db_path = db_path
        # Initialize schema by connecting
        with _connect(self.db_path) as _:
            pass

    def get_market_bars(self, symbol: str, timeframe: str, provider: str, start: datetime, end: datetime) -> pd.DataFrame:
        return read_market_bars(
            category="market_data", # Default to market_data or infer from provider?
            provider=provider,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            db_path=self.db_path
        )

    def write_market_bars(self, symbol: str, timeframe: str, category: str, provider: str, df: pd.DataFrame, ttl_seconds: int = 3600):
        return write_market_bars(
            category=category,
            provider=provider,
            symbol=symbol,
            timeframe=timeframe,
            bars=df,
            ttl_seconds=ttl_seconds,
            db_path=self.db_path
        )

    def cleanup_expired(self):
        # duckdb_store functions don't currently support expiry directly in market_bars table, 
        # but they do in api_cache. 
        # For now, let's just checkpoint.
        compact_storage(self.db_path)

# Alias for backward compatibility with newer refactored code
DataStore = DuckDBStore
