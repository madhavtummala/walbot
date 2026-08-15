from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.data.universe import resolve_project_path


logger = logging.getLogger(__name__)

DUCKDB_STATE_PATH = os.getenv("STATE_DUCKDB_PATH", "data/walbot.duckdb")
MARKET_BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"]

#: A daily bar's resolution, in minutes. A calendar day rather than a 390-minute session so
#: the ordering "1m < 5m < 15m < 1d" holds and a coarser tier is always a larger number.
DAILY_INTERVAL_MINUTES = 1440

#: Where a regular US equities session ends, in exchange-local time. Daily bars are stamped
#: here rather than at whatever midnight their provider chose -- see ``bar_end_timestamps``.
MARKET_TIMEZONE = "America/New_York"
SESSION_CLOSE_HOUR = 16


def bar_end_timestamps(timestamps: pd.Series, interval_minutes: int) -> pd.Series:
    """Move bar-start stamps to the instant the bar's close actually happened.

    Providers stamp a bar at its *start* but report ``close`` as the price at its *end*. For a
    5-minute bar that skew is five minutes; for a daily bar it is most of a day, and the
    providers do not even agree on which midnight to use -- Alpaca files a session at 00:00
    ET, Schwab at 01:00, yfinance at 19:00 or 20:00. Three rows for one fact, spread over
    twenty hours.

    Left uncorrected that is not merely untidy. A lookup asking "what was the price at 10:00"
    against a daily bar stamped 00:00 gets that day's *closing* price -- six hours of
    lookahead, silently, with a number that looks entirely reasonable. And no two providers'
    rows for the same session can ever be recognised as the same observation.

    So every bar is stamped when its close occurred: bar start plus the interval for intraday,
    and the 16:00 exchange-local session close for dailies.
    """
    stamps = pd.to_datetime(timestamps, utc=True, errors="coerce")
    if int(interval_minutes) < DAILY_INTERVAL_MINUTES:
        return stamps + pd.Timedelta(minutes=int(interval_minutes))
    # The session is identified by the bar's *UTC* date, not its exchange-local one. Each
    # provider stamps a session somewhere inside that UTC day -- Schwab 05:00, Alpaca 04:00,
    # yfinance 00:00 -- so the UTC date is the one thing all three agree on. Reading the local
    # date instead would push yfinance's 00:00 UTC stamp back to the previous evening in ET
    # and file the whole series one session early.
    sessions = stamps.dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    closes = sessions + pd.Timedelta(hours=SESSION_CLOSE_HOUR)
    return closes.dt.tz_localize(MARKET_TIMEZONE, nonexistent="shift_forward", ambiguous=True).dt.tz_convert("UTC")


def parse_interval_minutes(timeframe: Any, default: int = 0) -> int:
    """Resolution in minutes from either a number or a timeframe string like ``15m``/``1d``.

    Kept tolerant because the same value arrives as an int from algorithms, as ``"15m"`` from
    provider code, and as whatever was written to the cache before this column existed.
    """
    if isinstance(timeframe, (int, float)) and not isinstance(timeframe, bool):
        return int(timeframe) if timeframe > 0 else default
    text = str(timeframe or "").strip().lower()
    if not text:
        return default
    try:
        if text.endswith("min"):
            return int(text[:-3])
        if text.endswith("m"):
            return int(text[:-1])
        if text.endswith("h"):
            return int(text[:-1]) * 60
        if text.endswith("d"):
            return int(text[:-1]) * DAILY_INTERVAL_MINUTES
        return int(text)
    except ValueError:
        return default


def _drop_unclosed_bars(bars: pd.DataFrame, interval_minutes: int, now: datetime | None = None) -> pd.DataFrame:
    """Discard bars whose interval has not finished yet.

    A bar fetched mid-interval is a partial aggregate -- its close is simply the last trade so
    far -- but nothing ever revisits it, so it is cached as though it were final and stays
    wrong forever. Timestamps now name the bar's end, which makes the test trivial: a bar
    ending in the future has not happened yet.
    """
    if bars.empty:
        return bars
    cutoff = pd.Timestamp(now or _now())
    return bars.loc[pd.to_datetime(bars["timestamp"], utc=True) <= cutoff].reset_index(drop=True)


def timeframe_label(interval_minutes: int) -> str:
    """Human-facing name for a resolution, for logs and the cache summary."""
    minutes = int(interval_minutes)
    if minutes % DAILY_INTERVAL_MINUTES == 0:
        return f"{minutes // DAILY_INTERVAL_MINUTES}d"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


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


def _market_bars_columns(connection) -> set[str]:
    try:
        rows = connection.execute("PRAGMA table_info('market_bars')").fetchall()
    except Exception:
        return set()
    return {str(row[1]) for row in rows}


def _migrate_market_bars(connection) -> None:
    """Fold the old ``(category, timeframe)`` key into a single ``interval_minutes`` column.

    EOD and intraday were separate categories with separate read paths, which is exactly the
    split this store no longer makes: both are timestamped bars, differing only in resolution.
    The migration is row-preserving because the cache accumulates intraday history from live
    runs and the backtester's depth is that history -- see ``src/execution/replay.py``.
    """
    columns = _market_bars_columns(connection)
    if not columns or "interval_minutes" in columns:
        return

    timeframes = [str(row[0]) for row in connection.execute("SELECT DISTINCT timeframe FROM market_bars").fetchall()]
    connection.execute("ALTER TABLE market_bars RENAME TO market_bars_legacy")
    _create_market_bars(connection)
    # CASE over the distinct timeframes actually present, so the mapping runs in DuckDB rather
    # than pulling every row through Python.
    if timeframes:
        cases = " ".join(
            f"WHEN timeframe = '{timeframe}' THEN {parse_interval_minutes(timeframe, DAILY_INTERVAL_MINUTES)}"
            for timeframe in timeframes
        )
        interval_sql = f"CASE {cases} ELSE {DAILY_INTERVAL_MINUTES} END"
        connection.execute(
            f"""
            INSERT OR REPLACE INTO market_bars
                (provider, symbol, interval_minutes, timestamp, open, high, low, close, volume,
                 adjusted_close, raw_json)
            SELECT provider, symbol, {interval_sql}, timestamp, open, high, low, close, volume,
                   adjusted_close, raw_json
            FROM market_bars_legacy
            """
        )
    moved = int(connection.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0] or 0)
    connection.execute("DROP TABLE market_bars_legacy")
    logger.info("Migrated %s cached market bars onto interval_minutes", moved)


def _create_market_bars(connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS market_bars (
            provider VARCHAR NOT NULL,
            symbol VARCHAR NOT NULL,
            interval_minutes INTEGER NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL,
            open DOUBLE NOT NULL,
            high DOUBLE NOT NULL,
            low DOUBLE NOT NULL,
            close DOUBLE NOT NULL,
            volume DOUBLE NOT NULL,
            adjusted_close DOUBLE,
            raw_json VARCHAR,
            PRIMARY KEY (provider, symbol, interval_minutes, timestamp)
        )
        """
    )


#: Bumped when stored rows need rewriting rather than just re-reading. Recorded in app_state
#: so a migration runs once per database instead of on every connect.
MARKET_BARS_SCHEMA_VERSION = 2


def _migrate_bar_timestamps(connection) -> None:
    """Restamp already-cached bars from bar-start to bar-end, once.

    Everything written before this ran carries the provider's own convention, which is what
    made Alpaca, Schwab and yfinance disagree by up to twenty hours about the same session.
    Rewriting in place is safe because the transformation is a pure function of
    ``(timestamp, interval_minutes)`` -- see :func:`bar_end_timestamps`.
    """
    try:
        row = connection.execute(
            "SELECT value FROM app_state WHERE key = 'market_bars_schema_version'"
        ).fetchone()
    except Exception:
        return
    if row and int(str(row[0]).strip('"')) >= MARKET_BARS_SCHEMA_VERSION:
        return

    before = int(connection.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0] or 0)
    # Rebuilt through INSERT OR REPLACE rather than updated in place: two stamps can normalise
    # onto the same session (a provider that filed one session at both 00:00 and 01:00), and
    # an in-place UPDATE would abort on the primary key instead of collapsing the duplicate.
    connection.execute("ALTER TABLE market_bars RENAME TO market_bars_restamp")
    _create_market_bars(connection)
    # Interpolated rather than bound: DuckDB will not take a parameter inside INTERVAL or
    # AT TIME ZONE. All three are module constants, never caller input.
    connection.execute(
        f"""
        INSERT OR REPLACE INTO market_bars
            (provider, symbol, interval_minutes, timestamp, open, high, low, close, volume,
             adjusted_close, raw_json)
        SELECT provider, symbol, interval_minutes,
               CASE WHEN interval_minutes < {DAILY_INTERVAL_MINUTES}
                    THEN timestamp + INTERVAL (interval_minutes) MINUTE
                    ELSE (CAST(timestamp AT TIME ZONE 'UTC' AS DATE)
                          + INTERVAL {SESSION_CLOSE_HOUR} HOUR) AT TIME ZONE '{MARKET_TIMEZONE}'
               END,
               open, high, low, close, volume, adjusted_close, raw_json
        FROM market_bars_restamp
        """
    )
    after = int(connection.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0] or 0)
    connection.execute("DROP TABLE market_bars_restamp")
    connection.execute(
        "INSERT OR REPLACE INTO app_state (key, value, updated_at) VALUES (?, ?, ?)",
        ["market_bars_schema_version", str(MARKET_BARS_SCHEMA_VERSION), _now()],
    )
    logger.info(
        "Restamped %s cached bars to bar-end timestamps (%s collapsed as duplicates)",
        after, before - after,
    )


def initialize_schema(connection) -> None:
    _create_market_bars(connection)
    _migrate_market_bars(connection)
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
    # Imported here rather than at module scope: ``src.data.dividends`` reads ``_connect``
    # from this module, so a top-level import would close the cycle.
    from src.data.dividends import create_dividends_table

    create_dividends_table(connection)
    connection.execute("UPDATE market_bars SET adjusted_close = close WHERE adjusted_close IS NULL")
    # Last, because it records its own completion in app_state, which is created just above.
    _migrate_bar_timestamps(connection)


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
    keep = MARKET_BAR_COLUMNS + (["interval_minutes"] if "interval_minutes" in work else [])
    return (
        work.dropna(subset=["timestamp", "open", "high", "low", "close"])
        [keep]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def available_intervals(
    symbol: str,
    *,
    provider: str | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> list[int]:
    """Cached resolutions for ``symbol``, finest first."""
    clauses = ["symbol = ?"]
    params: list[Any] = [symbol.upper()]
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    query = f"""
        SELECT DISTINCT interval_minutes FROM market_bars
        WHERE {' AND '.join(clauses)}
        ORDER BY interval_minutes ASC
    """
    with _connect(db_path) as connection:
        return [int(row[0]) for row in connection.execute(query, params).fetchall()]


def read_bars(
    symbol: str,
    *,
    interval_minutes: int | None = None,
    provider: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> pd.DataFrame:
    """Bars for one symbol at one resolution, oldest first.

    ``limit`` counts back from the newest bar, which is what every caller wants: the most
    recent N observations, not the first N in the window.
    """
    clauses = ["symbol = ?"]
    params: list[Any] = [symbol.upper()]
    if interval_minutes is not None:
        clauses.append("interval_minutes = ?")
        params.append(int(interval_minutes))
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    if start is not None:
        clauses.append("timestamp >= ?")
        params.append(pd.Timestamp(start).to_pydatetime())
    if end is not None:
        clauses.append("timestamp <= ?")
        params.append(pd.Timestamp(end).to_pydatetime())
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    # One row per timestamp even when several providers hold the same bar, preferring the one
    # that carries an adjusted close.
    query = f"""
        SELECT timestamp, open, high, low, close, volume, adjusted_close, interval_minutes
        FROM (
            SELECT timestamp, open, high, low, close, volume, adjusted_close, interval_minutes,
                   ROW_NUMBER() OVER (PARTITION BY timestamp, interval_minutes
                                      ORDER BY adjusted_close IS NOT NULL DESC, provider) as rnk
            FROM market_bars
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp DESC
            {limit_sql}
        )
        WHERE rnk = 1
        ORDER BY timestamp ASC
    """
    with _connect(db_path) as connection:
        frame = _normalize_bars(connection.execute(query, params).fetchdf())
    # Derived here rather than stored: ``market_bars`` holds only what the market printed.
    from src.data.bars import attach_total_return

    return attach_total_return(frame, symbol)


def read_history(
    symbol: str,
    *,
    lookback_minutes: int,
    end: datetime | None = None,
    max_interval_minutes: int | None = None,
    provider: str | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> pd.DataFrame:
    """The best available view of ``symbol`` over the trailing window, blending resolutions.

    ``lookback_minutes`` is market time -- minutes the exchange was open -- because that is
    what a momentum horizon means; the calendar span to search is derived from it. Resolutions
    are walked finest first, each taking what it covers, with the next coarser tier filling
    the stretch still missing. So a window longer than the intraday cache reaches is served by
    daily bars at its far end rather than coming back truncated.

    That blend is the point of stating horizons in minutes: a lookup only needs *an*
    observation near the moment it asks about, not one on a particular grid.
    """
    from src.data.bars import calendar_days_for, coverage_minutes

    end_ts = pd.Timestamp(end or _now())
    floor = end_ts - pd.Timedelta(days=calendar_days_for(lookback_minutes))
    intervals = [
        interval
        for interval in available_intervals(symbol, provider=provider, db_path=db_path)
        if max_interval_minutes is None or interval <= int(max_interval_minutes)
    ]
    if not intervals:
        return pd.DataFrame(columns=MARKET_BAR_COLUMNS + ["interval_minutes"])

    frames: list[pd.DataFrame] = []
    # Filled from the newest end backwards; each coarser tier only supplies what is still
    # uncovered, so a fine bar always wins over a coarse one for the same instant. The walk
    # stops as soon as the window is covered in market minutes, which is what keeps a coarse
    # tier out of a frame the fine bars already answer in full.
    uncovered_end = end_ts
    covered = 0
    for interval in intervals:
        if uncovered_end <= floor or covered >= lookback_minutes:
            break
        bars = read_bars(
            symbol,
            interval_minutes=interval,
            provider=provider,
            start=floor,
            end=uncovered_end,
            db_path=db_path,
        )
        if bars.empty:
            continue
        frames.append(bars)
        covered += coverage_minutes(bars)
        uncovered_end = pd.Timestamp(bars["timestamp"].min()) - pd.Timedelta(minutes=1)

    if not frames:
        return pd.DataFrame(columns=MARKET_BAR_COLUMNS + ["interval_minutes"])
    merged = pd.concat(frames, ignore_index=True)
    return (
        merged.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="first")
        .reset_index(drop=True)
    )


def write_market_bars(
    provider: str,
    symbol: str,
    interval_minutes: int,
    bars: pd.DataFrame,
    *,
    ttl_seconds: int | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> int:
    normalized = _normalize_bars(bars)
    if normalized.empty:
        return 0
    resolution = int(interval_minutes)
    # Timestamps arrive already stamped at bar-end -- ``connectors.service._provider_bars``
    # does it where a payload becomes bars, so a fetcher's return value and its stored rows
    # cannot disagree. Doing it again here would shift everything a second interval.
    normalized = _drop_unclosed_bars(normalized, resolution)
    if normalized.empty:
        return 0
    rows = []
    for row in normalized.to_dict(orient="records"):
        rows.append(
            (
                provider,
                symbol.upper(),
                resolution,
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
                (provider, symbol, interval_minutes, timestamp, open, high, low, close, volume,
                 adjusted_close, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def _market_bar_filters(
    provider: str | None,
    symbols: list[str] | None,
    interval_minutes: int | None,
) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    if symbols:
        wanted = sorted({symbol.upper() for symbol in symbols})
        clauses.append("symbol IN (" + ",".join(["?"] * len(wanted)) + ")")
        params.extend(wanted)
    if interval_minutes is not None:
        clauses.append("interval_minutes = ?")
        params.append(int(interval_minutes))
    return (f"WHERE {' AND '.join(clauses)}" if clauses else ""), params


def clear_market_bars(
    *,
    provider: str | None = None,
    symbols: list[str] | None = None,
    interval_minutes: int | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> int:
    where, params = _market_bar_filters(provider, symbols, interval_minutes)
    with _connect(db_path) as connection:
        deleted = int(connection.execute(f"SELECT COUNT(*) FROM market_bars {where}", params).fetchone()[0] or 0)
        connection.execute(f"DELETE FROM market_bars {where}", params)
        return deleted


def market_bars_summary(
    *,
    provider: str | None = None,
    symbols: list[str] | None = None,
    interval_minutes: int | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> list[dict[str, Any]]:
    where, params = _market_bar_filters(provider, symbols, interval_minutes)
    query = f"""
        SELECT provider, symbol, interval_minutes, COUNT(*) AS rows,
               MIN(timestamp) AS start, MAX(timestamp) AS end
        FROM market_bars
        {where}
        GROUP BY provider, symbol, interval_minutes
        ORDER BY provider, symbol, interval_minutes
    """
    with _connect(db_path) as connection:
        frame = connection.execute(query, params).fetchdf()
    return [
        {
            "provider": str(row["provider"]),
            "symbol": str(row["symbol"]),
            "interval_minutes": int(row["interval_minutes"]),
            "timeframe": timeframe_label(int(row["interval_minutes"])),
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
