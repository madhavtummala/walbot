"""Reading and writing what a provider already told us.

Two layers: the DuckDB bar store is durable and keyed by resolution; ``provider_cache`` is a
short-lived payload cache in front of it, for answers that are not bars. Both are keyed by
provider, so one provider's staleness never masks another's.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..core.interfaces import MARKET_TZ
from ..data.duckdb_store import DAILY_INTERVAL_MINUTES, bar_end_timestamps
from .frames import _empty_bars, filter_bar_range

logger = logging.getLogger(__name__)

INTRADAY_CACHE_TTL_SECONDS = 900
EOD_CACHE_TTL_SECONDS = 1800
EOD_BAR_FRESH_FOR_DAYS = 3

#: Regular US equity session, market-local. Intraday bars only appear between these.
_SESSION_OPEN = (9, 30)
_SESSION_CLOSE = (16, 0)


def _in_session(local: pd.Timestamp) -> bool:
    """Whether the regular session is running at ``local`` (market-local, weekdays only).

    Holidays read as in-session, which is the safe direction: it only costs a fetch that comes
    back with nothing new.
    """
    if local.weekday() >= 5:
        return False
    return (local.hour, local.minute) >= _SESSION_OPEN and (local.hour, local.minute) <= _SESSION_CLOSE


def _last_session_close(local: pd.Timestamp) -> pd.Timestamp:
    """The most recent regular close at or before ``local``, market-local."""
    close_today = local.normalize() + pd.Timedelta(hours=_SESSION_CLOSE[0], minutes=_SESSION_CLOSE[1])
    candidate = close_today if local >= close_today else close_today - pd.Timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= pd.Timedelta(days=1)
    return candidate

def _quote_cache_key(symbol: str) -> str:
    return symbol.upper()


def _provider_bars(
    parsed: pd.DataFrame,
    interval_minutes: int,
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Turn a parsed provider payload into bars this project's conventions agree with.

    Stamp at bar *end* (providers timestamp at bar start), clip to the requested range, keep
    the newest N. Must run before a bar is stored or returned so both paths share one
    convention -- stamping only on write let a fresh read and a cached one disagree by one
    interval and double up the same bar.
    """
    if parsed.empty:
        return parsed
    work = parsed.copy()
    work["timestamp"] = bar_end_timestamps(work["timestamp"], interval_minutes)
    # Dividends are cash events recorded separately; a bar's adjusted_close, if a provider
    # supplies one, is kept as-is and never derived here.
    if "adjusted_close" not in work:
        work["adjusted_close"] = pd.to_numeric(work["close"], errors="coerce")
    work = filter_bar_range(work, start_date, end_date)
    if limit:
        work = work.tail(limit)
    return work.reset_index(drop=True)


def last_complete_bar_end(interval_minutes: int, now: datetime | None = None) -> pd.Timestamp:
    """End stamp of the newest bar that can possibly be complete right now, in UTC.

    This is what replaces a TTL for bar data. A completed bar is immutable -- Wednesday's
    14:35 five-minute bar will never differ from what the exchange printed -- so asking how
    *old* a cached bar is answers the wrong question. The only thing that decides whether the
    cache is behind is whether a newer bar could exist yet, and outside the session the answer
    is simply no.
    """
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    local = now_ts.tz_convert(MARKET_TZ)

    # A daily bar is only complete once its session has closed.
    if int(interval_minutes) >= DAILY_INTERVAL_MINUTES or not _in_session(local):
        return _last_session_close(local).tz_convert("UTC")

    grid = max(int(interval_minutes or 1), 1)
    open_minute = _SESSION_OPEN[0] * 60 + _SESSION_OPEN[1]
    elapsed = (local.hour * 60 + local.minute) - open_minute
    completed = max((elapsed // grid) * grid, 0)
    frontier = local.normalize() + pd.Timedelta(minutes=open_minute + completed)
    return frontier.tz_convert("UTC")


def cached_bars_frontier(bars: pd.DataFrame) -> pd.Timestamp | None:
    """The newest bar stamp held, or ``None`` when there is nothing cached."""
    if bars is None or bars.empty or "timestamp" not in bars:
        return None
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dropna()
    return timestamps.max() if not timestamps.empty else None


def cache_is_current(
    bars: pd.DataFrame, interval_minutes: int, *, now: datetime | None = None
) -> bool:
    """Whether the cache already holds every bar that could have printed.

    ``True`` means no request can return anything new, so the fetch is skipped entirely.
    ``False`` means only the *gap* needs fetching -- see ``BaseConnector.bars`` -- not the whole
    window, because everything before the frontier is immutable and already stored.
    """
    latest = cached_bars_frontier(bars)
    if latest is None:
        return False
    return latest >= last_complete_bar_end(interval_minutes, now)


def _news_cache_key(symbols: list[str]) -> str:
    return ",".join(sorted({symbol.upper() for symbol in symbols}))


def _read_duckdb_bars(
    provider: str,
    symbol: str,
    interval_minutes: int,
    *,
    limit: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    try:
        from ..data.duckdb_store import read_bars

        bars = read_bars(
            symbol,
            interval_minutes=int(interval_minutes),
            provider=provider,
            limit=limit,
            start=start,
            end=end,
        )
        if bars.empty:
            logger.debug(
                "DuckDB market cache miss provider=%s symbol=%s interval=%sm limit=%s start=%s end=%s",
                provider,
                symbol,
                interval_minutes,
                limit,
                start,
                end,
            )
        return bars
    except Exception as exc:
        logger.warning(
            "DuckDB market cache read failed provider=%s symbol=%s interval=%sm limit=%s start=%s end=%s: %s",
            provider,
            symbol,
            interval_minutes,
            limit,
            start,
            end,
            exc,
        )
    return _empty_bars()


def _write_duckdb_bars(
    provider: str,
    symbol: str,
    interval_minutes: int,
    bars: pd.DataFrame,
    *,
    ttl_seconds: int | None,
) -> None:
    if bars.empty:
        return
    try:
        from ..data.duckdb_store import write_market_bars

        write_market_bars(provider, symbol, int(interval_minutes), bars, ttl_seconds=ttl_seconds)
    except Exception as exc:
        logger.warning(
            "DuckDB market cache write failed provider=%s symbol=%s interval=%sm rows=%s: %s",
            provider,
            symbol,
            interval_minutes,
            len(bars),
            exc,
        )


def _read_duckdb_sentiment(provider: str, symbols: list[str]) -> list[dict[str, Any]]:
    try:
        from ..data.duckdb_store import read_sentiment_records

        return read_sentiment_records(provider, symbols)
    except RuntimeError as exc:
        logger.debug("DuckDB sentiment cache unavailable: %s", exc)
    return []


def _write_duckdb_sentiment(provider: str, records: list[dict[str, Any]], *, ttl_seconds: int | None) -> None:
    if not records:
        return
    try:
        from ..data.duckdb_store import write_sentiment_records

        write_sentiment_records(provider, records, ttl_seconds=ttl_seconds)
    except RuntimeError as exc:
        logger.debug("DuckDB sentiment cache unavailable: %s", exc)
