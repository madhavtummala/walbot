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

#: Trading minutes in a regular session, for converting a bar count to calendar time.
_SESSION_MINUTES = 390


def window_start_for(bar_count: int, interval_minutes: int, window_end: pd.Timestamp) -> pd.Timestamp:
    """The calendar instant ``bar_count`` bars of ``interval_minutes`` reaches back to.

    Counted in *sessions* rather than by a slack multiplier: bars only exist for 390 minutes of
    each weekday, so the conversion is how many sessions the count covers and then how far back
    those weekdays sit. A flat multiplier had to over-reach to stay safe, which asked the
    provider for more history than the window needed on every cold fetch.
    """
    if bar_count <= 0 or interval_minutes <= 0:
        return window_end
    minutes = int(bar_count) * int(interval_minutes)
    if int(interval_minutes) >= DAILY_INTERVAL_MINUTES:
        sessions = int(bar_count)
    else:
        sessions = max(int(-(-minutes // _SESSION_MINUTES)), 1)
    # Weekends carry no bars, so N sessions span N*7/5 calendar days. One extra session of
    # slack absorbs holidays, which are rare enough not to be worth a calendar for.
    calendar_days = int(-(-(sessions + 1) * 7 // 5))
    local = window_end.tz_convert(MARKET_TZ)
    start = (local.normalize() - pd.Timedelta(days=calendar_days)) + pd.Timedelta(
        hours=_SESSION_OPEN[0], minutes=_SESSION_OPEN[1]
    )
    return start.tz_convert("UTC")


def _merge_bars(held: pd.DataFrame, fetched: pd.DataFrame) -> pd.DataFrame:
    """Combine cached and freshly fetched bars, newest write winning on a shared timestamp."""
    frames = [frame for frame in (held, fetched) if frame is not None and not frame.empty]
    if not frames:
        return _empty_bars()
    if len(frames) == 1:
        return frames[0].sort_values("timestamp").reset_index(drop=True)
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def _horizon_key(provider: str, symbol: str, interval_minutes: int) -> str:
    return f"bar_horizon:{provider}:{str(symbol).upper()}:{int(interval_minutes)}"


def _provider_horizon(provider: str, symbol: str, interval_minutes: int) -> pd.Timestamp | None:
    """The earliest bar this provider has ever served for a symbol, if we have learned it.

    A provider's history is finite -- Schwab serves 259 days of intraday -- so a window
    reaching past it has a permanent leading gap. Without remembering the horizon, every call
    re-requests it and gets the same nothing back, which is precisely the repeated-work
    failure the TTL had.
    """
    from ..data.state_store import load_state

    stored = load_state(_horizon_key(provider, symbol, interval_minutes), None)
    if not stored:
        return None
    try:
        return pd.Timestamp(stored).tz_convert("UTC")
    except (TypeError, ValueError):
        return None


def _record_horizon(
    provider: str, symbol: str, interval_minutes: int, bars: pd.DataFrame, asked_from: pd.Timestamp
) -> None:
    """Remember that ``asked_from`` reached further back than the provider actually goes."""
    from ..data.state_store import save_state

    earliest = None
    if bars is not None and not bars.empty:
        stamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dropna()
        earliest = stamps.min() if not stamps.empty else None
    if earliest is None or earliest <= asked_from:
        return
    save_state(_horizon_key(provider, symbol, interval_minutes), earliest.isoformat())


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


def missing_ranges(
    bars: pd.DataFrame,
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    interval_minutes: int,
    earliest_available: pd.Timestamp | None = None,
    now: datetime | None = None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """The sub-ranges of ``[window_start, window_end]`` the cache cannot answer.

    **Only the leading and trailing edges can be real holes.** Anything bracketed by cached
    bars was already inside a fetched span, so if it is empty the market was shut -- measured
    against the live cache, every one of GLD's nine interior gaps across 195 sessions is a
    holiday (Christmas, New Year, MLK, Presidents' Day, Good Friday, Memorial Day, Juneteenth,
    July 3rd). Treating those as holes would re-request them on every call, forever, and get
    nothing back each time: the same failure the TTL had, in a new place.

    So this returns at most two ranges, which is also the right shape for the cost. A provider
    call is latency-bound -- one session costs 0.81s and eighty cost 1.10s -- so asking for a
    generous contiguous span is nearly free while an extra round trip is not. Splitting a
    holiday-riddled window into a dozen exact requests would be strictly slower than two.
    """
    frontier = last_complete_bar_end(interval_minutes, now)
    window_end = min(window_end, frontier)
    if window_end <= window_start:
        return []

    latest = cached_bars_frontier(bars)
    if latest is None:
        return [(window_start, window_end)]

    stamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dropna()
    earliest = stamps.min()
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    # Leading edge: history older than anything held. Skipped once the provider has told us it
    # has nothing further back -- otherwise every call re-probes a horizon that will not move.
    reach = window_start
    if earliest_available is not None:
        reach = max(reach, earliest_available)
    if earliest - reach > pd.Timedelta(minutes=int(interval_minutes)):
        gaps.append((reach, earliest))

    # Trailing edge: bars that have completed since the last fetch.
    if latest < frontier:
        gaps.append((latest, window_end))

    return gaps


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
) -> None:
    if bars.empty:
        return
    try:
        from ..data.duckdb_store import write_market_bars

        write_market_bars(provider, symbol, int(interval_minutes), bars)
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
