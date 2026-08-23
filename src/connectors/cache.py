"""Reading and writing what a provider already told us.

Two layers, deliberately. The DuckDB bar store is the durable one and is keyed by resolution;
``provider_cache`` is a short-lived payload cache in front of it, for the calls whose answers
are not bars. Both are keyed by provider, so one provider's staleness never masks another's.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..data.duckdb_store import DAILY_INTERVAL_MINUTES, bar_end_timestamps
from .frames import _empty_bars, filter_bar_range

logger = logging.getLogger(__name__)

INTRADAY_CACHE_TTL_SECONDS = 900
EOD_CACHE_TTL_SECONDS = 1800
EOD_BAR_FRESH_FOR_DAYS = 3

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

    Every fetcher ends the same way -- stamp, clip to the requested range, keep the newest N --
    so it lives here once rather than in ten near-identical tails.

    The stamping is the part that matters. Providers timestamp a bar at its start; the close
    is the price at its end. Doing that here, at the boundary where a payload becomes bars,
    means the frame a fetcher *returns* and the rows it *stores* carry the same convention.
    Normalising only on the way into the store left those two disagreeing by one interval,
    and a read that blended fresh provider bars with cached ones got both -- the same bar
    twice, an interval apart.
    """
    if parsed.empty:
        return parsed
    work = parsed.copy()
    work["timestamp"] = bar_end_timestamps(work["timestamp"], interval_minutes)
    # Kept only so a provider that supplies its own adjusted series (Alpaca sends
    # ``Adjustment.ALL``) has somewhere to put it, and so the frame's shape never depends on
    # the symbol's dividend policy. Nothing in this module writes it any more: distributions
    # are cash events, recorded in the ``dividends`` table and booked by the ledger, so a
    # cached bar stays whatever the market actually printed.
    if "adjusted_close" not in work:
        work["adjusted_close"] = pd.to_numeric(work["close"], errors="coerce")
    work = filter_bar_range(work, start_date, end_date)
    if limit:
        work = work.tail(limit)
    return work.reset_index(drop=True)


def _fresh_cached_bars(
    bars: pd.DataFrame,
    interval_minutes: int,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Cached bars if they are recent enough for their own resolution, else nothing.

    One rule for every grid rather than one per category: a bar is stale once several of its
    own intervals have passed. Daily bars get the session-boundary allowance they always had,
    since the next one does not exist until the market closes again.
    """
    if bars.empty or "timestamp" not in bars:
        return pd.DataFrame()
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return pd.DataFrame()
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    latest = timestamps.max()
    if int(interval_minutes) >= DAILY_INTERVAL_MINUTES:
        return bars if now_ts - latest <= pd.Timedelta(days=EOD_BAR_FRESH_FOR_DAYS) else pd.DataFrame()
    if latest.date() == now_ts.date():
        return bars
    max_age = pd.Timedelta(seconds=max(INTRADAY_CACHE_TTL_SECONDS, int(interval_minutes or 15) * 60 * 3))
    return bars if now_ts - latest <= max_age else pd.DataFrame()


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


