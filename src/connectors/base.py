"""What a market-data provider is, and what it gets for free.

Every provider answers two questions: ``price(symbols)`` (what each symbol trades at now) and
``bars(symbols, interval_minutes=..., lookback_bars=...)`` (OHLCV history at a resolution -- one
method, since intraday and EOD are the same request at different grids).

The base class owns caching, resolution negotiation and normalisation; a subclass implements
``fetch_price`` and ``fetch_bars`` and returns raw vendor output.

Bars read through to DuckDB (the durable, TTL'd store, keyed by provider and resolution).
Quotes keep a short-lived payload cache instead, since there is no bar store for them to live in.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from src.common.timeutils import utc_now

from ..core.config import Config
from ..data.duckdb_store import DAILY_INTERVAL_MINUTES
from ..data.provider_cache import load_cached_payload, save_cached_payload
from .cache import (
    EOD_CACHE_TTL_SECONDS,
    INTRADAY_CACHE_TTL_SECONDS,
    _CALENDAR_SLACK,
    _merge_bars,
    _provider_horizon,
    _record_horizon,
    last_complete_bar_end,
    cached_bars_frontier,
    missing_ranges,
    _provider_bars,
    _quote_cache_key,
    _read_duckdb_bars,
    _write_duckdb_bars,
)
from .frames import _empty_bars, normalize_intraday_frame
from .grid import bars_for_minutes, resolve_bar_minutes
from .sources import MARKET_CATEGORY

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    """A source of prices and bars, with the read-through cache supplied.

    Subclasses implement ``fetch_price`` and ``fetch_bars`` for whatever the cache could not
    answer; everything else -- resolution negotiation, cache reads/writes, normalisation --
    happens here.
    """

    #: Registry key. Names this provider in ``config.*_provider_order``, in the bar store and
    #: in the rate-limit table.
    name: str = ""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} name={self.name!r}>"

    # -- what callers use -------------------------------------------------------------------

    def price(self, symbols: list[str], *, force_refresh: bool = False, **extra: Any) -> dict[str, dict[str, Any]]:
        """``{symbol: {"price": float, "timestamp": ..., "current": bool, ...}}``, cache first.

        A symbol this provider cannot price is simply absent, so the caller falls through to
        the next provider for that symbol alone.
        """
        wanted = [str(symbol).upper() for symbol in symbols]
        resolved: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for symbol in wanted:
            cached = None if force_refresh else load_cached_payload(
                MARKET_CATEGORY, self.name, _quote_cache_key(symbol)
            )
            if isinstance(cached, dict) and cached:
                resolved[symbol] = cached
            else:
                missing.append(symbol)

        for symbol, quote in (self.fetch_price(missing, **extra) if missing else {}).items():
            key = str(symbol).upper()
            if not quote:
                continue
            save_cached_payload(
                MARKET_CATEGORY,
                self.name,
                _quote_cache_key(key),
                quote,
                ttl_seconds=int(getattr(self.config, "market_data_cache_ttl_seconds", 0) or 0),
            )
            resolved[key] = quote
        return resolved

    def bars(
        self,
        symbols: list[str],
        *,
        interval_minutes: int = DAILY_INTERVAL_MINUTES,
        lookback_bars: int | None = None,
        lookback_minutes: int | None = None,
        force_refresh: bool = False,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **extra: Any,
    ) -> dict[str, pd.DataFrame]:
        """OHLCV at ``interval_minutes``, oldest first, served from the store where possible.

        ``interval_minutes`` is a request, not a guarantee: a provider that cannot serve the
        requested grid gets its nearest coarser one instead of failing. Give either
        ``lookback_bars`` or ``lookback_minutes`` -- the latter is converted once the grid is
        known.

        Every frame returned has ``timestamp, open, high, low, close, volume, adjusted_close``,
        stamped at bar *end* in UTC. Bars are what the market printed; distributions are
        recorded separately and booked as cash.
        """
        grid = resolve_bar_minutes(self.name, interval_minutes)
        if lookback_bars is None:
            lookback_bars = bars_for_minutes(int(lookback_minutes or 0), grid)
        ttl_seconds = self._ttl_seconds(grid)
        wanted = [str(symbol).upper() for symbol in symbols]

        # The request as an absolute window. Callers state a relative lookback ("the last N
        # minutes"), which cannot be asked of a store keyed by timestamp -- so it is resolved
        # once, here, and everything below reasons about a range.
        window_end = pd.Timestamp(end_date or utc_now())
        window_end = window_end.tz_localize("UTC") if window_end.tzinfo is None else window_end.tz_convert("UTC")
        # Clamped to the newest bar that can exist *before* the span is measured back from it.
        # Measuring from "now" instead put the whole window past the frontier out of hours --
        # a two-bar request at 08:00 asked for 06:00-08:00, which is after Wednesday's close,
        # so it read as entirely in the future and reported no gaps at all.
        window_end = min(window_end, last_complete_bar_end(grid))
        if start_date is not None:
            window_start = pd.Timestamp(start_date)
            window_start = window_start.tz_localize("UTC") if window_start.tzinfo is None else window_start.tz_convert("UTC")
        else:
            # Calendar span for the bars asked for. Generous on purpose: nights and weekends
            # carry no bars, so a span measured in trading minutes under-reaches badly.
            window_start = window_end - pd.Timedelta(minutes=int(lookback_bars or 0) * grid * _CALENDAR_SLACK)

        resolved: dict[str, pd.DataFrame] = {}
        for symbol in wanted:
            held = (
                _empty_bars() if force_refresh
                else _read_duckdb_bars(
                    self.name, symbol, grid,
                    start=window_start.to_pydatetime(), end=window_end.to_pydatetime(),
                )
            )
            gaps = (
                [(window_start, window_end)] if force_refresh
                else missing_ranges(
                    held,
                    window_start=window_start,
                    window_end=window_end,
                    interval_minutes=grid,
                    earliest_available=_provider_horizon(self.name, symbol, grid),
                )
            )
            for gap_start, gap_end in gaps:
                fetched = self._fetch_range(
                    symbol, grid, gap_start, gap_end, lookback_bars, ttl_seconds, **extra
                )
                held = _merge_bars(held, fetched)
                # A leading fetch that came back no earlier than what we already had is the
                # provider saying it has nothing further back. Recorded so the next call does
                # not re-probe a horizon that cannot move -- Schwab serves 259 days and a
                # longer window would otherwise pay for that discovery on every run.
                if gap_start < (cached_bars_frontier(held) or gap_end):
                    _record_horizon(self.name, symbol, grid, held, gap_start)

            resolved[symbol] = held.tail(lookback_bars).reset_index(drop=True) if lookback_bars else held

        return {symbol: resolved.get(symbol, _empty_bars()) for symbol in wanted}

    def _fetch_range(
        self, symbol: str, grid: int, start: Any, end: Any,
        lookback_bars: int | None, ttl_seconds: int, **extra: Any,
    ) -> pd.DataFrame:
        """One provider call for one gap, normalised and stored."""
        raw = self.fetch_bars(
            [symbol],
            interval_minutes=grid,
            lookback_bars=lookback_bars,
            start_date=start.to_pydatetime() if hasattr(start, "to_pydatetime") else start,
            end_date=end.to_pydatetime() if hasattr(end, "to_pydatetime") else end,
            **extra,
        )
        # Keyed by name, never by truthiness: a DataFrame has no boolean value, so ``a or b``
        # raises rather than falling through.
        answers = raw or {}
        payload = answers.get(symbol)
        if payload is None:
            payload = answers.get(str(symbol).upper())
        if payload is None:
            return _empty_bars()
        frame = _provider_bars(normalize_intraday_frame(payload), grid, limit=None)
        if not frame.empty:
            _write_duckdb_bars(self.name, str(symbol).upper(), grid, frame, ttl_seconds=ttl_seconds)
        return frame

    def _ttl_seconds(self, interval_minutes: int) -> int:
        """How long a bar at this resolution stays fresh."""
        if interval_minutes >= DAILY_INTERVAL_MINUTES:
            return int(getattr(self.config, "eod_market_data_cache_ttl_seconds", EOD_CACHE_TTL_SECONDS))
        return int(getattr(self.config, "intraday_market_data_cache_ttl_seconds", INTRADAY_CACHE_TTL_SECONDS))

    # -- what a provider implements ---------------------------------------------------------

    @abstractmethod
    def fetch_price(self, symbols: list[str], **extra: Any) -> dict[str, dict[str, Any]]:
        """Live quotes from the vendor, for the symbols the cache could not answer.

        Build each one with :func:`~src.connectors.frames._normalize_quote` so provenance
        (``timestamp``, ``current``) is recorded the same way by every provider.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_bars(
        self,
        symbols: list[str],
        *,
        interval_minutes: int,
        lookback_bars: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **extra: Any,
    ) -> dict[str, pd.DataFrame]:
        """Raw frames from the vendor at ``interval_minutes``, for the symbols still missing.

        Return whatever shape the vendor gives; normalisation runs on it. A symbol the vendor
        has nothing for is simply absent from the mapping. Raise
        :class:`~src.connectors.sources.ProviderUnavailable` when the provider cannot answer at
        all, so the dispatcher moves to the next one.
        """
        raise NotImplementedError
