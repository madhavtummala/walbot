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

from ..core.config import Config
from ..data.duckdb_store import DAILY_INTERVAL_MINUTES
from ..data.provider_cache import load_cached_payload, save_cached_payload
from .cache import (
    EOD_CACHE_TTL_SECONDS,
    INTRADAY_CACHE_TTL_SECONDS,
    cache_is_current,
    cached_bars_frontier,
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
        explicit_range = start_date is not None or end_date is not None
        resolved: dict[str, pd.DataFrame] = {}
        stored: dict[str, pd.DataFrame] = {}
        missing: list[str] = []

        for symbol in wanted:
            if force_refresh:
                missing.append(symbol)
                continue
            held = _read_duckdb_bars(self.name, symbol, grid, limit=lookback_bars)
            # An explicit range is a cache-warming request for a specific window, so it is
            # always served from the provider rather than from what happens to be stored.
            if not explicit_range and cache_is_current(held, grid):
                resolved[symbol] = held.tail(lookback_bars).reset_index(drop=True)
            else:
                stored[symbol] = held
                missing.append(symbol)

        if missing:
            # Fetch the *gap*, not the window. Every bar before the cached frontier is complete
            # and immutable -- a printed bar never changes -- so re-requesting the whole
            # lookback re-downloads thousands of rows to arrive back at what is already stored.
            # The batch starts at the earliest frontier among the symbols that need one, which
            # over-fetches slightly for the more current of them and still bounds the request by
            # the gap rather than by the horizon.
            gap_start = start_date
            if gap_start is None and not force_refresh:
                frontiers = [
                    frontier for frontier in
                    (cached_bars_frontier(stored.get(symbol, _empty_bars())) for symbol in missing)
                    if frontier is not None
                ]
                # Only when every symbol has history; one cold symbol needs the full window.
                if frontiers and len(frontiers) == len(missing):
                    gap_start = min(frontiers).to_pydatetime()

            fresh = self.fetch_bars(
                missing,
                interval_minutes=grid,
                lookback_bars=lookback_bars,
                start_date=gap_start,
                end_date=end_date,
                **extra,
            )
            for symbol in missing:
                key = str(symbol).upper()
                raw = (fresh or {}).get(symbol, (fresh or {}).get(key))
                frame = _provider_bars(
                    normalize_intraday_frame(raw) if raw is not None else _empty_bars(),
                    grid,
                    start_date=start_date,
                    end_date=end_date,
                    limit=lookback_bars,
                )
                if not frame.empty:
                    _write_duckdb_bars(self.name, key, grid, frame, ttl_seconds=ttl_seconds)
                # Merged with what was already held, since the fetch covered only the gap.
                previous = stored.get(key, _empty_bars())
                if not previous.empty and not explicit_range:
                    frame = (
                        pd.concat([previous, frame], ignore_index=True)
                        .drop_duplicates(subset="timestamp", keep="last")
                        .sort_values("timestamp")
                        .tail(lookback_bars)
                        .reset_index(drop=True)
                    )
                resolved[key] = frame

        return {symbol: resolved.get(symbol, _empty_bars()) for symbol in wanted}

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
