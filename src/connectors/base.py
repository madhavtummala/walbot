"""What a market-data provider is, and what it gets for free.

Two questions, and every provider answers both:

``price(symbols)``
    What each symbol trades at now.
``bars(symbols, interval_minutes=..., lookback_bars=...)``
    OHLCV history at a resolution. **One method, not two.** Intraday and EOD were never two
    kinds of data, only two grids -- ``interval_minutes=5`` and ``interval_minutes=1440`` are
    the same request -- and the bar store has always been keyed by resolution for exactly that
    reason. Splitting them at the provider produced two registries, two config sections and two
    copies of the same read-through in every provider.

The contract is a *template method*, not a list of signatures. An earlier version of this file
declared abstract methods and provided nothing, so implementing it was pure overhead and nothing
ever did. Every provider stayed a loose function and each hand-rolled its own caching -- which
drifted, as duplicated logic does: two of four bar providers checked a payload cache the other
two did not, so the same question got a different answer depending on who served it.

So the base owns the caching and asks a provider for the one thing only it knows: how to get raw
frames out of its vendor. Declare ``name``, implement ``fetch_bars`` and ``fetch_price``, and
never touch a cache, a TTL or a normaliser again.

**Bars read through to DuckDB.** It is the durable store, keyed by provider and resolution, and
``_fresh_cached_bars`` already applies the TTL. A short-lived payload cache in front of it only
bought a second place for a stale bar to hide. Quotes keep the payload cache: they are not bars,
and there is no bar store for them to live in.
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
    _fresh_cached_bars,
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

    Subclasses implement ``fetch_price`` and ``fetch_bars``, which are called only for what the
    cache could not answer and return raw vendor output. Everything else -- resolution
    negotiation, cache reads, normalisation, cache writes -- happens here, once.
    """

    #: Registry key. The same string names this provider in ``config.*_provider_order``, in the
    #: bar store and in the rate-limit table, so one identifier accounts for it everywhere. An
    #: attribute rather than an argument because it used to be a string literal repeated through
    #: each provider -- fourteen times in the worst of them.
    name: str = ""

    def __init__(self, config: Config) -> None:
        self.config = config

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} name={self.name!r}>"

    # -- what callers use -------------------------------------------------------------------

    def price(self, symbols: list[str], *, force_refresh: bool = False, **extra: Any) -> dict[str, dict[str, Any]]:
        """``{symbol: {"price": float, "timestamp": ..., "current": bool, ...}}``, cache first.

        A symbol this provider cannot price is simply *absent*, so the caller falls through to
        the next provider for that symbol alone rather than discarding the whole batch.
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
            save_cached_payload(MARKET_CATEGORY, self.name, _quote_cache_key(key), quote)
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
        requested grid gets its nearest *coarser* one instead of failing, because horizons are
        stated in market minutes and a coarser answer still answers the question. Give either
        ``lookback_bars`` or ``lookback_minutes`` -- the latter is converted once the grid is
        known, which is the only point at which the conversion is correct.

        Every frame returned has ``timestamp, open, high, low, close, volume, adjusted_close``,
        stamped at bar *end* in UTC. Bars are what the market printed: nothing here folds a
        distribution into a price -- those are recorded separately and booked as cash.
        """
        grid = resolve_bar_minutes(self.name, interval_minutes)
        # Converted *after* resolution, never before: the caller asks for a preferred grid and
        # the provider may serve a coarser one, so a bar count derived from the request would be
        # wrong by exactly that ratio -- 235 five-minute bars asked of a provider that answers
        # in fifteens, which needs 79.
        if lookback_bars is None:
            lookback_bars = bars_for_minutes(int(lookback_minutes or 0), grid)
        ttl_seconds = self._ttl_seconds(grid)
        wanted = [str(symbol).upper() for symbol in symbols]
        resolved: dict[str, pd.DataFrame] = {}
        missing: list[str] = []

        for symbol in wanted:
            cached = (
                _empty_bars()
                if force_refresh
                else _fresh_cached_bars(
                    _read_duckdb_bars(self.name, symbol, grid, limit=lookback_bars), grid
                )
            )
            if cached.empty:
                missing.append(symbol)
            else:
                resolved[symbol] = cached.tail(lookback_bars).reset_index(drop=True)

        if missing:
            # The provider is handed the missing symbols as a *list*, so a vendor with a batch
            # endpoint keeps its single call and one that must loop still loops -- internally,
            # where that is its own business rather than a difference visible in the cache path.
            fresh = self.fetch_bars(
                missing,
                interval_minutes=grid,
                lookback_bars=lookback_bars,
                start_date=start_date,
                end_date=end_date,
                **extra,
            )
            for symbol, raw in (fresh or {}).items():
                key = str(symbol).upper()
                frame = _provider_bars(
                    normalize_intraday_frame(raw),
                    grid,
                    start_date=start_date,
                    end_date=end_date,
                    limit=lookback_bars,
                )
                if not frame.empty:
                    _write_duckdb_bars(self.name, key, grid, frame, ttl_seconds=ttl_seconds)
                resolved[key] = frame

        return {symbol: resolved.get(symbol, _empty_bars()) for symbol in wanted}

    def _ttl_seconds(self, interval_minutes: int) -> int:
        """How long a bar at this resolution stays fresh.

        The one place the two grids are still treated differently, and for a real reason: a
        daily bar is final once the session closes, a five-minute bar is stale in minutes.
        """
        if interval_minutes >= DAILY_INTERVAL_MINUTES:
            return int(getattr(self.config, "eod_market_data_cache_ttl_seconds", EOD_CACHE_TTL_SECONDS))
        return int(getattr(self.config, "intraday_market_data_cache_ttl_seconds", INTRADAY_CACHE_TTL_SECONDS))

    # -- what a provider implements ---------------------------------------------------------

    @abstractmethod
    def fetch_price(self, symbols: list[str], **extra: Any) -> dict[str, dict[str, Any]]:
        """Live quotes from the vendor, for the symbols the cache could not answer.

        Build each one with :func:`~src.connectors.frames._normalize_quote` so provenance --
        ``timestamp`` and ``current`` -- is recorded the same way by every provider.
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

        Return whatever shape the vendor gives -- normalisation runs on it. A symbol the vendor
        has nothing for is simply absent from the mapping. Raise
        :class:`~src.connectors.sources.ProviderUnavailable` when the provider cannot answer at
        all, so the dispatcher moves to the next one rather than recording an empty result.
        """
        raise NotImplementedError
