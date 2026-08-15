"""What a data connector has to provide.

These contracts are deliberately *derived* rather than designed. An earlier attempt at this
layer defined an abstraction first and left the real providers where they were: the registry
ended up holding one 47-line yfinance wrapper while Schwab, Alpaca, Finnhub and Alpha Vantage
stayed as loose functions in ``service.py``, and the whole stack was eventually deleted as dead
weight. So every signature below is the one the existing fetchers already share, and each
provider moves behind it by extraction, never by reimplementation.

Capabilities are split across three ABCs rather than gathered into one because they genuinely
differ: Alpha Vantage serves quotes and news but no bars, yfinance serves bars but is not a
quote source. A single interface would force every provider to declare methods it cannot
answer, which is how an abstraction starts lying about what it can do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd


class Connector(ABC):
    """Shared identity and configuration for every data provider."""

    #: Registry key. Matches the name used in ``config.*_provider_order`` and in the cache and
    #: rate-limit tables, so one string identifies the provider everywhere it is accounted for.
    name: str = ""

    def __init__(self, config: Any) -> None:
        self.config = config

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"<{type(self).__name__} name={self.name!r}>"


class BarsProvider(Connector):
    """A source of OHLCV bars, at one or both resolutions.

    Both methods return ``{symbol: DataFrame}`` with the columns ``timestamp, open, high, low,
    close, volume, adjusted_close``, timestamps stamped at bar *end* in UTC. Bars are what the
    market printed: a provider must never fold a distribution into a price here -- those are
    recorded separately by :class:`~src.core.interfaces.DividendProvider` and booked as cash.
    """

    @abstractmethod
    def fetch_intraday_bars(
        self,
        symbols: list[str],
        *,
        lookback_minutes: int,
        bar_minutes: int | None = None,
        force_refresh: bool = False,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **extra: Any,
    ) -> dict[str, pd.DataFrame]:
        """Fine-grained bars. ``bar_minutes`` is a request, not a guarantee.

        A provider that cannot serve the requested grid returns its nearest *coarser* one
        rather than failing; horizons are stated in market minutes and resolved against
        whatever grid arrives, so a coarser answer is usable and a missing one is not.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_eod_bars(
        self,
        symbols: list[str],
        *,
        lookback_bars: int,
        force_refresh: bool = False,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **extra: Any,
    ) -> dict[str, pd.DataFrame]:
        """Daily bars, oldest first."""
        raise NotImplementedError


class QuoteProvider(Connector):
    """A source of the latest traded price."""

    @abstractmethod
    def fetch_quotes(self, symbols: list[str], **extra: Any) -> dict[str, dict[str, Any]]:
        """``{symbol: {"price": float, ...}}``.

        A symbol the provider cannot price is simply absent, so the caller can fall through to
        the next provider for that symbol alone rather than discarding the whole batch.
        """
        raise NotImplementedError


class NewsProvider(Connector):
    """A source of headline sentiment."""

    @abstractmethod
    def fetch_news(self, symbols: list[str], **extra: Any) -> list[dict[str, Any]]:
        raise NotImplementedError
