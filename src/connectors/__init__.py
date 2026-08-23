"""Data connectors: market bars, quotes, headline sentiment, distributions.

Names are resolved on attribute access rather than at import. Importing this package used to
pull in every provider's third-party dependency -- yfinance, alpaca-py, the HTTP stack -- for
any caller that wanted a single symbol lookup, and made one unconfigured provider an import
error for the whole application. ``__getattr__`` keeps the public API identical while letting
a provider stay unimported until something actually calls it.

To add a market-data provider: subclass :class:`~src.connectors.base.MarketDataProvider` in
``market/<name>.py``, implement ``fetch_price`` and ``fetch_bars``, and add one line to
``registry.py``. The read-through cache, resolution negotiation and normalisation come with the
base; nothing in ``service.py`` needs to change.

The shared plumbing lives in four modules, by what it does rather than in one called
``support``: ``grid`` (which bar resolutions a provider serves), ``sources`` (which provider to
try, whether it is usable, and its credentials), ``http`` (making the call, and telling a rate
limit from an outage), ``cache`` (the DuckDB bar store and the payload cache in front of it) and
``frames`` (shaping whatever came back into the one frame everything else reads).
"""

from __future__ import annotations

from typing import Any

from .registry import register_market_provider, register_news_fetcher

#: Public name -> the module that defines it. Kept explicit rather than star-imported so the
#: package's surface is legible in one place and a typo is an error rather than a silent miss.
_EXPORTS = {
    "EOD_MARKET_CATEGORY": "src.connectors.sources",
    "INTRADAY_MARKET_CATEGORY": "src.connectors.sources",
    "INTRADAY_CACHE_TTL_SECONDS": "src.connectors.cache",
    "MARKET_CATEGORY": "src.connectors.sources",
    "NEWS_CATEGORY": "src.connectors.sources",
    "ProviderRateLimited": "src.connectors.sources",
    "ProviderUnavailable": "src.connectors.sources",
    "bars_for_minutes": "src.connectors.grid",
    "default_bar_minutes": "src.connectors.grid",
    "resolve_bar_minutes": "src.connectors.grid",
    "MARKET_DATA": "src.connectors.registry",
    "market_provider": "src.connectors.registry",
    "NEWS_FETCHERS": "src.connectors.service",
    "append_latest_quotes_to_bars": "src.connectors.service",
    "fetch_eod_market_bars": "src.connectors.service",
    "fetch_market_history": "src.connectors.service",
    "load_current_prices": "src.connectors.service",
    "load_latest_prices": "src.connectors.service",
    "prices_from_store": "src.connectors.service",
    "fetch_latest_news_sentiment": "src.connectors.service",
    "ensure_stream": "src.connectors.streaming",
    "stream_status": "src.connectors.streaming",
    "stream_store": "src.connectors.streaming",
    "Schwab": "src.connectors.market.schwab",
}


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(
    set(_EXPORTS)
    | {"register_market_provider", "register_news_fetcher"}
)
