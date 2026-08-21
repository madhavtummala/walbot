"""Data connectors: market bars, quotes, headline sentiment, distributions.

Names are resolved on attribute access rather than at import. Importing this package used to
pull in every provider's third-party dependency -- yfinance, alpaca-py, the HTTP stack -- for
any caller that wanted a single symbol lookup, and made one unconfigured provider an import
error for the whole application. ``__getattr__`` keeps the public API identical while letting
a provider stay unimported until something actually calls it.

To add a provider: write ``market/<name>.py`` (or ``news/<name>.py``) implementing the
contracts in ``base.py``, then add one line to ``registry.py``. Nothing in ``service.py``
needs to change.
"""

from __future__ import annotations

from typing import Any

from .base import BarsProvider, Connector, NewsProvider, QuoteProvider
from .registry import (
    register_eod_fetcher,
    register_intraday_fetcher,
    register_news_fetcher,
    register_quote_fetcher,
)

#: Public name -> the module that defines it. Kept explicit rather than star-imported so the
#: package's surface is legible in one place and a typo is an error rather than a silent miss.
_EXPORTS = {
    "EOD_MARKET_CATEGORY": "src.connectors.support",
    "INTRADAY_MARKET_CATEGORY": "src.connectors.support",
    "INTRADAY_CACHE_TTL_SECONDS": "src.connectors.support",
    "MARKET_CATEGORY": "src.connectors.support",
    "NEWS_CATEGORY": "src.connectors.support",
    "ProviderRateLimited": "src.connectors.support",
    "ProviderUnavailable": "src.connectors.support",
    "bars_for_minutes": "src.connectors.support",
    "default_bar_minutes": "src.connectors.support",
    "resolve_bar_minutes": "src.connectors.support",
    "MARKET_FETCHERS": "src.connectors.service",
    "NEWS_FETCHERS": "src.connectors.service",
    "append_latest_quotes_to_bars": "src.connectors.service",
    "fetch_eod_market_bars": "src.connectors.service",
    "fetch_market_history": "src.connectors.service",
    "load_current_prices": "src.connectors.service",
    "load_latest_prices": "src.connectors.service",
    "prices_from_store": "src.connectors.service",
    "fetch_latest_news_sentiment": "src.connectors.service",
    "fetch_schwab_eod_bars": "src.connectors.market.schwab",
    "fetch_schwab_intraday_bars": "src.connectors.market.schwab",
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
    | {
        "BarsProvider",
        "Connector",
        "NewsProvider",
        "QuoteProvider",
        "register_eod_fetcher",
        "register_intraday_fetcher",
        "register_news_fetcher",
        "register_quote_fetcher",
    }
)
