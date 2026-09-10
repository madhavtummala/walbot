"""Where providers are declared.

The one file to edit when adding a data source. Entries are lazy ``"module:name"`` paths, so
importing this module does not import yfinance, alpaca-py, or anything else a provider happens
to need -- a provider that is unconfigured or whose dependency is missing fails only for the
call that wanted it, not at application start.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from ..common.registry import Registry
from .base import MarketDataProvider

#: Market-data providers. One class per vendor, answering both price and bars via the
#: read-through cache supplied by :class:`~src.connectors.base.MarketDataProvider`.
MARKET_PROVIDERS: dict[str, str] = {
    "alpaca": "src.connectors.market.alpaca:Alpaca",
    "schwab": "src.connectors.market.schwab:Schwab",
    "yfinance": "src.connectors.market.yfinance:YFinance",
    "finnhub": "src.connectors.market.finnhub:Finnhub",
    "alpha_vantage": "src.connectors.market.alpha_vantage:AlphaVantage",
}

#: Headline sentiment fetchers, by provider name.
NEWS_FETCHERS_PATHS: dict[str, str] = {
    "marketaux": "src.connectors.news.marketaux:_fetch_marketaux_news",
    "newsapi": "src.connectors.news.newsapi:_fetch_newsapi_news",
    "stocktwits": "src.connectors.news.stocktwits:_fetch_stocktwits_news",
}


def _resolve(path: str) -> Callable[..., Any]:
    module_path, _, attribute = path.partition(":")
    if not module_path or not attribute:
        raise ValueError(f"Invalid provider path {path!r}; expected 'module:name'")
    return getattr(import_module(module_path), attribute)


class _LazyFetchers(dict):
    """A mapping that imports a provider the first time it is actually called.

    Presents as a plain dict -- ``in``, ``[]`` and iteration all behave normally -- but the
    import happens on lookup.
    """

    def __init__(self, paths: dict[str, str]):
        super().__init__({name: None for name in paths})
        self._paths = dict(paths)

    def __getitem__(self, name: str) -> Callable[..., Any]:
        resolved = super().get(name)
        if resolved is None:
            resolved = _resolve(self._paths[name])
            self[name] = resolved
        return resolved

    def get(self, name, default=None):  # type: ignore[override]
        return self[name] if name in self._paths else default

    # Must override values()/items()/copy() too, or they'd return the unresolved None
    # placeholders instead of resolving like __getitem__ does.
    def values(self):  # type: ignore[override]
        return [self[name] for name in self._paths]

    def items(self):  # type: ignore[override]
        return [(name, self[name]) for name in self._paths]

    def copy(self) -> dict:  # type: ignore[override]
        return dict(self.items())


NEWS_FETCHER_REGISTRY = _LazyFetchers(NEWS_FETCHERS_PATHS)

MARKET_DATA: Registry[MarketDataProvider] = Registry(
    "market data provider", MarketDataProvider, MARKET_PROVIDERS
)


def market_provider(name: str, config) -> MarketDataProvider:
    """The provider registered under ``name``, constructed against ``config``."""
    return MARKET_DATA.create(name, config)


def register_market_provider(name: str, provider: type[MarketDataProvider] | str) -> None:
    MARKET_DATA.register(name, provider)


def register_news_fetcher(name: str, fetcher: Callable[..., Any] | str) -> None:
    _register(NEWS_FETCHER_REGISTRY, name, fetcher)


def _register(registry: _LazyFetchers, name: str, fetcher: Callable[..., Any] | str) -> None:
    normalized = str(name or "").strip().lower()
    if not normalized:
        raise ValueError("provider name is required")
    if isinstance(fetcher, str):
        registry._paths[normalized] = fetcher
        dict.__setitem__(registry, normalized, None)
    else:
        registry._paths[normalized] = f"<callable {normalized}>"
        dict.__setitem__(registry, normalized, fetcher)
