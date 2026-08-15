"""Where providers are declared.

The one file to edit when adding a data source. Entries are lazy ``"module:name"`` paths, so
importing this module does not import yfinance, alpaca-py, or anything else a provider happens
to need -- a provider that is unconfigured or whose dependency is missing fails only for the
call that wanted it, not at application start.

An earlier version of this file held a single ``YFinanceConnector`` while every real provider
lived as a loose function in ``service.py``. The difference now is that these entries point at
the *actual* fetchers: there is no second implementation anywhere for them to drift from.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

#: Latest-price fetchers, by provider name. Signature: ``(symbols, config) -> {symbol: quote}``,
#: except Alpaca which also accepts a preconstructed ``data_client``.
QUOTE_FETCHERS: dict[str, str] = {
    "alpaca": "src.connectors.market.alpaca:_fetch_alpaca_quotes",
    "schwab": "src.connectors.market.schwab:_fetch_schwab_quotes",
    "finnhub": "src.connectors.market.finnhub:_fetch_finnhub_quotes",
    "alpha_vantage": "src.connectors.market.alpha_vantage:_fetch_alpha_vantage_quotes",
}

#: Fine-grained bar fetchers, by provider name.
INTRADAY_BAR_FETCHERS: dict[str, str] = {
    "yfinance": "src.connectors.market.yfinance:fetch_yfinance_intraday_bars",
    "alpaca": "src.connectors.market.alpaca:fetch_alpaca_intraday_bars",
    "schwab": "src.connectors.market.schwab:fetch_schwab_intraday_bars",
    "finnhub": "src.connectors.market.finnhub:fetch_finnhub_intraday_bars",
}

#: Daily bar fetchers, by provider name.
EOD_BAR_FETCHERS: dict[str, str] = {
    "yfinance": "src.connectors.market.yfinance:fetch_yfinance_eod_bars",
    "alpaca": "src.connectors.market.alpaca:fetch_alpaca_eod_bars",
    "schwab": "src.connectors.market.schwab:fetch_schwab_eod_bars",
    "finnhub": "src.connectors.market.finnhub:fetch_finnhub_eod_bars",
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

    Presents as a plain dict because that is what the dispatch code already expects -- ``in``,
    ``[]`` and iteration all behave normally, and the import happens on lookup. Keeping the
    dict interface is what let the providers move out without the orchestration changing shape
    around them.
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


QUOTE_FETCHER_REGISTRY = _LazyFetchers(QUOTE_FETCHERS)
INTRADAY_BAR_REGISTRY = _LazyFetchers(INTRADAY_BAR_FETCHERS)
EOD_BAR_REGISTRY = _LazyFetchers(EOD_BAR_FETCHERS)
NEWS_FETCHER_REGISTRY = _LazyFetchers(NEWS_FETCHERS_PATHS)


def register_quote_fetcher(name: str, fetcher: Callable[..., Any] | str) -> None:
    _register(QUOTE_FETCHER_REGISTRY, name, fetcher)


def register_intraday_fetcher(name: str, fetcher: Callable[..., Any] | str) -> None:
    _register(INTRADAY_BAR_REGISTRY, name, fetcher)


def register_eod_fetcher(name: str, fetcher: Callable[..., Any] | str) -> None:
    _register(EOD_BAR_REGISTRY, name, fetcher)


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
