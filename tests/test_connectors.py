from __future__ import annotations

import logging


import pandas as pd

from src.brokerages.alpaca import client as alpaca_client
from src.connectors import service as connectors
from src.connectors.market import alpaca as market_alpaca
from src.connectors.market import finnhub as market_finnhub
from src.connectors.market import schwab as market_schwab
from src.connectors.news import stocktwits as news_stocktwits
from src.connectors.cache import INTRADAY_CACHE_TTL_SECONDS, _fresh_cached_bars
from src.connectors.frames import normalize_intraday_frame
from src.connectors.grid import bars_for_minutes, resolve_bar_minutes
from src.core.config import Config



# The registry is the seam now: a test supplies a provider the same way a deployment would,
# which means these tests exercise the extension point rather than reaching around it.
def no_bar_cache(monkeypatch, stored=None):
    """Silence the read-through cache for one test.

    One seam instead of one per provider: every provider reads and writes bars through
    ``MarketDataProvider``, so patching it here covers all of them and a new provider needs no
    new patch.
    """
    from src.connectors import base

    monkeypatch.setattr(base, "_read_duckdb_bars", lambda *_a, **_kw: stored if stored is not None else pd.DataFrame())
    monkeypatch.setattr(base, "_write_duckdb_bars", lambda *_a, **_kw: None)


def use_provider(monkeypatch, name, *, bars=None, price=None):
    """Register a stub :class:`MarketDataProvider` under ``name`` for one test."""
    from src.connectors.base import MarketDataProvider
    from src.connectors.registry import MARKET_DATA

    class _Stub(MarketDataProvider):
        def fetch_bars(self, symbols, **kwargs):
            return bars(symbols, self.config, **kwargs) if bars else {}

        def fetch_price(self, symbols, **kwargs):
            return price(symbols, self.config, **kwargs) if price else {}

    _Stub.name = name
    monkeypatch.setitem(MARKET_DATA._entries, name, _Stub)
    return _Stub


def test_a_freshly_fetched_quote_is_actually_cached(monkeypatch) -> None:
    """The cache write on the fetch path has to be callable, not merely reached.

    ``save_cached_payload`` takes ``ttl_seconds`` as a required positional argument, and the
    quote path omitted it. Nothing caught that: the call only runs when a symbol is *not*
    already cached, and the ``TypeError`` was swallowed by the provider-fallback handler, which
    reported "provider schwab failed" and discarded the whole batch. Live, that surfaced as
    "quotes for only 5/7 symbols" -- the five that happened to be cached -- and downstream as an
    options bracket with no mark to price itself from.
    """
    from src.connectors import base
    from src.connectors.base import MarketDataProvider

    saved: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(base, "load_cached_payload", lambda *_a, **_kw: None)
    monkeypatch.setattr(base, "save_cached_payload",
                        lambda *args, **kwargs: saved.append((args, kwargs)))

    class _Stub(MarketDataProvider):
        name = "ttl_probe"

        def fetch_bars(self, symbols, **kwargs):
            return {}

        def fetch_price(self, symbols, **kwargs):
            return {symbol: {"price": 1.23, "current": True} for symbol in symbols}

    resolved = _Stub(Config()).price(["AAA", "BBB"], force_refresh=True)

    assert sorted(resolved) == ["AAA", "BBB"], "a fetched quote must reach the caller"
    assert len(saved) == 2, "and must be written to the cache"
    for args, kwargs in saved:
        # However the argument is passed, it has to be there: this is the whole bug.
        assert "ttl_seconds" in kwargs or len(args) >= 5, f"no ttl_seconds in {args}/{kwargs}"


def test_market_quote_fetch_falls_back_to_next_provider(monkeypatch, caplog) -> None:
    calls = []
    saved = []
    config = Config(
        market_data_provider_order=["finnhub", "alpha_vantage"],
        data_source_configs={
            "market_data": {
                "providers": {
                    "finnhub": {"enabled": True, "api_key": "finnhub-key"},
                    "alpha_vantage": {"enabled": True, "api_key": "alpha-key"},
                }
            }
        },
    )

    def limited_fetch(_symbols, _config):
        calls.append("finnhub")
        raise connectors.ProviderRateLimited("quota")

    def fallback_fetch(symbols, _config):
        calls.append("alpha_vantage")
        return {
            symbol: {
                "symbol": symbol,
                "price": 101.0,
                "timestamp": "2026-05-21T15:30:00+00:00",
                "provider": "alpha_vantage",
                "raw": {},
            }
            for symbol in symbols
        }

    use_provider(monkeypatch, "finnhub", price=lambda symbols, config, **kw: limited_fetch(symbols, config))
    use_provider(monkeypatch, "alpha_vantage", price=lambda symbols, config, **kw: fallback_fetch(symbols, config))
    from src.connectors import base

    monkeypatch.setattr(connectors, "provider_is_limited", lambda _provider: False)
    monkeypatch.setattr(base, "load_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(base, "save_cached_payload", lambda *args, **_kwargs: saved.append(args))

    with caplog.at_level(logging.INFO):
        quotes = connectors.load_latest_prices(["SPY"], config)

    assert calls == ["finnhub", "alpha_vantage"]
    assert quotes["SPY"]["price"] == 101.0
    assert quotes["SPY"]["current"] is True
    assert saved[0][1] == "alpha_vantage"
    assert "Market data provider finnhub hit its rate limit; falling back to alpha_vantage" in caplog.text


def _quote_config() -> Config:
    return Config(
        market_data_provider_order=["schwab"],
        data_source_configs={
            "market_data": {"providers": {"schwab": {"enabled": True}}}
        },
    )


def test_load_latest_prices_marks_live_prints_and_skips_the_store(monkeypatch) -> None:
    """Within market hours a provider quote is 'current'; stored bars are not consulted."""
    monkeypatch.setattr(
        "src.brokerages.alpaca.client.create_trading_client", lambda _config: object()
    )
    monkeypatch.setattr("src.brokerages.alpaca.client.is_market_open", lambda _client: True)
    monkeypatch.setattr(
        connectors,
        "load_current_prices",
        lambda symbols, _config, **_kwargs: {
            symbol: {"symbol": symbol, "price": 500.0, "timestamp": "2026-08-12T15:00:00+00:00", "provider": "schwab"}
            for symbol in symbols
        },
    )

    def _fail(_symbols):
        raise AssertionError("store fallback consulted while live quotes were available")

    monkeypatch.setattr(connectors, "prices_from_store", _fail)

    quotes = connectors.load_latest_prices(["SPY"], _quote_config())

    assert quotes["SPY"]["price"] == 500.0
    assert quotes["SPY"]["current"] is True


def test_load_latest_prices_degrades_to_stored_bars_off_hours(monkeypatch) -> None:
    """Off hours the chain reads the bar store and marks the price as not current."""
    monkeypatch.setattr(
        "src.brokerages.alpaca.client.create_trading_client", lambda _config: object()
    )
    monkeypatch.setattr("src.brokerages.alpaca.client.is_market_open", lambda _client: False)

    def _fail(*_args, **_kwargs):
        raise AssertionError("live providers contacted while the market was closed")

    monkeypatch.setattr(connectors, "load_current_prices", _fail)
    monkeypatch.setattr(
        connectors,
        "prices_from_store",
        lambda symbols: {
            symbol: {
                "symbol": symbol,
                "price": 498.0,
                "timestamp": "2026-08-11T20:00:00+00:00",
                "provider": "duckdb_cache",
                "cached": True,
                "current": False,
            }
            for symbol in symbols
        },
    )

    quotes = connectors.load_latest_prices(["SPY"], _quote_config())

    assert quotes["SPY"]["price"] == 498.0
    assert quotes["SPY"]["current"] is False


def test_prices_from_store_reads_the_finest_cached_interval(monkeypatch) -> None:
    from src.data import duckdb_store

    monkeypatch.setattr(duckdb_store, "available_intervals", lambda _symbol: [5, 1440])
    reads = []

    def fake_read_bars(symbol, interval_minutes, limit):
        reads.append(interval_minutes)
        if interval_minutes == 1440:
            return pd.DataFrame({"close": [499.0], "timestamp": pd.Timestamp("2026-08-11", tz="UTC")})
        return pd.DataFrame({"close": [500.25], "timestamp": pd.Timestamp("2026-08-12", tz="UTC")})

    monkeypatch.setattr(duckdb_store, "read_bars", fake_read_bars)

    prices = connectors.prices_from_store(["SPY"])

    # Finest first: the 5-minute bar answers before the daily bar is ever read.
    assert reads == [5]
    assert prices["SPY"]["price"] == 500.25
    assert prices["SPY"]["interval_minutes"] == 5
    assert prices["SPY"]["current"] is False


def test_append_latest_quotes_to_bars_adds_intraday_row() -> None:
    bars = {
        "SPY": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-05-20"], utc=True),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
            }
        )
    }
    quotes = {"SPY": {"price": 102.0, "timestamp": "2026-05-21T15:00:00+00:00"}}

    merged = connectors.append_latest_quotes_to_bars(bars, quotes)

    assert merged["SPY"]["close"].tolist() == [100.5, 102.0]
    assert merged["SPY"]["timestamp"].iloc[-1].isoformat() == "2026-05-21T15:00:00+00:00"


def test_normalize_intraday_frame_preserves_adjusted_close() -> None:
    raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-05-20", "2026-05-21"]),
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Adj Close": [99.75, None],
            "Volume": [1000, 2000],
        }
    )

    bars = normalize_intraday_frame(raw)

    assert bars["adjusted_close"].tolist() == [99.75, 101.5]


def test_fresh_cached_bars_rejects_stale_eod_rows() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-04T23:00:00-05:00"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
        }
    )

    fresh = _fresh_cached_bars(
        bars,
        connectors.DAILY_INTERVAL_MINUTES,
        now=pd.Timestamp("2026-06-12T12:00:00-05:00"),
    )

    assert fresh.empty


def test_fetch_alpaca_eod_bars_fetches_when_duckdb_rows_are_stale(monkeypatch) -> None:
    config = Config(alpaca_data_feed="iex")
    stale = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-04T23:00:00-05:00"]),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
        }
    )
    fresh = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-06-11T23:00:00-05:00"]),
            "open": [110.0],
            "high": [111.0],
            "low": [109.0],
            "close": [110.5],
            "volume": [2000],
        }
    )
    calls = []

    no_bar_cache(monkeypatch, stored=stale)
    monkeypatch.setattr(alpaca_client, "create_data_client", lambda _config: object())

    def fake_historical_daily_bars(symbols, **_kwargs):
        calls.append(symbols)
        return {"SPY": fresh}

    monkeypatch.setattr(alpaca_client, "get_historical_daily_bars", fake_historical_daily_bars)

    bars = market_alpaca.Alpaca(config).bars(["SPY"], lookback_bars=1)

    assert calls == [["SPY"]]
    assert bars["SPY"]["close"].tolist() == [110.5]


def test_fetch_finnhub_intraday_bars_parses_and_caches_candles(monkeypatch) -> None:
    saved = {}
    config = Config(
        data_source_configs={
            "market_data": {
                "providers": {
                    "finnhub": {"enabled": True, "api_key": "finnhub-key"},
                }
            }
        }
    )

    def fake_request(provider, category, url, params=None, headers=None):
        assert provider == "finnhub"
        assert category == connectors.INTRADAY_MARKET_CATEGORY
        assert url.endswith("/stock/candle")
        assert params["symbol"] == "SPY"
        assert params["resolution"] == "30"
        return {
            "s": "ok",
            "t": [1_779_999_000, 1_780_000_800],
            "o": [100.0, 101.0],
            "h": [101.0, 102.0],
            "l": [99.0, 100.0],
            "c": [100.5, 101.5],
            "v": [1000, 2000],
        }

    monkeypatch.setattr(market_finnhub, "_request_json", fake_request)
    from src.connectors import base

    monkeypatch.setattr(base, "_read_duckdb_bars", lambda *_a, **_kw: pd.DataFrame())
    monkeypatch.setattr(
        base, "_write_duckdb_bars",
        lambda *args, **kwargs: saved.update({"args": args, "kwargs": kwargs}),
    )

    bars = market_finnhub.Finnhub(config).bars(["SPY"], interval_minutes=30, lookback_bars=2)

    assert bars["SPY"]["close"].tolist() == [100.5, 101.5]
    # Written straight to the bar store, keyed by provider and resolution -- no payload cache
    # in front of it any more.
    assert saved["args"][:3] == ("finnhub", "SPY", 30)
    assert saved["kwargs"]["ttl_seconds"] == INTRADAY_CACHE_TTL_SECONDS


def test_fetch_market_history_uses_yfinance_provider(monkeypatch) -> None:
    calls = {}
    config = Config(intraday_market_data_provider_order=["yfinance"])

    def fake_yfinance(symbols, _config, *, interval_minutes, lookback_bars, **_kwargs):
        calls.update({"symbols": symbols, "interval_minutes": interval_minutes, "lookback_bars": lookback_bars})
        return {"SPY": pd.DataFrame({"timestamp": pd.to_datetime(["2026-06-04T20:00:00Z"]), "close": [101.0]})}

    use_provider(monkeypatch, "yfinance", bars=fake_yfinance)
    # The window is fully covered by the provider, so no cached back-fill is consulted.
    monkeypatch.setattr(connectors, "_extend_with_cached_history", lambda bars, *_args: bars)

    bars = connectors.fetch_market_history(["SPY"], config, lookback_minutes=1170, force_refresh=True)

    assert bars["SPY"]["close"].iloc[-1] == 101.0
    # The grid the provider was asked for, snapped to what yfinance actually serves, and the
    # bar count the base derived from the market-minutes window.
    assert calls["symbols"] == ["SPY"]
    assert calls["interval_minutes"] == 15
    assert calls["lookback_bars"] == bars_for_minutes(1170, 15)


def test_a_provider_that_cannot_serve_the_grid_snaps_to_a_coarser_one() -> None:
    """Horizons are wall-clock, so a coarser bar still answers them -- never an error."""
    assert resolve_bar_minutes("schwab", 5) == 5
    assert resolve_bar_minutes("yfinance", 5) == 15
    # Below everything a provider offers, take its finest rather than refusing.
    assert resolve_bar_minutes("yfinance", 1) == 15


def test_fetch_eod_market_bars_falls_back_to_next_provider(monkeypatch, caplog) -> None:
    calls = []
    config = Config(eod_market_data_provider_order=["finnhub", "yfinance"])

    def failing_finnhub(*_args, **_kwargs):
        calls.append("finnhub")
        raise connectors.ProviderUnavailable("down")

    def working_yfinance(symbols, _config, *, interval_minutes, lookback_bars, **_kwargs):
        calls.append("yfinance")
        assert symbols == ["SPY"]
        assert interval_minutes == 1440
        assert lookback_bars == 3
        return {"SPY": pd.DataFrame({"timestamp": pd.to_datetime(["2026-05-01"], utc=True), "close": [101.0]})}

    use_provider(monkeypatch, "finnhub", bars=failing_finnhub)
    use_provider(monkeypatch, "yfinance", bars=working_yfinance)

    with caplog.at_level(logging.INFO):
        bars = connectors.fetch_eod_market_bars(["SPY"], config, lookback_bars=3, force_refresh=True)

    assert calls == ["finnhub", "yfinance"]
    assert bars["SPY"]["close"].iloc[-1] == 101.0
    assert "EOD market data provider finnhub failed; falling back to yfinance" in caplog.text


def test_fetch_schwab_intraday_bars_parses_price_history(monkeypatch) -> None:
    captured = {}
    config = Config(
        data_source_configs={
            "intraday_market_data": {
                "providers": {
                    "schwab": {"access_token": "token"},
                }
            }
        }
    )

    def fake_request(provider, category, url, params=None, headers=None):
        captured.update({"provider": provider, "category": category, "url": url, "params": params, "headers": headers})
        return {
            "candles": [
                {"datetime": 1_780_000_000_000, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
                {"datetime": 1_780_000_900_000, "open": 101, "high": 102, "low": 100, "close": 101.5, "volume": 2000},
            ]
        }

    monkeypatch.setattr(market_schwab, "_request_json", fake_request)
    no_bar_cache(monkeypatch)

    bars = market_schwab.Schwab(config).bars(["SPY"], interval_minutes=15, lookback_bars=2)

    assert bars["SPY"]["close"].tolist() == [100.5, 101.5]
    assert captured["provider"] == "schwab"
    assert captured["category"] == connectors.INTRADAY_MARKET_CATEGORY
    assert captured["url"].endswith("/marketdata/v1/pricehistory")
    assert captured["params"]["frequencyType"] == "minute"
    assert captured["params"]["frequency"] == 15
    assert captured["headers"]["Authorization"] == "Bearer token"


def test_fetch_schwab_eod_bars_parses_price_history(monkeypatch) -> None:
    captured = {}
    config = Config(
        data_source_configs={
            "eod_market_data": {
                "providers": {
                    "schwab": {"access_token": "token"},
                }
            }
        }
    )

    def fake_request(provider, category, url, params=None, headers=None):
        # The EOD fetch makes two calls now: dividend metadata, then the price history.
        if url.endswith("/quotes"):
            return {"SPY": {"fundamental": {"divAmount": 0.0, "divFreq": 0}}}
        captured.update({"provider": provider, "category": category, "params": params, "headers": headers})
        return {
            "candles": [
                {"datetime": 1_780_000_000_000, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
            ]
        }

    monkeypatch.setattr(market_schwab, "_request_json", fake_request)
    no_bar_cache(monkeypatch)

    bars = market_schwab.Schwab(config).bars(["SPY"], lookback_bars=1)

    assert bars["SPY"]["close"].tolist() == [100.5]
    assert captured["category"] == connectors.EOD_MARKET_CATEGORY
    assert captured["params"]["frequencyType"] == "daily"
    assert captured["params"]["frequency"] == 1
    assert captured["headers"]["Authorization"] == "Bearer token"


def test_alpaca_quote_fetch_skips_symbols_without_latest_price(monkeypatch) -> None:
    def fake_latest_price(symbol, _client, data_feed=None):
        if symbol == "SPY":
            raise RuntimeError("No latest bar found for SPY")
        return 50.0

    monkeypatch.setattr("src.brokerages.alpaca.client.get_latest_price", fake_latest_price)
    monkeypatch.setattr(market_alpaca, "record_provider_success", lambda *_args, **_kwargs: None)

    quotes = market_alpaca.Alpaca(Config()).fetch_price(["SPY", "QQQ"], data_client=object())

    assert "SPY" not in quotes
    assert quotes["QQQ"]["price"] == 50.0


def test_news_sentiment_fetch_falls_back_to_stocktwits(monkeypatch, caplog) -> None:
    config = Config(
        news_sentiment_provider_order=["marketaux", "stocktwits"],
        data_source_configs={
            "news_sentiment": {
                "providers": {
                    "marketaux": {"enabled": True, "api_key": "marketaux-key"},
                    "stocktwits": {"enabled": True},
                }
            }
        },
    )

    def limited_news(_symbols, _config):
        raise connectors.ProviderRateLimited("quota")

    def stocktwits_news(symbols, _config):
        return [
            {
                "timestamp": "2026-05-21T15:00:00+00:00",
                "symbol": symbols[0],
                "mentions": 1.0,
                "sentiment": 1.0,
                "social_score": 1.0,
                "provider": "stocktwits",
                "title": "bullish",
                "url": "",
                "raw": {},
            }
        ]

    monkeypatch.setitem(connectors.NEWS_FETCHERS, "marketaux", limited_news)
    monkeypatch.setitem(connectors.NEWS_FETCHERS, "stocktwits", stocktwits_news)
    monkeypatch.setattr(connectors, "provider_is_limited", lambda _provider: False)
    monkeypatch.setattr(connectors, "load_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(connectors, "save_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(connectors, "_read_duckdb_sentiment", lambda *_args, **_kwargs: [])

    with caplog.at_level(logging.INFO):
        records = connectors.fetch_latest_news_sentiment(["SPY"], config)

    assert records[0]["provider"] == "stocktwits"
    assert records[0]["sentiment"] == 1.0
    assert "News/sentiment provider marketaux hit its rate limit; falling back to stocktwits" in caplog.text


def test_news_sentiment_fetch_combines_configured_providers(monkeypatch) -> None:
    config = Config(
        news_sentiment_provider_order=["newsapi", "stocktwits"],
        data_source_configs={
            "news_sentiment": {
                "providers": {
                    "newsapi": {"enabled": True, "api_key": "newsapi-key"},
                    "stocktwits": {"enabled": True},
                }
            }
        },
    )

    def newsapi_news(symbols, _config):
        return [
            {
                "timestamp": "2026-05-27T15:00:00+00:00",
                "symbol": symbols[0],
                "mentions": 1.0,
                "sentiment": 0.0,
                "social_score": 0.0,
                "provider": "newsapi",
                "title": "older headline",
                "url": "",
                "raw": {},
            }
        ]

    def stocktwits_news(symbols, _config):
        return [
            {
                "timestamp": "2026-05-28T15:00:00+00:00",
                "symbol": symbols[0],
                "mentions": 42.0,
                "sentiment": 0.4,
                "social_score": 0.4,
                "provider": "stocktwits",
                "title": "current sentiment",
                "url": "",
                "raw": {},
            }
        ]

    monkeypatch.setitem(connectors.NEWS_FETCHERS, "newsapi", newsapi_news)
    monkeypatch.setitem(connectors.NEWS_FETCHERS, "stocktwits", stocktwits_news)
    monkeypatch.setattr(connectors, "provider_is_limited", lambda _provider: False)
    monkeypatch.setattr(connectors, "load_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(connectors, "save_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(connectors, "_read_duckdb_sentiment", lambda *_args, **_kwargs: [])

    records = connectors.fetch_latest_news_sentiment(["SPY"], config)

    assert [record["provider"] for record in records] == ["newsapi", "stocktwits"]
    assert records[-1]["sentiment"] == 0.4


def test_sentiment_data_alias_uses_new_provider_order(monkeypatch) -> None:
    config = Config(
        sentiment_data_provider_order=["newsapi"],
        data_source_configs={
            "sentiment_data": {
                "providers": {
                    "newsapi": {"enabled": True, "api_key": "newsapi-key"},
                }
            }
        },
    )

    def newsapi_news(symbols, _config):
        return [
            {
                "timestamp": "2026-05-27T15:00:00+00:00",
                "symbol": symbols[0],
                "mentions": 1.0,
                "sentiment": 0.2,
                "social_score": 0.2,
                "provider": "newsapi",
                "title": "headline",
                "url": "",
                "raw": {},
            }
        ]

    monkeypatch.setitem(connectors.NEWS_FETCHERS, "newsapi", newsapi_news)
    monkeypatch.setattr(connectors, "provider_is_limited", lambda _provider: False)
    monkeypatch.setattr(connectors, "load_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(connectors, "save_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(connectors, "_read_duckdb_sentiment", lambda *_args, **_kwargs: [])

    records = connectors.fetch_latest_news_sentiment(["SPY"], config)

    assert records[0]["provider"] == "newsapi"
    assert records[0]["sentiment"] == 0.2


def test_news_sentiment_fetch_logs_unsupported_provider(caplog) -> None:
    config = Config(news_sentiment_provider_order=["newsapi_ai"])

    with caplog.at_level(logging.WARNING):
        records = connectors.fetch_latest_news_sentiment(["SPY"], config)

    assert records == []
    assert "News/sentiment provider newsapi_ai is not supported; skipping" in caplog.text


def test_provider_order_does_not_enable_missing_provider_section(monkeypatch) -> None:
    config = Config(
        market_data_provider_order=["alpaca"],
        data_source_configs={"market_data": {"providers": {}}},
    )
    use_provider(monkeypatch, "alpaca", price=lambda symbols, config, **kw: {"SPY": {}})
    monkeypatch.setattr(connectors, "prices_from_store", lambda _symbols: {})

    assert connectors.load_latest_prices(["SPY"], config) == {}


def test_stocktwits_basic_auth_sentiment_endpoint(monkeypatch) -> None:
    captured = {}
    config = Config(
        data_source_configs={
            "news_sentiment": {
                "providers": {
                    "stocktwits": {"username": "user@example.com", "password": "secret"}
                }
            }
        }
    )

    def fake_request(provider, category, url, params=None, headers=None):
        captured.update({"provider": provider, "url": url, "headers": headers or {}})
        return {"data": {"sentiment_score": 0.25, "message_volume": 42, "updated_at": "2026-05-21T15:00:00Z"}}

    monkeypatch.setattr(news_stocktwits, "_request_json", fake_request)

    records = news_stocktwits._fetch_stocktwits_news(["SPY"], config)

    assert records[0]["sentiment"] == 0.25
    assert records[0]["mentions"] == 42
    assert "Authorization" in captured["headers"]
    assert captured["url"].endswith("/SPY/detail")


def test_the_ladder_fills_missing_symbols_from_the_next_provider() -> None:
    """Per symbol, not per batch.

    A provider that answered for most of the request used to win it outright, and the symbols
    it had nothing for came back empty with no fallback attempted -- indistinguishable
    downstream from a symbol with no history.
    """
    import pandas as pd

    from src.connectors.service import _run_provider_fallback

    def frame(value: float) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-02", tz="UTC")], "close": [value]})

    fetchers = {
        "first": lambda: {"AAA": frame(1.0), "BBB": pd.DataFrame()},
        "second": lambda: {"BBB": frame(2.0), "AAA": frame(99.0)},
    }

    bars = _run_provider_fallback(["AAA", "BBB"], ["first", "second"], fetchers, Config(), category="x", label="test")

    assert float(bars["AAA"]["close"].iloc[0]) == 1.0, "the preferred provider keeps what it answered"
    assert float(bars["BBB"]["close"].iloc[0]) == 2.0, "the gap is filled from the next one"


def test_a_symbol_no_provider_can_answer_comes_back_empty() -> None:
    import pandas as pd

    from src.connectors.service import _run_provider_fallback

    fetchers = {"only": lambda: {"AAA": pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-02", tz="UTC")], "close": [1.0]})}}

    bars = _run_provider_fallback(["AAA", "MISSING"], ["only"], fetchers, Config(), category="x", label="test")

    assert not bars["AAA"].empty
    assert bars["MISSING"].empty


def test_cached_schwab_bars_are_never_dividend_adjusted() -> None:
    """The price cache holds what the market printed, and nothing else.

    Adjustment used to happen here, which cost more than it bought: it rewrote history every
    time a payment landed, it made the daily and intraday tiers disagree about the same
    instant, and it disguised real cash as price appreciation so the replay booked no income
    at all. Distributions are cash events now -- see ``src.data.dividends``.
    """
    import src.connectors.service as service

    assert not hasattr(service, "_apply_dividend_adjustment")
    assert not hasattr(service, "_schwab_dividend_schedule")


def test_provider_bars_default_adjusted_close_to_the_raw_close() -> None:
    """The column survives for feeds that supply their own, but is never synthesised."""
    import pandas as pd

    from src.connectors.cache import _provider_bars

    bars = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="D", tz="UTC"),
        "open": [10.0, 11.0, 12.0], "high": [10.0, 11.0, 12.0],
        "low": [10.0, 11.0, 12.0], "close": [10.0, 11.0, 12.0],
        "volume": [1.0, 1.0, 1.0],
    })

    out = _provider_bars(bars, 1440)

    assert out["adjusted_close"].tolist() == out["close"].tolist()


def test_schwab_counts_as_configured_without_an_api_key() -> None:
    """It authenticates with an OAuth token, not a key in the connector config.

    _enabled treated "no api_key" as "not configured", so Schwab was skipped for quotes
    however high it sat in the provider order -- the watchlist and the prices used to size
    orders silently kept coming from Alpaca's IEX feed.
    """
    from src.connectors.sources import EXTERNAL_AUTH_PROVIDERS, MARKET_CATEGORY, _enabled

    config = Config(data_source_configs={"market_data": {"providers": {"schwab": {}}}})

    assert "schwab" in EXTERNAL_AUTH_PROVIDERS
    assert _enabled(config, MARKET_CATEGORY, "schwab", uses_external_auth=True) is True


def test_a_provider_switched_off_explicitly_stays_off() -> None:
    from src.connectors.sources import MARKET_CATEGORY, _enabled

    config = Config(data_source_configs={"market_data": {"providers": {"schwab": {"enabled": False}}}})

    assert _enabled(config, MARKET_CATEGORY, "schwab", uses_external_auth=True) is False


def test_a_new_provider_needs_no_edit_to_the_dispatch(monkeypatch) -> None:
    """The property the connector layer exists to have.

    An earlier version of this abstraction ended up holding a single yfinance wrapper while
    every real provider stayed as a loose function in ``service.py`` -- so "pluggable" was
    true of the registry and false of the system. This asserts the registry is the thing the
    dispatch actually reads.
    """
    def third_party(symbols, _config, **_kwargs):
        return {
            symbol: pd.DataFrame({
                "timestamp": pd.to_datetime(["2026-08-14"], utc=True),
                "open": [42.0], "high": [42.0], "low": [42.0], "close": [42.0], "volume": [1.0],
            })
            for symbol in symbols
        }

    use_provider(monkeypatch, "third_party", bars=third_party)

    bars = connectors.fetch_eod_market_bars(
        ["SPY"],
        Config(eod_market_data_provider_order=["third_party"]),
        lookback_bars=1,
        force_refresh=True,
    )

    assert bars["SPY"]["close"].iloc[0] == 42.0


def test_importing_connectors_does_not_import_every_provider_dependency() -> None:
    """A provider's third-party dependency is that provider's problem, not the app's.

    Eager imports meant one unconfigured or uninstalled provider was an import error for
    everything, and every process paid for every SDK regardless of what it used.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys, src.connectors; "
         "print(int('yfinance' in sys.modules or any(m.startswith('alpaca') for m in sys.modules)))"],
        capture_output=True, text=True,
    )
    assert result.stdout.strip() == "0", result.stdout + result.stderr


def test_the_lazy_registry_resolves_under_every_dict_accessor() -> None:
    """It subclasses ``dict``, so it has to behave like one everywhere, not just on ``[]``.

    Entries start as ``None`` placeholders and resolve on lookup. The inherited ``values()``,
    ``items()`` and ``copy()`` returned those placeholders -- a mapping that answered ``in``
    and ``[]`` correctly while reporting ``None`` for anything not yet imported. Nothing in the
    codebase calls them today, which is exactly why it would have been found late and from a
    confusing direction.
    """
    from src.connectors.registry import NEWS_FETCHER_REGISTRY as EOD_BAR_REGISTRY

    assert all(callable(f) for f in EOD_BAR_REGISTRY.values())
    assert all(callable(f) for _, f in EOD_BAR_REGISTRY.items())
    assert all(callable(f) for f in EOD_BAR_REGISTRY.copy().values())
    assert all(callable(EOD_BAR_REGISTRY[name]) for name in EOD_BAR_REGISTRY)


def test_an_expired_cache_row_is_a_miss_not_a_hit(tmp_path) -> None:
    """``expires_at`` was written on every row and read on none.

    Every TTL in the config was decorative and a cached quote was served for ever. Found live: a
    run on 2026-08-25 priced IBIT at 36.15 from a row fetched on 2026-08-17 and expired thirty
    minutes later, while the venue quoted 45.04 -- an eight-day-old price handed to an algorithm
    that sizes real orders from it.
    """
    from src.data.provider_cache import load_cached_payload, save_cached_payload

    db = str(tmp_path / "cache.duckdb")
    save_cached_payload("market_data", "schwab", "IBIT", {"price": 36.15}, 3600, db_path=db)
    assert load_cached_payload("market_data", "schwab", "IBIT", db_path=db) == {"price": 36.15}

    # Same row, written with a TTL that has already elapsed.
    save_cached_payload("market_data", "schwab", "IBIT", {"price": 36.15}, -1, db_path=db)
    assert load_cached_payload("market_data", "schwab", "IBIT", db_path=db) is None
