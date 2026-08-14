from __future__ import annotations

import logging

import pandas as pd

from src.brokerages import alpaca_client
from src.connectors import service as connectors
from src.core.config import Config


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

    monkeypatch.setitem(connectors.MARKET_FETCHERS, "finnhub", limited_fetch)
    monkeypatch.setitem(connectors.MARKET_FETCHERS, "alpha_vantage", fallback_fetch)
    monkeypatch.setattr(connectors, "provider_is_limited", lambda _provider: False)
    monkeypatch.setattr(connectors, "load_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(connectors, "save_cached_payload", lambda *args, **_kwargs: saved.append(args))

    with caplog.at_level(logging.INFO):
        quotes = connectors.fetch_latest_market_quotes(["SPY"], config)

    assert calls == ["finnhub", "alpha_vantage"]
    assert quotes["SPY"]["price"] == 101.0
    assert saved[0][1] == "alpha_vantage"
    assert "Market data provider finnhub hit its rate limit; falling back to alpha_vantage" in caplog.text


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

    bars = connectors._normalize_intraday_frame(raw)

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

    fresh = connectors._fresh_cached_bars(
        bars,
        connectors.EOD_MARKET_CATEGORY,
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

    monkeypatch.setattr(connectors, "_read_duckdb_bars", lambda *_args, **_kwargs: stale)
    monkeypatch.setattr(connectors, "_write_duckdb_bars", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(alpaca_client, "create_data_client", lambda _config: object())

    def fake_historical_daily_bars(symbols, **_kwargs):
        calls.append(symbols)
        return {"SPY": fresh}

    monkeypatch.setattr(alpaca_client, "get_historical_daily_bars", fake_historical_daily_bars)

    bars = connectors.fetch_alpaca_eod_bars(["SPY"], config, lookback_bars=1)

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

    monkeypatch.setattr(connectors, "_request_json", fake_request)
    monkeypatch.setattr(connectors, "load_cached_payload", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(connectors, "save_cached_payload", lambda *args, **kwargs: saved.update({"args": args, "kwargs": kwargs}))
    monkeypatch.setattr(connectors, "_read_duckdb_bars", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(connectors, "_write_duckdb_bars", lambda *_args, **_kwargs: None)

    bars = connectors.fetch_finnhub_intraday_bars(["SPY"], config, lookback_bars=2, bar_minutes=30)

    assert bars["SPY"]["close"].tolist() == [100.5, 101.5]
    assert saved["args"][0] == connectors.INTRADAY_MARKET_CATEGORY
    assert saved["args"][1] == "finnhub"
    assert saved["kwargs"]["ttl_seconds"] == connectors.INTRADAY_CACHE_TTL_SECONDS


def test_fetch_intraday_market_bars_uses_yfinance_provider(monkeypatch) -> None:
    calls = {}
    config = Config(intraday_market_data_provider_order=["yfinance"])

    def fake_yfinance(symbols, _config, *, lookback_bars, bar_minutes, force_refresh=False):
        calls.update(
            {
                "symbols": symbols,
                "lookback_bars": lookback_bars,
                "bar_minutes": bar_minutes,
                "force_refresh": force_refresh,
            }
        )
        return {"SPY": pd.DataFrame({"close": [101.0]})}

    monkeypatch.setattr(connectors, "fetch_yfinance_intraday_bars", fake_yfinance)

    bars = connectors.fetch_intraday_market_bars(["SPY"], config, lookback_bars=78, bar_minutes=15, force_refresh=True)

    assert bars["SPY"]["close"].iloc[-1] == 101.0
    assert calls == {
        "symbols": ["SPY"],
        "lookback_bars": 78,
        "bar_minutes": 15,
        "force_refresh": True,
    }


def test_fetch_eod_market_bars_falls_back_to_next_provider(monkeypatch, caplog) -> None:
    calls = []
    config = Config(eod_market_data_provider_order=["finnhub", "yfinance"])

    def failing_finnhub(*_args, **_kwargs):
        calls.append("finnhub")
        raise connectors.ProviderUnavailable("down")

    def working_yfinance(symbols, _config, *, lookback_bars, force_refresh=False):
        calls.append("yfinance")
        assert symbols == ["SPY"]
        assert lookback_bars == 3
        assert force_refresh is True
        return {"SPY": pd.DataFrame({"timestamp": pd.to_datetime(["2026-05-01"], utc=True), "close": [101.0]})}

    monkeypatch.setattr(connectors, "fetch_finnhub_eod_bars", failing_finnhub)
    monkeypatch.setattr(connectors, "fetch_yfinance_eod_bars", working_yfinance)

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

    monkeypatch.setattr(connectors, "_request_json", fake_request)
    monkeypatch.setattr(connectors, "_read_duckdb_bars", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(connectors, "_write_duckdb_bars", lambda *_args, **_kwargs: None)

    bars = connectors.fetch_schwab_intraday_bars(["SPY"], config, lookback_bars=2, bar_minutes=15)

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
        captured.update({"provider": provider, "category": category, "params": params, "headers": headers})
        return {
            "candles": [
                {"datetime": 1_780_000_000_000, "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
            ]
        }

    monkeypatch.setattr(connectors, "_request_json", fake_request)
    monkeypatch.setattr(connectors, "_read_duckdb_bars", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(connectors, "_write_duckdb_bars", lambda *_args, **_kwargs: None)

    bars = connectors.fetch_schwab_eod_bars(["SPY"], config, lookback_bars=1)

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

    monkeypatch.setattr("src.brokerages.alpaca_client.get_latest_price", fake_latest_price)
    monkeypatch.setattr(connectors, "record_provider_success", lambda *_args, **_kwargs: None)

    quotes = connectors._fetch_alpaca_quotes(["SPY", "QQQ"], Config(), data_client=object())

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
    monkeypatch.setattr(connectors, "_read_duckdb_bars", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(connectors, "_write_duckdb_bars", lambda *_args, **_kwargs: None)

    with caplog.at_level(logging.INFO):
        records = connectors.fetch_latest_news_sentiment(["SPY"], config)
    frames = connectors.news_records_to_social_frames(records)

    assert records[0]["provider"] == "stocktwits"
    assert frames["SPY"]["sentiment"].iloc[-1] == 1.0
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
    monkeypatch.setattr(connectors, "_read_duckdb_bars", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(connectors, "_write_duckdb_bars", lambda *_args, **_kwargs: None)

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
    monkeypatch.setattr(connectors, "_read_duckdb_bars", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(connectors, "_write_duckdb_bars", lambda *_args, **_kwargs: None)

    records = connectors.fetch_latest_sentiment_data(["SPY"], config)

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
    monkeypatch.setitem(connectors.MARKET_FETCHERS, "alpaca", lambda *_args, **_kwargs: {"SPY": {}})

    assert connectors.fetch_latest_market_quotes(["SPY"], config) == {}


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

    monkeypatch.setattr(connectors, "_request_json", fake_request)

    records = connectors._fetch_stocktwits_news(["SPY"], config)

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
