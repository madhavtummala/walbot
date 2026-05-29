from __future__ import annotations

import logging

import pandas as pd

from src import connectors
from src.config import Config


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


def test_alpaca_quote_fetch_skips_symbols_without_latest_price(monkeypatch) -> None:
    def fake_latest_price(symbol, _client, data_feed=None):
        if symbol == "SPY":
            raise RuntimeError("No latest bar found for SPY")
        return 50.0

    monkeypatch.setattr("src.alpaca_client.get_latest_price", fake_latest_price)
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

    records = connectors.fetch_latest_news_sentiment(["SPY"], config)

    assert [record["provider"] for record in records] == ["newsapi", "stocktwits"]
    assert records[-1]["sentiment"] == 0.4


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
