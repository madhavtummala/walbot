from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone
from typing import Any

from src.brokerages.alpaca_client import create_data_client, get_latest_price
from src.connectors import (
    fetch_intraday_market_bars,
    fetch_latest_market_quotes,
    fetch_latest_news_sentiment,
)
from src.core.interfaces import AlgorithmContext
from src.data import fetch_daily_bars

logger = logging.getLogger(__name__)


def load_latest_prices(symbols: list[str], config, data_client) -> dict[str, float]:
    """Latest price per symbol, preferring live quotes and falling back to the data client.

    Symbols that cannot be priced are omitted rather than raising: a single illiquid or
    newly listed ticker must not take down a whole algorithm run or the dashboard. Step 2
    rejects an unpriced symbol by name when it actually matters for sizing.
    """
    latest_quotes = fetch_latest_market_quotes(symbols, config, data_client=data_client)
    prices: dict[str, float] = {}
    for symbol in symbols:
        quote = latest_quotes.get(symbol)
        if quote and quote.get("price"):
            prices[symbol] = float(quote["price"])
            continue
        try:
            prices[symbol] = float(get_latest_price(symbol, data_client, data_feed=config.alpaca_data_feed))
        except Exception as exc:
            logger.warning("Could not price %s; excluding it from this run: %s", symbol, exc)
    return prices


def load_sentiment_scores(symbols: list[str], config) -> tuple[dict[str, float], float]:
    """Per-symbol sentiment plus the market average, from whichever provider answers.

    A provider outage degrades to neutral rather than failing the run: sentiment is one
    weighted term in a composite score, and losing it should not stop the algorithm trading.
    """
    from src.algorithms.fast_momentum import sentiment_scores_from_records

    records: list[dict[str, Any]] = []
    providers = [str(item).lower() for item in getattr(config, "sentiment_data_provider_order", [])]
    if not providers:
        providers = [str(item).lower() for item in getattr(config, "news_sentiment_provider_order", [])]
    for provider in providers or [""]:
        provider_config = config
        if provider and dataclasses.is_dataclass(config):
            provider_config = dataclasses.replace(config, news_sentiment_provider_order=[provider])
        try:
            records.extend(fetch_latest_news_sentiment(symbols, provider_config))
        except Exception as exc:
            logger.warning(
                "Sentiment provider %s unavailable; continuing with neutral fallback: %s",
                provider or "default",
                exc,
            )

    lookback_minutes = int(getattr(config, "sentiment_lookback_minutes", 60) or 60)
    by_symbol, market_sentiment, _metadata, _providers = sentiment_scores_from_records(
        symbols, records, lookback_minutes
    )
    return by_symbol, market_sentiment


def build_algorithm_context(
    config,
    requirements,
    *,
    positions: dict[str, int] | None = None,
    equity: float = 0.0,
    data_client: Any = None,
) -> AlgorithmContext:
    """Satisfy ``requirements`` from live data and return the context ``analyze`` will read.

    The single place a live context is assembled. The live runner, the dashboard signal view,
    and the MCP agent all go through here, so an algorithm that declares a new data need gets
    it everywhere at once. The backtester builds the same context from cached history instead
    -- see ``src/execution/replay.py``.
    """
    positions = positions or {}
    data_client = data_client or create_data_client(config)

    price_symbols = sorted(set(requirements.price_symbols or config.symbols) | set(positions))
    latest_prices = load_latest_prices(price_symbols, config, data_client)

    bars_by_symbol: dict[str, Any] = {}
    if requirements.daily_lookback_days:
        bars_by_symbol = fetch_daily_bars(
            # The symbols the algorithm asked for, not the global universe: an algorithm
            # trading a name outside config.symbols used to receive no daily bars for it and
            # silently score it as flat, which reads as a market fact rather than a data gap.
            price_symbols,
            requirements.daily_lookback_days,
            ma_days=requirements.daily_ma_days,
            extra_buffer_days=requirements.daily_extra_buffer_days,
            alpaca_data_client=data_client,
            data_feed=config.alpaca_data_feed,
            include_latest=requirements.include_latest_daily,
            config=config,
        )

    intraday_bars_by_symbol: dict[str, Any] = {}
    if requirements.intraday_lookback_bars:
        intraday_bars_by_symbol = fetch_intraday_market_bars(
            price_symbols,
            config,
            lookback_bars=requirements.intraday_lookback_bars,
            bar_minutes=requirements.intraday_bar_minutes,
            data_client=data_client,
        )

    sentiment_scores: dict[str, float] = {}
    market_sentiment = 0.0
    if requirements.needs_sentiment:
        sentiment_scores, market_sentiment = load_sentiment_scores(price_symbols, config)

    return AlgorithmContext(
        config=config,
        bars_by_symbol=bars_by_symbol,
        intraday_bars_by_symbol=intraday_bars_by_symbol,
        sentiment_scores=sentiment_scores,
        market_sentiment=market_sentiment,
        positions=positions,
        latest_prices=latest_prices,
        equity=equity,
        account_id=getattr(config, "account_id", ""),
        timestamp=datetime.now(timezone.utc),
        extra={"data_client": data_client},
    )
