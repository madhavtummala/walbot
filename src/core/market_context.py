from __future__ import annotations

import logging
from typing import Any

from src.brokerages.alpaca_client import create_data_client, get_latest_price
from src.connectors import (
    fetch_latest_market_quotes,
    fetch_latest_news_sentiment,
    merge_social_frames,
    news_records_to_social_frames,
)
from src.data import fetch_daily_bars
from src.data.social import load_social_trends_csv

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


def load_sentiment_frames(config, symbols: list[str]):
    return merge_social_frames(
        load_social_trends_csv(config.social_trends_csv, symbols),
        news_records_to_social_frames(fetch_latest_news_sentiment(symbols, config)),
    )


def load_algorithm_context(
    config,
    requirements,
    current_positions: dict[str, int] | None = None,
    data_client: Any = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], Any]:
    """Load bars, sentiment, and latest prices for an algorithm run without needing a brokerage.

    Returns ``(bars_by_symbol, sentiment_by_symbol, latest_prices, data_client)``.
    """
    current_positions = current_positions or {}
    data_client = data_client or create_data_client(config)

    price_symbols = sorted(set(requirements.price_symbols or config.symbols) | set(current_positions))
    latest_prices = load_latest_prices(price_symbols, config, data_client)

    bars_by_symbol: dict[str, Any] = {}
    if requirements.daily_lookback_days:
        bars_by_symbol = fetch_daily_bars(
            config.symbols,
            requirements.daily_lookback_days,
            ma_days=requirements.daily_ma_days,
            extra_buffer_days=requirements.daily_extra_buffer_days,
            alpaca_data_client=data_client,
            data_feed=config.alpaca_data_feed,
            include_latest=requirements.include_latest_daily,
            config=config,
        )

    sentiment_by_symbol: dict[str, Any] = {}
    if requirements.needs_sentiment:
        sentiment_by_symbol = load_sentiment_frames(config, config.symbols)

    return bars_by_symbol, sentiment_by_symbol, latest_prices, data_client
