from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.signals.sentiment import sentiment_scores_from_records

import dataclasses
import logging
from datetime import datetime, timezone
from typing import Any

from src.brokerages.alpaca_client import create_data_client, get_latest_price
from src.connectors import (
    fetch_latest_market_quotes,
    fetch_latest_news_sentiment,
    fetch_market_history,
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


class ContextSource(ABC):
    """Where an :class:`AlgorithmContext`'s data comes from.

    ``build_algorithm_context`` used to do two jobs at once -- decide *what* an algorithm's
    ``requirements`` mean, and *fetch* it from live feeds. Only the second differs between a
    live run and a replay, but because they were fused the backtester had to assemble its own
    context by hand. That is the kind of duplication that goes stale silently: an algorithm
    declaring a new data need got it live and not in the backtest, and nothing failed.

    Sourcing is the seam, so it is the thing with implementations.
    """

    @abstractmethod
    def timestamp(self) -> datetime:
        """The moment this context describes. ``analyze`` may read no data after it."""

    @abstractmethod
    def latest_prices(self, symbols: list[str], config) -> dict[str, float]:
        ...

    @abstractmethod
    def daily_bars(self, symbols: list[str], requirements, config) -> dict[str, Any]:
        ...

    @abstractmethod
    def history_bars(self, symbols: list[str], requirements, config) -> dict[str, Any]:
        ...

    def sentiment(self, symbols: list[str], config) -> tuple[dict[str, float], float]:
        """Neutral by default: only a live run has a sentiment feed to read."""
        return {}, 0.0

    def extra(self) -> dict[str, Any]:
        return {}


class LiveContextSource(ContextSource):
    """Satisfies requirements from the live feeds, through the connector layer."""

    def __init__(self, data_client: Any = None, config=None) -> None:
        self._data_client = data_client or (create_data_client(config) if config is not None else None)

    @property
    def data_client(self) -> Any:
        return self._data_client

    def timestamp(self) -> datetime:
        return datetime.now(timezone.utc)

    def latest_prices(self, symbols: list[str], config) -> dict[str, float]:
        return load_latest_prices(symbols, config, self._data_client)

    def daily_bars(self, symbols: list[str], requirements, config) -> dict[str, Any]:
        return fetch_daily_bars(
            # The symbols the algorithm asked for, not the global universe: an algorithm
            # trading a name outside config.symbols used to receive no daily bars for it and
            # silently score it as flat, which reads as a market fact rather than a data gap.
            symbols,
            requirements.daily_lookback_days,
            ma_days=requirements.daily_ma_days,
            extra_buffer_days=requirements.daily_extra_buffer_days,
            data_client=self._data_client,
            include_latest=requirements.include_latest_daily,
            config=config,
        )

    def history_bars(self, symbols: list[str], requirements, config) -> dict[str, Any]:
        return fetch_market_history(
            symbols,
            config,
            lookback_minutes=requirements.history_lookback_minutes,
            bar_minutes=requirements.preferred_bar_minutes,
            data_client=self._data_client,
        )

    def sentiment(self, symbols: list[str], config) -> tuple[dict[str, float], float]:
        return load_sentiment_scores(symbols, config)

    def extra(self) -> dict[str, Any]:
        return {"data_client": self._data_client}


def build_algorithm_context(
    config,
    requirements,
    *,
    positions: dict[str, int] | None = None,
    equity: float = 0.0,
    data_client: Any = None,
    source: ContextSource | None = None,
) -> AlgorithmContext:
    """Satisfy ``requirements`` and return the context ``analyze`` will read.

    The single place a context is assembled, live or replayed. The live runner, the dashboard
    signal view, the MCP agent and the backtester all go through here, so an algorithm that
    declares a new data need gets it everywhere at once -- which was the original intent, and
    was true of everything except the backtester until ``source`` existed.
    """
    positions = positions or {}
    source = source or LiveContextSource(data_client=data_client, config=config)

    price_symbols = sorted(set(requirements.price_symbols or config.symbols) | set(positions))
    latest_prices = source.latest_prices(price_symbols, config)

    bars_by_symbol: dict[str, Any] = {}
    if requirements.daily_lookback_days:
        bars_by_symbol = source.daily_bars(price_symbols, requirements, config)

    history_bars_by_symbol: dict[str, Any] = {}
    if requirements.history_lookback_minutes:
        history_bars_by_symbol = source.history_bars(price_symbols, requirements, config)

    sentiment_scores: dict[str, float] = {}
    market_sentiment = 0.0
    if requirements.needs_sentiment:
        sentiment_scores, market_sentiment = source.sentiment(price_symbols, config)

    return AlgorithmContext(
        config=config,
        bars_by_symbol=bars_by_symbol,
        history_bars_by_symbol=history_bars_by_symbol,
        sentiment_scores=sentiment_scores,
        market_sentiment=market_sentiment,
        positions=positions,
        latest_prices=latest_prices,
        equity=equity,
        account_id=getattr(config, "account_id", ""),
        timestamp=source.timestamp(),
        extra=source.extra(),
    )
