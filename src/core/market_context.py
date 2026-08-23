from __future__ import annotations

from abc import ABC, abstractmethod

from src.data.signals.sentiment import sentiment_scores_from_records

import dataclasses
import logging
from datetime import datetime, timezone
from typing import Any

from src.brokerages.alpaca.client import create_data_client
from src.connectors import (
    fetch_latest_news_sentiment,
    fetch_market_history,
)
from src.core.interfaces import AlgorithmContext
from src.data import fetch_daily_bars
from src.data.state_store import algorithm_state_key, load_state

logger = logging.getLogger(__name__)


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
        """The moment this context describes. ``plan`` may read no data after it."""

    @abstractmethod
    def latest_prices(self, symbols: list[str], config) -> dict[str, float]:
        ...

    @abstractmethod
    def daily_bars(self, symbols: list[str], requirements, config) -> dict[str, Any]:
        ...

    @abstractmethod
    def intraday_bars(self, symbols: list[str], requirements, config) -> dict[str, Any]:
        ...

    def sentiment(self, symbols: list[str], config) -> tuple[dict[str, float], float]:
        """Neutral by default: only a live run has a sentiment feed to read."""
        return {}, 0.0

    def extra(self) -> dict[str, Any]:
        return {}


class LiveContextSource(ContextSource):
    """Satisfies requirements from the live feeds, through the connector layer.

    Prices come from the one entry point, :func:`load_latest_prices`: live quotes while
    the market is open, stored bars otherwise or on failure. Quote metadata rides along
    via :meth:`extra` so a caller can show *which* kind of price it got -- the dashboard
    uses this to flag prices that are not live prints. With ``as_of`` set, bars are
    fetched ending at that timestamp and prices come from the bar store only, since a
    snapshot of a past moment must not price against "now".
    """

    def __init__(
        self,
        data_client: Any = None,
        config=None,
        as_of: datetime | None = None,
    ) -> None:
        self._data_client = data_client or (create_data_client(config) if config is not None else None)
        self._as_of = as_of
        self._price_quotes: dict[str, dict[str, Any]] = {}
        self._price_symbols: list[str] = []

    @property
    def data_client(self) -> Any:
        return self._data_client

    def timestamp(self) -> datetime:
        return self._as_of or datetime.now(timezone.utc)

    def latest_prices(self, symbols: list[str], config) -> dict[str, float]:
        from ..connectors import load_latest_prices, prices_from_store

        self._price_symbols = [symbol.upper() for symbol in symbols]
        if self._as_of is not None:
            quotes = prices_from_store([symbol.upper() for symbol in symbols])
        else:
            quotes = load_latest_prices(symbols, config, data_client=self._data_client)
        self._price_quotes = quotes

        prices: dict[str, float] = {}
        for symbol, quote in quotes.items():
            try:
                price = float(quote.get("price") or 0.0)
            except (TypeError, ValueError):
                continue
            if price > 0:
                prices[str(symbol).upper()] = price
        return prices

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
            include_latest=self._as_of is None,
            config=config,
            end_date=self._as_of,
        )

    def intraday_bars(self, symbols: list[str], requirements, config) -> dict[str, Any]:
        return fetch_market_history(
            symbols,
            config,
            lookback_minutes=requirements.intraday_lookback_minutes,
            bar_minutes=requirements.preferred_bar_minutes,
            data_client=self._data_client,
            end_date=self._as_of,
        )

    def sentiment(self, symbols: list[str], config) -> tuple[dict[str, float], float]:
        return load_sentiment_scores(symbols, config)

    def extra(self) -> dict[str, Any]:
        extra: dict[str, Any] = {"data_client": self._data_client, "price_quotes": self._price_quotes}
        from ..connectors.streaming import stream_status, stream_store

        # Real-time microstructure, when the stream is running: the book and the trade
        # prints behind the latest bar. An algorithm that never reads these pays nothing,
        # and a replay (no live stream) sees neither.
        store = stream_store()
        if store is not None:
            symbols = self._price_symbols or None
            extra["order_books"] = store.snapshot_books(symbols)
            extra["stream_trades"] = store.snapshot_trades(symbols)
        extra["stream_status"] = stream_status()
        return extra


def build_algorithm_context(
    config,
    requirements,
    *,
    algorithm_id: str = "",
    positions: dict[str, int] | None = None,
    equity: float = 0.0,
    data_client: Any = None,
    source: ContextSource | None = None,
    as_of: datetime | None = None,
) -> AlgorithmContext:
    """Satisfy ``requirements`` and return the context ``plan`` will read.

    The single place a context is assembled, live or replayed. The live runner, the dashboard
    signal view, the MCP agent and the backtester all go through here, so an algorithm that
    declares a new data need gets it everywhere at once -- which was the original intent, and
    was true of everything except the backtester until ``source`` existed.

    When ``as_of`` is provided without an explicit ``source``, a ``LiveContextSource`` is
    created that fetches bars ending at that timestamp and prices from the bar store.
    """
    positions = positions or {}
    source = source or LiveContextSource(data_client=data_client, config=config, as_of=as_of)

    price_symbols = sorted(set(requirements.price_symbols or config.symbols) | set(positions))
    latest_prices = source.latest_prices(price_symbols, config)

    daily_bars_by_symbol: dict[str, Any] = {}
    if requirements.daily_lookback_days:
        daily_bars_by_symbol = source.daily_bars(price_symbols, requirements, config)

    intraday_bars_by_symbol: dict[str, Any] = {}
    if requirements.intraday_lookback_minutes:
        intraday_bars_by_symbol = source.intraday_bars(price_symbols, requirements, config)

    sentiment_scores: dict[str, float] = {}
    market_sentiment = 0.0
    if requirements.needs_sentiment:
        sentiment_scores, market_sentiment = source.sentiment(price_symbols, config)

    # Read here, so an algorithm never reaches into the store during its own run. Under a
    # replay this resolves against the throwaway dict ``ephemeral_state`` installs, which is
    # what keeps a backtest from reading -- and then overwriting -- the live account's memory.
    state: dict[str, Any] = {}
    if requirements.needs_state:
        loaded = load_state(algorithm_state_key(algorithm_id, getattr(config, "account_id", "")), {})
        state = dict(loaded) if isinstance(loaded, dict) else {}

    return AlgorithmContext(
        config=config,
        daily_bars_by_symbol=daily_bars_by_symbol,
        intraday_bars_by_symbol=intraday_bars_by_symbol,
        sentiment_scores=sentiment_scores,
        market_sentiment=market_sentiment,
        positions=positions,
        latest_prices=latest_prices,
        equity=equity,
        account_id=getattr(config, "account_id", ""),
        state=state,
        timestamp=source.timestamp(),
        extra=source.extra(),
    )
