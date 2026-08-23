"""Alpaca market data.

Bars come back batched -- one call for every missing symbol -- because Alpaca's endpoint takes a
symbol list. The base hands ``fetch_bars`` exactly the symbols the cache could not answer, so
that batching survives intact.

Daily bars are dividend-adjusted at the source (``Adjustment.ALL``), which is why SGOV shows its
yield here rather than a flat sawtooth.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from ...data.duckdb_store import DAILY_INTERVAL_MINUTES
from ...data.provider_cache import record_provider_success
from ..base import MarketDataProvider
from ..frames import _normalize_quote
from ..sources import MARKET_CATEGORY

logger = logging.getLogger(__name__)


class Alpaca(MarketDataProvider):
    name = "alpaca"

    def fetch_price(self, symbols: list[str], *, data_client=None, **extra: Any) -> dict[str, dict[str, Any]]:
        from src.brokerages.alpaca.client import create_data_client, get_latest_price

        client = data_client or create_data_client(self.config)
        feed = self.config.alpaca_data_feed
        quotes: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            try:
                price = get_latest_price(symbol, client, data_feed=feed)
            except Exception as exc:  # noqa: BLE001 - one symbol must not fail the batch
                logger.info("Alpaca latest price unavailable for %s: %s", symbol, exc)
                continue
            quote = _normalize_quote(self.name, symbol, price, {"price": price, "feed": feed})
            if quote:
                quotes[symbol.upper()] = quote
        if quotes:
            record_provider_success(MARKET_CATEGORY, self.name)
        return quotes

    def fetch_bars(
        self,
        symbols: list[str],
        *,
        interval_minutes: int,
        lookback_bars: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        data_client=None,
        **extra: Any,
    ) -> dict[str, pd.DataFrame]:
        from src.brokerages.alpaca.client import (
            create_data_client,
            get_historical_daily_bars,
            get_historical_intraday_bars,
        )

        client = data_client or create_data_client(self.config)
        if interval_minutes >= DAILY_INTERVAL_MINUTES:
            return get_historical_daily_bars(
                symbols,
                lookback_days=lookback_bars,
                extra_buffer_days=0,
                data_client=client,
                start_date=start_date,
                end_date=end_date,
                data_feed=self.config.alpaca_data_feed,
            )
        return get_historical_intraday_bars(
            symbols,
            lookback_bars=lookback_bars,
            bar_minutes=interval_minutes,
            data_client=client,
            start_date=start_date,
            end_date=end_date,
            data_feed=self.config.alpaca_data_feed,
        )
