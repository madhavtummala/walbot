"""Finnhub market data, over the ``/api/v1`` REST endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ...data.bars import calendar_days_for
from ...data.duckdb_store import DAILY_INTERVAL_MINUTES
from ..base import MarketDataProvider
from ..frames import _empty_bars, _normalize_quote
from ..http import _request_json
from ..sources import (
    EOD_MARKET_CATEGORY,
    INTRADAY_MARKET_CATEGORY,
    MARKET_CATEGORY,
    ProviderUnavailable,
    _api_key,
)

logger = logging.getLogger(__name__)

QUOTE_URL = "https://finnhub.io/api/v1/quote"
CANDLE_URL = "https://finnhub.io/api/v1/stock/candle"


def _candles_to_bars(payload: Any) -> pd.DataFrame:
    """Finnhub returns parallel arrays, and ``s`` is the status field, not a symbol."""
    if not isinstance(payload, dict) or payload.get("s") != "ok":
        return _empty_bars()
    rows = [
        {
            "timestamp": pd.to_datetime(int(timestamp), unit="s", utc=True),
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        }
        for timestamp, open_, high, low, close, volume in zip(
            payload.get("t") or [], payload.get("o") or [], payload.get("h") or [],
            payload.get("l") or [], payload.get("c") or [], payload.get("v") or [],
        )
    ]
    if not rows:
        return _empty_bars()
    return pd.DataFrame.from_records(rows).sort_values("timestamp").reset_index(drop=True)


class Finnhub(MarketDataProvider):
    name = "finnhub"

    def _key(self) -> str:
        key = _api_key(self.config, MARKET_CATEGORY, self.name)
        if not key:
            raise ProviderUnavailable("FINNHUB_API_KEY is not configured")
        return key

    def fetch_price(self, symbols: list[str], **extra: Any) -> dict[str, dict[str, Any]]:
        key = self._key()
        quotes: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            payload = _request_json(self.name, MARKET_CATEGORY, QUOTE_URL, {"symbol": symbol, "token": key})
            quote = _normalize_quote(self.name, symbol, payload.get("c"), payload, payload.get("t"))
            if quote:
                quotes[symbol.upper()] = quote
        return quotes

    def fetch_bars(
        self,
        symbols: list[str],
        *,
        interval_minutes: int,
        lookback_bars: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **extra: Any,
    ) -> dict[str, pd.DataFrame]:
        key = self._key()
        daily = interval_minutes >= DAILY_INTERVAL_MINUTES
        end = end_date or datetime.now(timezone.utc)
        span_days = lookback_bars if daily else calendar_days_for(lookback_bars * interval_minutes)
        start = start_date or end - timedelta(days=max(span_days, 1))
        # "D" for daily, the minute count itself for anything finer. The category follows the
        # resolution too: it is the rate-limit namespace and the config section, so a daily
        # fetch must not be accounted against the intraday budget.
        resolution = "D" if daily else str(interval_minutes)
        category = EOD_MARKET_CATEGORY if daily else INTRADAY_MARKET_CATEGORY

        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            payload = _request_json(
                self.name, category, CANDLE_URL,
                {
                    "symbol": symbol,
                    "resolution": resolution,
                    "from": int(start.timestamp()),
                    "to": int(end.timestamp()),
                    "token": key,
                },
            )
            frames[symbol] = _candles_to_bars(payload)
        return frames
