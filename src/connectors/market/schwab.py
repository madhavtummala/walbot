"""Schwab market data, over the ``marketdata/v1`` REST endpoints.

Schwab's ``pricehistory`` takes a fixed set of minute frequencies and 400s on anything else, so
:data:`~src.connectors.grid.PROVIDER_BAR_MINUTES` snaps a request to the nearest grid it serves
rather than letting it fail. With horizons stated in market minutes, a coarser bar still answers
the question.

The realtime stream is separate -- see ``market/schwab_stream.py``. This module is the REST
fallback that answers when the stream is not running.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ...common.config_utils import json_number
from ...data.bars import calendar_days_for
from ...data.duckdb_store import DAILY_INTERVAL_MINUTES
from ..base import MarketDataProvider
from ..frames import _empty_bars, _normalize_quote
from ..http import _bearer_auth_header, _request_json
from ..sources import (
    EOD_MARKET_CATEGORY,
    INTRADAY_MARKET_CATEGORY,
    MARKET_CATEGORY,
    ProviderUnavailable,
    _schwab_token,
)

logger = logging.getLogger(__name__)

PRICE_HISTORY_URL = "https://api.schwabapi.com/marketdata/v1/pricehistory"
QUOTES_URL = "https://api.schwabapi.com/marketdata/v1/quotes"


def _candles_to_bars(payload: Any) -> pd.DataFrame:
    candles = payload.get("candles") if isinstance(payload, dict) else []
    rows = [
        {
            "timestamp": pd.to_datetime(int(candle["datetime"]), unit="ms", utc=True),
            "open": float(candle.get("open", 0.0)),
            "high": float(candle.get("high", 0.0)),
            "low": float(candle.get("low", 0.0)),
            "close": float(candle.get("close", 0.0)),
            "volume": float(candle.get("volume", 0.0)),
        }
        for candle in (candles or [])
        if candle.get("datetime") is not None
    ]
    if not rows:
        return _empty_bars()
    return pd.DataFrame.from_records(rows).sort_values("timestamp").reset_index(drop=True)


class Schwab(MarketDataProvider):
    name = "schwab"

    def fetch_price(self, symbols: list[str], **extra: Any) -> dict[str, dict[str, Any]]:
        """Latest quotes, preferring the last trade and falling back to the bid/ask mid."""
        token = _schwab_token(self.config, MARKET_CATEGORY)
        if not token:
            raise ProviderUnavailable("Schwab access token is not configured")

        wanted = [symbol.upper() for symbol in symbols if symbol]
        payload = _request_json(
            self.name, MARKET_CATEGORY, QUOTES_URL,
            {"symbols": ",".join(wanted)}, headers=_bearer_auth_header(token),
        ) or {}

        quotes: dict[str, dict[str, Any]] = {}
        for symbol in wanted:
            row = payload.get(symbol) or {}
            raw = row.get("quote", row) or {}
            price = json_number(raw.get("lastPrice"))
            if not price or price <= 0:
                bid = json_number(raw.get("bidPrice")) or 0.0
                ask = json_number(raw.get("askPrice")) or 0.0
                price = (bid + ask) / 2 if bid > 0 and ask > 0 else json_number(raw.get("closePrice"))
            stamp = raw.get("quoteTime") or raw.get("tradeTime")
            quote = _normalize_quote(
                self.name, symbol, price, row,
                pd.to_datetime(stamp, unit="ms", utc=True, errors="coerce") if stamp else None,
            )
            if quote:
                quotes[symbol] = quote
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
        daily = interval_minutes >= DAILY_INTERVAL_MINUTES
        category = EOD_MARKET_CATEGORY if daily else INTRADAY_MARKET_CATEGORY
        token = _schwab_token(self.config, category)
        if not token:
            raise ProviderUnavailable("Schwab access token is not configured")

        end = end_date or datetime.now(timezone.utc)
        span_days = lookback_bars if daily else calendar_days_for(lookback_bars * interval_minutes)
        start = start_date or end - timedelta(days=max(span_days, 1))
        grid = (
            {"periodType": "year", "frequencyType": "daily", "frequency": 1}
            if daily
            else {"frequencyType": "minute", "frequency": interval_minutes}
        )

        # One request per symbol: pricehistory takes a single ``symbol``, unlike the quotes
        # endpoint above.
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            payload = _request_json(
                self.name, category, PRICE_HISTORY_URL,
                {
                    "symbol": symbol,
                    **grid,
                    "startDate": int(start.timestamp() * 1000),
                    "endDate": int(end.timestamp() * 1000),
                    "needExtendedHoursData": "false",
                    "needPreviousClose": "false",
                },
                headers=_bearer_auth_header(token),
            )
            frames[symbol] = _candles_to_bars(payload)
        return frames
