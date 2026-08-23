"""Alpha Vantage market data.

Quotes only. Alpha Vantage serves bars too, but behind a rate limit strict enough that it has
never been worth putting in a provider order for them -- so ``fetch_bars`` reports nothing and
the dispatcher falls through.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from ..base import MarketDataProvider
from ..frames import _normalize_quote
from ..http import _request_json
from ..sources import MARKET_CATEGORY, ProviderUnavailable, _api_key

logger = logging.getLogger(__name__)

QUERY_URL = "https://www.alphavantage.co/query"


class AlphaVantage(MarketDataProvider):
    name = "alpha_vantage"

    def fetch_price(self, symbols: list[str], **extra: Any) -> dict[str, dict[str, Any]]:
        key = _api_key(self.config, MARKET_CATEGORY, self.name)
        if not key:
            raise ProviderUnavailable("ALPHA_VANTAGE_API_KEY is not configured")
        quotes: dict[str, dict[str, Any]] = {}
        for symbol in symbols:
            payload = _request_json(
                self.name, MARKET_CATEGORY, QUERY_URL,
                {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key},
            )
            row = payload.get("Global Quote", {}) if isinstance(payload, dict) else {}
            quote = _normalize_quote(
                self.name, symbol, row.get("05. price"), payload, row.get("07. latest trading day")
            )
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
        return {}
