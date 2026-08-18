"""Alpha Vantage market data.

Extracted verbatim from ``service.py``: this is the same code, in a file named after the
provider it belongs to. Adding a provider is now a new module plus a registry line, rather
than an edit to the module that dispatches to every provider.
"""

from __future__ import annotations

import logging
from typing import Any


from ...core.config import Config
from ..support import (
    MARKET_CATEGORY, ProviderUnavailable, _api_key,
    _normalize_quote, _request_json,
)

logger = logging.getLogger(__name__)


def _fetch_alpha_vantage_quotes(symbols: list[str], config: Config) -> dict[str, dict[str, Any]]:
    key = _api_key(config, MARKET_CATEGORY, "alpha_vantage")
    if not key:
        raise ProviderUnavailable("ALPHA_VANTAGE_API_KEY is not configured")
    quotes: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        payload = _request_json(
            "alpha_vantage",
            MARKET_CATEGORY,
            "https://www.alphavantage.co/query",
            {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key},
        )
        quote_payload = payload.get("Global Quote", {}) if isinstance(payload, dict) else {}
        quote = _normalize_quote(
            "alpha_vantage",
            symbol,
            quote_payload.get("05. price"),
            payload,
            quote_payload.get("07. latest trading day"),
        )
        if quote:
            quotes[symbol.upper()] = quote
    return quotes
