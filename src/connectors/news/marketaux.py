"""Marketaux market data.

Extracted verbatim from ``service.py``: this is the same code, in a file named after the
provider it belongs to. Adding a provider is now a new module plus a registry line, rather
than an edit to the module that dispatches to every provider.
"""

from __future__ import annotations

import logging
from typing import Any


from ...core.config import Config
from ..support import (
    NEWS_CATEGORY, ProviderUnavailable, _api_key,
    _news_record, _request_json,
)

logger = logging.getLogger(__name__)


def _fetch_marketaux_news(symbols: list[str], config: Config) -> list[dict[str, Any]]:
    key = _api_key(config, NEWS_CATEGORY, "marketaux")
    if not key:
        raise ProviderUnavailable("MARKETAUX_API_KEY is not configured")
    payload = _request_json(
        "marketaux",
        NEWS_CATEGORY,
        "https://api.marketaux.com/v1/news/all",
        {
            "api_token": key,
            "symbols": ",".join(symbols),
            "filter_entities": "true",
            "language": "en",
            "limit": 50,
        },
    )
    records: list[dict[str, Any]] = []
    for item in payload.get("data", []) if isinstance(payload, dict) else []:
        entities = item.get("entities") or []
        matched_symbols = [str(entity.get("symbol", "")).upper() for entity in entities if str(entity.get("symbol", "")).upper() in symbols]
        if not matched_symbols:
            matched_symbols = [symbol for symbol in symbols if symbol in f"{item.get('title', '')} {item.get('description', '')}".upper()]
        for symbol in matched_symbols:
            entity = next((entity for entity in entities if str(entity.get("symbol", "")).upper() == symbol), {})
            records.append(
                _news_record(
                    "marketaux",
                    symbol,
                    item.get("published_at"),
                    str(item.get("title") or ""),
                    str(item.get("url") or ""),
                    entity.get("sentiment_score", item.get("sentiment_score", 0.0)),
                    item,
                )
            )
    return records
