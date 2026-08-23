"""NewsAPI market data.

Extracted verbatim from ``service.py``: this is the same code, in a file named after the
provider it belongs to. Adding a provider is now a new module plus a registry line, rather
than an edit to the module that dispatches to every provider.
"""

from __future__ import annotations

import logging
from typing import Any


from ...core.config import Config
from ..frames import _news_record
from ..http import _request_json
from ..sources import NEWS_CATEGORY, ProviderUnavailable, _api_key

logger = logging.getLogger(__name__)


def _fetch_newsapi_news(symbols: list[str], config: Config) -> list[dict[str, Any]]:
    key = _api_key(config, NEWS_CATEGORY, "newsapi")
    if not key:
        raise ProviderUnavailable("NEWSAPI_API_KEY is not configured")
    query = " OR ".join(symbols[:20])
    payload = _request_json(
        "newsapi",
        NEWS_CATEGORY,
        "https://newsapi.org/v2/everything",
        {"apiKey": key, "q": query, "language": "en", "sortBy": "publishedAt", "pageSize": 100},
    )
    records: list[dict[str, Any]] = []
    for item in payload.get("articles", []) if isinstance(payload, dict) else []:
        text = f"{item.get('title', '')} {item.get('description', '')}".upper()
        for symbol in [symbol for symbol in symbols if symbol in text]:
            records.append(
                _news_record(
                    "newsapi",
                    symbol,
                    item.get("publishedAt"),
                    str(item.get("title") or ""),
                    str(item.get("url") or ""),
                    0.0,
                    item,
                )
            )
    return records
