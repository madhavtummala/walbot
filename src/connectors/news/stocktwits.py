"""StockTwits market data.

Extracted verbatim from ``service.py``: this is the same code, in a file named after the
provider it belongs to. Adding a provider is now a new module plus a registry line, rather
than an edit to the module that dispatches to every provider.
"""

from __future__ import annotations

import logging
import os
from typing import Any


from ...core.config import Config
from ..support import (
    NEWS_CATEGORY, _basic_auth_header, _finite,
    _news_record, _request_json, _provider_config,
)

logger = logging.getLogger(__name__)


def _fetch_stocktwits_news(symbols: list[str], _config: Config) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    provider_config = _provider_config(_config, NEWS_CATEGORY, "stocktwits")
    username_env = str(provider_config.get("username_env") or "").strip()
    password_env = str(provider_config.get("password_env") or "").strip()
    username = str(provider_config.get("username") or os.getenv(username_env, "")).strip()
    password = str(provider_config.get("password") or os.getenv(password_env, "")).strip()
    if username and password:
        for symbol in symbols:
            payload = _request_json(
                "stocktwits",
                NEWS_CATEGORY,
                f"https://api-gw-prd.stocktwits.com/api-middleware/external/sentiment/v2/{symbol}/detail",
                headers=_basic_auth_header(username, password),
            )
            data = payload.get("data", payload) if isinstance(payload, dict) else {}
            sentiment = (
                data.get("sentiment_score")
                or data.get("sentiment")
                or data.get("bullish_sentiment")
                or data.get("score")
                or 0.0
            )
            mentions = data.get("message_volume") or data.get("mentions") or data.get("volume") or 1.0
            record = _news_record(
                "stocktwits",
                symbol,
                data.get("timestamp") or data.get("updated_at") or data.get("created_at"),
                f"StockTwits sentiment detail for {symbol}",
                "",
                sentiment,
                payload,
            )
            record["mentions"] = float(_finite(mentions) or 1.0)
            records.append(record)
        if records:
            return records

    for symbol in symbols:
        payload = _request_json("stocktwits", NEWS_CATEGORY, f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json")
        for item in payload.get("messages", []) if isinstance(payload, dict) else []:
            sentiment = item.get("entities", {}).get("sentiment")
            basic = str((sentiment or {}).get("basic") or "").lower()
            score = 1.0 if basic == "bullish" else -1.0 if basic == "bearish" else 0.0
            records.append(
                _news_record(
                    "stocktwits",
                    symbol,
                    item.get("created_at"),
                    str(item.get("body") or "")[:240],
                    f"https://stocktwits.com/message/{item.get('id')}",
                    score,
                    item,
                )
            )
    return records

