"""Social and news sentiment views.

Split out of the single ``api_payloads`` module, which had grown to 1253 lines covering nine
unrelated domains. The public names are unchanged and still importable from ``api_payloads``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ...connectors.sentiment.alpha_vantage import collect_alpha_vantage_news, write_social_trends_csv
from ...core.config import (

    get_config,
)
from ...data.social import load_social_trends_csv

logger = logging.getLogger(__name__)
from .system import _display_path





def social_payload(limit: int = 250) -> dict[str, Any]:
    config = get_config()
    social_by_symbol = load_social_trends_csv(config.social_trends_csv, config.symbols)
    summary: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for symbol, df in social_by_symbol.items():
        if df.empty:
            continue
        work_df = df.copy()
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)
        latest = work_df.sort_values("timestamp").iloc[-1]
        summary.append(
            {
                "symbol": symbol,
                "latest_timestamp": latest["timestamp"].isoformat(),
                "mentions": float(latest.get("mentions", 0.0)),
                "sentiment": float(latest.get("sentiment", 0.0)),
                "social_score": float(latest.get("social_score", 0.0)),
            }
        )
        for _, row in work_df.tail(limit).iterrows():
            rows.append(
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "symbol": symbol,
                    "mentions": float(row.get("mentions", 0.0)),
                    "sentiment": float(row.get("sentiment", 0.0)),
                    "social_score": float(row.get("social_score", 0.0)),
                }
            )

    rows.sort(key=lambda item: (item["timestamp"], item["symbol"]), reverse=True)
    summary.sort(key=lambda item: item["symbol"])
    return {"summary": summary, "rows": rows[:limit]}


def refresh_social_payload(body: dict[str, Any]) -> dict[str, Any]:
    config = get_config()
    symbols = body.get("symbols") or config.symbols
    if isinstance(symbols, str):
        symbols = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]

    limit = int(body.get("limit") or config.alpha_vantage_news_limit)
    max_symbols = int(body.get("max_symbols") or min(config.alpha_vantage_max_symbols, len(symbols)))
    lookback_days = int(body.get("lookback_days") or config.alpha_vantage_news_lookback_days)
    delay = float(body.get("delay") if body.get("delay") is not None else config.alpha_vantage_request_delay_seconds)
    output = str(body.get("output") or config.alpha_vantage_news_csv)

    df = collect_alpha_vantage_news(
        config.alpha_vantage_api_key,
        symbols,
        lookback_days=lookback_days,
        limit=limit,
        max_symbols=max_symbols,
        request_delay_seconds=delay,
    )
    csv_path = write_social_trends_csv(df, output)
    return {
        "rows": len(df),
        "symbols_requested": min(len(symbols), max_symbols),
        "output": _display_path(csv_path),
    }
