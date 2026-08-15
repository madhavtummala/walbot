"""Normalising raw provider sentiment records into scores.

Lives in the data layer, not in an algorithm. It is a pure transform over already-fetched
records -- its own docstring said so -- but it sat inside ``fast_momentum``, so
``core.market_context`` had to reach into a concrete strategy to use it. That import was the
last cycle in the codebase: base -> market_context -> fast_momentum -> base.

Nothing here knows which algorithm is asking.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def sentiment_scores_from_records(
    symbols: list[str],
    records: list[dict[str, Any]],
    lookback_minutes: int,
) -> tuple[dict[str, float], float, dict[str, Any], list[str]]:
    """Normalise raw provider sentiment records into per-symbol and market scores.

    Pure: takes already-fetched records so callers control provider selection and fetching.
    Returns ``(by_symbol, market_sentiment, metadata, providers_seen)``. Market sentiment
    falls back to the universe average when SPY is not covered.
    """
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=max(lookback_minutes, 1))
    by_symbol: dict[str, list[float]] = {symbol.upper(): [] for symbol in symbols}
    providers_seen: list[str] = []
    used = 0

    for record in records or []:
        provider = str(record.get("provider", "")).lower()
        if provider and provider not in providers_seen:
            providers_seen.append(provider)
        symbol = str(record.get("symbol", "")).upper()
        if symbol not in by_symbol:
            continue
        timestamp = pd.to_datetime(record.get("timestamp"), utc=True, errors="coerce")
        if not pd.isna(timestamp) and timestamp < cutoff:
            continue
        try:
            sentiment = float(record.get("social_score", record.get("sentiment", 0.0)))
        except (TypeError, ValueError):
            sentiment = 0.0
        by_symbol[symbol].append(max(-1.0, min(1.0, sentiment)))
        used += 1

    symbol_sentiment = {
        symbol: (sum(values) / len(values) if values else 0.0)
        for symbol, values in by_symbol.items()
    }
    market_sentiment = symbol_sentiment.get("SPY")
    if market_sentiment is None:
        values = list(symbol_sentiment.values())
        market_sentiment = sum(values) / len(values) if values else 0.0

    metadata = {
        "records_seen": len(records or []),
        "records_used": used,
        "covered_symbols": sum(1 for values in by_symbol.values() if values),
    }
    return symbol_sentiment, float(market_sentiment), metadata, providers_seen
