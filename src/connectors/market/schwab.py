"""Schwab market data.

Extracted verbatim from ``service.py``: this is the same code, in a file named after the
provider it belongs to. Adding a provider is now a new module plus a registry line, rather
than an edit to the module that dispatches to every provider.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ...core.config import Config
from ..support import (
    EOD_CACHE_TTL_SECONDS, EOD_MARKET_CATEGORY, INTRADAY_CACHE_TTL_SECONDS,
    INTRADAY_MARKET_CATEGORY, MARKET_CATEGORY, ProviderUnavailable, _bearer_auth_header,
    _empty_bars, _finite,
    _fresh_cached_bars, _normalize_quote, _provider_bars, _read_duckdb_bars, _request_json,
    _schwab_token, _write_duckdb_bars, bars_for_minutes, calendar_days_for, default_bar_minutes,
    DAILY_INTERVAL_MINUTES, resolve_bar_minutes,
)

logger = logging.getLogger(__name__)


def _schwab_candles_to_bars(payload: Any) -> pd.DataFrame:
    candles = payload.get("candles") if isinstance(payload, dict) else []
    rows = []
    for candle in candles or []:
        timestamp = candle.get("datetime")
        if timestamp is None:
            continue
        rows.append(
            {
                "timestamp": pd.to_datetime(int(timestamp), unit="ms", utc=True),
                "open": float(candle.get("open", 0.0)),
                "high": float(candle.get("high", 0.0)),
                "low": float(candle.get("low", 0.0)),
                "close": float(candle.get("close", 0.0)),
                "volume": float(candle.get("volume", 0.0)),
            }
        )
    if not rows:
        return _empty_bars()
    return pd.DataFrame.from_records(rows).sort_values("timestamp").reset_index(drop=True)


def fetch_schwab_intraday_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_minutes: int,
    bar_minutes: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    token = _schwab_token(config, INTRADAY_MARKET_CATEGORY)
    if not token:
        raise ProviderUnavailable("Schwab access token is not configured")
    if lookback_minutes <= 0:
        raise ValueError("lookback_minutes must be positive")
    # Schwab's pricehistory takes a fixed set of minute frequencies and 400s on anything else,
    # so the request is snapped to the nearest grid it serves rather than rejected: with the
    # horizon stated in minutes, a coarser bar still answers the question.
    interval_minutes = resolve_bar_minutes("schwab", bar_minutes or default_bar_minutes(config))
    lookback_bars = bars_for_minutes(lookback_minutes, interval_minutes)
    ttl_seconds = int(getattr(config, "intraday_market_data_cache_ttl_seconds", INTRADAY_CACHE_TTL_SECONDS))
    end = end_date or datetime.now(timezone.utc)
    start = start_date or end - timedelta(days=calendar_days_for(lookback_minutes))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in [item.upper() for item in symbols]:
        cached = (
            _empty_bars()
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars("schwab", symbol, interval_minutes, limit=lookback_bars),
                interval_minutes,
            )
        )
        if not cached.empty:
            bars_by_symbol[symbol] = cached.tail(lookback_bars).reset_index(drop=True)
        else:
            missing.append(symbol)
    for symbol in missing:
        payload = _request_json(
            "schwab",
            INTRADAY_MARKET_CATEGORY,
            "https://api.schwabapi.com/marketdata/v1/pricehistory",
            {
                "symbol": symbol,
                "frequencyType": "minute",
                "frequency": interval_minutes,
                "startDate": int(start.timestamp() * 1000),
                "endDate": int(end.timestamp() * 1000),
                "needExtendedHoursData": "false",
                "needPreviousClose": "false",
            },
            headers=_bearer_auth_header(token),
        )
        bars = _provider_bars(_schwab_candles_to_bars(payload), interval_minutes,
                              start_date=start_date, end_date=end_date, limit=lookback_bars)
        _write_duckdb_bars("schwab", symbol, interval_minutes, bars, ttl_seconds=ttl_seconds)
        bars_by_symbol[symbol] = bars
    return bars_by_symbol


def fetch_schwab_eod_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_bars: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    token = _schwab_token(config, EOD_MARKET_CATEGORY)
    if not token:
        raise ProviderUnavailable("Schwab access token is not configured")
    ttl_seconds = int(getattr(config, "eod_market_data_cache_ttl_seconds", EOD_CACHE_TTL_SECONDS))
    end = end_date or datetime.now(timezone.utc)
    start = start_date or end - timedelta(days=max(int(lookback_bars * 2), 30))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    wanted = [item.upper() for item in symbols]
    cached_bars = {}
    for symbol in wanted:
        cached = (
            _empty_bars()
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars("schwab", symbol, DAILY_INTERVAL_MINUTES, limit=lookback_bars),
                DAILY_INTERVAL_MINUTES,
            )
        )
        if not cached.empty:
            cached_bars[symbol] = cached.tail(lookback_bars).reset_index(drop=True)
    for symbol in wanted:
        if symbol in cached_bars:
            bars_by_symbol[symbol] = cached_bars[symbol]
            continue
        payload = _request_json(
            "schwab",
            EOD_MARKET_CATEGORY,
            "https://api.schwabapi.com/marketdata/v1/pricehistory",
            {
                "symbol": symbol,
                "periodType": "year",
                "frequencyType": "daily",
                "frequency": 1,
                "startDate": int(start.timestamp() * 1000),
                "endDate": int(end.timestamp() * 1000),
                "needExtendedHoursData": "false",
                "needPreviousClose": "false",
            },
            headers=_bearer_auth_header(token),
        )
        bars = _provider_bars(_schwab_candles_to_bars(payload), DAILY_INTERVAL_MINUTES,
                              start_date=start_date, end_date=end_date, limit=lookback_bars)
        _write_duckdb_bars("schwab", symbol, DAILY_INTERVAL_MINUTES, bars, ttl_seconds=ttl_seconds)
        bars_by_symbol[symbol] = bars
    return bars_by_symbol


def _fetch_schwab_quotes(symbols: list[str], config: Config) -> dict[str, dict[str, Any]]:
    """Latest Schwab quotes, preferring last trade and falling back to the bid/ask mid."""
    token = _schwab_token(config, MARKET_CATEGORY)
    if not token:
        raise ProviderUnavailable("Schwab access token is not configured")

    wanted = [symbol.upper() for symbol in symbols if symbol]
    payload = _request_json(
        "schwab",
        MARKET_CATEGORY,
        "https://api.schwabapi.com/marketdata/v1/quotes",
        {"symbols": ",".join(wanted)},
        headers=_bearer_auth_header(token),
    ) or {}

    quotes: dict[str, dict[str, Any]] = {}
    for symbol in wanted:
        row = payload.get(symbol) or {}
        raw_quote = row.get("quote", row) or {}
        price = _finite(raw_quote.get("lastPrice"))
        if not price or price <= 0:
            bid = _finite(raw_quote.get("bidPrice")) or 0.0
            ask = _finite(raw_quote.get("askPrice")) or 0.0
            price = (bid + ask) / 2 if bid > 0 and ask > 0 else _finite(raw_quote.get("closePrice"))
        timestamp = raw_quote.get("quoteTime") or raw_quote.get("tradeTime")
        quote = _normalize_quote(
            "schwab",
            symbol,
            price,
            row,
            pd.to_datetime(timestamp, unit="ms", utc=True, errors="coerce") if timestamp else None,
        )
        if quote:
            quotes[symbol] = quote
    return quotes

