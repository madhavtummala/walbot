"""Finnhub market data.

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
    INTRADAY_MARKET_CATEGORY, MARKET_CATEGORY, ProviderUnavailable, _api_key,
    _bars_to_records, _empty_bars,
    _fresh_cached_bars, _intraday_cache_key, _normalize_quote, _provider_bars, _read_duckdb_bars,
    _records_to_bars, _request_json, _write_duckdb_bars, bars_for_minutes, calendar_days_for,
    default_bar_minutes, load_cached_payload, DAILY_INTERVAL_MINUTES, resolve_bar_minutes, save_cached_payload,
)

logger = logging.getLogger(__name__)


def _fetch_finnhub_quotes(symbols: list[str], config: Config) -> dict[str, dict[str, Any]]:
    key = _api_key(config, MARKET_CATEGORY, "finnhub")
    if not key:
        raise ProviderUnavailable("FINNHUB_API_KEY is not configured")
    quotes: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        payload = _request_json("finnhub", MARKET_CATEGORY, "https://finnhub.io/api/v1/quote", {"symbol": symbol, "token": key})
        quote = _normalize_quote("finnhub", symbol, payload.get("c"), payload, payload.get("t"))
        if quote:
            quotes[symbol.upper()] = quote
    return quotes


def _finnhub_candles_to_bars(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, dict) or payload.get("s") != "ok":
        return _empty_bars()
    timestamps = payload.get("t") or []
    opens = payload.get("o") or []
    highs = payload.get("h") or []
    lows = payload.get("l") or []
    closes = payload.get("c") or []
    volumes = payload.get("v") or []
    rows = []
    for timestamp, open_, high, low, close, volume in zip(timestamps, opens, highs, lows, closes, volumes):
        rows.append(
            {
                "timestamp": pd.to_datetime(int(timestamp), unit="s", utc=True),
                "open": float(open_),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
            }
        )
    if not rows:
        return _empty_bars()
    return pd.DataFrame.from_records(rows).sort_values("timestamp").reset_index(drop=True)


def fetch_finnhub_eod_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_bars: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    key = _api_key(config, EOD_MARKET_CATEGORY, "finnhub")
    if not key:
        raise ProviderUnavailable("FINNHUB_API_KEY is not configured")
    ttl_seconds = int(getattr(config, "eod_market_data_cache_ttl_seconds", EOD_CACHE_TTL_SECONDS))
    end = end_date or datetime.now(timezone.utc)
    start = start_date or end - timedelta(days=max(int(lookback_bars * 2), 30))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in [item.upper() for item in symbols]:
        cached = (
            _empty_bars()
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars("finnhub", symbol, DAILY_INTERVAL_MINUTES, limit=lookback_bars),
                DAILY_INTERVAL_MINUTES,
            )
        )
        if not cached.empty:
            bars_by_symbol[symbol] = cached.tail(lookback_bars).reset_index(drop=True)
            continue
        try:
            payload = _request_json(
                "finnhub",
                EOD_MARKET_CATEGORY,
                "https://finnhub.io/api/v1/stock/candle",
                {
                    "symbol": symbol,
                    "resolution": "D",
                    "from": int(start.timestamp()),
                    "to": int(end.timestamp()),
                    "token": key,
                },
            )
            bars = _provider_bars(_finnhub_candles_to_bars(payload), DAILY_INTERVAL_MINUTES,
                                  start_date=start_date, end_date=end_date, limit=lookback_bars)
        except ProviderUnavailable as exc:
            logger.warning("Skipping Finnhub EOD bars for %s after provider error: %s", symbol, exc)
            bars = _empty_bars()
        _write_duckdb_bars("finnhub", symbol, DAILY_INTERVAL_MINUTES, bars, ttl_seconds=ttl_seconds)
        bars_by_symbol[symbol] = bars
    return bars_by_symbol


def fetch_finnhub_intraday_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_minutes: int,
    bar_minutes: int | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch recent intraday OHLCV bars from Finnhub with a short-lived cache."""
    key = _api_key(config, INTRADAY_MARKET_CATEGORY, "finnhub")
    if not key:
        raise ProviderUnavailable("FINNHUB_API_KEY is not configured")
    if lookback_minutes <= 0:
        raise ValueError("lookback_minutes must be positive")

    interval_minutes = resolve_bar_minutes("finnhub", bar_minutes or default_bar_minutes(config))
    lookback_bars = bars_for_minutes(lookback_minutes, interval_minutes)
    end = end_date or datetime.now(timezone.utc)
    start = start_date or end - timedelta(days=calendar_days_for(lookback_minutes))
    resolution = str(interval_minutes)
    bars_by_symbol: dict[str, pd.DataFrame] = {}

    for symbol in [item.upper() for item in symbols]:
        cache_key = _intraday_cache_key(symbol, interval_minutes, lookback_bars)
        if not force_refresh:
            cached = load_cached_payload(INTRADAY_MARKET_CATEGORY, "finnhub", cache_key)
            cached_bars = _records_to_bars(cached)
            if not cached_bars.empty:
                bars_by_symbol[symbol] = cached_bars.tail(lookback_bars).reset_index(drop=True)
                continue
            duckdb_bars = _fresh_cached_bars(
                _read_duckdb_bars("finnhub", symbol, interval_minutes, limit=lookback_bars),
                interval_minutes,
            )
            if not duckdb_bars.empty:
                bars_by_symbol[symbol] = duckdb_bars.tail(lookback_bars).reset_index(drop=True)
                continue
        try:
            payload = _request_json(
                "finnhub",
                INTRADAY_MARKET_CATEGORY,
                "https://finnhub.io/api/v1/stock/candle",
                {
                    "symbol": symbol,
                    "resolution": resolution,
                    "from": int(start.timestamp()),
                    "to": int(end.timestamp()),
                    "token": key,
                },
            )
            bars = _provider_bars(_finnhub_candles_to_bars(payload), interval_minutes,
                                  start_date=start_date, end_date=end_date, limit=lookback_bars)
        except ProviderUnavailable as exc:
            logger.warning("Skipping Finnhub intraday bars for %s after provider error: %s", symbol, exc)
            bars = _empty_bars()
        if not bars.empty:
            save_cached_payload(
                INTRADAY_MARKET_CATEGORY,
                "finnhub",
                cache_key,
                _bars_to_records(bars),
                ttl_seconds=INTRADAY_CACHE_TTL_SECONDS,
            )
            _write_duckdb_bars(
                "finnhub",
                symbol,
                interval_minutes,
                bars,
                ttl_seconds=INTRADAY_CACHE_TTL_SECONDS,
            )
        bars_by_symbol[symbol] = bars
    return bars_by_symbol
