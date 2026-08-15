"""Alpaca market data.

Extracted verbatim from ``service.py``: this is the same code, in a file named after the
provider it belongs to. Adding a provider is now a new module plus a registry line, rather
than an edit to the module that dispatches to every provider.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from ...core.config import Config
from ..support import (
    EOD_CACHE_TTL_SECONDS, INTRADAY_CACHE_TTL_SECONDS, MARKET_CATEGORY,
    _empty_bars, _fresh_cached_bars, _normalize_quote, _provider_bars,
    _read_duckdb_bars, _write_duckdb_bars,
    bars_for_minutes, default_bar_minutes, DAILY_INTERVAL_MINUTES, normalize_intraday_frame, record_provider_success,
    resolve_bar_minutes,
)

logger = logging.getLogger(__name__)


def _fetch_alpaca_quotes(symbols: list[str], config: Config, data_client=None) -> dict[str, dict[str, Any]]:
    from src.brokerages.alpaca_client import create_data_client, get_latest_price

    client = data_client or create_data_client(config)
    quotes: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        try:
            price = get_latest_price(symbol, client, data_feed=config.alpaca_data_feed)
        except Exception as exc:
            logger.info("Alpaca latest price unavailable for %s: %s", symbol, exc)
            continue
        quote = _normalize_quote("alpaca", symbol, price, {"price": price, "feed": config.alpaca_data_feed})
        if quote:
            quotes[symbol.upper()] = quote
    if quotes:
        record_provider_success(MARKET_CATEGORY, "alpaca")
    return quotes


def fetch_alpaca_intraday_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_minutes: int,
    bar_minutes: int | None = None,
    force_refresh: bool = False,
    data_client=None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    from src.brokerages.alpaca_client import create_data_client, get_historical_intraday_bars

    interval_minutes = resolve_bar_minutes("alpaca", bar_minutes or default_bar_minutes(config))
    lookback_bars = bars_for_minutes(lookback_minutes, interval_minutes)
    ttl_seconds = int(getattr(config, "intraday_market_data_cache_ttl_seconds", INTRADAY_CACHE_TTL_SECONDS))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in [item.upper() for item in symbols]:
        cached = (
            _empty_bars()
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars("alpaca", symbol, interval_minutes, limit=lookback_bars),
                interval_minutes,
            )
        )
        if not cached.empty:
            bars_by_symbol[symbol] = cached.tail(lookback_bars).reset_index(drop=True)
        else:
            missing.append(symbol)
    if missing:
        client = data_client or create_data_client(config)
        fresh = get_historical_intraday_bars(
            missing,
            lookback_bars=lookback_bars,
            bar_minutes=interval_minutes,
            data_client=client,
            start_date=start_date,
            end_date=end_date,
            data_feed=config.alpaca_data_feed,
        )
        for symbol, bars in fresh.items():
            normalized = _provider_bars(normalize_intraday_frame(bars), interval_minutes,
                                        start_date=start_date, end_date=end_date, limit=lookback_bars)
            _write_duckdb_bars("alpaca", symbol, interval_minutes, normalized, ttl_seconds=ttl_seconds)
            bars_by_symbol[symbol.upper()] = normalized
    return bars_by_symbol


def fetch_alpaca_eod_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_bars: int,
    force_refresh: bool = False,
    data_client=None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    from src.brokerages.alpaca_client import create_data_client, get_historical_daily_bars

    ttl_seconds = int(getattr(config, "eod_market_data_cache_ttl_seconds", EOD_CACHE_TTL_SECONDS))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in [item.upper() for item in symbols]:
        cached = (
            _empty_bars()
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars("alpaca", symbol, DAILY_INTERVAL_MINUTES, limit=lookback_bars),
                DAILY_INTERVAL_MINUTES,
            )
        )
        if not cached.empty:
            bars_by_symbol[symbol] = cached.tail(lookback_bars).reset_index(drop=True)
        else:
            missing.append(symbol)
    if missing:
        client = data_client or create_data_client(config)
        fresh = get_historical_daily_bars(
            missing,
            lookback_days=lookback_bars,
            extra_buffer_days=0,
            data_client=client,
            start_date=start_date,
            end_date=end_date,
            data_feed=config.alpaca_data_feed,
        )
        for symbol, bars in fresh.items():
            normalized = _provider_bars(normalize_intraday_frame(bars), DAILY_INTERVAL_MINUTES,
                                        start_date=start_date, end_date=end_date, limit=lookback_bars)
            _write_duckdb_bars("alpaca", symbol, DAILY_INTERVAL_MINUTES, normalized, ttl_seconds=ttl_seconds)
            bars_by_symbol[symbol.upper()] = normalized
    return bars_by_symbol
