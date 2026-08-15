"""yfinance market data.

Extracted verbatim from ``service.py``: this is the same code, in a file named after the
provider it belongs to. Adding a provider is now a new module plus a registry line, rather
than an edit to the module that dispatches to every provider.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from ...core.config import Config
from ..support import (
    EOD_CACHE_TTL_SECONDS, INTRADAY_CACHE_TTL_SECONDS, INTRADAY_MARKET_CATEGORY,
    ProviderUnavailable, _bars_to_records, _empty_bars, _fresh_cached_bars,
    _intraday_cache_key, _provider_bars,
    _read_duckdb_bars, _records_to_bars, _write_duckdb_bars, bars_for_minutes, calendar_days_for,
    default_bar_minutes, load_cached_payload, DAILY_INTERVAL_MINUTES, normalize_intraday_frame, resolve_bar_minutes,
    save_cached_payload,
)

logger = logging.getLogger(__name__)


def _yfinance_period_for(lookback_minutes: int) -> str:
    # yfinance caps intraday history at 60 days regardless of what is asked for.
    return f"{min(calendar_days_for(lookback_minutes), 59)}d"


def fetch_yfinance_intraday_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_minutes: int,
    bar_minutes: int | None = None,
    force_refresh: bool = False,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch recent intraday OHLCV bars from yfinance with a short-lived cache."""
    if lookback_minutes <= 0:
        raise ValueError("lookback_minutes must be positive")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ProviderUnavailable("yfinance is not installed") from exc

    interval_minutes = resolve_bar_minutes("yfinance", bar_minutes or default_bar_minutes(config))
    lookback_bars = bars_for_minutes(lookback_minutes, interval_minutes)
    ttl_seconds = int(getattr(config, "intraday_market_data_cache_ttl_seconds", INTRADAY_CACHE_TTL_SECONDS))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    interval = f"{interval_minutes}m"
    period = _yfinance_period_for(lookback_minutes)
    for symbol in [item.upper() for item in symbols]:
        cache_key = _intraday_cache_key(symbol, interval_minutes, lookback_bars)
        if not force_refresh:
            cached = load_cached_payload(INTRADAY_MARKET_CATEGORY, "yfinance", cache_key)
            cached_bars = _records_to_bars(cached)
            if not cached_bars.empty:
                bars_by_symbol[symbol] = cached_bars.tail(lookback_bars).reset_index(drop=True)
                continue
            duckdb_bars = _fresh_cached_bars(
                _read_duckdb_bars("yfinance", symbol, interval_minutes, limit=lookback_bars),
                interval_minutes,
            )
            if not duckdb_bars.empty:
                bars_by_symbol[symbol] = duckdb_bars.tail(lookback_bars).reset_index(drop=True)
                continue
        try:
            kwargs = {
                "interval": interval,
                "auto_adjust": False,
                "progress": False,
                "threads": False,
            }
            if start_date is not None or end_date is not None:
                kwargs.update({"start": start_date, "end": end_date})
            else:
                kwargs["period"] = period
            raw = yf.download(symbol, **kwargs)
            bars = _provider_bars(normalize_intraday_frame(raw), interval_minutes,
                                  start_date=start_date, end_date=end_date, limit=lookback_bars)
        except Exception as exc:
            logger.warning("Skipping yfinance intraday bars for %s after provider error: %s", symbol, exc)
            bars = _empty_bars()
        if not bars.empty:
            save_cached_payload(
                INTRADAY_MARKET_CATEGORY,
                "yfinance",
                cache_key,
                _bars_to_records(bars),
                ttl_seconds=ttl_seconds,
            )
            _write_duckdb_bars("yfinance", symbol, interval_minutes, bars, ttl_seconds=ttl_seconds)
        bars_by_symbol[symbol] = bars
    return bars_by_symbol


def fetch_yfinance_eod_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_bars: int,
    force_refresh: bool = False,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ProviderUnavailable("yfinance is not installed") from exc

    ttl_seconds = int(getattr(config, "eod_market_data_cache_ttl_seconds", EOD_CACHE_TTL_SECONDS))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    period = f"{max(int(lookback_bars * 2), 30)}d"
    for symbol in [item.upper() for item in symbols]:
        cached = (
            _empty_bars()
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars("yfinance", symbol, DAILY_INTERVAL_MINUTES, limit=lookback_bars),
                DAILY_INTERVAL_MINUTES,
            )
        )
        if not cached.empty:
            bars_by_symbol[symbol] = cached.tail(lookback_bars).reset_index(drop=True)
            continue
        try:
            kwargs = {"interval": "1d", "auto_adjust": False, "progress": False, "threads": False}
            if start_date is not None or end_date is not None:
                kwargs.update({"start": start_date, "end": end_date})
            else:
                kwargs["period"] = period
            raw = yf.download(symbol, **kwargs)
            bars = _provider_bars(normalize_intraday_frame(raw), DAILY_INTERVAL_MINUTES,
                                  start_date=start_date, end_date=end_date, limit=lookback_bars)
        except Exception as exc:
            logger.warning("Skipping yfinance EOD bars for %s after provider error: %s", symbol, exc)
            bars = _empty_bars()
        _write_duckdb_bars("yfinance", symbol, DAILY_INTERVAL_MINUTES, bars, ttl_seconds=ttl_seconds)
        bars_by_symbol[symbol] = bars
    return bars_by_symbol
