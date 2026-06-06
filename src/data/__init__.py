from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from src.data.state_store import load_state, save_state


MARKET_DATA_CACHE_STATE_KEY = "market_data_cache"
MARKET_DATA_CACHE_VERSION = 1
MARKET_DATA_CACHE_FRESH_FOR = pd.Timedelta(days=1)
DEFAULT_LONG_MA_DAYS = 200
BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLUMNS)


def _feed_key(data_feed: str | None) -> str:
    raw = getattr(data_feed, "value", data_feed) or "default"
    return str(raw).lower()


def _cache_item_key(symbol: str, data_feed: str | None) -> str:
    return f"{_feed_key(data_feed)}:{symbol.upper()}"


def _load_market_data_cache() -> dict[str, Any]:
    cache = load_state(
        MARKET_DATA_CACHE_STATE_KEY,
        {"version": MARKET_DATA_CACHE_VERSION, "items": {}},
    )
    if cache.get("version") == MARKET_DATA_CACHE_VERSION and isinstance(cache.get("items"), dict):
        return cache
    return {"version": MARKET_DATA_CACHE_VERSION, "items": {}}


def _save_market_data_cache(cache: dict[str, Any]) -> None:
    save_state(MARKET_DATA_CACHE_STATE_KEY, cache)


def _as_utc_timestamp(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def _normalize_bars(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_bars()

    work = df.copy()
    if "timestamp" not in work:
        return _empty_bars()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"])
    if work.empty:
        return _empty_bars()

    for column in ("open", "high", "low", "close", "volume"):
        if column not in work:
            work[column] = 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["open", "high", "low", "close"])
    if work.empty:
        return _empty_bars()

    return (
        work[BAR_COLUMNS]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def _records_to_bars(records: list[dict[str, Any]] | None) -> pd.DataFrame:
    if not records:
        return _empty_bars()
    return _normalize_bars(pd.DataFrame.from_records(records))


def _bars_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = _normalize_bars(df)
    records: list[dict[str, Any]] = []
    for row in normalized.to_dict(orient="records"):
        records.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )
    return records


def _target_start(end: datetime, lookback_days: int, ma_days: int, extra_buffer_days: int) -> datetime:
    total_lookback = lookback_days + ma_days + extra_buffer_days
    calendar_days = max(total_lookback + extra_buffer_days, 1) * 2
    return end - timedelta(days=calendar_days)


def _is_cache_stale(item: dict[str, Any] | None, now: pd.Timestamp) -> bool:
    if not item:
        return True
    updated_at = _as_utc_timestamp(item.get("updated_at"))
    if updated_at is None:
        return True
    return now - updated_at >= MARKET_DATA_CACHE_FRESH_FOR


def _filter_starting_at(df: pd.DataFrame, start: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    start_ts = pd.Timestamp(start)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    return df.loc[df["timestamp"] >= start_ts].reset_index(drop=True)


def refresh_market_data_cache(
    symbols: list[str],
    lookback_days: int,
    ma_days: int = DEFAULT_LONG_MA_DAYS,
    extra_buffer_days: int = 250,
    alpaca_data_client=None,
    end_date: datetime | None = None,
    data_feed: str | None = "iex",
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Refresh stale/missing cached daily bars and return the cached range."""
    from src.brokerages.alpaca_client import get_historical_daily_bars

    end = end_date or datetime.now(timezone.utc)
    target_start = _target_start(end, lookback_days, ma_days, extra_buffer_days)
    now = pd.Timestamp.now(tz="UTC")
    cache = _load_market_data_cache()
    fetched_by_symbol: dict[str, pd.DataFrame] = {}

    for symbol in [item.upper() for item in symbols]:
        item_key = _cache_item_key(symbol, data_feed)
        item = cache["items"].get(item_key, {})
        cached = _records_to_bars(item.get("bars"))
        cached = _filter_starting_at(cached, target_start)
        missing_history = cached.empty or pd.Timestamp(cached["timestamp"].iloc[0]) > pd.Timestamp(target_start)
        stale = _is_cache_stale(item, now)
        needs_fetch = force_refresh or missing_history or stale

        if not needs_fetch:
            fetched_by_symbol[symbol] = cached
            continue

        request_start = target_start
        if not missing_history and not cached.empty:
            latest_cached = pd.Timestamp(cached["timestamp"].max()).to_pydatetime()
            request_start = latest_cached + timedelta(days=1)
            if request_start > end:
                request_start = end - timedelta(days=1)

        try:
            fresh = get_historical_daily_bars(
                [symbol],
                lookback_days=lookback_days + ma_days + extra_buffer_days,
                extra_buffer_days=extra_buffer_days,
                data_client=alpaca_data_client,
                end_date=end,
                start_date=request_start,
                data_feed=data_feed,
            ).get(symbol, _empty_bars())
        except Exception:
            fresh = _empty_bars()
        merged = _normalize_bars(pd.concat([cached, fresh], ignore_index=True))
        merged = _filter_starting_at(merged, target_start)
        cache["items"][item_key] = {
            "symbol": symbol,
            "data_feed": _feed_key(data_feed),
            "updated_at": now.isoformat(),
            "bars": _bars_to_records(merged),
        }
        fetched_by_symbol[symbol] = merged

    _save_market_data_cache(cache)
    return fetched_by_symbol


def fetch_daily_bars(
    symbols: list[str],
    lookback_days: int,
    ma_days: int = DEFAULT_LONG_MA_DAYS,
    extra_buffer_days: int = 250,
    alpaca_data_client=None,
    end_date: datetime | None = None,
    data_feed: str | None = "iex",
    use_cache: bool = True,
    force_refresh: bool = False,
    include_latest: bool = False,
    config=None,
) -> dict[str, pd.DataFrame]:
    """Fetch enough daily bars for a momentum signal and moving average calculation."""
    from src.brokerages.alpaca_client import get_historical_daily_bars

    total_lookback = lookback_days + ma_days + extra_buffer_days
    if config is not None:
        from ..connectors import fetch_eod_market_bars

        bars_by_symbol = fetch_eod_market_bars(
            symbols=symbols,
            config=config,
            lookback_bars=total_lookback,
            force_refresh=force_refresh,
            data_client=alpaca_data_client,
        )
    elif use_cache:
        bars_by_symbol = refresh_market_data_cache(
            symbols=symbols,
            lookback_days=lookback_days,
            ma_days=ma_days,
            extra_buffer_days=extra_buffer_days,
            alpaca_data_client=alpaca_data_client,
            end_date=end_date,
            data_feed=data_feed,
            force_refresh=force_refresh,
        )
    else:
        bars_by_symbol = get_historical_daily_bars(
            symbols=symbols,
            lookback_days=total_lookback,
            extra_buffer_days=extra_buffer_days,
            data_client=alpaca_data_client,
            end_date=end_date,
            data_feed=data_feed,
        )

    normalized: dict[str, pd.DataFrame] = {}
    for symbol, df in bars_by_symbol.items():
        if not df.empty:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        normalized[symbol] = df
    if include_latest:
        from src.core.config import get_config
        from ..connectors import append_latest_quotes_to_bars, fetch_latest_market_quotes

        runtime_config = config or get_config()
        quotes = fetch_latest_market_quotes(
            symbols,
            runtime_config,
            data_client=alpaca_data_client,
            force_refresh=force_refresh,
        )
        normalized = append_latest_quotes_to_bars(normalized, quotes)
    return normalized
