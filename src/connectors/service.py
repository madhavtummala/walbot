from __future__ import annotations

import json
import logging
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ..core.config import Config
from .utils import filter_bar_range, normalize_intraday_frame
_filter_bar_range = filter_bar_range
_normalize_intraday_frame = normalize_intraday_frame
from ..data.provider_cache import (
    load_cached_payload,
    provider_is_limited,
    record_provider_limited,
    record_provider_success,
    save_cached_payload,
)


logger = logging.getLogger(__name__)
MARKET_CATEGORY = "market_data"
INTRADAY_MARKET_CATEGORY = "intraday_market_data"
EOD_MARKET_CATEGORY = "eod_market_data"
NEWS_CATEGORY = "news_sentiment"
SENTIMENT_CATEGORY = "sentiment_data"
INTRADAY_CACHE_TTL_SECONDS = 900
EOD_CACHE_TTL_SECONDS = 1800
EOD_BAR_FRESH_FOR_DAYS = 3

_CATEGORY_ALIASES = {
    MARKET_CATEGORY: (MARKET_CATEGORY, EOD_MARKET_CATEGORY),
    EOD_MARKET_CATEGORY: (EOD_MARKET_CATEGORY, MARKET_CATEGORY),
    NEWS_CATEGORY: (NEWS_CATEGORY, SENTIMENT_CATEGORY),
    SENTIMENT_CATEGORY: (SENTIMENT_CATEGORY, NEWS_CATEGORY),
    INTRADAY_MARKET_CATEGORY: (INTRADAY_MARKET_CATEGORY, EOD_MARKET_CATEGORY, MARKET_CATEGORY),
}


class ProviderUnavailable(RuntimeError):
    pass


class ProviderRateLimited(ProviderUnavailable):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_config(config: Config, category: str, provider: str) -> dict[str, Any]:
    root = config.data_source_configs if isinstance(config.data_source_configs, dict) else {}
    for candidate_category in _CATEGORY_ALIASES.get(category, (category,)):
        section = root.get(candidate_category, {}) if isinstance(root.get(candidate_category, {}), dict) else {}
        providers = section.get("providers", {}) if isinstance(section.get("providers", {}), dict) else {}
        provider_config = providers.get(provider, {}) if isinstance(providers.get(provider, {}), dict) else {}
        if provider_config or provider in providers:
            return provider_config
    return {}


def _provider_configured(config: Config, category: str, provider: str) -> bool:
    root = config.data_source_configs if isinstance(config.data_source_configs, dict) else {}
    for candidate_category in _CATEGORY_ALIASES.get(category, (category,)):
        section = root.get(candidate_category, {}) if isinstance(root.get(candidate_category, {}), dict) else {}
        providers = section.get("providers", {}) if isinstance(section.get("providers", {}), dict) else {}
        if provider in providers:
            return True
    return False


def _enabled(config: Config, category: str, provider: str, *, uses_external_auth: bool = False) -> bool:
    if not _provider_configured(config, category, provider):
        return False
    provider_config = _provider_config(config, category, provider)
    enabled = provider_config.get("enabled")
    if enabled is not None:
        return bool(enabled)
    return uses_external_auth or bool(_api_key(config, category, provider))


def _next_provider_name(
    providers: list[str],
    current_index: int,
    fetchers: dict[str, Any],
    config: Config,
    category: str,
    external_auth_providers: set[str],
) -> str:
    for candidate in providers[current_index + 1 :]:
        if candidate not in fetchers or provider_is_limited(candidate):
            continue
        if _enabled(config, category, candidate, uses_external_auth=candidate in external_auth_providers):
            return candidate
    return ""


def _fallback_suffix(next_provider: str) -> str:
    if next_provider:
        return f"; falling back to {next_provider}"
    return "; no configured fallback provider remains"


def _api_key(config: Config, category: str, provider: str) -> str:
    provider_config = _provider_config(config, category, provider)
    defaults = {
        "finnhub": "FINNHUB_API_KEY",
        "alpha_vantage": "ALPHA_VANTAGE_API_KEY",
        "marketaux": "MARKETAUX_API_KEY",
        "newsapi": "NEWSAPI_API_KEY",
    }
    env_name = str(provider_config.get("api_key_env") or defaults.get(provider, "")).strip()
    configured_key = str(provider_config.get("api_key") or "").strip()
    return os.getenv(env_name, configured_key).strip() if env_name else configured_key


def _request_json(
    provider: str,
    category: str,
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    query = urllib.parse.urlencode({key: value for key, value in (params or {}).items() if value not in {None, ""}})
    full_url = f"{url}?{query}" if query else url
    request_headers = {"User-Agent": "walbot/1.0", **(headers or {})}
    request = urllib.request.Request(full_url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code in {402, 403, 429}:
            record_provider_limited(category, provider, error or str(exc), retry_after_seconds=3600)
            raise ProviderRateLimited(error or str(exc)) from exc
        raise ProviderUnavailable(error or str(exc)) from exc
    except urllib.error.URLError as exc:
        raise ProviderUnavailable(str(exc)) from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderUnavailable("provider returned non-JSON response") from exc

    if _looks_limited(payload):
        record_provider_limited(category, provider, json.dumps(payload)[:500], retry_after_seconds=3600)
        raise ProviderRateLimited("provider quota appears exhausted")
    record_provider_success(category, provider)
    return payload


def _looks_limited(payload: Any) -> bool:
    text = json.dumps(payload).lower() if isinstance(payload, (dict, list)) else str(payload).lower()
    markers = ("rate limit", "rate-limit", "too many requests", "quota", "api call frequency", "limit reached")
    return any(marker in text for marker in markers)


def _basic_auth_header(username: str, password: str) -> dict[str, str]:
    token = b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}", "Accept": "application/json"}


def _bearer_auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}


def _access_token(config: Config, category: str, provider: str) -> str:
    provider_config = _provider_config(config, category, provider)
    env_name = str(provider_config.get("access_token_env") or provider_config.get("bearer_token_env") or "").strip()
    configured_token = str(provider_config.get("access_token") or provider_config.get("bearer_token") or "").strip()
    return os.getenv(env_name, configured_token).strip() if env_name else configured_token


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _quote_cache_key(symbol: str) -> str:
    return symbol.upper()


def _intraday_cache_key(symbol: str, bar_minutes: int, lookback_bars: int) -> str:
    return f"{symbol.upper()}:{int(bar_minutes)}:{int(lookback_bars)}"


def _timeframe_from_minutes(bar_minutes: int) -> str:
    return f"{int(bar_minutes)}m"


def _fresh_cached_bars(
    bars: pd.DataFrame,
    category: str,
    *,
    bar_minutes: int | None = None,
    now: datetime | None = None,
) -> pd.DataFrame:
    if bars.empty or "timestamp" not in bars:
        return pd.DataFrame()
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return pd.DataFrame()
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    latest = timestamps.max()
    if category == INTRADAY_MARKET_CATEGORY:
        if latest.date() == now_ts.date():
            return bars
        max_age = pd.Timedelta(seconds=max(INTRADAY_CACHE_TTL_SECONDS, int(bar_minutes or 15) * 60 * 3))
        return bars if now_ts - latest <= max_age else pd.DataFrame()
    if category == EOD_MARKET_CATEGORY:
        return bars if now_ts - latest <= pd.Timedelta(days=EOD_BAR_FRESH_FOR_DAYS) else pd.DataFrame()
    return bars


def _news_cache_key(symbols: list[str]) -> str:
    return ",".join(sorted({symbol.upper() for symbol in symbols}))


def _read_duckdb_bars(
    category: str,
    provider: str,
    symbol: str,
    timeframe: str,
    *,
    lookback_bars: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    try:
        from ..data.duckdb_store import read_market_bars

        bars = read_market_bars(
            category,
            provider,
            symbol,
            timeframe,
            lookback_bars=lookback_bars,
            start=start,
            end=end,
        )
        if bars.empty:
            logger.debug(
                "DuckDB market cache miss category=%s provider=%s symbol=%s timeframe=%s lookback=%s start=%s end=%s",
                category,
                provider,
                symbol,
                timeframe,
                lookback_bars,
                start,
                end,
            )
        return bars
    except Exception as exc:
        logger.warning(
            "DuckDB market cache read failed category=%s provider=%s symbol=%s timeframe=%s lookback=%s start=%s end=%s: %s",
            category,
            provider,
            symbol,
            timeframe,
            lookback_bars,
            start,
            end,
            exc,
        )
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def _write_duckdb_bars(
    category: str,
    provider: str,
    symbol: str,
    timeframe: str,
    bars: pd.DataFrame,
    *,
    ttl_seconds: int | None,
) -> None:
    if bars.empty:
        return
    try:
        from ..data.duckdb_store import write_market_bars

        write_market_bars(category, provider, symbol, timeframe, bars, ttl_seconds=ttl_seconds)
    except Exception as exc:
        logger.warning(
            "DuckDB market cache write failed category=%s provider=%s symbol=%s timeframe=%s rows=%s: %s",
            category,
            provider,
            symbol,
            timeframe,
            len(bars),
            exc,
        )


def _read_duckdb_sentiment(provider: str, symbols: list[str]) -> list[dict[str, Any]]:
    try:
        from ..data.duckdb_store import read_sentiment_records

        return read_sentiment_records(provider, symbols)
    except RuntimeError as exc:
        logger.debug("DuckDB sentiment cache unavailable: %s", exc)
    return []


def _write_duckdb_sentiment(provider: str, records: list[dict[str, Any]], *, ttl_seconds: int | None) -> None:
    if not records:
        return
    try:
        from ..data.duckdb_store import write_sentiment_records

        write_sentiment_records(provider, records, ttl_seconds=ttl_seconds)
    except RuntimeError as exc:
        logger.debug("DuckDB sentiment cache unavailable: %s", exc)


def _normalize_quote(provider: str, symbol: str, price: Any, raw: Any, timestamp: Any = None) -> dict[str, Any] | None:
    parsed_price = _finite(price)
    if parsed_price is None or parsed_price <= 0:
        return None
    parsed_timestamp = pd.to_datetime(timestamp, utc=True, errors="coerce") if timestamp is not None else pd.NaT
    if pd.isna(parsed_timestamp):
        parsed_timestamp = pd.Timestamp.now(tz="UTC")
    return {
        "symbol": symbol.upper(),
        "price": parsed_price,
        "timestamp": parsed_timestamp.isoformat(),
        "provider": provider,
        "raw": raw,
    }


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
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
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
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.DataFrame.from_records(rows).sort_values("timestamp").reset_index(drop=True)


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
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return pd.DataFrame.from_records(rows).sort_values("timestamp").reset_index(drop=True)


def _yfinance_period_for(lookback_bars: int, bar_minutes: int) -> str:
    trading_minutes_per_day = 390
    needed_sessions = max(1, math.ceil((bar_minutes * lookback_bars) / trading_minutes_per_day))
    days = min(max(needed_sessions * 3, 5), 59)
    return f"{days}d"


def fetch_yfinance_intraday_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_bars: int,
    bar_minutes: int = 15,
    force_refresh: bool = False,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch recent intraday OHLCV bars from yfinance with a short-lived cache."""
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ProviderUnavailable("yfinance is not installed") from exc

    ttl_seconds = int(getattr(config, "intraday_market_data_cache_ttl_seconds", INTRADAY_CACHE_TTL_SECONDS))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    interval = f"{int(bar_minutes)}m"
    timeframe = _timeframe_from_minutes(bar_minutes)
    period = _yfinance_period_for(lookback_bars, bar_minutes)
    for symbol in [item.upper() for item in symbols]:
        cache_key = _intraday_cache_key(symbol, bar_minutes, lookback_bars)
        if not force_refresh:
            cached = load_cached_payload(INTRADAY_MARKET_CATEGORY, "yfinance", cache_key)
            cached_bars = _records_to_bars(cached)
            if not cached_bars.empty:
                bars_by_symbol[symbol] = cached_bars.tail(lookback_bars).reset_index(drop=True)
                continue
            duckdb_bars = _read_duckdb_bars(
                INTRADAY_MARKET_CATEGORY,
                "yfinance",
                symbol,
                timeframe,
                lookback_bars=lookback_bars,
            )
            duckdb_bars = _fresh_cached_bars(duckdb_bars, INTRADAY_MARKET_CATEGORY, bar_minutes=bar_minutes)
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
            bars = filter_bar_range(
                normalize_intraday_frame(raw),
                start_date,
                end_date,
            ).tail(lookback_bars).reset_index(drop=True)
        except Exception as exc:
            logger.warning("Skipping yfinance intraday bars for %s after provider error: %s", symbol, exc)
            bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if not bars.empty:
            save_cached_payload(
                INTRADAY_MARKET_CATEGORY,
                "yfinance",
                cache_key,
                _bars_to_records(bars),
                ttl_seconds=ttl_seconds,
            )
            _write_duckdb_bars(INTRADAY_MARKET_CATEGORY, "yfinance", symbol, timeframe, bars, ttl_seconds=ttl_seconds)
        bars_by_symbol[symbol] = bars
    return bars_by_symbol


def fetch_intraday_market_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_bars: int,
    bar_minutes: int = 15,
    force_refresh: bool = False,
    provider: str | None = None,
    data_client=None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    logger.info("Fetching intraday bars for %s symbols (lookback=%s, timeframe=%sm)", len(symbols), lookback_bars, bar_minutes)
    range_kwargs = {
        key: value
        for key, value in {"start_date": start_date, "end_date": end_date}.items()
        if value is not None
    }
    providers = [provider.lower()] if provider else [
        item.lower()
        for item in getattr(config, "intraday_market_data_provider_order", [])
    ] or ["yfinance"]
    fetchers = {
        "yfinance": lambda: fetch_yfinance_intraday_bars(
            symbols,
            config,
            lookback_bars=lookback_bars,
            bar_minutes=bar_minutes,
            force_refresh=force_refresh,
            **range_kwargs,
        ),
        "finnhub": lambda: fetch_finnhub_intraday_bars(
            symbols,
            config,
            lookback_bars=lookback_bars,
            bar_minutes=bar_minutes,
            force_refresh=force_refresh,
            **range_kwargs,
        ),
        "alpaca": lambda: fetch_alpaca_intraday_bars(
            symbols,
            config,
            lookback_bars=lookback_bars,
            bar_minutes=bar_minutes,
            force_refresh=force_refresh,
            data_client=data_client,
            **range_kwargs,
        ),
        "schwab": lambda: fetch_schwab_intraday_bars(
            symbols,
            config,
            lookback_bars=lookback_bars,
            bar_minutes=bar_minutes,
            force_refresh=force_refresh,
            **range_kwargs,
        ),
    }
    empty = {symbol.upper(): pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]) for symbol in symbols}
    for index, provider_name in enumerate(providers):
        if provider_name not in fetchers:
            logger.warning("Intraday market data provider %s is not supported; skipping", provider_name)
            continue
        if _provider_configured(config, INTRADAY_MARKET_CATEGORY, provider_name):
            provider_config = _provider_config(config, INTRADAY_MARKET_CATEGORY, provider_name)
            if provider_config.get("enabled") is False:
                continue
        try:
            bars = fetchers[provider_name]()
        except ProviderUnavailable as exc:
            next_provider = next((item for item in providers[index + 1 :] if item in fetchers), "")
            log = logger.info if next_provider else logger.warning
            log("Intraday market data provider %s failed%s: %s", provider_name, _fallback_suffix(next_provider), exc)
            continue
        except Exception as exc:
            next_provider = next((item for item in providers[index + 1 :] if item in fetchers), "")
            log = logger.info if next_provider else logger.warning
            log("Intraday market data provider %s failed%s: %s", provider_name, _fallback_suffix(next_provider), exc)
            continue
        if any(not frame.empty for frame in bars.values()):
            return bars
        next_provider = next((item for item in providers[index + 1 :] if item in fetchers), "")
        logger.info("Intraday market data provider %s returned no bars%s", provider_name, _fallback_suffix(next_provider))
    return empty


def fetch_alpaca_intraday_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_bars: int,
    bar_minutes: int = 30,
    force_refresh: bool = False,
    data_client=None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    from src.brokerages.alpaca_client import create_data_client, get_historical_intraday_bars

    ttl_seconds = int(getattr(config, "intraday_market_data_cache_ttl_seconds", INTRADAY_CACHE_TTL_SECONDS))
    timeframe = _timeframe_from_minutes(bar_minutes)
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in [item.upper() for item in symbols]:
        cached = (
            pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars(INTRADAY_MARKET_CATEGORY, "alpaca", symbol, timeframe, lookback_bars=lookback_bars),
                INTRADAY_MARKET_CATEGORY,
                bar_minutes=bar_minutes,
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
            bar_minutes=bar_minutes,
            data_client=client,
            start_date=start_date,
            end_date=end_date,
            data_feed=config.alpaca_data_feed,
        )
        for symbol, bars in fresh.items():
            normalized = filter_bar_range(
                normalize_intraday_frame(bars),
                start_date,
                end_date,
            ).tail(lookback_bars).reset_index(drop=True)
            _write_duckdb_bars(INTRADAY_MARKET_CATEGORY, "alpaca", symbol, timeframe, normalized, ttl_seconds=ttl_seconds)
            bars_by_symbol[symbol.upper()] = normalized
    return bars_by_symbol


def fetch_schwab_intraday_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_bars: int,
    bar_minutes: int = 15,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    token = _access_token(config, INTRADAY_MARKET_CATEGORY, "schwab")
    if not token:
        raise ProviderUnavailable("Schwab access token is not configured")
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")
    ttl_seconds = int(getattr(config, "intraday_market_data_cache_ttl_seconds", INTRADAY_CACHE_TTL_SECONDS))
    end = end_date or datetime.now(timezone.utc)
    trading_minutes_per_day = 390
    needed_sessions = max(1, math.ceil((bar_minutes * lookback_bars) / trading_minutes_per_day))
    start = start_date or end - timedelta(days=max(7, (needed_sessions * 3) + 2))
    timeframe = _timeframe_from_minutes(bar_minutes)
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in [item.upper() for item in symbols]:
        cached = (
            pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars(INTRADAY_MARKET_CATEGORY, "schwab", symbol, timeframe, lookback_bars=lookback_bars),
                INTRADAY_MARKET_CATEGORY,
                bar_minutes=bar_minutes,
            )
        )
        if not cached.empty:
            bars_by_symbol[symbol] = cached.tail(lookback_bars).reset_index(drop=True)
            continue
        payload = _request_json(
            "schwab",
            INTRADAY_MARKET_CATEGORY,
            "https://api.schwabapi.com/marketdata/v1/pricehistory",
            {
                "symbol": symbol,
                "frequencyType": "minute",
                "frequency": int(bar_minutes),
                "startDate": int(start.timestamp() * 1000),
                "endDate": int(end.timestamp() * 1000),
                "needExtendedHoursData": "false",
                "needPreviousClose": "false",
            },
            headers=_bearer_auth_header(token),
        )
        bars = filter_bar_range(
            _schwab_candles_to_bars(payload),
            start_date,
            end_date,
        ).tail(lookback_bars).reset_index(drop=True)
        _write_duckdb_bars(INTRADAY_MARKET_CATEGORY, "schwab", symbol, timeframe, bars, ttl_seconds=ttl_seconds)
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
            pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars(EOD_MARKET_CATEGORY, "yfinance", symbol, "1d", lookback_bars=lookback_bars),
                EOD_MARKET_CATEGORY,
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
            bars = filter_bar_range(
                normalize_intraday_frame(raw),
                start_date,
                end_date,
            ).tail(lookback_bars).reset_index(drop=True)
        except Exception as exc:
            logger.warning("Skipping yfinance EOD bars for %s after provider error: %s", symbol, exc)
            bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        _write_duckdb_bars(EOD_MARKET_CATEGORY, "yfinance", symbol, "1d", bars, ttl_seconds=ttl_seconds)
        bars_by_symbol[symbol] = bars
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
            pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars(EOD_MARKET_CATEGORY, "alpaca", symbol, "1d", lookback_bars=lookback_bars),
                EOD_MARKET_CATEGORY,
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
            normalized = filter_bar_range(
                normalize_intraday_frame(bars),
                start_date,
                end_date,
            ).tail(lookback_bars).reset_index(drop=True)
            _write_duckdb_bars(EOD_MARKET_CATEGORY, "alpaca", symbol, "1d", normalized, ttl_seconds=ttl_seconds)
            bars_by_symbol[symbol.upper()] = normalized
    return bars_by_symbol


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
            pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars(EOD_MARKET_CATEGORY, "finnhub", symbol, "1d", lookback_bars=lookback_bars),
                EOD_MARKET_CATEGORY,
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
            bars = filter_bar_range(
                _finnhub_candles_to_bars(payload),
                start_date,
                end_date,
            ).tail(lookback_bars).reset_index(drop=True)
        except ProviderUnavailable as exc:
            logger.warning("Skipping Finnhub EOD bars for %s after provider error: %s", symbol, exc)
            bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        _write_duckdb_bars(EOD_MARKET_CATEGORY, "finnhub", symbol, "1d", bars, ttl_seconds=ttl_seconds)
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
    token = _access_token(config, EOD_MARKET_CATEGORY, "schwab")
    if not token:
        raise ProviderUnavailable("Schwab access token is not configured")
    ttl_seconds = int(getattr(config, "eod_market_data_cache_ttl_seconds", EOD_CACHE_TTL_SECONDS))
    end = end_date or datetime.now(timezone.utc)
    start = start_date or end - timedelta(days=max(int(lookback_bars * 2), 30))
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in [item.upper() for item in symbols]:
        cached = (
            pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
            if force_refresh
            else _fresh_cached_bars(
                _read_duckdb_bars(EOD_MARKET_CATEGORY, "schwab", symbol, "1d", lookback_bars=lookback_bars),
                EOD_MARKET_CATEGORY,
            )
        )
        if not cached.empty:
            bars_by_symbol[symbol] = cached.tail(lookback_bars).reset_index(drop=True)
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
        bars = filter_bar_range(
            _schwab_candles_to_bars(payload),
            start_date,
            end_date,
        ).tail(lookback_bars).reset_index(drop=True)
        _write_duckdb_bars(EOD_MARKET_CATEGORY, "schwab", symbol, "1d", bars, ttl_seconds=ttl_seconds)
        bars_by_symbol[symbol] = bars
    return bars_by_symbol


def fetch_eod_market_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_bars: int,
    force_refresh: bool = False,
    provider: str | None = None,
    data_client=None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    logger.info("Fetching EOD bars for %s symbols (lookback=%s)", len(symbols), lookback_bars)
    range_kwargs = {
        key: value
        for key, value in {"start_date": start_date, "end_date": end_date}.items()
        if value is not None
    }
    providers = [provider.lower()] if provider else [
        item.lower()
        for item in getattr(config, "eod_market_data_provider_order", [])
    ] or [item.lower() for item in getattr(config, "market_data_provider_order", [])] or ["alpaca"]
    fetchers = {
        "yfinance": lambda: fetch_yfinance_eod_bars(symbols, config, lookback_bars=lookback_bars, force_refresh=force_refresh, **range_kwargs),
        "alpaca": lambda: fetch_alpaca_eod_bars(
                symbols,
                config,
                lookback_bars=lookback_bars,
                force_refresh=force_refresh,
                data_client=data_client,
                **range_kwargs,
            ),
        "finnhub": lambda: fetch_finnhub_eod_bars(symbols, config, lookback_bars=lookback_bars, force_refresh=force_refresh, **range_kwargs),
        "schwab": lambda: fetch_schwab_eod_bars(symbols, config, lookback_bars=lookback_bars, force_refresh=force_refresh, **range_kwargs),
    }
    empty = {symbol.upper(): pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]) for symbol in symbols}
    for index, provider_name in enumerate(providers):
        if provider_name not in fetchers:
            logger.warning("EOD market data provider %s is not supported; skipping", provider_name)
            continue
        if _provider_configured(config, EOD_MARKET_CATEGORY, provider_name):
            provider_config = _provider_config(config, EOD_MARKET_CATEGORY, provider_name)
            if provider_config.get("enabled") is False:
                continue
        try:
            bars = fetchers[provider_name]()
        except ProviderUnavailable as exc:
            next_provider = next((item for item in providers[index + 1 :] if item in fetchers), "")
            log = logger.info if next_provider else logger.warning
            log("EOD market data provider %s failed%s: %s", provider_name, _fallback_suffix(next_provider), exc)
            continue
        except Exception as exc:
            next_provider = next((item for item in providers[index + 1 :] if item in fetchers), "")
            log = logger.info if next_provider else logger.warning
            log("EOD market data provider %s failed%s: %s", provider_name, _fallback_suffix(next_provider), exc)
            continue
        if any(not frame.empty for frame in bars.values()):
            return bars
        next_provider = next((item for item in providers[index + 1 :] if item in fetchers), "")
        logger.info("EOD market data provider %s returned no bars%s", provider_name, _fallback_suffix(next_provider))
    return empty


def _bars_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records = []
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"])
    for row in work.to_dict(orient="records"):
        records.append(
            {
                "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "adjusted_close": float(row.get("adjusted_close", row["close"])),
            }
        )
    return records


def _records_to_bars(records: Any) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"])
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"])
    df["timestamp"] = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "adjusted_close"):
        if column not in df.columns:
            df[column] = df["close"] if column == "adjusted_close" else 0.0
        df[column] = pd.to_numeric(df.get(column), errors="coerce")
    df["adjusted_close"] = df["adjusted_close"].fillna(df["close"])
    return (
        df.dropna(subset=["timestamp", "open", "high", "low", "close"])
        [["timestamp", "open", "high", "low", "close", "volume", "adjusted_close"]]
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


def fetch_finnhub_intraday_bars(
    symbols: list[str],
    config: Config,
    *,
    lookback_bars: int,
    bar_minutes: int = 30,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch recent intraday OHLCV bars from Finnhub with a short-lived cache."""
    key = _api_key(config, INTRADAY_MARKET_CATEGORY, "finnhub")
    if not key:
        raise ProviderUnavailable("FINNHUB_API_KEY is not configured")
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")

    end = end_date or datetime.now(timezone.utc)
    trading_minutes_per_day = 390
    needed_sessions = max(1, math.ceil((bar_minutes * lookback_bars) / trading_minutes_per_day))
    calendar_days = max(7, (needed_sessions * 3) + 2)
    start = start_date or end - timedelta(days=calendar_days)
    resolution = str(int(bar_minutes))
    timeframe = _timeframe_from_minutes(bar_minutes)
    bars_by_symbol: dict[str, pd.DataFrame] = {}

    for symbol in [item.upper() for item in symbols]:
        cache_key = _intraday_cache_key(symbol, bar_minutes, lookback_bars)
        if not force_refresh:
            cached = load_cached_payload(INTRADAY_MARKET_CATEGORY, "finnhub", cache_key)
            cached_bars = _records_to_bars(cached)
            if not cached_bars.empty:
                bars_by_symbol[symbol] = cached_bars.tail(lookback_bars).reset_index(drop=True)
                continue
            duckdb_bars = _read_duckdb_bars(
                INTRADAY_MARKET_CATEGORY,
                "finnhub",
                symbol,
                timeframe,
                lookback_bars=lookback_bars,
            )
            duckdb_bars = _fresh_cached_bars(duckdb_bars, INTRADAY_MARKET_CATEGORY, bar_minutes=bar_minutes)
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
            bars = filter_bar_range(
                _finnhub_candles_to_bars(payload),
                start_date,
                end_date,
            ).tail(lookback_bars).reset_index(drop=True)
        except ProviderUnavailable as exc:
            logger.warning("Skipping Finnhub intraday bars for %s after provider error: %s", symbol, exc)
            bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if not bars.empty:
            save_cached_payload(
                INTRADAY_MARKET_CATEGORY,
                "finnhub",
                cache_key,
                _bars_to_records(bars),
                ttl_seconds=INTRADAY_CACHE_TTL_SECONDS,
            )
            _write_duckdb_bars(
                INTRADAY_MARKET_CATEGORY,
                "finnhub",
                symbol,
                timeframe,
                bars,
                ttl_seconds=INTRADAY_CACHE_TTL_SECONDS,
            )
        bars_by_symbol[symbol] = bars
    return bars_by_symbol


def _fetch_alpha_vantage_quotes(symbols: list[str], config: Config) -> dict[str, dict[str, Any]]:
    key = _api_key(config, MARKET_CATEGORY, "alpha_vantage")
    if not key:
        raise ProviderUnavailable("ALPHA_VANTAGE_API_KEY is not configured")
    quotes: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        payload = _request_json(
            "alpha_vantage",
            MARKET_CATEGORY,
            "https://www.alphavantage.co/query",
            {"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": key},
        )
        quote_payload = payload.get("Global Quote", {}) if isinstance(payload, dict) else {}
        quote = _normalize_quote(
            "alpha_vantage",
            symbol,
            quote_payload.get("05. price"),
            payload,
            quote_payload.get("07. latest trading day"),
        )
        if quote:
            quotes[symbol.upper()] = quote
    return quotes


MARKET_FETCHERS = {
    "alpaca": _fetch_alpaca_quotes,
    "finnhub": _fetch_finnhub_quotes,
    "alpha_vantage": _fetch_alpha_vantage_quotes,
}


def fetch_latest_market_quotes(
    symbols: list[str],
    config: Config,
    *,
    data_client=None,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    wanted = [symbol.upper() for symbol in symbols]
    quotes: dict[str, dict[str, Any]] = {}
    providers = [item.lower() for item in config.market_data_provider_order]

    from src.brokerages.alpaca_client import create_trading_client, is_market_open

    has_enabled_provider = any(
        provider in MARKET_FETCHERS
        and _enabled(config, MARKET_CATEGORY, provider, uses_external_auth=(provider == "alpaca"))
        for provider in providers
    )

    # Check if we should even try live providers
    provider_fetch_allowed = force_refresh
    if not force_refresh:
        try:
            # Use a cached market status if possible to avoid excessive clock calls
            trading_client = create_trading_client(config)
            provider_fetch_allowed = is_market_open(trading_client)
        except Exception:
            provider_fetch_allowed = True

    if provider_fetch_allowed:
        for index, provider in enumerate(providers):
            if provider not in MARKET_FETCHERS:
                continue
            next_provider = _next_provider_name(providers, index, MARKET_FETCHERS, config, MARKET_CATEGORY, {"alpaca"})
            if provider_is_limited(provider):
                log = logger.info if next_provider else logger.warning
                log(
                    "Market data provider %s is marked rate-limited%s",
                    provider,
                    _fallback_suffix(next_provider),
                )
                continue
            if not _enabled(config, MARKET_CATEGORY, provider, uses_external_auth=(provider == "alpaca")):
                continue

            missing = [symbol for symbol in wanted if symbol not in quotes]
            if not missing:
                break

            if not force_refresh:
                for symbol in list(missing):
                    cached = load_cached_payload(MARKET_CATEGORY, provider, _quote_cache_key(symbol))
                    if cached:
                        quotes[symbol] = {**cached, "cached": True}
                missing = [symbol for symbol in wanted if symbol not in quotes]
                if not missing:
                    break

            try:
                if provider == "alpaca":
                    fresh = MARKET_FETCHERS[provider](missing, config, data_client)
                else:
                    fresh = MARKET_FETCHERS[provider](missing, config)
            except ProviderRateLimited as exc:
                log = logger.info if next_provider else logger.warning
                log(
                    "Market data provider %s hit its rate limit%s: %s",
                    provider,
                    _fallback_suffix(next_provider),
                    exc,
                )
                continue
            except ProviderUnavailable as exc:
                log = logger.info if next_provider else logger.warning
                log(
                    "Market data provider %s failed%s: %s",
                    provider,
                    _fallback_suffix(next_provider),
                    exc,
                )
                continue
            except Exception as exc:
                log = logger.info if next_provider else logger.warning
                log(
                    "Market data provider %s failed%s: %s",
                    provider,
                    _fallback_suffix(next_provider),
                    exc,
                )
                continue

            for symbol, quote in fresh.items():
                quotes[symbol] = quote
                save_cached_payload(
                    MARKET_CATEGORY,
                    provider,
                    _quote_cache_key(symbol),
                    quote,
                    ttl_seconds=config.market_data_cache_ttl_seconds,
                )

    still_missing = [symbol for symbol in wanted if symbol not in quotes]
    if still_missing and has_enabled_provider:
        # Final fallback: Check DuckDB market_bars for the last known price
        from ..data.duckdb_store import read_market_bars

        for symbol in still_missing:
            # Try intraday first for higher frequency, then EOD
            for category, timeframe in [(INTRADAY_MARKET_CATEGORY, "15m"), (EOD_MARKET_CATEGORY, "1d")]:
                try:
                    df = read_market_bars(category, None, symbol, timeframe, lookback_bars=1)
                    if not df.empty:
                        price = float(df["close"].iloc[-1])
                        quotes[symbol] = {
                            "symbol": symbol,
                            "price": price,
                            "timestamp": df["timestamp"].iloc[-1].isoformat(),
                            "provider": "duckdb_cache",
                            "cached": True,
                        }
                        break
                except Exception:
                    continue

    still_missing = [symbol for symbol in wanted if symbol not in quotes]
    if still_missing:
        logger.warning(
            "Market data providers returned quotes for only %s/%s symbols; %s still missing",
            len(quotes),
            len(wanted),
            len(still_missing),
        )
    return quotes


def append_latest_quotes_to_bars(
    bars_by_symbol: dict[str, pd.DataFrame],
    quotes_by_symbol: dict[str, dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    merged: dict[str, pd.DataFrame] = {}
    for symbol, df in bars_by_symbol.items():
        quote = quotes_by_symbol.get(symbol.upper())
        if not quote:
            merged[symbol] = df
            continue
        timestamp = pd.to_datetime(quote["timestamp"], utc=True, errors="coerce")
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.now(tz="UTC")
        price = float(quote["price"])
        quote_row = pd.DataFrame(
            [
                {
                    "timestamp": timestamp,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 0.0,
                }
            ]
        )
        work = pd.concat([df, quote_row], ignore_index=True)
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
        work = work.dropna(subset=["timestamp"]).sort_values("timestamp")
        merged[symbol] = work.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return merged


def _news_record(provider: str, symbol: str, timestamp: Any, title: str, url: str, sentiment: Any, raw: Any) -> dict[str, Any]:
    parsed_timestamp = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if pd.isna(parsed_timestamp):
        parsed_timestamp = pd.Timestamp.now(tz="UTC")
    sentiment_value = _finite(sentiment)
    sentiment_value = 0.0 if sentiment_value is None else max(-1.0, min(1.0, sentiment_value))
    return {
        "timestamp": parsed_timestamp.isoformat(),
        "symbol": symbol.upper(),
        "mentions": 1.0,
        "sentiment": sentiment_value,
        "social_score": sentiment_value,
        "provider": provider,
        "title": title,
        "url": url,
        "raw": raw,
    }


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


NEWS_FETCHERS = {
    "marketaux": _fetch_marketaux_news,
    "newsapi": _fetch_newsapi_news,
    "stocktwits": _fetch_stocktwits_news,
}


def fetch_latest_news_sentiment(
    symbols: list[str],
    config: Config,
    *,
    force_refresh: bool = False,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    wanted = [symbol.upper() for symbol in symbols]
    cache_key = _news_cache_key(wanted)
    providers = [provider.lower()] if provider else (
        [item.lower() for item in getattr(config, "sentiment_data_provider_order", [])]
        or [item.lower() for item in config.news_sentiment_provider_order]
    )
    all_records: list[dict[str, Any]] = []
    for index, provider in enumerate(providers):
        if provider not in NEWS_FETCHERS:
            logger.warning("News/sentiment provider %s is not supported; skipping", provider)
            continue
        next_provider = _next_provider_name(providers, index, NEWS_FETCHERS, config, NEWS_CATEGORY, {"stocktwits"})
        if provider_is_limited(provider):
            log = logger.info if next_provider else logger.warning
            log(
                "News/sentiment provider %s is marked rate-limited%s",
                provider,
                _fallback_suffix(next_provider),
            )
            continue
        if not _enabled(config, NEWS_CATEGORY, provider, uses_external_auth=(provider == "stocktwits")):
            continue
        if not force_refresh:
            cached = load_cached_payload(NEWS_CATEGORY, provider, cache_key)
            if cached:
                all_records.extend({**record, "cached": True} for record in cached)
                continue
            duckdb_records = _read_duckdb_sentiment(provider, wanted)
            if duckdb_records:
                all_records.extend({**record, "cached": True} for record in duckdb_records)
                continue
        try:
            records = NEWS_FETCHERS[provider](wanted, config)
        except ProviderRateLimited as exc:
            log = logger.info if next_provider else logger.warning
            log(
                "News/sentiment provider %s hit its rate limit%s: %s",
                provider,
                _fallback_suffix(next_provider),
                exc,
            )
            continue
        except ProviderUnavailable as exc:
            log = logger.info if next_provider else logger.warning
            log(
                "News/sentiment provider %s failed%s: %s",
                provider,
                _fallback_suffix(next_provider),
                exc,
            )
            continue
        except Exception as exc:
            log = logger.info if next_provider else logger.warning
            log(
                "News/sentiment provider %s failed%s: %s",
                provider,
                _fallback_suffix(next_provider),
                exc,
            )
            continue
        if records:
            save_cached_payload(
                NEWS_CATEGORY,
                provider,
                cache_key,
                records,
                ttl_seconds=config.news_sentiment_cache_ttl_seconds,
            )
            _write_duckdb_sentiment(
                provider,
                records,
                ttl_seconds=getattr(config, "sentiment_data_cache_ttl_seconds", config.news_sentiment_cache_ttl_seconds),
            )
            all_records.extend(records)
            continue
        log = logger.info if next_provider else logger.warning
        log(
            "News/sentiment provider %s returned no records%s",
            provider,
            _fallback_suffix(next_provider),
        )
    return all_records


def fetch_latest_sentiment_data(
    symbols: list[str],
    config: Config,
    *,
    force_refresh: bool = False,
    provider: str | None = None,
) -> list[dict[str, Any]]:
    return fetch_latest_news_sentiment(symbols, config, force_refresh=force_refresh, provider=provider)


def news_records_to_social_frames(records: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_symbol.setdefault(str(record["symbol"]).upper(), []).append(record)
    frames: dict[str, pd.DataFrame] = {}
    for symbol, rows in by_symbol.items():
        frame = pd.DataFrame(rows)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frames[symbol] = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frames


def merge_social_frames(
    base: dict[str, pd.DataFrame],
    fresh: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    merged = {
        symbol: frame.copy()
        for symbol, frame in base.items()
        if isinstance(frame, pd.DataFrame)
    }
    for symbol, frame in fresh.items():
        current = merged.get(symbol, pd.DataFrame())
        combined = pd.concat([current, frame], ignore_index=True)
        if combined.empty:
            merged[symbol] = combined
            continue
        combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
        merged[symbol] = (
            combined.dropna(subset=["timestamp"])
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp", "symbol", "title"], keep="last")
            .reset_index(drop=True)
        )
    return merged
