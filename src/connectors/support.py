"""Shared plumbing every data provider needs.

Extracted from ``service.py`` so a provider can be its own module. Everything here was already
being shared by the fetchers -- HTTP with rate-limit accounting, the cache read/write pair, bar
normalisation, provider enablement -- but living in the same file as the orchestration meant a
provider could not move out without importing the module that dispatches to it.

The split is by direction of dependency, not by topic: providers depend on this, and the
service depends on providers. Nothing here may import a provider or the service.
"""

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
from ..data.bars import TRADING_MINUTES_PER_DAY, calendar_days_for
from ..data.duckdb_store import DAILY_INTERVAL_MINUTES, bar_end_timestamps
from ..data.provider_cache import (
    load_cached_payload,
    provider_is_limited,
    record_provider_limited,
    record_provider_success,
    save_cached_payload,
)
from .utils import filter_bar_range, normalize_intraday_frame

logger = logging.getLogger(__name__)

#: These name *provider configuration sections* and rate-limit namespaces -- which providers to
#: try, in what order, for fine-grained versus daily data. They are no longer cache categories:
#: bars all land in one store keyed by resolution, because intraday and EOD were never two
#: kinds of data, only two grids. See ``src/data/duckdb_store.py``.
MARKET_CATEGORY = "market_data"
INTRADAY_MARKET_CATEGORY = "intraday_market_data"
EOD_MARKET_CATEGORY = "eod_market_data"
NEWS_CATEGORY = "news_sentiment"
SENTIMENT_CATEGORY = "sentiment_data"
INTRADAY_CACHE_TTL_SECONDS = 900
#: Minute frequencies Schwab's pricehistory endpoint accepts. Anything else is a 400.
SCHWAB_MINUTE_FREQUENCIES = frozenset({1, 5, 10, 15, 30})
EOD_CACHE_TTL_SECONDS = 1800
EOD_BAR_FRESH_FOR_DAYS = 3

#: Bar resolutions each provider can actually serve, finest first. yfinance's sub-15m data is
#: capped at about a week of history, so 15 minutes stays its practical floor; Schwab has no
#: such cap, which is what makes a finer default worth taking.
PROVIDER_BAR_MINUTES: dict[str, tuple[int, ...]] = {
    "schwab": tuple(sorted(SCHWAB_MINUTE_FREQUENCIES)),
    "alpaca": (1, 5, 15, 30, 60),
    "finnhub": (1, 5, 15, 30, 60),
    "yfinance": (15, 30, 60),
}

#: Providers that authenticate with an OAuth token rather than an API key in config.
EXTERNAL_AUTH_PROVIDERS = {"alpaca", "schwab"}

_filter_bar_range = filter_bar_range
_normalize_intraday_frame = normalize_intraday_frame


def default_bar_minutes(config: Config) -> int:
    """The configured preferred resolution for fine-grained bars."""
    return max(int(getattr(config, "market_data_bar_minutes", 5) or 5), 1)


def resolve_bar_minutes(provider: str, wanted: int) -> int:
    """The finest grid ``provider`` can serve at or below ``wanted``.

    Coarser rather than an error: horizons are stated in minutes now, so a provider that
    cannot hit the requested grid still answers the question, just with less resolution.
    """
    supported = PROVIDER_BAR_MINUTES.get(provider.lower())
    if not supported:
        return max(int(wanted), 1)
    eligible = [minutes for minutes in supported if minutes <= int(wanted)]
    return max(eligible) if eligible else min(supported)


def bars_for_minutes(lookback_minutes: int, bar_minutes: int) -> int:
    """How many bars of ``bar_minutes`` a window of ``lookback_minutes`` of market time spans.

    Lookbacks count minutes the market was open, so a 4800-minute window is about twelve
    sessions -- and asking a provider for 4800/5 = 960 five-minute bars would fetch nearly
    four times what it needs. One session is 390 minutes however finely it is sliced.
    """
    if lookback_minutes <= 0 or bar_minutes <= 0:
        return 0
    sessions = lookback_minutes / TRADING_MINUTES_PER_DAY
    per_session = max(TRADING_MINUTES_PER_DAY // bar_minutes, 1)
    return max(int(math.ceil(sessions * per_session)) + 1, 1)

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


#: Providers that authenticate with something other than an api_key in the connector config:
#: Alpaca reads its key pair from the environment, Schwab holds an OAuth token. Without this
#: they look unconfigured and get skipped, however high they sit in the provider order.
EXTERNAL_AUTH_PROVIDERS = {"alpaca", "schwab"}


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


def _schwab_token(config: Config, category: str) -> str:
    """Schwab bearer token, refreshed via OAuth when app credentials are configured.

    Schwab access tokens last ~30 minutes, so a statically configured token goes stale almost
    immediately; it is kept only as a fallback for manual testing.

    Gated on the app credentials alone, not on a configured refresh token: consent completed
    through the dashboard stores its refresh token in the state store, which is where
    ``SchwabSession`` looks when the config has none. Requiring SCHWAB_REFRESH_TOKEN here made
    the connector unusable for exactly the flow the dashboard exists to drive.
    """
    if getattr(config, "schwab_app_key", "") and getattr(config, "schwab_app_secret", ""):
        from src.brokerages.schwab_client import SchwabAuthError, SchwabSession

        try:
            return SchwabSession(config).access_token()
        except SchwabAuthError as error:
            # No consent yet, or the refresh token expired: fall through to the ladder rather
            # than taking down the whole fetch.
            logger.info("Schwab OAuth session unavailable, falling back: %s", error)
            return ""
    return _access_token(config, category, "schwab")


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


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def _provider_bars(
    parsed: pd.DataFrame,
    interval_minutes: int,
    *,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Turn a parsed provider payload into bars this project's conventions agree with.

    Every fetcher ends the same way -- stamp, clip to the requested range, keep the newest N --
    so it lives here once rather than in ten near-identical tails.

    The stamping is the part that matters. Providers timestamp a bar at its start; the close
    is the price at its end. Doing that here, at the boundary where a payload becomes bars,
    means the frame a fetcher *returns* and the rows it *stores* carry the same convention.
    Normalising only on the way into the store left those two disagreeing by one interval,
    and a read that blended fresh provider bars with cached ones got both -- the same bar
    twice, an interval apart.
    """
    if parsed.empty:
        return parsed
    work = parsed.copy()
    work["timestamp"] = bar_end_timestamps(work["timestamp"], interval_minutes)
    # Kept only so a provider that supplies its own adjusted series (Alpaca sends
    # ``Adjustment.ALL``) has somewhere to put it, and so the frame's shape never depends on
    # the symbol's dividend policy. Nothing in this module writes it any more: distributions
    # are cash events, recorded in the ``dividends`` table and booked by the ledger, so a
    # cached bar stays whatever the market actually printed.
    if "adjusted_close" not in work:
        work["adjusted_close"] = pd.to_numeric(work["close"], errors="coerce")
    work = filter_bar_range(work, start_date, end_date)
    if limit:
        work = work.tail(limit)
    return work.reset_index(drop=True)


def _fresh_cached_bars(
    bars: pd.DataFrame,
    interval_minutes: int,
    *,
    now: datetime | None = None,
) -> pd.DataFrame:
    """Cached bars if they are recent enough for their own resolution, else nothing.

    One rule for every grid rather than one per category: a bar is stale once several of its
    own intervals have passed. Daily bars get the session-boundary allowance they always had,
    since the next one does not exist until the market closes again.
    """
    if bars.empty or "timestamp" not in bars:
        return pd.DataFrame()
    timestamps = pd.to_datetime(bars["timestamp"], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return pd.DataFrame()
    now_ts = pd.Timestamp(now or datetime.now(timezone.utc))
    now_ts = now_ts.tz_localize("UTC") if now_ts.tzinfo is None else now_ts.tz_convert("UTC")
    latest = timestamps.max()
    if int(interval_minutes) >= DAILY_INTERVAL_MINUTES:
        return bars if now_ts - latest <= pd.Timedelta(days=EOD_BAR_FRESH_FOR_DAYS) else pd.DataFrame()
    if latest.date() == now_ts.date():
        return bars
    max_age = pd.Timedelta(seconds=max(INTRADAY_CACHE_TTL_SECONDS, int(interval_minutes or 15) * 60 * 3))
    return bars if now_ts - latest <= max_age else pd.DataFrame()


def _news_cache_key(symbols: list[str]) -> str:
    return ",".join(sorted({symbol.upper() for symbol in symbols}))


def _read_duckdb_bars(
    provider: str,
    symbol: str,
    interval_minutes: int,
    *,
    limit: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    try:
        from ..data.duckdb_store import read_bars

        bars = read_bars(
            symbol,
            interval_minutes=int(interval_minutes),
            provider=provider,
            limit=limit,
            start=start,
            end=end,
        )
        if bars.empty:
            logger.debug(
                "DuckDB market cache miss provider=%s symbol=%s interval=%sm limit=%s start=%s end=%s",
                provider,
                symbol,
                interval_minutes,
                limit,
                start,
                end,
            )
        return bars
    except Exception as exc:
        logger.warning(
            "DuckDB market cache read failed provider=%s symbol=%s interval=%sm limit=%s start=%s end=%s: %s",
            provider,
            symbol,
            interval_minutes,
            limit,
            start,
            end,
            exc,
        )
    return _empty_bars()


def _write_duckdb_bars(
    provider: str,
    symbol: str,
    interval_minutes: int,
    bars: pd.DataFrame,
    *,
    ttl_seconds: int | None,
) -> None:
    if bars.empty:
        return
    try:
        from ..data.duckdb_store import write_market_bars

        write_market_bars(provider, symbol, int(interval_minutes), bars, ttl_seconds=ttl_seconds)
    except Exception as exc:
        logger.warning(
            "DuckDB market cache write failed provider=%s symbol=%s interval=%sm rows=%s: %s",
            provider,
            symbol,
            interval_minutes,
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
