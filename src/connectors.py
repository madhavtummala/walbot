from __future__ import annotations

import json
import logging
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import Config
from .provider_cache import (
    load_cached_payload,
    provider_is_limited,
    record_provider_limited,
    record_provider_success,
    save_cached_payload,
)


logger = logging.getLogger(__name__)
MARKET_CATEGORY = "market_data"
NEWS_CATEGORY = "news_sentiment"


class ProviderUnavailable(RuntimeError):
    pass


class ProviderRateLimited(ProviderUnavailable):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _provider_config(config: Config, category: str, provider: str) -> dict[str, Any]:
    root = config.data_source_configs if isinstance(config.data_source_configs, dict) else {}
    section = root.get(category, {}) if isinstance(root.get(category, {}), dict) else {}
    providers = section.get("providers", {}) if isinstance(section.get("providers", {}), dict) else {}
    provider_config = providers.get(provider, {}) if isinstance(providers.get(provider, {}), dict) else {}
    return provider_config


def _provider_configured(config: Config, category: str, provider: str) -> bool:
    root = config.data_source_configs if isinstance(config.data_source_configs, dict) else {}
    section = root.get(category, {}) if isinstance(root.get(category, {}), dict) else {}
    providers = section.get("providers", {}) if isinstance(section.get("providers", {}), dict) else {}
    return provider in providers


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
    request_headers = {"User-Agent": "trading-bot/1.0", **(headers or {})}
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


def _finite(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _quote_cache_key(symbol: str) -> str:
    return symbol.upper()


def _news_cache_key(symbols: list[str]) -> str:
    return ",".join(sorted({symbol.upper() for symbol in symbols}))


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
    from .alpaca_client import create_data_client, get_latest_price

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
        still_missing = [symbol for symbol in missing if symbol not in fresh]
        if still_missing:
            log = logger.info if next_provider else logger.warning
            log(
                "Market data provider %s returned quotes for %s/%s requested symbols%s",
                provider,
                len(fresh),
                len(missing),
                _fallback_suffix(next_provider),
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
    username = str(provider_config.get("username") or "").strip()
    password = str(provider_config.get("password") or "").strip()
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
) -> list[dict[str, Any]]:
    wanted = [symbol.upper() for symbol in symbols]
    cache_key = _news_cache_key(wanted)
    providers = [item.lower() for item in config.news_sentiment_provider_order]
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
            all_records.extend(records)
            continue
        log = logger.info if next_provider else logger.warning
        log(
            "News/sentiment provider %s returned no records%s",
            provider,
            _fallback_suffix(next_provider),
        )
    return all_records


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
