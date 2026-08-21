from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from ..core.config import Config
from ..data.provider_cache import (
    load_cached_payload,
    provider_is_limited,
    save_cached_payload,
)
from .utils import filter_bar_range, normalize_intraday_frame

_filter_bar_range = filter_bar_range
_normalize_intraday_frame = normalize_intraday_frame


logger = logging.getLogger(__name__)

# Providers are declared in ``registry`` and imported on first use. ``MARKET_FETCHERS`` and
# ``NEWS_FETCHERS`` keep their names and their dict behaviour because callers and tests reach
# for them directly; what changed is that the implementations now live in one module per
# provider instead of in this file.
from .registry import (  # noqa: E402
    EOD_BAR_REGISTRY,
    INTRADAY_BAR_REGISTRY,
    NEWS_FETCHER_REGISTRY as NEWS_FETCHERS,
    QUOTE_FETCHER_REGISTRY as MARKET_FETCHERS,
)

# Provider plumbing, and the category and resolution constants, live in ``support`` so a
# provider can be its own module without importing the service that dispatches to it.
# Re-exported here because callers across the codebase -- and the tests -- have long imported
# these names from ``connectors.service``.
from .support import (  # noqa: E402,F401
    DAILY_INTERVAL_MINUTES,
    EOD_BAR_FRESH_FOR_DAYS,
    EOD_CACHE_TTL_SECONDS,
    EOD_MARKET_CATEGORY,
    EXTERNAL_AUTH_PROVIDERS,
    INTRADAY_CACHE_TTL_SECONDS,
    INTRADAY_MARKET_CATEGORY,
    MARKET_CATEGORY,
    NEWS_CATEGORY,
    PROVIDER_BAR_MINUTES,
    ProviderRateLimited,
    ProviderUnavailable,
    SCHWAB_MINUTE_FREQUENCIES,
    SENTIMENT_CATEGORY,
    _access_token,
    _api_key,
    _bars_to_records,
    _basic_auth_header,
    _bearer_auth_header,
    _empty_bars,
    _enabled,
    _fallback_suffix,
    _filter_bar_range,
    json_number,
    _fresh_cached_bars,
    _intraday_cache_key,
    _looks_limited,
    _news_cache_key,
    _news_record,
    _next_provider_name,
    _normalize_intraday_frame,
    _normalize_quote,
    _provider_bars,
    _provider_config,
    _provider_configured,
    _quote_cache_key,
    _read_duckdb_bars,
    _read_duckdb_sentiment,
    _records_to_bars,
    _request_json,
    _schwab_token,
    _write_duckdb_bars,
    _write_duckdb_sentiment,
    bars_for_minutes,
    default_bar_minutes,
    resolve_bar_minutes,
)


def _run_provider_fallback(
    symbols: list[str],
    providers: list[str],
    fetchers: dict[str, Any],
    config: Config,
    *,
    category: str,
    label: str,
) -> dict[str, pd.DataFrame]:
    """Walk the provider order, keeping the best result *per symbol*.

    Per symbol rather than per batch: a provider that answered for twelve of fourteen symbols
    used to win the whole request, and the two it had nothing for were simply returned empty
    -- no fallback, no error, and downstream they read as a symbol with no history rather
    than one nobody asked properly. Later providers now fill only what is still missing, and
    the walk stops as soon as every symbol is covered.
    """
    resolved: dict[str, pd.DataFrame] = {}
    wanted = [symbol.upper() for symbol in symbols]

    for index, provider_name in enumerate(providers):
        if provider_name not in fetchers:
            logger.warning("%s market data provider %s is not supported; skipping", label, provider_name)
            continue
        if _provider_configured(config, category, provider_name):
            if _provider_config(config, category, provider_name).get("enabled") is False:
                continue
        next_provider = next((item for item in providers[index + 1 :] if item in fetchers), "")
        try:
            bars = fetchers[provider_name]()
        except Exception as exc:
            log = logger.info if next_provider else logger.warning
            log("%s market data provider %s failed%s: %s", label, provider_name, _fallback_suffix(next_provider), exc)
            continue

        supplied = 0
        for symbol, frame in (bars or {}).items():
            key = str(symbol).upper()
            if key in resolved or frame is None or frame.empty:
                continue
            resolved[key] = frame
            supplied += 1

        missing = [symbol for symbol in wanted if symbol not in resolved]
        if not missing:
            return {symbol: resolved[symbol] for symbol in wanted}
        if supplied:
            logger.info(
                "%s market data provider %s covered %s of %s symbols; %s for %s",
                label, provider_name, len(resolved), len(wanted),
                f"falling back to {next_provider}" if next_provider else "no provider left",
                ", ".join(missing[:8]) + ("..." if len(missing) > 8 else ""),
            )
        else:
            logger.info("%s market data provider %s returned no bars%s", label, provider_name, _fallback_suffix(next_provider))

    if not resolved:
        return {symbol: _empty_bars() for symbol in wanted}
    return {symbol: resolved.get(symbol, _empty_bars()) for symbol in wanted}


def _bind_fetchers(registry, symbols, config, **kwargs):
    """``{provider: callable}`` for every registered provider, each bound to what it accepts.

    Providers do not take identical keyword arguments -- only Alpaca wants a preconstructed
    ``data_client``, and a third-party provider added downstream will want neither that nor
    anything else this module knows about. Rather than maintaining a hand-written lambda per
    provider (which is what made adding one an edit in three places), each fetcher is inspected
    and handed exactly the arguments it declares. A fetcher with ``**kwargs`` gets everything.
    """
    import inspect

    def bind(name):
        def call():
            fetcher = registry[name]
            try:
                parameters = inspect.signature(fetcher).parameters
            except (TypeError, ValueError):  # builtins and C callables have no signature
                return fetcher(symbols, config, **kwargs)
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                accepted = kwargs
            else:
                accepted = {k: v for k, v in kwargs.items() if k in parameters}
            return fetcher(symbols, config, **accepted)

        return call

    return {name: bind(name) for name in registry}


def fetch_market_history(
    symbols: list[str],
    config: Config,
    *,
    lookback_minutes: int,
    bar_minutes: int | None = None,
    force_refresh: bool = False,
    provider: str | None = None,
    data_client=None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict[str, pd.DataFrame]:
    """Timestamped bars covering the trailing ``lookback_minutes``, at the best grid available.

    The single entry point for "what has this symbol done recently", replacing the old split
    between an intraday fetch and an EOD one. Providers are tried in order for fine-grained
    bars; whatever any of them supply is written to the shared store, and the store is then
    asked for the window -- so a lookback longer than the fine bars reach is completed with
    the coarser daily series already cached rather than coming back short.
    """
    requested_minutes = bar_minutes or default_bar_minutes(config)
    logger.info(
        "Fetching %s minutes of history for %s symbols (preferred grid %sm)",
        lookback_minutes,
        len(symbols),
        requested_minutes,
    )
    range_kwargs = {
        key: value
        for key, value in {"start_date": start_date, "end_date": end_date}.items()
        if value is not None
    }
    providers = [provider.lower()] if provider else [
        item.lower()
        for item in getattr(config, "intraday_market_data_provider_order", [])
    ] or ["yfinance"]
    fetchers = _bind_fetchers(
        INTRADAY_BAR_REGISTRY,
        symbols,
        config,
        lookback_minutes=lookback_minutes,
        bar_minutes=requested_minutes,
        force_refresh=force_refresh,
        data_client=data_client,
        **range_kwargs,
    )
    fine = _run_provider_fallback(
        symbols, providers, fetchers, config, category=INTRADAY_MARKET_CATEGORY, label="History"
    )
    if start_date is not None or end_date is not None:
        # An explicit range is a cache-warming request for one grid, not a signal window.
        return fine
    return _extend_with_cached_history(fine, lookback_minutes, requested_minutes)


def _extend_with_cached_history(
    fine_bars: dict[str, pd.DataFrame],
    lookback_minutes: int,
    requested_minutes: int,
) -> dict[str, pd.DataFrame]:
    """Back-fill each symbol's window from coarser cached bars where the fine ones run out.

    This is what makes a minute-stated horizon honest across the board: a 4800-minute lookback
    is roughly twelve sessions, which the intraday cache reaches once it has been running, but
    a fresh deployment or a long-horizon knob would otherwise score every symbol flat. Daily
    bars answer the far end of the window perfectly well for a return measured in minutes.
    """
    from ..data.bars import coverage_minutes
    from ..data.bars import read_history

    end = datetime.now(timezone.utc)
    extended: dict[str, pd.DataFrame] = {}
    for symbol, bars in fine_bars.items():
        if not bars.empty and coverage_minutes(bars) >= lookback_minutes:
            extended[symbol] = bars
            continue
        try:
            blended = read_history(symbol, lookback_minutes=lookback_minutes, end=end)
        except Exception as exc:
            logger.warning("Cached history read failed for %s; using fine bars alone: %s", symbol, exc)
            extended[symbol] = bars
            continue
        if blended.empty:
            extended[symbol] = bars
            continue
        if not bars.empty:
            work = bars.copy()
            if "interval_minutes" not in work:
                work["interval_minutes"] = int(requested_minutes)
            blended = pd.concat([blended, work], ignore_index=True)
        extended[symbol] = (
            blended.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
            .reset_index(drop=True)
        )
    return extended


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
    # Built from the registry, not from a hand-written list: a provider registered at runtime
    # is dispatchable immediately, which is the whole point of the registry existing.
    fetchers = _bind_fetchers(
        EOD_BAR_REGISTRY,
        symbols,
        config,
        lookback_bars=lookback_bars,
        force_refresh=force_refresh,
        data_client=data_client,
        **range_kwargs,
    )
    return _run_provider_fallback(
        symbols, providers, fetchers, config, category=EOD_MARKET_CATEGORY, label="EOD"
    )


def load_current_prices(
    symbols: list[str],
    config: Config,
    *,
    data_client=None,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fresh quotes from the enabled providers, in configured order.

    The live leg of :func:`load_latest_prices` -- no market-hours gating, no bar-store
    fallback. Each quote is written through to the payload cache as it is fetched, so a
    later off-hours read has something to degrade to.
    """
    wanted = [symbol.upper() for symbol in symbols]
    quotes: dict[str, dict[str, Any]] = {}
    providers = [item.lower() for item in config.market_data_provider_order]

    has_enabled_provider = any(
        provider in MARKET_FETCHERS
        and _enabled(config, MARKET_CATEGORY, provider, uses_external_auth=(provider in EXTERNAL_AUTH_PROVIDERS))
        for provider in providers
    )

    for index, provider in enumerate(providers):
        if provider not in MARKET_FETCHERS:
            continue
        next_provider = _next_provider_name(providers, index, MARKET_FETCHERS, config, MARKET_CATEGORY, EXTERNAL_AUTH_PROVIDERS)
        if provider_is_limited(provider):
            log = logger.info if next_provider else logger.warning
            log(
                "Market data provider %s is marked rate-limited%s",
                provider,
                _fallback_suffix(next_provider),
            )
            continue
        if not _enabled(config, MARKET_CATEGORY, provider, uses_external_auth=(provider in EXTERNAL_AUTH_PROVIDERS)):
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
                "Market data provider %s unavailable%s: %s",
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
    if still_missing and not has_enabled_provider:
        logger.warning("No market data provider is enabled; quotes must come from stored bars")
    return quotes


def prices_from_store(symbols: list[str]) -> dict[str, dict[str, Any]]:
    """Closest stored price per symbol, finest cached interval first.

    The last resort of :func:`load_latest_prices`, and the only price source for
    point-in-time (``as_of``) contexts, where a live call would describe the wrong
    moment. A recent minute bar stands in for a quote better than yesterday's daily
    close -- but either beats having no price at all.
    """
    from ..data.duckdb_store import available_intervals, read_bars

    prices: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        # Best-effort by construction: a store that is locked or missing must leave the
        # symbol unpriced rather than raise. The caller drops unpriced symbols; letting
        # this throw would fail the whole run.
        try:
            intervals = available_intervals(symbol)
        except Exception as exc:
            logger.debug("Stored-price fallback unavailable for %s: %s", symbol, exc)
            continue
        for interval in intervals:
            try:
                df = read_bars(symbol, interval_minutes=interval, limit=1)
            except Exception:
                continue
            if df.empty:
                continue
            prices[symbol] = {
                "symbol": symbol,
                "price": float(df["close"].iloc[-1]),
                "timestamp": df["timestamp"].iloc[-1].isoformat(),
                "provider": "duckdb_cache",
                "interval_minutes": int(interval),
                "cached": True,
                "current": False,
            }
            break
    return prices


def load_latest_prices(
    symbols: list[str],
    config: Config,
    *,
    data_client=None,
    force_refresh: bool = False,
) -> dict[str, dict[str, Any]]:
    """The one price entry point.

    Within market hours (or on ``force_refresh``) quotes come from
    :func:`load_current_prices`; every quote it returns is a live print and is marked
    ``current``. Off hours, or for any symbol the providers could not price, the chain
    degrades -- payload cache, then :func:`prices_from_store` -- and those quotes are
    marked ``current: False`` so the dashboard can say so rather than pass a stale
    number off as live.
    """
    wanted = [symbol.upper() for symbol in symbols]

    provider_fetch_allowed = force_refresh
    if not force_refresh:
        try:
            # Use a cached market status if possible to avoid excessive clock calls
            from src.brokerages.alpaca_client import create_trading_client, is_market_open

            trading_client = create_trading_client(config)
            provider_fetch_allowed = is_market_open(trading_client)
        except Exception:
            provider_fetch_allowed = True

    quotes: dict[str, dict[str, Any]] = {}
    if provider_fetch_allowed:
        quotes = load_current_prices(wanted, config, data_client=data_client, force_refresh=force_refresh)
        for quote in quotes.values():
            quote["current"] = True

    still_missing = [symbol for symbol in wanted if symbol not in quotes]
    if still_missing:
        quotes.update(prices_from_store(still_missing))

    still_missing = [symbol for symbol in wanted if symbol not in quotes]
    if still_missing:
        logger.warning(
            "Price sources returned quotes for only %s/%s symbols; %s still missing",
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
