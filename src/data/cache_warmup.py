from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from src.connectors.service import INTRADAY_MARKET_CATEGORY, resolve_bar_minutes
from src.core.config import MARKET_DATA_BAR_MINUTES, get_config, load_algorithms_config
from src.data.duckdb_store import DAILY_INTERVAL_MINUTES, clear_market_bars, market_bars_summary
from src.data.provider_cache import clear_cached_payloads

logger = logging.getLogger(__name__)

DEFAULT_BACKTEST_BUFFER_BARS = 10
#: Three trading sessions, the shortest window any algorithm's fast horizons need.
DEFAULT_LOOKBACK_MINUTES = 1170
DEFAULT_BAR_MINUTES = MARKET_DATA_BAR_MINUTES
TRADING_MINUTES_PER_DAY = 390
MARKET_TIMEZONE = ZoneInfo("America/Chicago")
EOD_TIMEZONE = ZoneInfo("America/New_York")


def _algorithm_symbols(algorithm_id: str | None = None) -> list[str]:
    raw = load_algorithms_config()
    algorithms = raw.get("algorithms", raw) if isinstance(raw, dict) else {}
    if algorithm_id:
        if not isinstance(algorithms, dict) or algorithm_id not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_id}")
        algorithms = {algorithm_id: algorithms[algorithm_id]}
    symbols: set[str] = set()
    for config in algorithms.values() if isinstance(algorithms, dict) else []:
        if not isinstance(config, dict):
            continue
        for key in (
            "risk_on_universe",
            "defensive_universe",
            "equity_income_universe",
            "crisis_hedge_universe",
            "symbols",
        ):
            values = config.get(key)
            if isinstance(values, list):
                symbols.update(str(value).strip().upper() for value in values if str(value).strip())
        for key in ("spy_symbol", "benchmark_symbol"):
            value = str(config.get(key) or "").strip().upper()
            if value:
                symbols.add(value)
    return sorted(symbols)


def _wanted_symbols(symbols: list[str] | None, algorithm_id: str | None = None) -> list[str]:
    """Which symbols to warm: explicit list, else one algorithm's universe, else everything.

    "Everything" means the tradable universe, not the union of what the algorithms currently
    hold. That is the whole point of the universe -- its own config comment says it "decides
    what gets priced and cached", so that widening it is an edit rather than a data-gathering
    exercise. Defaulting to the algorithm universes quietly broke that promise: a symbol added
    to the universe was never warmed until some algorithm already traded it, which is backwards.
    """
    explicit = [symbol.strip().upper() for symbol in (symbols or []) if symbol.strip()]
    if explicit:
        return sorted(set(explicit))
    if algorithm_id:
        return sorted(set(_algorithm_symbols(algorithm_id)))
    universe = sorted(set(get_config().symbols))
    return universe or sorted(set(_algorithm_symbols(None)))


def _count_rows(bars_by_symbol: dict[str, Any]) -> dict[str, int]:
    return {symbol: int(0 if bars.empty else len(bars)) for symbol, bars in bars_by_symbol.items()}


def _default_eod_lookback_bars() -> int:
    period = str(getattr(get_config(), "backtest_period", "4m") or "4m").strip().lower()
    match = re.fullmatch(r"([1-9][0-9]*)m", period)
    months = int(match.group(1)) if match else 4
    return max(int(round(months * 22)) + DEFAULT_BACKTEST_BUFFER_BARS, 2)


def _parse_date_range(
    start_date: str | None,
    end_date: str | None,
    *,
    timezone_name: ZoneInfo = MARKET_TIMEZONE,
) -> tuple[datetime | None, datetime | None]:
    try:
        start_day = date.fromisoformat(start_date) if start_date else None
        end_day = date.fromisoformat(end_date) if end_date else None
    except ValueError as exc:
        raise ValueError("Dates must use YYYY-MM-DD format") from exc
    if start_day and end_day and start_day > end_day:
        raise ValueError("start date must be on or before end date")
    start = datetime.combine(start_day, time.min, timezone_name).astimezone(timezone.utc) if start_day else None
    end = (
        datetime.combine(end_day + timedelta(days=1), time.min, timezone_name).astimezone(timezone.utc)
        if end_day
        else None
    )
    return start, end


def _eod_timezone(provider: str) -> ZoneInfo:
    return ZoneInfo("UTC") if provider.lower() in {"yfinance", "finnhub"} else EOD_TIMEZONE


def _range_lookbacks(
    start_date: datetime | None,
    end_date: datetime | None,
) -> tuple[int | None, int | None]:
    """Daily bars and trading minutes spanned by an explicit date range."""
    if start_date is None or end_date is None:
        return None, None
    calendar_days = max((end_date.date() - start_date.date()).days, 1)
    return calendar_days + 1, calendar_days * TRADING_MINUTES_PER_DAY


def _clear_market_cache(
    symbols: list[str],
    *,
    eod_provider: str,
    intraday_provider: str,
    bar_minutes: int,
    warm_eod: bool,
    warm_intraday: bool,
) -> dict[str, int]:
    deleted: dict[str, int] = {}
    if warm_eod:
        deleted["eod_duckdb_rows"] = clear_market_bars(
            provider=eod_provider,
            symbols=symbols,
            interval_minutes=DAILY_INTERVAL_MINUTES,
        )
    if warm_intraday:
        deleted["intraday_duckdb_rows"] = clear_market_bars(
            provider=intraday_provider,
            symbols=symbols,
            interval_minutes=bar_minutes,
        )
        prefixes = [f"{symbol.upper()}:{int(bar_minutes)}:" for symbol in symbols]
        deleted["intraday_payload_rows"] = clear_cached_payloads(
            category=INTRADAY_MARKET_CATEGORY,
            provider=intraday_provider,
            cache_key_prefixes=prefixes,
        )
    return deleted


def warm_market_data_cache(
    symbols: list[str] | None = None,
    *,
    algorithm_id: str | None = None,
    provider: str | None = None,
    eod_provider: str | None = None,
    intraday_provider: str | None = None,
    eod_lookback_bars: int | None = None,
    lookback_minutes: int | None = None,
    bar_minutes: int = DEFAULT_BAR_MINUTES,
    start_date: str | None = None,
    end_date: str | None = None,
    warm_eod: bool = True,
    warm_intraday: bool = True,
    clear: bool = False,
) -> dict[str, Any]:
    from src.connectors import fetch_eod_market_bars, fetch_market_history

    if not warm_eod and not warm_intraday:
        raise ValueError("At least one of EOD or intraday warming must be enabled")
    if symbols and algorithm_id:
        raise ValueError("Use either explicit symbols or an algorithm, not both")
    wanted = _wanted_symbols(symbols, algorithm_id)
    config = get_config()

    actual_eod_provider = (
        eod_provider or provider or (config.eod_market_data_provider_order[0] if config.eod_market_data_provider_order else "alpaca")
    ).lower()
    actual_intraday_provider = (
        intraday_provider or provider or (config.intraday_market_data_provider_order[0] if config.intraday_market_data_provider_order else "yfinance")
    ).lower()

    range_start, range_end = _parse_date_range(start_date, end_date)
    eod_range_start, eod_range_end = _parse_date_range(
        start_date,
        end_date,
        timezone_name=_eod_timezone(actual_eod_provider),
    )
    range_eod_bars, range_minutes = _range_lookbacks(range_start, range_end)
    eod_lookback_bars = int(eod_lookback_bars or range_eod_bars or _default_eod_lookback_bars())
    lookback_minutes = int(lookback_minutes or range_minutes or DEFAULT_LOOKBACK_MINUTES)
    # The grid actually written, so ``--clear`` targets the rows the fetch will replace rather
    # than the ones the caller asked for and the provider could not serve.
    actual_bar_minutes = resolve_bar_minutes(actual_intraday_provider, bar_minutes)

    logger.info("Warming market data cache for %s symbols...", len(wanted))
    if warm_eod:
        logger.info("  Daily: provider=%s, lookback=%s bars", actual_eod_provider, eod_lookback_bars)
    if warm_intraday:
        logger.info("  Intraday: provider=%s, lookback=%s minutes, grid=%sm",
                    actual_intraday_provider, lookback_minutes, actual_bar_minutes)

    deleted = (
        _clear_market_cache(
            wanted,
            eod_provider=actual_eod_provider,
            intraday_provider=actual_intraday_provider,
            bar_minutes=actual_bar_minutes,
            warm_eod=warm_eod,
            warm_intraday=warm_intraday,
        )
        if clear
        else {}
    )
    if deleted:
        logger.info("Cleared existing cache: %s", deleted)

    eod = (
        fetch_eod_market_bars(
            wanted,
            config,
            lookback_bars=eod_lookback_bars,
            force_refresh=True,
            provider=actual_eod_provider,
            start_date=eod_range_start,
            end_date=eod_range_end,
        )
        if warm_eod
        else {}
    )
    if warm_eod:
        total_eod = sum(_count_rows(eod).values())
        logger.info("Fetched %s EOD bars total", total_eod)

    intraday = (
        fetch_market_history(
            wanted,
            config,
            lookback_minutes=lookback_minutes,
            bar_minutes=bar_minutes,
            force_refresh=True,
            provider=actual_intraday_provider,
            start_date=range_start,
            end_date=range_end,
        )
        if warm_intraday
        else {}
    )
    if warm_intraday:
        total_intraday = sum(_count_rows(intraday).values())
        logger.info("Fetched %s intraday bars total", total_intraday)

    logger.info("Market data cache warming complete.")
    return {
        "algorithm": algorithm_id,
        "provider": {
            "eod": actual_eod_provider,
            "intraday": actual_intraday_provider,
        },
        "symbols": wanted,
        "date_range": {
            "start_date": start_date,
            "end_date": end_date,
        },
        "warmed": {
            "daily": warm_eod,
            "intraday": warm_intraday,
        },
        "bar_minutes": actual_bar_minutes,
        "lookback_minutes": lookback_minutes,
        "cleared": deleted,
        "fetched": {
            "eod_rows": _count_rows(eod),
            "intraday_rows": _count_rows(intraday),
        },
        "cache": market_bars_summary(
            symbols=wanted,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear, warm, and verify local market-data cache.")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--symbols", nargs="*", help="Symbols to warm. Defaults to the whole tradable universe.")
    scope.add_argument("--algorithm", help="Warm only the symbols configured for one algorithm, such as rally_rotation.")
    parser.add_argument("--provider", default=None, help="Global provider override for both EOD and intraday.")
    parser.add_argument("--eod-provider", help="Provider for EOD data (e.g., alpaca, yfinance).")
    parser.add_argument("--intraday-provider", help="Provider for intraday data (e.g., yfinance).")
    parser.add_argument("--start-date", help="First America/Chicago market date to fetch, in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Last America/Chicago market date to fetch, in YYYY-MM-DD format.")
    parser.add_argument("--eod", action="store_true", help="Warm daily bars only, unless --intraday is also supplied.")
    parser.add_argument("--intraday", action="store_true", help="Warm intraday bars only, unless --eod is also supplied.")
    parser.add_argument("--eod-lookback-bars", default=None, type=int)
    parser.add_argument(
        "--lookback-minutes",
        default=DEFAULT_LOOKBACK_MINUTES,
        type=int,
        help="Trading minutes of intraday history to warm. 1170 is three sessions.",
    )
    parser.add_argument(
        "--bar-minutes",
        default=DEFAULT_BAR_MINUTES,
        type=int,
        help="Preferred bar resolution. Snapped down to whatever the provider serves.",
    )
    parser.add_argument("--clear", action="store_true", help="Clear matching local cache rows before fetching.")
    args = parser.parse_args()
    warm_eod = args.eod or not args.intraday
    warm_intraday = args.intraday or not args.eod
    print(
        json.dumps(
            warm_market_data_cache(
                args.symbols,
                algorithm_id=args.algorithm,
                provider=args.provider,
                eod_provider=args.eod_provider,
                intraday_provider=args.intraday_provider,
                eod_lookback_bars=args.eod_lookback_bars,
                lookback_minutes=args.lookback_minutes,
                bar_minutes=args.bar_minutes,
                start_date=args.start_date,
                end_date=args.end_date,
                warm_eod=warm_eod,
                warm_intraday=warm_intraday,
                clear=args.clear,
            ),
            indent=2,
            sort_keys=True,
        )
    )



if __name__ == "__main__":
    main()
