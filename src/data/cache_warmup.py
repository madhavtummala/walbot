from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from src.connectors.service import EOD_MARKET_CATEGORY, INTRADAY_MARKET_CATEGORY
from src.core.config import get_config, load_algorithms_config
from src.data.duckdb_store import clear_market_bars, market_bars_summary
from src.data.provider_cache import clear_cached_payloads


DEFAULT_BACKTEST_BUFFER_BARS = 10
DEFAULT_INTRADAY_LOOKBACK_BARS = 78
DEFAULT_INTRADAY_BAR_MINUTES = 15
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
    explicit = [symbol.strip().upper() for symbol in (symbols or []) if symbol.strip()]
    if explicit:
        return sorted(set(explicit))
    configured = _algorithm_symbols(algorithm_id)
    if configured:
        return sorted(set(configured))
    return sorted(set(get_config().symbols))


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


def _range_lookback_bars(
    start_date: datetime | None,
    end_date: datetime | None,
    *,
    intraday_bar_minutes: int,
) -> tuple[int | None, int | None]:
    if start_date is None or end_date is None:
        return None, None
    calendar_days = max((end_date.date() - start_date.date()).days, 1)
    eod_bars = calendar_days + 1
    intraday_bars = calendar_days * max(390 // max(intraday_bar_minutes, 1), 1)
    return eod_bars, intraday_bars


def _clear_market_cache(
    symbols: list[str],
    *,
    eod_provider: str,
    intraday_provider: str,
    intraday_bar_minutes: int,
    warm_eod: bool,
    warm_intraday: bool,
) -> dict[str, int]:
    deleted: dict[str, int] = {}
    if warm_eod:
        deleted["eod_duckdb_rows"] = clear_market_bars(
            category=EOD_MARKET_CATEGORY,
            provider=eod_provider,
            symbols=symbols,
            timeframe="1d",
        )
    if warm_intraday:
        deleted["intraday_duckdb_rows"] = clear_market_bars(
            category=INTRADAY_MARKET_CATEGORY,
            provider=intraday_provider,
            symbols=symbols,
            timeframe=f"{int(intraday_bar_minutes)}m",
        )
        prefixes = [f"{symbol.upper()}:{int(intraday_bar_minutes)}:" for symbol in symbols]
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
    intraday_lookback_bars: int | None = None,
    intraday_bar_minutes: int = DEFAULT_INTRADAY_BAR_MINUTES,
    start_date: str | None = None,
    end_date: str | None = None,
    warm_eod: bool = True,
    warm_intraday: bool = True,
    clear: bool = False,
) -> dict[str, Any]:
    from src.connectors import fetch_eod_market_bars, fetch_intraday_market_bars

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
    range_eod_bars, range_intraday_bars = _range_lookback_bars(
        range_start,
        range_end,
        intraday_bar_minutes=intraday_bar_minutes,
    )
    eod_lookback_bars = int(eod_lookback_bars or range_eod_bars or _default_eod_lookback_bars())
    intraday_lookback_bars = int(intraday_lookback_bars or range_intraday_bars or DEFAULT_INTRADAY_LOOKBACK_BARS)
    deleted = (
        _clear_market_cache(
            wanted,
            eod_provider=actual_eod_provider,
            intraday_provider=actual_intraday_provider,
            intraday_bar_minutes=intraday_bar_minutes,
            warm_eod=warm_eod,
            warm_intraday=warm_intraday,
        )
        if clear
        else {}
    )
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
    intraday = (
        fetch_intraday_market_bars(
            wanted,
            config,
            lookback_bars=intraday_lookback_bars,
            bar_minutes=intraday_bar_minutes,
            force_refresh=True,
            provider=actual_intraday_provider,
            start_date=range_start,
            end_date=range_end,
        )
        if warm_intraday
        else {}
    )
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
        "categories": {
            "eod": warm_eod,
            "intraday": warm_intraday,
        },
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
    scope.add_argument("--symbols", nargs="*", help="Symbols to warm. Defaults to configured algorithm universes.")
    scope.add_argument("--algorithm", help="Warm only the symbols configured for one algorithm, such as fast_momentum.")
    parser.add_argument("--provider", default=None, help="Global provider override for both EOD and intraday.")
    parser.add_argument("--eod-provider", help="Provider for EOD data (e.g., alpaca, yfinance).")
    parser.add_argument("--intraday-provider", help="Provider for intraday data (e.g., yfinance).")
    parser.add_argument("--start-date", help="First America/Chicago market date to fetch, in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Last America/Chicago market date to fetch, in YYYY-MM-DD format.")
    parser.add_argument("--eod", action="store_true", help="Warm EOD bars only, unless --intraday is also supplied.")
    parser.add_argument("--intraday", action="store_true", help="Warm intraday bars only, unless --eod is also supplied.")
    parser.add_argument("--eod-lookback-bars", default=None, type=int)
    parser.add_argument("--intraday-lookback-bars", default=DEFAULT_INTRADAY_LOOKBACK_BARS, type=int)
    parser.add_argument("--intraday-bar-minutes", default=DEFAULT_INTRADAY_BAR_MINUTES, type=int)
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
                intraday_lookback_bars=args.intraday_lookback_bars,
                intraday_bar_minutes=args.intraday_bar_minutes,
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
