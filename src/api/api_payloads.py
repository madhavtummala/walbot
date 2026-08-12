from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ..connectors.sentiment.alpha_vantage import collect_alpha_vantage_news, write_social_trends_csv
from ..brokerages.alpaca_client import create_data_client, create_trading_client, get_account_equity, get_positions
from ..brokerages.schwab_auth import auth_status, begin_authorization, complete_authorization
from ..data import fetch_daily_bars
from ..execution.backtest import calculate_performance_metrics
from ..execution.replay import replay
from ..core.bot_runtime import bot_runtime
from ..core.config import DEFAULT_STRATEGY_ID, get_config, save_universe_symbols
from ..core.interfaces import Schedule
from ..algorithms.registry import canonical_algorithm_id, get_algorithm_class
from ..connectors import fetch_latest_news_sentiment
from src.api.controls import load_controls, save_controls
from ..algorithms.dca import DCA_ALGORITHMS, allocation_preview, load_dca_plan, save_dca_plan
from ..algorithms.dca.accrual import HOURS_IN_MONTH, min_executable
from ..data.social import load_social_trends_csv
from ..data.duckdb_store import read_market_bars
from ..data.state_store import load_state, save_state
from ..core.strategy_models import STRATEGY_LABELS, prepared_strategy_frame
from ..algorithms.fast_momentum import (
    DefensiveMomentumConfig,
    compute_composite_scores,
    compute_price_features,
    decide_target_weights,
)
from ..algorithms.invest_spy import (
    InvestSpyConfig,
    classify_spy_state,
    compute_invest_spy_price_features,
    decide_invest_spy_weights,
)
from ..core.orders import plan_position_orders
from ..data.universe import load_tradable_names, resolve_project_path
from ..data.universe_selector import candidate_specs_by_symbol, preferred_symbols, recommend_universe_rows

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_CACHE_PATH = "data/backtest_cache.json"
BACKTEST_CACHE_STATE_KEY = "backtest_cache"
BACKTEST_CACHE_VERSION = 9
BACKTEST_STARTING_EQUITY = 10_000.0


def _backtest_starting_equity() -> float:
    value = float(get_config().backtest_starting_equity or BACKTEST_STARTING_EQUITY)
    return max(value, 1.0)


def _redact(value: str) -> str:
    config = get_config()
    secrets = [
        config.alpaca_api_key,
        config.alpaca_api_secret,
        config.alpha_vantage_api_key,
    ]
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    redacted = re.sub(r"(apikey=)[^&\s]+", r"\1[redacted]", redacted, flags=re.IGNORECASE)
    return redacted


def _safe_error(error: Exception) -> dict[str, str]:
    return {"error": _redact(str(error))}


def _file_info(path: str) -> dict[str, Any]:
    if not path:
        return {"path": path, "exists": False}
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return {"path": path, "exists": False}
    return {
        "path": path,
        "exists": True,
        "updated_at": pd.Timestamp.fromtimestamp(resolved.stat().st_mtime, tz="UTC").isoformat(),
        "size_bytes": resolved.stat().st_size,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def status_payload() -> dict[str, Any]:
    controls = load_controls()
    strategy = str(controls.get("active_strategy") or DEFAULT_STRATEGY_ID)
    config = get_config(strategy_id=strategy)
    social_info = _file_info(config.social_trends_csv)
    social_rows = 0
    social_symbols: set[str] = set()
    latest_social_timestamp = None

    if social_info["exists"]:
        try:
            df = pd.read_csv(resolve_project_path(config.social_trends_csv))
            if not df.empty:
                social_rows = len(df)
                if "symbol" in df:
                    social_symbols = set(df["symbol"].dropna().astype(str).str.upper())
                if "timestamp" in df:
                    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
                    if not timestamps.empty:
                        latest_social_timestamp = timestamps.max().isoformat()
        except Exception as exc:  # pragma: no cover - status should stay available.
            social_info["error"] = _redact(str(exc))

    config_summary = asdict(config)
    for secret_field in list(config_summary):
        if secret_field.endswith("_api_key") or secret_field.endswith("_api_secret"):
            config_summary[secret_field] = bool(config_summary.get(secret_field))

    return {
        "mode": "READ_ONLY" if config.kill_switch else "ORDER_ENABLED",
        "kill_switch": config.kill_switch,
        "trading_endpoint": config.alpaca_base_url,
        "trading_account": {"id": config.account_id, "label": config.account_label},
        "bot": bot_runtime.snapshot(),
        "config": config_summary,
        "universe": {
            "count": len(config.symbols),
            "symbols": config.symbols,
            "master_list": config.tradables_csv,
        },
        "social": {
            **social_info,
            "rows": social_rows,
            "symbol_count": len(social_symbols),
            "latest_timestamp": latest_social_timestamp,
        },
        "risk": {
            "max_weight_per_symbol": config.max_weight_per_symbol,
            "max_portfolio_exposure": config.max_portfolio_exposure,
            "max_longs": config.max_longs,
            "target_annual_vol": config.target_annual_vol,
            "cash_buffer": config.cash_buffer,
        },
    }


def universe_payload() -> dict[str, Any]:
    config = get_config()
    tradable_names = load_tradable_names(config.tradables_csv)
    tradables = set(tradable_names) if tradable_names else set(config.symbols)
    specs = candidate_specs_by_symbol(tradables)

    rows: list[dict[str, Any]] = []
    for symbol in config.symbols:
        spec = specs.get(symbol)
        rows.append(
            {
                "symbol": symbol,
                "name": tradable_names.get(symbol, symbol),
                "bucket": spec.bucket if spec else "",
                "tradable": symbol in tradables,
                "enabled": symbol in config.symbols,
            }
        )

    return {"rows": rows, "count": len(rows)}


def recommend_universe_payload(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    config = get_config()
    max_symbols = min(max(int(body.get("max_symbols") or 12), 3), 24)
    force_refresh = bool(body.get("refresh", True))
    tradable_names = load_tradable_names(config.tradables_csv)
    candidates = preferred_symbols(set(tradable_names))
    if not candidates:
        raise RuntimeError("No preferred universe candidates were present in the tradables CSV.")

    data_client = create_data_client(config)
    bars_by_symbol = fetch_daily_bars(
        candidates,
        lookback_days=320,
        ma_days=252,
        extra_buffer_days=60,
        alpaca_data_client=data_client,
        data_feed=config.alpaca_data_feed,
        force_refresh=force_refresh,
    )
    recommendation = recommend_universe_rows(
        tradable_names=tradable_names,
        bars_by_symbol=bars_by_symbol,
        max_symbols=max_symbols,
    )
    return {
        **recommendation,
        "max_symbols": max_symbols,
        "data_feed": config.alpaca_data_feed,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "current": universe_payload()["rows"],
    }


def apply_universe_payload(body: dict[str, Any]) -> dict[str, Any]:
    config = get_config()
    tradable_names = load_tradable_names(config.tradables_csv)
    specs = candidate_specs_by_symbol(set(tradable_names))
    validate_against_master = bool(tradable_names)
    raw_rows = body.get("rows") or []
    raw_symbols = body.get("symbols") or []
    if isinstance(raw_symbols, str):
        raw_symbols = [symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()]

    proposed_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    if raw_rows:
        for row in raw_rows:
            symbol = str(row.get("symbol", row.get("Ticker", ""))).strip().upper()
            if not symbol or symbol in seen:
                continue
            if validate_against_master and symbol not in tradable_names:
                raise ValueError(f"{symbol} is not present in the tradables CSV.")
            spec = specs.get(symbol)
            proposed_rows.append(
                {
                    "symbol": symbol,
                    "name": tradable_names.get(symbol) or str(row.get("name") or symbol),
                    "bucket": str(row.get("bucket") or (spec.bucket if spec else "")).strip(),
                }
            )
            seen.add(symbol)
    else:
        for symbol in [str(item).strip().upper() for item in raw_symbols]:
            if not symbol or symbol in seen:
                continue
            if validate_against_master and symbol not in tradable_names:
                raise ValueError(f"{symbol} is not present in the tradables CSV.")
            spec = specs.get(symbol)
            proposed_rows.append(
                {
                    "symbol": symbol,
                    "name": tradable_names.get(symbol) or symbol,
                    "bucket": spec.bucket if spec else "",
                }
            )
            seen.add(symbol)

    if len(proposed_rows) < 3:
        raise ValueError("Universe must include at least 3 tradable symbols.")
    if len(proposed_rows) > 24:
        raise ValueError("Universe must include 24 symbols or fewer.")

    config_path = save_universe_symbols([row["symbol"] for row in proposed_rows])
    return {
        "saved": True,
        "path": _display_path(config_path),
        "universe": universe_payload(),
    }


def dca_payload() -> dict[str, Any]:
    universe = universe_payload()["rows"]
    plan = load_dca_plan(universe)
    available = [row for row in universe if row["enabled"]]
    return {
        "plan": plan,
        "available": available,
        "preview": allocation_preview(plan),
    }


def save_dca_payload(body: dict[str, Any]) -> dict[str, Any]:
    universe = universe_payload()["rows"]
    plan = save_dca_plan(body.get("plan", body), universe)
    available = [row for row in universe if row["enabled"]]
    return {
        "plan": plan,
        "available": available,
        "preview": allocation_preview(plan),
    }


def controls_payload() -> dict[str, Any]:
    config = get_config()
    controls = load_controls()
    if not controls.get("trading_account_id"):
        controls["trading_account_id"] = config.account_id
    return {
        "controls": controls,
        "accounts": config.account_options,
        "bot": bot_runtime.snapshot(),
    }


def schwab_auth_payload() -> dict[str, Any]:
    return auth_status(get_config())


def start_schwab_auth_payload() -> dict[str, Any]:
    return begin_authorization(get_config())


def complete_schwab_auth_payload(code: str, state: str) -> dict[str, Any]:
    complete_authorization(get_config(), code=code, returned_state=state)
    return schwab_auth_payload()


def save_controls_payload(body: dict[str, Any]) -> dict[str, Any]:
    raw_controls = body.get("controls", body)
    if str(raw_controls.get("options_strategy") or "none") == "none":
        raw_controls = {**raw_controls, "options_trading_enabled": False}
    controls = save_controls(raw_controls)
    return {
        "controls": controls,
        "accounts": get_config(account_id=str(controls.get("trading_account_id") or "") or None).account_options,
        "bot": bot_runtime.snapshot(),
    }


def social_payload(limit: int = 250) -> dict[str, Any]:
    config = get_config()
    social_by_symbol = load_social_trends_csv(config.social_trends_csv, config.symbols)
    summary: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for symbol, df in social_by_symbol.items():
        if df.empty:
            continue
        work_df = df.copy()
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)
        latest = work_df.sort_values("timestamp").iloc[-1]
        summary.append(
            {
                "symbol": symbol,
                "latest_timestamp": latest["timestamp"].isoformat(),
                "mentions": float(latest.get("mentions", 0.0)),
                "sentiment": float(latest.get("sentiment", 0.0)),
                "social_score": float(latest.get("social_score", 0.0)),
            }
        )
        for _, row in work_df.tail(limit).iterrows():
            rows.append(
                {
                    "timestamp": row["timestamp"].isoformat(),
                    "symbol": symbol,
                    "mentions": float(row.get("mentions", 0.0)),
                    "sentiment": float(row.get("sentiment", 0.0)),
                    "social_score": float(row.get("social_score", 0.0)),
                }
            )

    rows.sort(key=lambda item: (item["timestamp"], item["symbol"]), reverse=True)
    summary.sort(key=lambda item: item["symbol"])
    return {"summary": summary, "rows": rows[:limit]}


def _json_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed) or parsed in {float("inf"), float("-inf")}:
        return None
    return round(parsed, 2)


def strategy_signals_payload(strategy: str = DEFAULT_STRATEGY_ID) -> dict[str, Any]:
    """Dashboard signal payload: dispatch to the algorithm's own ``signal_view``.

    ``none`` used to fall back to the DCA plan's view, because DCA was not selectable in the
    deck and would otherwise have had nowhere to render. DCA is an ordinary algorithm now, so
    a saved ``none`` simply resolves to it.
    """
    strategy = canonical_algorithm_id(strategy or DEFAULT_STRATEGY_ID)[:80]
    config = get_config(strategy_id=strategy)
    algorithm = get_algorithm_class(strategy).from_config(config)
    view = algorithm.signal_view(config)
    return {
        "strategy": strategy,
        "wired": view.wired,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "summary": view.summary,
        "leaders": view.leaders,
    }


def refresh_social_payload(body: dict[str, Any]) -> dict[str, Any]:
    config = get_config()
    symbols = body.get("symbols") or config.symbols
    if isinstance(symbols, str):
        symbols = [symbol.strip().upper() for symbol in symbols.split(",") if symbol.strip()]

    limit = int(body.get("limit") or config.alpha_vantage_news_limit)
    max_symbols = int(body.get("max_symbols") or min(config.alpha_vantage_max_symbols, len(symbols)))
    lookback_days = int(body.get("lookback_days") or config.alpha_vantage_news_lookback_days)
    delay = float(body.get("delay") if body.get("delay") is not None else config.alpha_vantage_request_delay_seconds)
    output = str(body.get("output") or config.alpha_vantage_news_csv)

    df = collect_alpha_vantage_news(
        config.alpha_vantage_api_key,
        symbols,
        lookback_days=lookback_days,
        limit=limit,
        max_symbols=max_symbols,
        request_delay_seconds=delay,
    )
    csv_path = write_social_trends_csv(df, output)
    return {
        "rows": len(df),
        "symbols_requested": min(len(symbols), max_symbols),
        "output": _display_path(csv_path),
    }


def _period_start(period: str) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    months = _period_months(period)
    if months:
        return now - pd.DateOffset(months=months)
    normalized = period.lower()
    if normalized == "ytd":
        return pd.Timestamp(now.year, 1, 1, tz="UTC")
    return now - pd.DateOffset(months=_period_months(_default_backtest_period()) or 4)


def _period_label(period: str) -> str:
    months = _period_months(period)
    if months:
        return f"{months}M"
    normalized = period.lower()
    if normalized == "ytd":
        return "YTD"
    return period.upper()


def _period_row_count(period: str) -> int:
    months = _period_months(period)
    if months:
        return max(int(round(months * 22)), 2)
    return max(int(round((_period_months(_default_backtest_period()) or 4) * 22)), 2)


def _period_months(period: str) -> int | None:
    normalized = str(period or "").strip().lower()
    aliases = {
        "2mo": 2,
        "2month": 2,
        "2months": 2,
        "4mo": 4,
        "4month": 4,
        "4months": 4,
        "6mo": 6,
        "6month": 6,
        "6months": 6,
    }
    if normalized in aliases:
        return aliases[normalized]
    match = re.fullmatch(r"([1-9][0-9]*)m", normalized)
    if match:
        return int(match.group(1))
    return None


def _default_backtest_period() -> str:
    return str(get_config().backtest_period or "4m").strip().lower() or "4m"


def _load_backtest_cache(path: str = BACKTEST_CACHE_PATH) -> dict[str, Any]:
    if path == BACKTEST_CACHE_PATH:
        cache = load_state(
            BACKTEST_CACHE_STATE_KEY,
            {"version": BACKTEST_CACHE_VERSION, "items": {}},
        )
        if cache.get("version") == BACKTEST_CACHE_VERSION and isinstance(cache.get("items"), dict):
            return cache
        return {"version": BACKTEST_CACHE_VERSION, "items": {}}

    cache_path = resolve_project_path(path)
    if not cache_path.exists():
        return {"version": BACKTEST_CACHE_VERSION, "items": {}}
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": BACKTEST_CACHE_VERSION, "items": {}}
    if cache.get("version") != BACKTEST_CACHE_VERSION or not isinstance(cache.get("items"), dict):
        return {"version": BACKTEST_CACHE_VERSION, "items": {}}
    return cache


def _save_backtest_cache(cache: dict[str, Any], path: str = BACKTEST_CACHE_PATH) -> None:
    if path == BACKTEST_CACHE_PATH:
        save_state(BACKTEST_CACHE_STATE_KEY, cache)
        return

    cache_path = resolve_project_path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


def _cache_key(strategy: str, period: str, dca_plan: dict[str, Any]) -> str:
    config = get_config(strategy_id=strategy)
    selected_algorithm_config = (
        config.algorithm_configs.get(strategy, {}) if isinstance(config.algorithm_configs, dict) else {}
    )
    algorithm_config = {
        "symbols": config.symbols,
        "selected_algorithm_config": selected_algorithm_config,
        "momentum_lookback_days": config.momentum_lookback_days,
        "short_momentum_lookback_days": config.short_momentum_lookback_days,
        "long_ma_days": config.long_ma_days,
        "volume_lookback_days": config.volume_lookback_days,
        "social_lookback_days": config.social_lookback_days,
        "price_momentum_weight": config.price_momentum_weight,
        "social_momentum_weight": config.social_momentum_weight,
        "volume_momentum_weight": config.volume_momentum_weight,
        "min_composite_score": config.min_composite_score,
        "max_weight_per_symbol": config.max_weight_per_symbol,
        "max_portfolio_exposure": config.max_portfolio_exposure,
        "max_longs": config.max_longs,
        "target_annual_vol": config.target_annual_vol,
        "cash_buffer": config.cash_buffer,
        "transaction_cost_bps": config.transaction_cost_bps,
        "starting_equity": _backtest_starting_equity(),
        "cash_account_only": True,
    }
    cache_basis = {
        "strategy": strategy,
        "period": period,
        "data_feed": config.alpaca_data_feed,
        "algorithm_config": algorithm_config,
        "dca_plan": dca_plan if strategy in DCA_ALGORITHMS else {},
    }
    encoded = json.dumps(cache_basis, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def _json_backtest_rows(history_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows_df = history_df.reset_index().copy()
    for column in rows_df.columns:
        if pd.api.types.is_datetime64_any_dtype(rows_df[column]):
            rows_df[column] = pd.to_datetime(rows_df[column], utc=True).dt.strftime("%Y-%m-%d")
    return json.loads(rows_df.to_json(orient="records"))


def _backtest_order_summary(history_df: pd.DataFrame) -> dict[str, Any]:
    turnover = pd.to_numeric(history_df.get("turnover", pd.Series(dtype=float)), errors="coerce").fillna(0.0).abs()
    order_counts = pd.to_numeric(history_df.get("order_count", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    planned_values = pd.to_numeric(
        history_df.get("planned_order_value", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    skipped_values = pd.to_numeric(
        history_df.get("skipped_order_value", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    scheduled_counts = pd.to_numeric(
        history_df.get("scheduled_order_count", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    skipped_counts = pd.to_numeric(
        history_df.get("skipped_order_count", pd.Series(dtype=float)),
        errors="coerce",
    ).fillna(0.0)
    gross_exposure = pd.to_numeric(
        history_df.get("gross_exposure", history_df.get("invested", pd.Series(dtype=float))),
        errors="coerce",
    ).fillna(0.0).abs()
    equity = pd.to_numeric(history_df.get("equity", pd.Series(dtype=float)), errors="coerce").replace(0, pd.NA)
    starting_equity_series = pd.to_numeric(history_df.get("equity", pd.Series(dtype=float)), errors="coerce").dropna()
    starting_equity = float(starting_equity_series.iloc[0]) if not starting_equity_series.empty else 0.0
    exposure_ratio = (gross_exposure / equity).replace([float("inf"), float("-inf")], pd.NA).dropna()
    order_days = int((turnover > 0).sum())
    total_order_value = float(turnover.sum())
    total_orders = int(order_counts.sum()) if not order_counts.empty else order_days
    max_gross_exposure = float(gross_exposure.max()) if not gross_exposure.empty else 0.0
    return {
        "total_orders": total_orders,
        "scheduled_orders": int(scheduled_counts.sum()) if not scheduled_counts.empty else total_orders,
        "skipped_orders": int(skipped_counts.sum()) if not skipped_counts.empty else 0,
        "order_days": order_days,
        "total_order_value": total_order_value,
        "cumulative_turnover": total_order_value,
        "planned_order_value": float(planned_values.sum()) if not planned_values.empty else total_order_value,
        "skipped_order_value": float(skipped_values.sum()) if not skipped_values.empty else 0.0,
        "average_order_day_value": total_order_value / order_days if order_days else 0.0,
        "max_order_day_value": float(turnover.max()) if not turnover.empty else 0.0,
        "max_gross_exposure": max_gross_exposure,
        "max_gross_exposure_pct": float(exposure_ratio.max()) if not exposure_ratio.empty else 0.0,
        "capital_limit": starting_equity,
        "max_capital_at_work": max_gross_exposure,
        "max_capital_at_work_pct_of_start": max_gross_exposure / starting_equity if starting_equity else 0.0,
    }


def _backtest_response(
    history_df: pd.DataFrame,
    *,
    strategy: str,
    label: str,
    period: str,
    source: str,
    cached: bool = False,
    updated_at: str | None = None,
) -> dict[str, Any]:
    sizing_config = get_config(strategy_id=strategy)
    if history_df.empty:
        raise RuntimeError("Backtest did not produce any equity history.")
    history_df = history_df.copy()
    backtest_cash_limit = _backtest_starting_equity()
    starting_raw_equity = float(history_df["equity"].iloc[0])
    if starting_raw_equity:
        scale = backtest_cash_limit / starting_raw_equity
        for column in (
            "equity",
            "cash",
            "invested",
            "turnover",
            "transaction_costs",
            "dca_contributions",
            "gross_exposure",
            "long_value",
            "short_value",
            "planned_order_value",
            "skipped_order_value",
        ):
            if column in history_df:
                history_df[column] = pd.to_numeric(history_df[column], errors="coerce") * scale
        if "positions" in history_df:
            history_df["positions"] = history_df["positions"].apply(
                lambda positions: {
                    str(symbol): float(value) * scale
                    for symbol, value in (positions or {}).items()
                    if abs(float(value or 0.0) * scale) > 0.005
                }
                if isinstance(positions, dict)
                else {}
            )
    metrics = calculate_performance_metrics(history_df["equity"])
    starting_equity = float(history_df["equity"].iloc[0])
    ending_equity = float(history_df["equity"].iloc[-1])
    ending_cash = float(pd.to_numeric(history_df.get("cash", pd.Series([0.0])), errors="coerce").fillna(0.0).iloc[-1])
    ending_invested = float(pd.to_numeric(history_df.get("invested", pd.Series([0.0])), errors="coerce").fillna(0.0).iloc[-1])
    profit_loss = ending_equity - starting_equity
    return {
        "period": period,
        "period_label": _period_label(period),
        "strategy": strategy,
        "strategy_label": label,
        "source": source,
        "cached": cached,
        "updated_at": updated_at or pd.Timestamp.now(tz="UTC").isoformat(),
        "metrics": metrics,
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "ending_cash": ending_cash,
        "ending_invested": ending_invested,
        "profit_loss": profit_loss,
        "total_return": (ending_equity / starting_equity - 1) if starting_equity else 0.0,
        "sizing": {
            "starting_equity": backtest_cash_limit,
            "cash_limit": backtest_cash_limit,
            "cash_account_only": True,
            "max_portfolio_exposure": sizing_config.max_portfolio_exposure,
            "max_weight_per_symbol": sizing_config.max_weight_per_symbol,
            "max_longs": sizing_config.max_longs,
            "cash_buffer": sizing_config.cash_buffer,
        },
        "orders": _backtest_order_summary(history_df),
        "rows": _json_backtest_rows(history_df),
    }


def _strategy_history_bars(strategy: str, period: str, config) -> dict[str, pd.DataFrame]:
    algorithm = get_algorithm_class(strategy).from_config(config)
    requirements = algorithm.requirements(config, {})
    symbols = requirements.price_symbols or config.symbols
    return fetch_daily_bars(
        symbols,
        config=config,
        lookback_days=int(requirements.daily_lookback_days or config.momentum_lookback_days),
        ma_days=int(requirements.daily_ma_days or 0),
        extra_buffer_days=int(requirements.daily_extra_buffer_days or 0) + _period_row_count(period) + 10,
        alpaca_data_client=create_data_client(config),
        data_feed=config.alpaca_data_feed,
        include_latest=True,
    )


def _configured_intraday_providers(config) -> list[str]:
    providers = [
        str(provider).strip().lower()
        for provider in getattr(config, "intraday_market_data_provider_order", [])
        if str(provider).strip()
    ]
    return providers or ["yfinance"]


def _compute_backtest(
    strategy: str,
    period: str,
    dca_plan: dict[str, Any],
) -> dict[str, Any]:
    """Backtest by replaying the algorithm itself.

    One path for every algorithm. There is no per-strategy branch here any more: whatever the
    algorithm declares in ``requirements()`` is what the replay loads, and whatever ``analyze``
    decides is what gets traded -- so a backtest cannot test different logic than the runtime.
    """
    starting_equity = _backtest_starting_equity()
    config = get_config(strategy_id=strategy)
    algorithm = get_algorithm_class(strategy).from_config(config)
    schedule = algorithm.schedule

    daily_history: dict[str, pd.DataFrame] = {}
    for symbol, frame in _strategy_history_bars(strategy, period, config).items():
        work = prepared_strategy_frame(frame)
        if work.empty:
            continue
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
        daily_history[symbol] = work.sort_values("timestamp").set_index("timestamp")
    if not daily_history:
        raise RuntimeError("No historical bars were available for the backtest.")

    start = _period_start(period)
    trade_dates = sorted(set.intersection(*(set(df.index) for df in daily_history.values())))
    in_period = [date for date in trade_dates if date >= start]
    trade_dates = in_period if len(in_period) >= 2 else trade_dates[-_period_row_count(period):]
    if len(trade_dates) < 2:
        raise RuntimeError("No common trading dates were available for the backtest period.")

    history_df, coverage = replay(
        algorithm,
        config,
        daily_history=daily_history,
        trade_dates=trade_dates,
        should_run=lambda date: int(date.dayofweek) in schedule.weekdays,
        starting_equity=starting_equity,
        intraday_providers=_configured_intraday_providers(config),
    )
    payload = _backtest_response(
        history_df,
        strategy=strategy,
        label=STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title()),
        period=period,
        source="algorithm",
    )
    # Surfaced, not buried: a window the intraday cache cannot reach scores every symbol near
    # zero, which would otherwise read as a poor strategy rather than an unsupported window.
    payload["coverage"] = coverage.as_dict()
    return payload


def backtest_payload(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    period = str(body.get("period") or _default_backtest_period()).lower()
    strategy = canonical_algorithm_id(str(body.get("strategy") or DEFAULT_STRATEGY_ID))[:80]
    refresh = bool(body.get("refresh"))
    cache_only = bool(body.get("cache_only") or body.get("cacheOnly"))
    dca_plan = load_dca_plan(universe_payload()["rows"])
    key = _cache_key(strategy, period, dca_plan)
    cache = _load_backtest_cache()

    if key in cache["items"] and not refresh:
        cached_payload = dict(cache["items"][key])
        cached_payload["cached"] = True
        return cached_payload

    if not refresh or cache_only:
        return {
            "strategy": strategy,
            "period": period,
            "period_label": _period_label(period),
            "cached": False,
            "error": f"No cached {_period_label(period)} backtest is available.",
        }

    payload = _compute_backtest(strategy, period, dca_plan)
    payload["cached"] = False
    cache["items"][key] = payload
    _save_backtest_cache(cache)
    return payload
