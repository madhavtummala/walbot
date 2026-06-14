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
from ..data import fetch_daily_bars
from ..execution.backtest import calculate_performance_metrics
from ..core.bot_runtime import bot_runtime
from ..core.config import get_config, save_universe_symbols
from ..algorithms.registry import get_algorithm_class
from ..connectors import (
    fetch_latest_news_sentiment,
    merge_social_frames,
    news_records_to_social_frames,
)
from src.api.controls import load_controls, save_controls
from ..algorithms.dca import allocation_preview, load_dca_plan, save_dca_plan
from ..core.portfolio import compute_target_weights
from ..data.signals.signals import compute_signals_for_universe
from ..data.social import load_social_trends_csv
from ..data.duckdb_store import read_market_bars
from ..data.state_store import load_state, save_state
from ..core.strategy_models import (
    STRATEGY_LABELS,
    prepared_strategy_frame,
    strategy_signal_rows,
    strategy_signal_rows_from_prepared,
    weights_from_strategy_rows,
)
from ..algorithms.fast_momentum import (
    DefensiveMomentumConfig,
    build_defensive_momentum_targets,
    compute_composite_scores,
    compute_price_features,
    decide_target_weights,
    get_daily_bars as get_defensive_daily_bars,
    get_intraday_bars,
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
    strategy = str(controls.get("active_strategy") or "momentum_social")
    config = get_config(strategy_id=strategy if strategy != "none" else None)
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


def save_controls_payload(body: dict[str, Any]) -> dict[str, Any]:
    raw_controls = body.get("controls", body)
    if str(raw_controls.get("active_strategy") or "none") == "none":
        raw_controls = {**raw_controls, "algorithm_enabled": False}
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


def _latest_signal_bars(strategy: str = "momentum_social") -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    config = get_config(strategy_id=strategy)
    data_client = create_data_client(config)
    bars_by_symbol = fetch_daily_bars(
        config.symbols,
        max(260, config.momentum_lookback_days),
        ma_days=max(200, config.long_ma_days),
        extra_buffer_days=config.history_extra_buffer_days,
        alpaca_data_client=data_client,
        data_feed=config.alpaca_data_feed,
        include_latest=True,
        config=config,
    )
    social_by_symbol = merge_social_frames(
        load_social_trends_csv(config.social_trends_csv, config.symbols),
        news_records_to_social_frames(fetch_latest_news_sentiment(config.symbols, config)),
    )
    return bars_by_symbol, social_by_symbol


def _sanitize_social_frames(social_by_symbol: dict[str, Any]) -> dict[str, pd.DataFrame | None]:
    return {
        symbol: frame if isinstance(frame, pd.DataFrame) else None
        for symbol, frame in social_by_symbol.items()
    }


def _latest_sma(df: pd.DataFrame, window: int) -> float | None:
    if df.empty or "close" not in df or len(df) < window:
        return None
    close = pd.to_numeric(df["close"], errors="coerce")
    return _json_number(close.rolling(window).mean().iloc[-1])


def strategy_signals_payload(strategy: str = "momentum_social") -> dict[str, Any]:
    strategy = (strategy or "none").lower()[:80]
    if strategy == "none":
        plan = load_dca_plan(universe_payload()["rows"])
        preview = allocation_preview(plan)
        return {
            "strategy": strategy,
            "wired": True,
            "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "summary": [
                {"label": "Mode", "value": "DCA" if plan.get("enabled") else "Flat"},
                {"label": "Schedule", "value": plan.get("schedule_pattern", "--")},
                {"label": "Planned", "value": f"${sum(float(row.get('notional', 0.0)) for row in preview):.0f}"},
            ],
            "leaders": [
                {
                    "symbol": row["symbol"],
                    "signal": row["action"].upper(),
                    "score": _json_number(row.get("weight")),
                    "target_weight": _json_number(row.get("weight")),
                    "ret_N": None,
                    "social_score": None,
                    "trend_ok": 1,
                }
                for row in preview[:5]
            ],
        }

    if strategy == "fast_momentum":
        return _defensive_momentum_signals_payload()
    if strategy == "invest_spy":
        return _invest_spy_signals_payload()

    if strategy != "momentum_social":
        config = get_config(strategy_id=strategy)
        bars_by_symbol, social_by_symbol = _latest_signal_bars(strategy)
        leaders = strategy_signal_rows(
            strategy,
            bars_by_symbol,
            social_by_symbol=social_by_symbol if strategy == "dual_momentum" else None,
            social_lookback_days=config.social_lookback_days,
            social_weight=config.social_momentum_weight if strategy == "dual_momentum" else 0.0,
        )
        long_count = sum(1 for row in leaders if row["signal"] == "LONG")
        short_count = sum(1 for row in leaders if row["signal"] == "SHORT")
        gross_count = max(long_count + short_count, 1)
        for row in leaders:
            row["target_weight"] = (1 / gross_count) if row["signal"] == "LONG" else (-1 / gross_count) if row["signal"] == "SHORT" else 0.0
        return {
            "strategy": strategy,
            "wired": True,
            "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "summary": [
                {"label": "Long", "value": str(long_count)},
                {"label": "Short", "value": str(short_count)},
                {"label": "Model", "value": STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title())},
            ],
            "leaders": leaders,
        }

    config = get_config(strategy_id=strategy)
    data_client = create_data_client(config)
    bars_by_symbol = fetch_daily_bars(
        config.symbols,
        config.momentum_lookback_days,
        ma_days=config.long_ma_days,
        extra_buffer_days=config.history_extra_buffer_days,
        alpaca_data_client=data_client,
        data_feed=config.alpaca_data_feed,
        include_latest=True,
        config=config,
    )
    social_by_symbol = _sanitize_social_frames(
        merge_social_frames(
            load_social_trends_csv(config.social_trends_csv, config.symbols),
            news_records_to_social_frames(fetch_latest_news_sentiment(config.symbols, config)),
        )
    )
    signals = compute_signals_for_universe(
        bars_by_symbol,
        config.momentum_lookback_days,
        config.long_ma_days,
        short_lookback_days=config.short_momentum_lookback_days,
        volume_lookback_days=config.volume_lookback_days,
        social_by_symbol=social_by_symbol,
        social_lookback_days=config.social_lookback_days,
        price_momentum_weight=config.price_momentum_weight,
        social_momentum_weight=config.social_momentum_weight,
        volume_momentum_weight=config.volume_momentum_weight,
        min_composite_score=config.min_composite_score,
    )
    weights = compute_target_weights(
        signals,
        config.max_weight_per_symbol,
        max_portfolio_exposure=config.max_portfolio_exposure,
        max_longs=config.max_longs,
        target_annual_vol=config.target_annual_vol,
    )
    ranked = sorted(
        signals.items(),
        key=lambda item: (int(item[1].get("signal", 0)), float(item[1].get("score", 0.0) or 0.0)),
        reverse=True,
    )
    active = [item for item in ranked if int(item[1].get("signal", 0))]
    leaders = [
        {
            "symbol": symbol,
            "signal": "LONG" if int(values.get("signal", 0)) else "FLAT",
            "close": _json_number(values.get("close")),
            "sma_20": _latest_sma(bars_by_symbol.get(symbol, pd.DataFrame()), 20),
            "sma_50": _latest_sma(bars_by_symbol.get(symbol, pd.DataFrame()), 50),
            "sma_long": _json_number(values.get("sma_long")),
            "score": _json_number(values.get("score")),
            "price_score": _json_number(values.get("price_score")),
            "social_score": _json_number(values.get("social_score")),
            "volume_score": _json_number(values.get("volume_score")),
            "ret_N": _json_number(values.get("ret_N")),
            "ret_short": _json_number(values.get("ret_short")),
            "realized_vol": _json_number(values.get("realized_vol")),
            "trend_ok": int(values.get("trend_ok", 0)),
            "target_weight": _json_number(weights.get(symbol, 0.0)),
            "reason": "Positive composite momentum above long trend" if int(values.get("signal", 0)) else "Watching for trend and score confirmation",
        }
        for symbol, values in ranked
    ]
    return {
        "strategy": strategy,
        "wired": True,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "summary": [
            {"label": "Active buys", "value": str(len(active))},
            {"label": "Universe", "value": str(len(signals))},
            {"label": "Exposure", "value": f"{sum(weights.values()) * 100:.0f}%"},
        ],
        "leaders": leaders,
    }


def _defensive_momentum_sentiment_from_records(
    symbols: list[str],
    records: list[dict[str, Any]],
    lookback_minutes: int,
) -> tuple[dict[str, float], float, dict[str, dict[str, Any]], list[str]]:
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=max(lookback_minutes, 1))
    by_symbol: dict[str, list[float]] = {symbol.upper(): [] for symbol in symbols}
    metadata: dict[str, dict[str, Any]] = {
        symbol.upper(): {"sentiment_records": 0, "sentiment_providers": []}
        for symbol in symbols
    }
    providers: set[str] = set()
    for record in records:
        symbol = str(record.get("symbol", "")).upper()
        if symbol not in by_symbol:
            continue
        timestamp = pd.to_datetime(record.get("timestamp"), utc=True, errors="coerce")
        if not pd.isna(timestamp) and timestamp < cutoff:
            continue
        try:
            sentiment = float(record.get("social_score", record.get("sentiment", 0.0)))
        except (TypeError, ValueError):
            sentiment = 0.0
        provider = str(record.get("provider") or "").strip().lower()
        if provider:
            providers.add(provider)
            if provider not in metadata[symbol]["sentiment_providers"]:
                metadata[symbol]["sentiment_providers"].append(provider)
        metadata[symbol]["sentiment_records"] += 1
        by_symbol[symbol].append(max(-1.0, min(1.0, sentiment)))

    symbol_sentiment = {
        symbol: (sum(values) / len(values) if values else 0.0)
        for symbol, values in by_symbol.items()
    }
    market_sentiment = symbol_sentiment.get("SPY")
    if market_sentiment is None:
        values = list(symbol_sentiment.values())
        market_sentiment = sum(values) / len(values) if values else 0.0
    return symbol_sentiment, float(market_sentiment), metadata, sorted(providers)


def _defensive_momentum_reason(
    row: dict[str, Any],
    weight: float,
    strategy_config: DefensiveMomentumConfig,
) -> str:
    symbol = str(row.get("symbol", "")).upper()
    if weight > 0:
        return "Top Rank"
    if not bool(row.get("macro_trend_ok")):
        return "Macro negative"
    score_floor = strategy_config.min_risk_on_score if symbol in {item.upper() for item in strategy_config.risk_on_universe} else strategy_config.min_defensive_score
    if float(row.get("score", 0.0)) < score_floor:
        return "Score too low"
    if symbol in {item.upper() for item in strategy_config.risk_on_universe} and float(row.get("micro_return", 0.0)) < strategy_config.min_risk_on_micro_return:
        return "Micro too low"
    return "No rank slot"


def _defensive_momentum_signals_payload() -> dict[str, Any]:
    config = get_config(strategy_id="fast_momentum")
    strategy_config = DefensiveMomentumConfig.from_runtime_config(config)
    symbols = strategy_config.symbols
    data_client = create_data_client(config)
    intraday_bars = get_intraday_bars(symbols, strategy_config.required_intraday_bars, config, data_client)
    daily_bars = get_defensive_daily_bars(symbols, strategy_config.required_daily_bars, config, data_client)
    sentiment_records = fetch_latest_news_sentiment(symbols, config)
    sentiment, market_sentiment, _, providers = _defensive_momentum_sentiment_from_records(
        symbols,
        sentiment_records,
        strategy_config.sentiment_lookback_minutes,
    )
    features = {
        symbol: compute_price_features(symbol, intraday_bars.get(symbol, pd.DataFrame()), daily_bars.get(symbol, pd.DataFrame()), strategy_config)
        for symbol in symbols
    }
    scores = compute_composite_scores(features, sentiment, strategy_config)
    weights = decide_target_weights(scores, strategy_config)
    allocation_mode = "Dynamic rank" if any(float(weight) > 0 for weight in weights.values()) else "Cash"
    leaders = []
    for symbol, row in scores.items():
        weight = float(weights.get(symbol, 0.0))
        components = row.get("components", {}) if isinstance(row.get("components"), dict) else {}
        leaders.append(
            {
                "symbol": symbol,
                "signal": "LONG" if weight > 0 else "FLAT",
                "close": _json_number(row.get("close")),
                "score": _json_number(row.get("score")),
                "score_components": {key: _json_number(value) for key, value in components.items()},
                "allocation_mode": allocation_mode,
                "realized_vol": _json_number(row.get("realized_volatility")),
                "trend_ok": int(bool(row.get("macro_trend_ok"))),
                "target_weight": _json_number(weight),
                "reason": _defensive_momentum_reason(row, weight, strategy_config),
            }
        )
    leaders.sort(key=lambda item: (item["signal"] != "LONG", -float(item.get("target_weight") or 0.0), -float(item.get("score") or 0.0)))
    return {
        "strategy": "fast_momentum",
        "wired": True,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "summary": [
            {"label": "Allocation", "value": allocation_mode},
            {"label": "Exposure", "value": f"{sum(float(weight) for weight in weights.values()) * 100:.0f}%"},
            {"label": "Sentiment", "value": ", ".join(providers) if providers else "No recent records"},
            {"label": "Universe", "value": str(len(symbols))},
        ],
        "leaders": leaders,
    }


def defensive_momentum_portfolio_preview() -> dict[str, Any]:
    """Compute what the fast momentum algorithm would do given current positions and stickiness.

    Fetches live market data and current brokerage positions, applies the full algorithm
    (including the min_score_delta_to_replace incumbency bonus), and returns the planned
    portfolio changes alongside the current state.
    """
    config = get_config(strategy_id="fast_momentum")
    data_client = create_data_client(config)
    trading_client = create_trading_client(config)
    current_positions = get_positions(trading_client)
    account_equity = get_account_equity(trading_client)
    equity = float(account_equity)
    strategy_config = DefensiveMomentumConfig.from_runtime_config(config)

    latest_prices: dict[str, float] = {}
    for symbol in strategy_config.symbols:
        bars = get_intraday_bars([symbol], strategy_config.required_intraday_bars, config, data_client)
        df = bars.get(symbol)
        if df is not None and not df.empty and "close" in df.columns:
            price = float(df["close"].iloc[-1])
            if price > 0:
                latest_prices[symbol] = price
    for symbol in set(current_positions) - set(latest_prices):
        bars = get_intraday_bars([symbol], 2, config, data_client)
        df = bars.get(symbol)
        if df is not None and not df.empty and "close" in df.columns:
            price = float(df["close"].iloc[-1])
            if price > 0:
                latest_prices[symbol] = price

    target_weights, signals, metadata = build_defensive_momentum_targets(
        config,
        data_client,
        current_positions,
        latest_prices,
        equity,
    )

    current_weights = {
        symbol: (shares * latest_prices.get(symbol, 0.0)) / max(equity, 1.0)
        for symbol, shares in current_positions.items()
        if latest_prices.get(symbol, 0.0) > 0
    }

    score_delta = max(strategy_config.min_score_delta_to_replace, 0.0)
    scores_data = metadata.get("scores", {})
    all_scored = [
        (s, float(scores_data[s].get("score", 0.0)))
        for s in strategy_config.symbols
        if s in scores_data
    ]
    without_bonus_top = set(
        s for s, _ in sorted(all_scored, key=lambda x: -x[1])[:max(strategy_config.max_positions, 0)]
    )
    current_held = {s for s, w in current_weights.items() if w > 0}
    target_set = {s for s, w in target_weights.items() if w > 0}

    planned_orders = plan_position_orders(
        latest_prices=latest_prices,
        current_positions=current_positions,
        target_weights=target_weights,
        equity=equity,
        cash_buffer=0.0,
        min_trade_dollars=strategy_config.per_trade_value_min,
        rebalance_threshold=strategy_config.rebalance_threshold,
    )

    positions_detail = []
    for symbol in strategy_config.symbols:
        cw = round(current_weights.get(symbol, 0.0), 4)
        tw = round(target_weights.get(symbol, 0.0), 4)
        in_current = symbol in current_held
        in_target = symbol in target_set

        if in_current and in_target:
            if score_delta > 0 and symbol not in without_bonus_top:
                status = "retained"
                sticky = True
                reason = "Sticky"
            else:
                status = "retained"
                sticky = False
                reason = "Retained"
        elif in_current and not in_target:
            status = "dropped"
            sticky = False
            reason = "Dropped"
        elif not in_current and in_target:
            status = "new"
            sticky = False
            reason = "New"
        else:
            status = "none"
            sticky = False
            reason = "None"

        positions_detail.append({
            "symbol": symbol,
            "current_weight": cw,
            "target_weight": tw,
            "status": status,
            "sticky": sticky,
            "reason": reason,
        })

    return {
        "strategy": "fast_momentum",
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "equity": equity,
        "current_weights": {s: round(w, 4) for s, w in sorted(current_weights.items()) if w > 0},
        "target_weights": {s: round(w, 4) for s, w in sorted(target_weights.items()) if w != 0},
        "positions": positions_detail,
        "planned_orders": [
            {
                "symbol": o["symbol"],
                "action": o["action"],
                "quantity": o["quantity"],
                "current_shares": o["current_shares"],
                "target_shares": o["target_shares"],
                "trade_dollars": round(float(o["trade_dollars"]), 2),
                "latest_price": float(o["latest_price"]),
            }
            for o in planned_orders
        ],
        "stickiness": {
            "delta": score_delta,
            "enabled": score_delta > 0,
        },
        "summary": [
            {"label": "Allocation", "value": metadata.get("allocation_mode", "N/A")},
            {"label": "Exposure", "value": f"{sum(float(w) for w in target_weights.values()) * 100:.0f}%"},
            {"label": "Positions", "value": str(len(current_positions))},
            {"label": "Changes", "value": str(len(planned_orders))},
        ],
    }


def _invest_spy_reason(symbol: str, weight: float, state: str, config: InvestSpyConfig) -> str:
    if weight > 0 and symbol == config.spy_symbol:
        return "SPY growth/pullback"
    if weight > 0 and symbol in {item.upper() for item in config.equity_income_universe}:
        return "Flat-market income"
    if weight > 0 and symbol in {item.upper() for item in config.crisis_hedge_universe}:
        return "Crisis hedge"
    if weight > 0:
        return "Defensive allocation"
    if symbol == config.spy_symbol and state in {"FALLING", "CRISIS"}:
        return "SPY state defensive"
    if symbol in {item.upper() for item in config.equity_income_universe} and state != "FLAT":
        return "Income waits for flat SPY"
    if symbol in {item.upper() for item in config.crisis_hedge_universe} and state != "CRISIS":
        return "Hedge waits for crisis"
    return "Outside state allocation"


def _invest_spy_signals_payload() -> dict[str, Any]:
    config = get_config(strategy_id="invest_spy")
    strategy_config = InvestSpyConfig.from_runtime_config(config)
    symbols = strategy_config.symbols
    data_client = create_data_client(config)
    intraday_bars = get_intraday_bars(symbols, strategy_config.required_intraday_bars, config, data_client)
    daily_bars = get_defensive_daily_bars(symbols, strategy_config.macro_trend_lookback_days, config, data_client)
    sentiment_records = fetch_latest_news_sentiment(symbols, config)
    sentiment, market_sentiment, _, providers = _defensive_momentum_sentiment_from_records(
        symbols,
        sentiment_records,
        strategy_config.sentiment_lookback_minutes,
    )
    features = {
        symbol: compute_invest_spy_price_features(symbol, intraday_bars.get(symbol, pd.DataFrame()), daily_bars.get(symbol, pd.DataFrame()), strategy_config)
        for symbol in symbols
    }
    scores = compute_composite_scores(features, sentiment, strategy_config)
    state = classify_spy_state(scores.get(strategy_config.spy_symbol, {}), sentiment.get(strategy_config.spy_symbol, market_sentiment), strategy_config)
    weights = decide_invest_spy_weights(scores, state, strategy_config)
    leaders = []
    for symbol, row in scores.items():
        weight = float(weights.get(symbol, 0.0))
        components = row.get("components", {}) if isinstance(row.get("components"), dict) else {}
        leaders.append(
            {
                "symbol": symbol,
                "signal": "LONG" if weight > 0 else "FLAT",
                "close": _json_number(row.get("close")),
                "score": _json_number(row.get("score")),
                "score_components": {key: _json_number(value) for key, value in components.items()},
                "spy_state": state,
                "realized_vol": _json_number(row.get("realized_volatility")),
                "trend_ok": int(bool(row.get("macro_trend_ok"))),
                "target_weight": _json_number(weight),
                "reason": _invest_spy_reason(symbol, weight, state, strategy_config),
            }
        )
    leaders.sort(key=lambda item: (item["signal"] != "LONG", -float(item.get("target_weight") or 0.0), -float(item.get("score") or 0.0)))
    return {
        "strategy": "invest_spy",
        "wired": True,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "summary": [
            {"label": "SPY state", "value": state.title()},
            {"label": "Exposure", "value": f"{sum(float(weight) for weight in weights.values()) * 100:.0f}%"},
            {"label": "Sentiment", "value": ", ".join(providers) if providers else "No recent records"},
            {"label": "Universe", "value": str(len(symbols))},
        ],
        "leaders": leaders,
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
    config = get_config(strategy_id=strategy if strategy != "none" else None)
    selected_algorithm_config = (
        config.algorithm_configs.get(strategy, {})
        if strategy != "none" and isinstance(config.algorithm_configs, dict)
        else {}
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
        "algorithm_config": algorithm_config if strategy != "none" else {},
        "dca_plan": dca_plan if strategy == "none" else {},
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
    sizing_config = get_config(strategy_id=strategy if strategy != "none" else None)
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


def _flat_backtest(period: str, starting_equity: float | None = None) -> pd.DataFrame:
    starting_equity = _backtest_starting_equity() if starting_equity is None else float(starting_equity)
    start = _period_start(period)
    end = pd.Timestamp.now(tz="UTC")
    dates = pd.bdate_range(start=start, end=end, tz="UTC")
    if len(dates) < 2:
        dates = pd.DatetimeIndex([start, end])
    return pd.DataFrame(
        [{"timestamp": date, "equity": starting_equity, "cash": starting_equity, "invested": 0.0} for date in dates]
    ).set_index("timestamp")


def _cron_value_matches(field: str, value: int) -> bool:
    field = field.strip()
    if field == "*":
        return True
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            if start_raw.isdigit() and end_raw.isdigit() and int(start_raw) <= value <= int(end_raw):
                return True
        elif part.isdigit() and int(part) == value:
            return True
    return False


def _frequency_runs_on_index(frequency: str, index: int) -> bool:
    schedule_every = 5 if frequency in {"weekly", "biweekly"} else 1
    if frequency == "biweekly":
        schedule_every = 10
    elif frequency == "monthly":
        schedule_every = 21
    return index % schedule_every == 0


def _dca_runs_on_trade_date(plan: dict[str, Any], trade_date: pd.Timestamp, index: int) -> bool:
    pattern = str(plan.get("schedule_pattern") or "").split()
    if len(pattern) == 5:
        _minute, _hour, day_of_month, month, day_of_week = pattern
        cron_dow = 0 if trade_date.dayofweek == 6 else trade_date.dayofweek + 1
        return (
            _cron_value_matches(month, int(trade_date.month))
            and _cron_value_matches(day_of_month, int(trade_date.day))
            and (_cron_value_matches(day_of_week, cron_dow) or (cron_dow == 0 and _cron_value_matches(day_of_week, 7)))
        )
    return _frequency_runs_on_index(str(plan.get("frequency") or "weekly"), index)


def _dca_backtest(
    plan: dict[str, Any],
    period: str,
    starting_equity: float | None = None,
) -> pd.DataFrame:
    starting_equity = _backtest_starting_equity() if starting_equity is None else float(starting_equity)
    preview = [row for row in allocation_preview(plan) if row["action"] == "buy" and row["notional"] > 0]
    if not plan.get("enabled") or not preview:
        return _flat_backtest(period, starting_equity)

    config = get_config()
    symbols = sorted({row["symbol"] for row in preview})
    data_client = create_data_client(config)
    bars_by_symbol = fetch_daily_bars(
        symbols,
        lookback_days=_period_row_count(period) + 10,
        ma_days=0,
        extra_buffer_days=10,
        alpaca_data_client=data_client,
        data_feed=config.alpaca_data_feed,
        config=config,
    )
    history_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol, df in bars_by_symbol.items():
        if df.empty:
            continue
        work_df = df.copy()
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)
        history_by_symbol[symbol] = work_df.sort_values("timestamp").set_index("timestamp")
    if not history_by_symbol:
        raise RuntimeError("No historical bars were available for the DCA backtest.")

    start = _period_start(period)
    common_dates = sorted(set.intersection(*(set(df.index) for df in history_by_symbol.values())))
    common_dates = [date for date in common_dates if date >= start]
    if not common_dates:
        raise RuntimeError("No common trading dates were available for the DCA backtest period.")

    cash = starting_equity
    shares = {symbol: 0.0 for symbol in history_by_symbol}
    invested_cash = 0.0
    records: list[dict[str, Any]] = []

    for index, trade_date in enumerate(common_dates):
        order_count = 0
        order_value = 0.0
        planned_order_value = 0.0
        skipped_order_value = 0.0
        scheduled_order_count = 0
        skipped_order_count = 0
        close_prices = {
            symbol: float(history_by_symbol[symbol].loc[trade_date, "close"])
            for symbol in history_by_symbol
            if trade_date in history_by_symbol[symbol].index
        }
        if _dca_runs_on_trade_date(plan, trade_date, index):
            for row in preview:
                symbol = row["symbol"]
                price = close_prices.get(symbol)
                notional = float(row["notional"])
                if notional > 0:
                    scheduled_order_count += 1
                    planned_order_value += notional
                if not price or price <= 0 or notional <= 0:
                    skipped_order_count += 1
                    skipped_order_value += notional
                    continue
                if cash < notional:
                    skipped_order_count += 1
                    skipped_order_value += notional
                    continue
                shares[symbol] = shares.get(symbol, 0.0) + notional / price
                cash -= notional
                invested_cash += notional
                order_count += 1
                order_value += notional

        market_value = sum(shares.get(symbol, 0.0) * price for symbol, price in close_prices.items())
        positions_value = {
            symbol: shares.get(symbol, 0.0) * price
            for symbol, price in close_prices.items()
            if abs(shares.get(symbol, 0.0) * price) > 0.005
        }
        records.append(
            {
                "timestamp": trade_date,
                "equity": cash + market_value,
                "cash": cash,
                "invested": market_value,
                "positions": positions_value,
                "dca_contributions": invested_cash,
                "turnover": order_value,
                "order_count": order_count,
                "scheduled_order_count": scheduled_order_count,
                "skipped_order_count": skipped_order_count,
                "planned_order_value": planned_order_value,
                "skipped_order_value": skipped_order_value,
                "gross_exposure": market_value,
                "long_value": market_value,
                "short_value": 0.0,
            }
        )

    return pd.DataFrame(records).set_index("timestamp").sort_index()


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


def _history_price_at(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    column: str,
    *,
    fallback_column: str | None = None,
) -> float:
    value = df.loc[timestamp, column] if column in df.columns else pd.NA
    if isinstance(value, pd.Series):
        value = value.dropna().iloc[-1] if not value.dropna().empty else pd.NA
    if pd.isna(value) and fallback_column:
        value = df.loc[timestamp, fallback_column] if fallback_column in df.columns else pd.NA
        if isinstance(value, pd.Series):
            value = value.dropna().iloc[-1] if not value.dropna().empty else pd.NA
    return float(value) if not pd.isna(value) else 0.0


def _configured_intraday_providers(config) -> list[str]:
    providers = [
        str(provider).strip().lower()
        for provider in getattr(config, "intraday_market_data_provider_order", [])
        if str(provider).strip()
    ]
    return providers or ["yfinance"]


def _read_backtest_intraday_bars(
    symbol: str,
    signal_date: pd.Timestamp,
    *,
    config,
    lookback_bars: int,
    bar_minutes: int,
) -> pd.DataFrame:
    timeframe = f"{int(bar_minutes)}m"
    providers = _configured_intraday_providers(config)
    intraday_end = signal_date + pd.Timedelta(hours=20)
    for provider in providers:
        try:
            bars = read_market_bars(
                "intraday_market_data",
                provider,
                symbol,
                timeframe,
                lookback_bars=lookback_bars,
                end=intraday_end,
            )
        except Exception as exc:
            logger.warning(
                "Fast Momentum backtest intraday cache read failed provider=%s symbol=%s signal_date=%s timeframe=%s lookback=%s: %s",
                provider,
                symbol,
                signal_date.isoformat(),
                timeframe,
                lookback_bars,
                exc,
            )
            continue
        if not bars.empty:
            logger.debug(
                "Fast Momentum backtest intraday cache hit provider=%s symbol=%s signal_date=%s rows=%s",
                provider,
                symbol,
                signal_date.isoformat(),
                len(bars),
            )
            return bars
        logger.debug(
            "Fast Momentum backtest intraday cache miss provider=%s symbol=%s signal_date=%s timeframe=%s lookback=%s",
            provider,
            symbol,
            signal_date.isoformat(),
            timeframe,
            lookback_bars,
        )
    logger.warning(
        "Fast Momentum backtest intraday cache miss symbol=%s signal_date=%s providers=%s; nano/micro returns will be 0",
        symbol,
        signal_date.isoformat(),
        ",".join(providers),
    )
    return pd.DataFrame()


def _strategy_backtest(
    strategy: str,
    period: str,
    starting_equity: float | None = None,
) -> pd.DataFrame:
    starting_equity = _backtest_starting_equity() if starting_equity is None else float(starting_equity)
    config = get_config(strategy_id=strategy)
    invest_spy_config = InvestSpyConfig.from_runtime_config(config) if strategy == "invest_spy" else None
    defensive_strategy_config = DefensiveMomentumConfig.from_runtime_config(config) if strategy == "fast_momentum" else None
    bars_by_symbol = _strategy_history_bars(strategy, period, config)
    history_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol, df in bars_by_symbol.items():
        work = prepared_strategy_frame(df)
        if work.empty:
            continue
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
        history_by_symbol[symbol] = work.sort_values("timestamp").set_index("timestamp")
    if not history_by_symbol:
        raise RuntimeError("No historical bars were available for the strategy backtest.")

    start = _period_start(period)
    common_dates = sorted(set.intersection(*(set(df.index) for df in history_by_symbol.values())))
    common_dates = [date for date in common_dates if date >= start]
    if len(common_dates) < 2:
        all_common_dates = sorted(set.intersection(*(set(df.index) for df in history_by_symbol.values())))
        common_dates = all_common_dates[-_period_row_count(period):]
    if len(common_dates) < 2:
        raise RuntimeError("No common trading dates were available for the strategy backtest period.")

    cash = starting_equity
    positions = {symbol: 0.0 for symbol in history_by_symbol}
    records: list[dict[str, Any]] = []
    max_exposure = min(
        max(float(
            invest_spy_config.max_gross_exposure
            if invest_spy_config
            else defensive_strategy_config.max_gross_exposure
            if defensive_strategy_config
            else config.max_portfolio_exposure
        ), 0.0),
        max(1.0 - max(float(config.cash_buffer), 0.0), 0.0),
        1.0,
    )
    max_per_symbol = min(
        max(float(
            invest_spy_config.max_single_position_weight
            if invest_spy_config
            else defensive_strategy_config.max_single_position_weight
            if defensive_strategy_config
            else config.max_weight_per_symbol
        ), 0.01),
        1.0,
    )
    max_longs = max(int(defensive_strategy_config.max_positions if defensive_strategy_config else config.max_longs), 0)

    signal_indexes = list(range(1, len(common_dates)))
    social_by_symbol = (
        load_social_trends_csv(config.social_trends_csv, list(history_by_symbol))
        if strategy == "momentum_social"
        else {}
    )

    def weights_for_index(index: int) -> tuple[int, dict[str, float]]:
        signal_date = common_dates[index - 1]
        if strategy == "momentum_social":
            snapshots = {symbol: df.loc[:signal_date].reset_index() for symbol, df in history_by_symbol.items()}
            signals = compute_signals_for_universe(
                snapshots,
                config.momentum_lookback_days,
                config.long_ma_days,
                short_lookback_days=config.short_momentum_lookback_days,
                volume_lookback_days=config.volume_lookback_days,
                social_by_symbol=social_by_symbol,
                social_lookback_days=config.social_lookback_days,
                price_momentum_weight=config.price_momentum_weight,
                social_momentum_weight=config.social_momentum_weight,
                volume_momentum_weight=config.volume_momentum_weight,
                min_composite_score=config.min_composite_score,
            )
            weights = compute_target_weights(
                signals,
                config.max_weight_per_symbol,
                max_portfolio_exposure=max_exposure,
                max_longs=config.max_longs,
                target_annual_vol=config.target_annual_vol,
            )
            return index, weights

        if invest_spy_config:
            snapshots = {symbol: df.loc[:signal_date].reset_index() for symbol, df in history_by_symbol.items()}
            features = {
                symbol: compute_invest_spy_price_features(symbol, snapshots.get(symbol, pd.DataFrame()), snapshots.get(symbol, pd.DataFrame()), invest_spy_config)
                for symbol in history_by_symbol
            }
            scores = compute_composite_scores(features, {}, invest_spy_config)
            state = classify_spy_state(scores.get(invest_spy_config.spy_symbol, {}), 0.0, invest_spy_config)
            return index, decide_invest_spy_weights(scores, state, invest_spy_config)

        if strategy == "fast_momentum" and defensive_strategy_config:
            snapshots = {symbol: df.loc[:signal_date].reset_index() for symbol, df in history_by_symbol.items()}
            intraday_by_symbol = {
                symbol: _read_backtest_intraday_bars(
                    symbol,
                    signal_date,
                    config=config,
                    lookback_bars=defensive_strategy_config.required_intraday_bars,
                    bar_minutes=15,
                )
                for symbol in snapshots
            }

            features = {
                symbol: compute_price_features(symbol, intraday_by_symbol.get(symbol, pd.DataFrame()), snapshots.get(symbol, pd.DataFrame()), defensive_strategy_config)
                for symbol in history_by_symbol
            }
            scores = compute_composite_scores(features, {}, defensive_strategy_config)
            return index, decide_target_weights(scores, defensive_strategy_config)

        snapshots = {symbol: df.loc[:signal_date] for symbol, df in history_by_symbol.items()}
        decisions = strategy_signal_rows_from_prepared(strategy, snapshots)
        return index, weights_from_strategy_rows(
            decisions,
            list(history_by_symbol),
            max_longs=max_longs,
            max_weight_per_symbol=max_per_symbol,
            max_portfolio_exposure=max_exposure,
        )

    if len(signal_indexes) > 1 and strategy != "fast_momentum":
        max_workers = min(8, len(signal_indexes))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            weights_by_index = dict(executor.map(weights_for_index, signal_indexes))
    else:
        weights_by_index = dict(weights_for_index(index) for index in signal_indexes)

    for index in signal_indexes:
        signal_date = common_dates[index - 1]
        trade_date = common_dates[index]
        weights = {
            symbol: max(float(weights_by_index[index].get(symbol, 0.0)), 0.0)
            for symbol in history_by_symbol
        }
        total_weight = sum(weights.values())
        if total_weight > max_exposure and total_weight > 0:
            scale = max_exposure / total_weight
            weights = {symbol: weight * scale for symbol, weight in weights.items()}

        open_prices = {symbol: _history_price_at(df, trade_date, "open", fallback_column="close") for symbol, df in history_by_symbol.items()}
        # Use adjusted_close for valuation if available to capture dividends
        close_prices = {
            symbol: _history_price_at(df, trade_date, "adjusted_close", fallback_column="close")
            for symbol, df in history_by_symbol.items()
        }
        equity_at_open = cash + sum(positions.get(symbol, 0.0) * open_prices[symbol] for symbol in history_by_symbol)
        target_positions = {
            symbol: (equity_at_open * weights.get(symbol, 0.0)) / open_prices[symbol]
            if open_prices[symbol] > 0
            else 0.0
            for symbol in history_by_symbol
        }
        turnover = 0.0
        costs = 0.0
        order_count = 0
        transaction_cost_rate = config.transaction_cost_bps / 10_000

        sell_diffs = {
            symbol: target_shares - positions.get(symbol, 0.0)
            for symbol, target_shares in target_positions.items()
            if target_shares < positions.get(symbol, 0.0)
        }
        buy_diffs = {
            symbol: target_shares - positions.get(symbol, 0.0)
            for symbol, target_shares in target_positions.items()
            if target_shares > positions.get(symbol, 0.0)
        }

        for symbol, diff in sell_diffs.items():
            notional = abs(diff) * open_prices[symbol]
            if notional <= 1e-9:
                continue
            cost = notional * transaction_cost_rate
            cash += notional - cost
            costs += cost
            turnover += notional
            order_count += 1
            positions[symbol] = target_positions[symbol]

        for symbol, desired_diff in buy_diffs.items():
            price = open_prices[symbol]
            diff = desired_diff
            notional = diff * price
            cost = notional * transaction_cost_rate
            if notional + cost > cash:
                diff = max(cash / (price * (1 + transaction_cost_rate)), 0.0)
                notional = diff * price
                cost = notional * transaction_cost_rate
            if diff <= 1e-9 or notional <= 1e-9:
                continue
            cash -= notional + cost
            if abs(cash) <= 1e-7:
                cash = 0.0
            costs += cost
            turnover += notional
            order_count += 1
            positions[symbol] = positions.get(symbol, 0.0) + diff

        values = {symbol: positions.get(symbol, 0.0) * close_prices[symbol] for symbol in history_by_symbol}
        invested = sum(values.values())
        long_value = sum(value for value in values.values() if value > 0)
        short_value = abs(sum(value for value in values.values() if value < 0))
        gross_exposure = long_value + short_value
        equity = cash + invested
        records.append(
            {
                "timestamp": trade_date,
                "signal_timestamp": signal_date,
                "equity": equity,
                "cash": cash,
                "invested": invested,
                "positions": {
                    symbol: value
                    for symbol, value in values.items()
                    if abs(value) > 0.005
                },
                "turnover": turnover,
                "transaction_costs": costs,
                "order_count": order_count,
                "gross_exposure": gross_exposure,
                "long_value": long_value,
                "short_value": short_value,
                **{f"weight_{symbol}": weights.get(symbol, 0.0) for symbol in history_by_symbol},
                **{f"shares_{symbol}": positions.get(symbol, 0.0) for symbol in history_by_symbol},
            }
        )

    return pd.DataFrame(records).set_index("timestamp").sort_index()


def _compute_backtest(
    strategy: str,
    period: str,
    dca_plan: dict[str, Any],
) -> dict[str, Any]:
    starting_equity = _backtest_starting_equity()
    if strategy == "none":
        if dca_plan.get("enabled"):
            history_df = _dca_backtest(dca_plan, period, starting_equity=starting_equity)
            return _backtest_response(
                history_df,
                strategy=strategy,
                label="None / DCA",
                period=period,
                source="dca",
            )
        return _backtest_response(
            _flat_backtest(period, starting_equity=starting_equity),
            strategy=strategy,
            label="None",
            period=period,
            source="flat",
        )

    history_df = _strategy_backtest(strategy, period, starting_equity=starting_equity)
    start = _period_start(period)
    if history_df.index.tz is None:
        start = start.tz_localize(None)
    period_df = history_df.loc[history_df.index >= start]
    if len(period_df) >= 2:
        history_df = period_df
    else:
        history_df = history_df.tail(_period_row_count(period))
    return _backtest_response(
        history_df,
        strategy=strategy,
        label=strategy.replace("_", " ").title(),
        period=period,
        source="algorithm",
    )


def backtest_payload(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    period = str(body.get("period") or _default_backtest_period()).lower()
    strategy = str(body.get("strategy") or "momentum_social").lower()[:80]
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
