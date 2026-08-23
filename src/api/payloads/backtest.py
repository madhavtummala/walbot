"""Replaying an algorithm over history, and caching the result.

Split out of the single ``api_payloads`` module, which had grown to 1253 lines covering nine
unrelated domains. The public names are unchanged and still importable from ``api_payloads``.
"""


from __future__ import annotations

from .strategy_config import config_for_strategy_view

from ...algorithms.registry import canonical_algorithm_id
from ...brokerages.alpaca.client import create_data_client
from ...algorithms.registry import get_algorithm_class
from ...data.universe import resolve_project_path

import json
import logging
import re
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd


from ...data import fetch_daily_bars
from ...core.cron import parse_cron
from ...data.bars import TRADING_MINUTES_PER_DAY
from ...execution.metrics import calculate_performance_metrics
from ...data.duckdb_store import pooled_connections
from ...execution.replay import replay
from ...core.config import (
    DEFAULT_STRATEGY_ID,
    get_config,
)
from ...data.state_store import load_state, save_state
from ...core.strategy_models import STRATEGY_LABELS, prepared_strategy_frame

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKTEST_CACHE_PATH = "data/backtest_cache.json"

BACKTEST_CACHE_STATE_KEY = "backtest_cache"

#: Bumped whenever the shape of a cached payload changes. ``_load_backtest_cache`` throws away
#: any cache whose version differs, which is the only thing stopping a row written under an
#: older schema from being served to a frontend that no longer understands it.
#:
#: 10: row timestamps carry the time of day.
#: 11: rows carry ``trade_shares`` beside ``trades``.
#: 12: the paper book fills whole shares only, so cached curves priced in fractional fills
#:     are stale.
BACKTEST_CACHE_VERSION = 12

BACKTEST_STARTING_EQUITY = 10_000.0


def _backtest_starting_equity() -> float:
    value = float(get_config().backtest_starting_equity or BACKTEST_STARTING_EQUITY)
    return max(value, 1.0)


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


def _cache_key(strategy: str, period: str, account_id: str = "") -> str:
    """Hash of everything a cached backtest's validity depends on.

    The algorithm declares its own half through ``config_fingerprint`` -- which is how DCA's
    plan gets in without this function knowing DCA exists. It used to be passed a ``dca_plan``
    argument threaded down from the request handler purely to reach this hash.

    The account is part of the basis because some of that config is per account: two DCA
    bindings on different accounts have different plans, and without this they collided on one
    cache entry, so whichever ran first answered for both.
    """
    config = config_for_strategy_view(strategy, account_id)
    try:
        algorithm_basis = get_algorithm_class(strategy).from_config(config).config_fingerprint(config)
    except (KeyError, TypeError, ValueError):
        # An unresolvable strategy still needs a stable key. Reporting it is the compute
        # step's job -- failing here would turn a bad request id into an error from a hash.
        algorithm_basis = {"unresolved": strategy}
    cache_basis = {
        "strategy": strategy,
        "period": period,
        "account": getattr(config, "account_id", ""),
        "data_feed": config.alpaca_data_feed,
        "starting_equity": _backtest_starting_equity(),
        # Shared knobs every algorithm is sized and costed by, whatever it decides.
        "portfolio": {
            "max_weight_per_symbol": config.max_weight_per_symbol,
            "max_portfolio_exposure": config.max_portfolio_exposure,
            "cash_buffer": config.cash_buffer,
            "transaction_cost_bps": config.transaction_cost_bps,
        },
        "algorithm": algorithm_basis,
    }
    encoded = json.dumps(cache_basis, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def _json_backtest_rows(history_df: pd.DataFrame) -> list[dict[str, Any]]:
    """The replay's rows as JSON, timestamps and all.

    ISO 8601 rather than the ``%Y-%m-%d`` this used to emit. A daily replay is unaffected --
    its bars land at midnight, so the frontend's ``new Date`` reads the same instant either way
    -- but an intraday one was destroyed by the old format: every bar in a session formatted to
    the same date string, so Options Flip's ~14,000 rows plotted at 179 distinct x-positions
    with ~78 points stacked on each. The equity line was drawn through whichever of them came
    last, and no marker or tooltip could ever address a single bar.
    """
    rows_df = history_df.reset_index().copy()
    for column in rows_df.columns:
        if pd.api.types.is_datetime64_any_dtype(rows_df[column]):
            rows_df[column] = pd.to_datetime(rows_df[column], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
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


def _fetch_backtest_history(strategy: str, period: str, config) -> dict[str, pd.DataFrame]:
    algorithm = get_algorithm_class(strategy).from_config(config)
    requirements = algorithm.requirements(config, {})
    symbols = requirements.price_symbols or config.symbols
    bars_by_symbol = fetch_daily_bars(
        symbols,
        config=config,
        lookback_days=int(requirements.daily_lookback_days or config.momentum_lookback_days),
        ma_days=int(requirements.daily_ma_days or 0),
        extra_buffer_days=int(requirements.daily_extra_buffer_days or 0) + _period_row_count(period) + 10,
        data_client=create_data_client(config),
        include_latest=True,
    )

    daily_history: dict[str, pd.DataFrame] = {}
    for symbol, frame in bars_by_symbol.items():
        work = prepared_strategy_frame(frame)
        if work.empty:
            continue
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
        daily_history[symbol] = work.sort_values("timestamp").set_index("timestamp")
    return daily_history


def _fetch_intraday_backtest_history(
    strategy: str, period: str, config, daily_history: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    """Fetch intraday bars for algorithms that need them (e.g. options_flip).

    Returns a dict keyed by symbol, indexed by intraday timestamps. Each frame has
    OHLCV columns. Falls back to an empty dict if no intraday data is available.
    """
    from ...data.bars import read_history

    algorithm = get_algorithm_class(strategy).from_config(config)
    requirements = algorithm.requirements(config, {})
    bar_minutes = requirements.preferred_bar_minutes or 15
    lookback_minutes = requirements.intraday_lookback_minutes or 0
    if lookback_minutes <= 0:
        return {}

    symbols = sorted(daily_history)
    start = _period_start(period)
    # Fetch enough history to cover the lookback + the period itself.
    span_days = max((pd.Timestamp.now(tz="UTC") - start).days + 10, 30)
    span_minutes = span_days * TRADING_MINUTES_PER_DAY + lookback_minutes

    intraday_history: dict[str, pd.DataFrame] = {}
    providers = _configured_history_providers(config)
    for symbol in symbols:
        best = pd.DataFrame()
        for provider in [*providers, None]:
            try:
                bars = read_history(
                    symbol,
                    lookback_minutes=span_minutes,
                    end=pd.Timestamp.now(tz="UTC"),
                    provider=provider,
                )
            except Exception as exc:
                logger.debug("Intraday fetch failed provider=%s symbol=%s: %s", provider, symbol, exc)
                continue
            if bars.empty:
                continue
            # Filter to the desired bar resolution.
            if "interval_minutes" in bars.columns:
                bars = bars[bars["interval_minutes"] == bar_minutes]
            if bars.empty:
                continue
            stamps = pd.DatetimeIndex(pd.to_datetime(bars["timestamp"], utc=True))
            work = bars.copy()
            work.index = stamps
            work = work.sort_index()
            if best.empty or len(work) > len(best):
                best = work
        if not best.empty:
            intraday_history[symbol] = best[["open", "high", "low", "close", "volume"]].copy()
    return intraday_history


def _configured_history_providers(config) -> list[str]:
    providers = [
        str(provider).strip().lower()
        for provider in getattr(config, "intraday_market_data_provider_order", [])
        if str(provider).strip()
    ]
    return providers or ["yfinance"]


def _compute_backtest(strategy: str, period: str, account_id: str = "") -> dict[str, Any]:
    """Backtest by replaying the algorithm itself.

    One path for every algorithm. There is no per-strategy branch here any more: whatever the
    algorithm declares in ``requirements()`` is what the replay loads, and whatever ``analyze``
    decides is what gets traded -- so a backtest cannot test different logic than the runtime.

    Replayed against the account the strategy is deployed on, for the same reason the signal
    view is: a DCA plan is per account, so the default account backtests a different plan than
    the one the dashboard edits.
    """
    starting_equity = _backtest_starting_equity()
    config = config_for_strategy_view(strategy, account_id)
    algorithm = get_algorithm_class(strategy).from_config(config)
    # The class default, not a binding's cron: a backtest describes the strategy, and running
    # the same period against two bindings' schedules would produce two curves for one strategy
    # under a cache key that does not distinguish them.
    cron = parse_cron(algorithm.cron)
    requirements = algorithm.requirements(config, {})

    # The fetch happens outside any pooled connection. It is network I/O that writes provider
    # bars into the cache, and holding the database file open across minutes of provider calls
    # locks every other process out -- the MCP server's reads abort with "Conflicting lock is
    # held". Short-lived connections keep the file free while the fetch runs.
    daily_history = _fetch_backtest_history(strategy, period, config)
    if not daily_history:
        raise RuntimeError("No historical bars were available for the backtest.")

    # Detect intraday algorithms: those that declare intraday_lookback_minutes > 0
    # get intraday bars and the replay steps at intraday intervals instead of daily.
    intraday_minutes = 0
    intraday_history: dict[str, pd.DataFrame] | None = None
    if requirements.intraday_lookback_minutes > 0:
        intraday_minutes = requirements.preferred_bar_minutes or 15
        intraday_history = _fetch_intraday_backtest_history(strategy, period, config, daily_history)

    start = _period_start(period)
    trade_dates = sorted(set.intersection(*(set(df.index) for df in daily_history.values())))
    in_period = [date for date in trade_dates if date >= start]
    trade_dates = in_period if len(in_period) >= 2 else trade_dates[-_period_row_count(period):]
    if len(trade_dates) < 2:
        raise RuntimeError("No common trading dates were available for the backtest period.")

    with pooled_connections(read_only=True):
        history_df, coverage = replay(
            algorithm,
            config,
            daily_history=daily_history,
            trade_dates=trade_dates,
            # Date-level only: the replay steps one bar per day and trades at its close, so
            # there is no clock time for the minute and hour fields to match against.
            should_run=lambda date: cron.matches_date(date),
            starting_equity=starting_equity,
            history_providers=_configured_history_providers(config),
            intraday_history=intraday_history,
            intraday_minutes=intraday_minutes,
        )
    payload = _backtest_response(
        history_df,
        strategy=strategy,
        label=STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").title()),
        period=period,
        source="algorithm",
    )
    # Surfaced, not buried: a window the bar cache cannot reach scores every symbol near
    # zero, which would otherwise read as a poor strategy rather than an unsupported window.
    payload["coverage"] = coverage.as_dict()
    # Which deployment this curve describes, for the same reason the signal view says so.
    payload["account_id"] = getattr(config, "account_id", "")
    return payload


def backtest_payload(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    period = str(body.get("period") or _default_backtest_period()).lower()
    strategy = canonical_algorithm_id(str(body.get("strategy") or DEFAULT_STRATEGY_ID))[:80]
    refresh = bool(body.get("refresh"))
    cache_only = bool(body.get("cache_only") or body.get("cacheOnly"))
    account_id = str(body.get("account_id") or body.get("accountId") or "")[:80]
    key = _cache_key(strategy, period, account_id)
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

    payload = _compute_backtest(strategy, period, account_id)
    payload["cached"] = False
    cache["items"][key] = payload
    _save_backtest_cache(cache)
    return payload
