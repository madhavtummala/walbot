"""Compare DCA vs Bursty DCA with $35k in SGOV."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.registry import get_algorithm_class
from src.api.payloads.backtest import (
    _backtest_starting_equity,
    _fetch_backtest_history,
    _configured_history_providers,
    _period_start,
    _period_label,
    _period_months,
    _period_row_count,
)
from src.api.payloads.strategy_config import config_for_strategy_view
from src.core.config import get_config
from src.data.duckdb_store import pooled_connections
from src.execution.replay import replay
from src.execution.metrics import calculate_performance_metrics

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def run_backtest(strategy: str, period: str, starting_equity: float) -> dict:
    config = config_for_strategy_view(strategy, "")
    algorithm = get_algorithm_class(strategy).from_config(config)
    schedule = algorithm.schedule

    daily_history = _fetch_backtest_history(strategy, period, config)
    if not daily_history:
        raise RuntimeError(f"No bars for {strategy}")

    # Ensure SGOV bars are available for open_in="SGOV"
    if "SGOV" not in daily_history:
        from src.data import fetch_daily_bars
        from src.brokerages.alpaca_client import create_data_client
        from src.core.strategy_models import prepared_strategy_frame
        cfg = config
        sgov_bars = fetch_daily_bars(
            ["SGOV"], config=cfg,
            lookback_days=int(algorithm.requirements(cfg, {}).daily_lookback_days or cfg.momentum_lookback_days),
            ma_days=int(algorithm.requirements(cfg, {}).daily_ma_days or 0),
            extra_buffer_days=int(algorithm.requirements(cfg, {}).daily_extra_buffer_days or 0) + _period_row_count(period) + 10,
            data_client=create_data_client(cfg),
            include_latest=True,
        )
        for sym, frame in sgov_bars.items():
            work = prepared_strategy_frame(frame)
            if not work.empty:
                work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
                daily_history[sym] = work.sort_values("timestamp").set_index("timestamp")

    start = _period_start(period)
    trade_dates = sorted(set.intersection(*(set(df.index) for df in daily_history.values())))
    in_period = [d for d in trade_dates if d >= start]
    trade_dates = in_period if len(in_period) >= 2 else trade_dates[-_period_row_count(period):]

    with pooled_connections(read_only=True):
        history_df, coverage = replay(
            algorithm,
            config,
            daily_history=daily_history,
            trade_dates=trade_dates,
            should_run=lambda date: int(date.dayofweek) in schedule.weekdays,
            starting_equity=starting_equity,
            history_providers=_configured_history_providers(config),
            open_in="SGOV",
        )

    if history_df.empty:
        raise RuntimeError(f"Empty history for {strategy}")

    # Scale to starting equity
    history_df = history_df.copy()
    scale = starting_equity / float(history_df["equity"].iloc[0]) if float(history_df["equity"].iloc[0]) else 1.0
    for col in ["equity", "cash", "invested", "turnover", "dca_contributions"]:
        if col in history_df.columns:
            history_df[col] = pd.to_numeric(history_df[col], errors="coerce") * scale
    if "positions" in history_df.columns:
        history_df["positions"] = history_df["positions"].apply(
            lambda p: {s: v * scale for s, v in p.items()} if isinstance(p, dict) else {}
        )

    metrics = calculate_performance_metrics(history_df["equity"])
    starting_eq = float(history_df["equity"].iloc[0])
    ending_eq = float(history_df["equity"].iloc[-1])
    total_return = (ending_eq / starting_eq - 1) if starting_eq else 0.0
    total_orders = int(pd.to_numeric(history_df.get("order_count", pd.Series([0])), errors="coerce").fillna(0).sum())
    total_dividends = float(pd.to_numeric(history_df.get("dividend_income", pd.Series([0])), errors="coerce").fillna(0).iloc[-1])

    return {
        "strategy": strategy,
        "period": period,
        "starting_equity": starting_eq,
        "ending_equity": ending_eq,
        "total_return": total_return,
        "cagr": metrics["cagr"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe": metrics["sharpe"],
        "total_orders": total_orders,
        "dividends": total_dividends,
        "coverage": coverage.history_ratio,
    }


def main():
    starting_equity = 35_000.0
    period = "12m"

    print(f"\n{'='*70}")
    print(f"  DCA vs BURSTY DCA — $35k SGOV — {period.upper()} backtest")
    print(f"{'='*70}\n")

    results = {}
    for strategy in ["bursty_dca"]:
        print(f"Running {strategy}...")
        try:
            results[strategy] = run_backtest(strategy, period, starting_equity)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("No results.")
        return

    print(f"\n{'='*70}")
    print(f"  RESULTS COMPARISON")
    print(f"{'='*70}")
    header = f"{'Metric':<25}"
    for s in results:
        header += f"  {s.upper():>15}"
    print(header)
    print(f"{'-'*70}")

    rows = [
        ("Starting Equity", "starting_equity", "${:,.0f}"),
        ("Ending Equity", "ending_equity", "${:,.2f}"),
        ("Total Return", "total_return", "{:.2%}"),
        ("CAGR", "cagr", "{:.2%}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Sharpe Ratio", "sharpe", "{:.2f}"),
        ("Total Orders", "total_orders", "{:,}"),
        ("Dividends", "dividends", "${:,.2f}"),
        ("Data Coverage", "coverage", "{:.1%}"),
    ]

    for label, key, fmt in rows:
        line = f"{label:<25}"
        for s in results:
            val = results[s].get(key, 0)
            line += f"  {fmt.format(val):>15}"
        print(line)

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
