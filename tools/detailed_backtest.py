"""Detailed 12-month Dual Momentum backtest with per-session and monthly reports."""

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
)
from src.core.config import get_config
from src.data.duckdb_store import pooled_connections
from src.execution.replay import replay
from src.execution.metrics import calculate_performance_metrics

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def run_detailed_backtest():
    strategy = "dual_momentum"
    period = "12m"
    starting_equity = _backtest_starting_equity()
    config = get_config(strategy_id=strategy)
    algorithm = get_algorithm_class(strategy).from_config(config)
    schedule = algorithm.schedule

    print(f"{'='*90}")
    print(f"  DUAL MOMENTUM 12-MONTH BACKTEST - DETAILED REPORT")
    print(f"{'='*90}")
    print(f"  Starting equity: ${starting_equity:,.2f}")
    print(f"  Period: 12 months")
    print(f"  Strategy: Dual Momentum")
    print(f"  Max positions: {getattr(config, 'max_positions', 'N/A')}")
    print(f"{'='*90}")
    print()

    print("Fetching historical bars...")
    daily_history = _fetch_backtest_history(strategy, period, config)
    if not daily_history:
        print("ERROR: No historical bars available.")
        return

    symbols_in_universe = sorted(daily_history.keys())
    print(f"Universe: {', '.join(symbols_in_universe)}")
    print()

    start = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=12)
    trade_dates_all = sorted(set.intersection(*(set(df.index) for df in daily_history.values())))
    in_period = [date for date in trade_dates_all if date >= start]
    trade_dates = in_period if len(in_period) >= 2 else trade_dates_all[-264:]
    if len(trade_dates) < 2:
        print("ERROR: Not enough trading dates.")
        return

    print(f"Trade dates: {trade_dates[0].strftime('%Y-%m-%d')} to {trade_dates[-1].strftime('%Y-%m-%d')} ({len(trade_dates)} sessions)")
    print()

    with pooled_connections(read_only=True):
        history_df, coverage = replay(
            algorithm,
            config,
            daily_history=daily_history,
            trade_dates=trade_dates,
            should_run=lambda date: int(date.dayofweek) in schedule.weekdays,
            starting_equity=starting_equity,
            history_providers=_configured_history_providers(config),
        )

    if history_df.empty:
        print("ERROR: Backtest produced no history.")
        return

    history_df = history_df.copy()
    scale = starting_equity / float(history_df["equity"].iloc[0]) if float(history_df["equity"].iloc[0]) else 1.0
    for col in ["equity", "cash", "invested", "turnover", "dca_contributions"]:
        if col in history_df.columns:
            history_df[col] = pd.to_numeric(history_df[col], errors="coerce") * scale
    if "positions" in history_df.columns:
        history_df["positions"] = history_df["positions"].apply(
            lambda p: {s: v * scale for s, v in p.items()} if isinstance(p, dict) else {}
        )
    if "trades" in history_df.columns:
        history_df["trades"] = history_df["trades"].apply(
            lambda t: {s: v * scale for s, v in t.items()} if isinstance(t, dict) else {}
        )

    metrics = calculate_performance_metrics(history_df["equity"])

    # ---- SUMMARY ----
    starting_eq = float(history_df["equity"].iloc[0])
    ending_eq = float(history_df["equity"].iloc[-1])
    total_return = (ending_eq / starting_eq - 1) if starting_eq else 0.0
    total_orders = int(pd.to_numeric(history_df.get("order_count", pd.Series([0])), errors="coerce").fillna(0).sum())
    total_turnover = float(pd.to_numeric(history_df.get("turnover", pd.Series([0])), errors="coerce").fillna(0).abs().sum())
    order_days = int((pd.to_numeric(history_df.get("turnover", pd.Series([0])), errors="coerce").fillna(0).abs() > 0).sum())
    total_dividends = float(pd.to_numeric(history_df.get("dividend_income", pd.Series([0])), errors="coerce").fillna(0).iloc[-1])

    print(f"{'='*90}")
    print(f"  PERFORMANCE SUMMARY")
    print(f"{'='*90}")
    print(f"  Starting equity:      ${starting_eq:>12,.2f}")
    print(f"  Ending equity:        ${ending_eq:>12,.2f}")
    print(f"  Total return:         {total_return:>12.2%}")
    print(f"  CAGR:                 {metrics['cagr']:>12.2%}")
    print(f"  Max drawdown:         {metrics['max_drawdown']:>12.2%}")
    print(f"  Sharpe ratio:         {metrics['sharpe']:>12.2f}")
    print(f"  Total orders:         {total_orders:>12}")
    print(f"  Days with trades:     {order_days:>12}")
    print(f"  Total turnover:       ${total_turnover:>12,.2f}")
    print(f"  Dividend income:      ${total_dividends:>12,.2f}")
    print(f"  History coverage:     {coverage.history_ratio:>12.1%}")
    if coverage.missing_symbols:
        print(f"  Missing symbols:      {', '.join(sorted(coverage.missing_symbols))}")
    print(f"{'='*90}")
    print()

    # ---- PER-SESSION DETAIL ----
    history_df_reset = history_df.reset_index()
    history_df_reset["month"] = pd.to_datetime(history_df_reset["timestamp"]).dt.to_period("M")
    history_df_reset["date_str"] = pd.to_datetime(history_df_reset["timestamp"]).dt.strftime("%Y-%m-%d")
    history_df_reset["dow"] = pd.to_datetime(history_df_reset["timestamp"]).dt.strftime("%a")

    months = history_df_reset["month"].unique()

    for month in months:
        month_df = history_df_reset[history_df_reset["month"] == month]
        month_start_eq = float(month_df.iloc[0]["equity"])
        month_end_eq = float(month_df.iloc[-1]["equity"])
        month_return = (month_end_eq / month_start_eq - 1) if month_start_eq else 0.0
        month_turnover = float(pd.to_numeric(month_df.get("turnover", pd.Series([0])), errors="coerce").fillna(0).abs().sum())
        month_orders = int(pd.to_numeric(month_df.get("order_count", pd.Series([0])), errors="coerce").fillna(0).sum())
        month_divs = float(pd.to_numeric(month_df.get("dividends_paid", pd.Series([0])), errors="coerce").fillna(0).sum())

        print(f"{'─'*90}")
        print(f"  {month}  |  Equity: ${month_start_eq:,.2f} -> ${month_end_eq:,.2f}  |  Return: {month_return:+.2%}  |  Orders: {month_orders}  |  Turnover: ${month_turnover:,.2f}  |  Divs: ${month_divs:,.2f}")
        print(f"{'─'*90}")

        for _, row in month_df.iterrows():
            date_str = row["date_str"]
            dow = row["dow"]
            equity = float(row["equity"])
            cash = float(row.get("cash", 0))
            invested = float(row.get("invested", 0))
            order_count = int(float(row.get("order_count", 0)))
            mode = str(row.get("allocation_mode", ""))
            turnover_val = float(row.get("turnover", 0))
            trades = row.get("trades", {})
            if isinstance(trades, str):
                try:
                    trades = json.loads(trades)
                except Exception:
                    trades = {}
            positions = row.get("positions", {})
            if isinstance(positions, str):
                try:
                    positions = json.loads(positions)
                except Exception:
                    positions = {}

            print(f"    {date_str} {dow}  Equity: ${equity:>10,.2f}  Cash: ${cash:>10,.2f}  Invested: ${invested:>10,.2f}  Mode: {mode}")

            if positions:
                pos_parts = []
                for sym in sorted(positions.keys()):
                    val = positions[sym]
                    pct = val / equity if equity else 0
                    pos_parts.append(f"{sym}: ${val:,.0f} ({pct:.1%})")
                print(f"             Positions: {' | '.join(pos_parts)}")

            if order_count > 0 and trades:
                order_parts = []
                for sym in sorted(trades.keys()):
                    val = trades[sym]
                    action = "BUY" if val > 0 else "SELL"
                    order_parts.append(f"{action} {sym}: ${abs(val):,.0f}")
                print(f"             ORDERS ({order_count}): {' | '.join(order_parts)}")
            elif order_count > 0:
                print(f"             ORDERS ({order_count}): [executed but trade details unavailable]")

        # Monthly position summary
        last_row = month_df.iloc[-1]
        last_positions = last_row.get("positions", {})
        if isinstance(last_positions, str):
            try:
                last_positions = json.loads(last_positions)
            except Exception:
                last_positions = {}
        if last_positions:
            print(f"    Month-end positions:")
            for sym in sorted(last_positions.keys()):
                val = last_positions[sym]
                pct = val / float(last_row["equity"]) if float(last_row["equity"]) else 0
                print(f"      {sym:>6s}  ${val:>10,.2f}  ({pct:>6.1%} of equity)")
        else:
            print(f"    Month-end positions: [empty - all cash]")
        print()

    # ---- FINAL POSITIONS ----
    last_row = history_df.iloc[-1]
    last_positions = last_row.get("positions", {})
    if isinstance(last_positions, str):
        try:
            last_positions = json.loads(last_positions)
        except Exception:
            last_positions = {}

    print(f"{'='*90}")
    print(f"  FINAL PORTFOLIO")
    print(f"{'='*90}")
    print(f"  Equity:   ${ending_eq:>12,.2f}")
    print(f"  Cash:     ${float(last_row.get('cash', 0)):>12,.2f}")
    print(f"  Invested: ${float(last_row.get('invested', 0)):>12,.2f}")
    if last_positions:
        print(f"  Positions:")
        for sym in sorted(last_positions.keys()):
            val = last_positions[sym]
            pct = val / ending_eq if ending_eq else 0
            print(f"    {sym:>6s}  ${val:>10,.2f}  ({pct:>6.1%})")
    print(f"{'='*90}")


if __name__ == "__main__":
    run_detailed_backtest()
