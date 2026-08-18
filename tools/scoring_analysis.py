"""Quick 2026 YTD backtest analysis: holdings, weights, turnover, SPY comparison."""

from __future__ import annotations

import json
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

import logging
logging.basicConfig(level=logging.WARNING)


def run():
    strategy = "rally_rotation"
    period = "2026-01-01:2026-12-31"
    starting_equity = _backtest_starting_equity()
    config = get_config(strategy_id=strategy)
    algorithm = get_algorithm_class(strategy).from_config(config)
    schedule = algorithm.schedule

    with pooled_connections(read_only=True):
        daily_history = _fetch_backtest_history(strategy, period, config)
        if not daily_history:
            print("ERROR: No historical bars available.")
            return

        trade_dates_all = sorted(set.intersection(*(set(df.index) for df in daily_history.values())))
        in_period = [d for d in trade_dates_all if d >= pd.Timestamp("2026-01-01", tz="UTC")]
        trade_dates = in_period if len(in_period) >= 2 else trade_dates_all[-264:]

        history_df, coverage = replay(
            algorithm, config, daily_history=daily_history, trade_dates=trade_dates,
            should_run=lambda date: int(date.dayofweek) in schedule.weekdays,
            starting_equity=starting_equity,
            history_providers=_configured_history_providers(config),
        )

        # SPY buy-and-hold
        spy_algo_class = get_algorithm_class("dca")
        spy_config = get_config(strategy_id="dca")

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

    # SPY return from same dates
    spy_start = float(daily_history["SPY"].loc[trade_dates[0], "close"])
    spy_end = float(daily_history["SPY"].loc[trade_dates[-1], "close"])
    spy_return = spy_end / spy_start - 1.0
    spy_bars = daily_history["SPY"].loc[trade_dates, "close"].astype(float)
    spy_daily_ret = spy_bars.pct_change().dropna()
    spy_sharpe = float(spy_daily_ret.mean() / spy_daily_ret.std() * (252**0.5)) if spy_daily_ret.std() > 0 else 0.0
    spy_peak = spy_bars.cummax()
    spy_dd = ((spy_bars - spy_peak) / spy_peak).min()

    # ---- Strategy metrics ----
    metrics = calculate_performance_metrics(history_df["equity"])
    starting_eq = float(history_df["equity"].iloc[0])
    ending_eq = float(history_df["equity"].iloc[-1])
    total_return = ending_eq / starting_eq - 1.0
    total_turnover = float(pd.to_numeric(history_df.get("turnover", pd.Series([0])), errors="coerce").fillna(0).abs().sum())
    total_orders = int(pd.to_numeric(history_df.get("order_count", pd.Series([0])), errors="coerce").fillna(0).sum())
    days_traded = int((pd.to_numeric(history_df.get("turnover", pd.Series([0])), errors="coerce").fillna(0).abs() > 0).sum())

    # ---- Per-month analysis ----
    hist = history_df.reset_index()
    hist["month"] = pd.to_datetime(hist["timestamp"]).dt.to_period("M")

    print(f"\n{'='*95}")
    print(f"  DUAL MOMENTUM vs SPY  |  2026 YTD  |  {trade_dates[0].strftime('%Y-%m-%d')} to {trade_dates[-1].strftime('%Y-%m-%d')}")
    print(f"{'='*95}")

    print(f"\n{'─── PERFORMANCE ───':─<95}")
    print(f"  {'':20s} {'Rally Rotation':>15s} {'SPY (buy-hold)':>15s}")
    print(f"  {'Total return':20s} {total_return:>15.2%} {spy_return:>15.2%}")
    print(f"  {'CAGR':20s} {metrics['cagr']:>15.2%} {'N/A':>15s}")
    print(f"  {'Max drawdown':20s} {metrics['max_drawdown']:>15.2%} {spy_dd:>15.2%}")
    print(f"  {'Sharpe':20s} {metrics['sharpe']:>15.2f} {spy_sharpe:>15.2f}")
    print(f"  {'Turnover ($)':20s} ${total_turnover:>14,.0f} {'--':>15s}")
    print(f"  {'Total orders':20s} {total_orders:>15d} {'--':>15s}")
    print(f"  {'Days with trades':20s} {days_traded:>15d} {'--':>15s}")

    print(f"\n{'─── MONTHLY BREAKDOWN ───':─<95}")
    print(f"  {'Month':10s} {'Return':>8s} {'# Holdings':>11s} {'# Changed':>10s} {'Turnover':>12s} {'Avg Weight':>11s} {'Top Holding':>14s}")

    monthly_stats = []
    prev_positions = {}

    for month in hist["month"].unique():
        mdf = hist[hist["month"] == month]
        m_start = float(mdf.iloc[0]["equity"])
        m_end = float(mdf.iloc[-1]["equity"])
        m_return = m_end / m_start - 1.0 if m_start else 0.0
        m_turnover = float(pd.to_numeric(mdf.get("turnover", pd.Series([0])), errors="coerce").fillna(0).abs().sum())

        # Positions at month-end
        last_pos = mdf.iloc[-1].get("positions", {})
        if isinstance(last_pos, str):
            try:
                last_pos = json.loads(last_pos)
            except Exception:
                last_pos = {}
        risk_on = {k: v for k, v in last_pos.items() if k not in ("SGOV", "BIL", "IEF", "AGG")}
        num_holdings = len(risk_on)

        # How many names changed vs prev month
        changed = 0
        if prev_positions:
            all_names = set(list(prev_positions.keys()) + list(risk_on.keys()))
            for name in all_names:
                old_w = prev_positions.get(name, 0.0)
                new_w = risk_on.get(name, 0.0)
                if abs(new_w - old_w) > 0.001:
                    changed += 1
        else:
            changed = num_holdings

        # Avg weight among held
        avg_w = (sum(risk_on.values()) / num_holdings / m_end) if num_holdings and m_end else 0.0

        # Top holding
        top = max(risk_on.items(), key=lambda x: x[1]) if risk_on else ("--", 0)
        top_name = top[0]
        top_pct = top[1] / m_end if m_end else 0

        print(f"  {str(month):10s} {m_return:>+8.2%} {num_holdings:>11d} {changed:>10d} ${m_turnover:>11,.0f} {avg_w:>11.1%} {top_name:>8s} ({top_pct:.0%})")

        monthly_stats.append({
            "month": str(month), "return": m_return, "num_holdings": num_holdings,
            "changed": changed, "turnover": m_turnover, "avg_weight": avg_w,
            "top": top_name, "top_pct": top_pct,
        })
        prev_positions = risk_on

    # ---- Aggregate stats ----
    avg_holdings = sum(s["num_holdings"] for s in monthly_stats) / len(monthly_stats) if monthly_stats else 0
    avg_changed = sum(s["changed"] for s in monthly_stats) / len(monthly_stats) if monthly_stats else 0
    avg_turnover = sum(s["turnover"] for s in monthly_stats) / len(monthly_stats) if monthly_stats else 0
    pct_months_positive = sum(1 for s in monthly_stats if s["return"] > 0) / len(monthly_stats) if monthly_stats else 0

    print(f"\n{'─── AGGREGATE STATS ───':─<95}")
    print(f"  Avg holdings per month:     {avg_holdings:.1f}")
    print(f"  Avg names changed/month:    {avg_changed:.1f}")
    print(f"  Avg turnover/month:         ${avg_turnover:,.0f}")
    print(f"  Months with trades:         {days_traded} / {len(trade_dates)} sessions")
    print(f"  Pct months positive:        {pct_months_positive:.0%}")
    print(f"{'='*95}\n")

    # ---- All-time holdings composition ----
    all_positions = {}
    for _, row in hist.iterrows():
        pos = row.get("positions", {})
        if isinstance(pos, str):
            try:
                pos = json.loads(pos)
            except Exception:
                pos = {}
        for sym, val in pos.items():
            if sym not in ("SGOV", "BIL", "IEF", "AGG"):
                all_positions.setdefault(sym, 0)
                all_positions[sym] += 1

    total_days = len(hist)
    if all_positions:
        print(f"{'─── HOLDINGS FREQUENCY (% of sessions held) ───':─<95}")
        for sym, count in sorted(all_positions.items(), key=lambda x: -x[1]):
            print(f"  {sym:>6s}  {count/total_days:>6.1%}  {'#' * int(count/total_days * 40)}")
        print()


if __name__ == "__main__":
    run()
