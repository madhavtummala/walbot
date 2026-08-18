"""Rerank-by-rerank detail for 2026 with 3-day throttle."""

from __future__ import annotations

import json
import sys
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

import logging
logging.basicConfig(level=logging.WARNING)


def parse_json(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return val if isinstance(val, dict) else {}


def run():
    strategy = "rally_rotation"
    period = "2026-01-01:2026-12-31"
    starting_equity = _backtest_starting_equity()
    config = get_config(strategy_id=strategy)
    algorithm = get_algorithm_class(strategy).from_config(config)
    schedule = algorithm.schedule

    with pooled_connections(read_only=True):
        daily_history = _fetch_backtest_history(strategy, period, config)
        trade_dates_all = sorted(set.intersection(*(set(df.index) for df in daily_history.values())))
        trade_dates = [d for d in trade_dates_all if d >= pd.Timestamp("2026-01-01", tz="UTC")]

        history_df, coverage = replay(
            algorithm, config, daily_history=daily_history, trade_dates=trade_dates,
            should_run=lambda date: int(date.dayofweek) in schedule.weekdays,
            starting_equity=starting_equity,
            history_providers=_configured_history_providers(config),
        )

    if history_df.empty:
        print("ERROR: No history.")
        return

    history_df = history_df.copy()
    scale = starting_equity / float(history_df["equity"].iloc[0]) if float(history_df["equity"].iloc[0]) else 1.0
    for col in ["equity", "cash", "invested", "turnover"]:
        if col in history_df.columns:
            history_df[col] = pd.to_numeric(history_df[col], errors="coerce") * scale
    if "positions" in history_df.columns:
        history_df["positions"] = history_df["positions"].apply(
            lambda p: {s: v * scale for s, v in p.items()} if isinstance(p, dict) else {}
        )

    # Daily returns for each symbol
    daily_returns = {}
    for sym in daily_history:
        df = daily_history[sym]
        daily_returns[sym] = df["close"].pct_change()

    # Monthly returns for each symbol
    monthly_returns = {}
    for sym in daily_history:
        df = daily_history[sym].copy()
        df.index = pd.to_datetime(df.index)
        monthly = df["close"].resample("ME").last().pct_change().dropna()
        monthly_returns[sym] = monthly

    hist = history_df.reset_index()
    hist["month"] = pd.to_datetime(hist["timestamp"]).dt.to_period("M")
    defensive = {"SGOV", "BIL", "IEF", "AGG"}

    # Simulate the 3-day rerank throttle to identify rerank days
    rerank_interval = 3
    run_index = 0
    last_rerank_run = -999
    rerank_days = []  # (run_index, date, month)

    for _, row in hist.iterrows():
        run_index += 1
        if run_index - last_rerank_run >= rerank_interval:
            date = pd.to_datetime(row["timestamp"])
            month = row["month"]
            rerank_days.append((run_index, date, month))
            last_rerank_run = run_index

    print(f"\n{'='*120}")
    print(f"  RERANK-BY-RERANK DETAIL  |  2026 YTD  |  rerank_interval_days = 3")
    print(f"  Total sessions: {len(hist)}  |  Total reranks: {len(rerank_days)}")
    print(f"{'='*120}")

    current_month = None
    rerank_count_in_month = 0

    for ri, (run_idx, date, month) in enumerate(rerank_days):
        # Find the row for this date
        date_str = date.strftime("%Y-%m-%d")
        row_match = hist[hist["timestamp"] == date]
        if row_match.empty:
            # Try nearest date
            row_match = hist.iloc[[min(ri, len(hist)-1)]]
        row = row_match.iloc[0]

        eq = float(row["equity"])
        pos = parse_json(row.get("positions", {}))
        risk = {k: v for k, v in pos.items() if k not in defensive}

        # Get scores from signals (we need to re-compute or approximate)
        # Since signals aren't stored, we'll use the positions and returns
        # Get the next few days' returns to see how the picks did
        row_idx = hist.index.get_loc(row.name) if row.name in hist.index else None

        # Find returns for held positions over next 3 sessions (until next rerank)
        next_returns = {}
        if row_idx is not None:
            for sym in risk:
                if sym in daily_returns:
                    future_rets = []
                    for offset in range(1, 4):
                        future_idx = row_idx + offset
                        if future_idx < len(hist):
                            future_date = pd.to_datetime(hist.iloc[future_idx]["timestamp"])
                            if future_date in daily_returns[sym].index:
                                ret = daily_returns[sym].loc[future_date]
                                if pd.notna(ret):
                                    future_rets.append(float(ret))
                    if future_rets:
                        cum = 1.0
                        for r in future_rets:
                            cum *= (1 + r)
                        next_returns[sym] = cum - 1

        # Month change
        if month != current_month:
            if current_month is not None:
                print()
            current_month = month
            rerank_count_in_month = 0

            # Get month return so far
            month_rows = hist[hist["month"] == month]
            m_start = float(month_rows.iloc[0]["equity"])
            m_end = float(month_rows.iloc[-1]["equity"])
            m_return = m_end / m_start - 1.0 if m_start else 0.0

            print(f"\n{'─'*120}")
            print(f"  {month}  |  Month Return: {m_return:+.2%}  |  Sessions: {len(month_rows)}")
            print(f"{'─'*120}")

        rerank_count_in_month += 1

        # Get symbol returns for the month so far
        month_rows = hist[hist["month"] == month]
        month_start_eq = float(month_rows.iloc[0]["equity"])

        # Format positions
        pos_str = " / ".join(f"{k}:{v/eq:.0%}" for k, v in sorted(risk.items(), key=lambda x: -x[1])) if risk else "[cash]"

        # Format next returns
        ret_str = ""
        if next_returns:
            ret_parts = [f"{s}:{r:+.1%}" for s, r in sorted(next_returns.items(), key=lambda x: -x[1])]
            ret_str = f"  3-day: {' / '.join(ret_parts)}"

        print(f"\n  Rerank #{rerank_count_in_month} | Run {run_idx:>3d} | {date_str} | Equity: ${eq:>10,.0f}")
        print(f"  Holdings: {pos_str}{ret_str}")

        # Show contribution since last rerank
        if ri > 0:
            prev_run_idx, prev_date, _ = rerank_days[ri - 1]
            prev_row = hist[hist["timestamp"] == prev_date]
            if not prev_row.empty:
                prev_eq = float(prev_row.iloc[0]["equity"])
                period_return = eq / prev_eq - 1.0 if prev_eq else 0.0
                print(f"  Since last rerank: {period_return:+.2%}")

    # Final summary
    print(f"\n{'='*120}")
    print(f"  SUMMARY: {len(rerank_days)} reranks over {len(hist)} sessions")
    print(f"  Avg positions per rerank: {sum(len(parse_json(r.get('positions', {}))) for _, r in hist.iterrows()) / len(hist):.1f}")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    run()
