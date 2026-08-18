"""Full average holdings breakdown per month -- how each position contributed."""

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

    hist = history_df.reset_index()
    hist["month"] = pd.to_datetime(hist["timestamp"]).dt.to_period("M")
    defensive = {"SGOV", "BIL", "IEF", "AGG"}

    print(f"\n{'='*110}")
    print(f"  FULL MONTHLY HOLDINGS BREAKDOWN  |  2026 YTD")
    print(f"{'='*110}")

    for month in hist["month"].unique():
        mdf = hist[hist["month"] == month]
        m_start_eq = float(mdf.iloc[0]["equity"])
        m_end_eq = float(mdf.iloc[-1]["equity"])
        m_return = m_end_eq / m_start_eq - 1.0 if m_start_eq else 0.0
        m_turnover = float(pd.to_numeric(mdf.get("turnover", pd.Series([0])), errors="coerce").fillna(0).abs().sum())

        # For each session, get weights
        month_dates = [pd.to_datetime(row["timestamp"]) for _, row in mdf.iterrows()]
        first_date = month_dates[0]
        last_date = month_dates[-1]

        # Collect per-symbol daily weight history
        sym_daily = {}  # sym -> [(date, weight, day_return)]
        for _, row in mdf.iterrows():
            pos = parse_json(row.get("positions", {}))
            eq = float(row["equity"])
            date = pd.to_datetime(row["timestamp"])
            for sym, val in pos.items():
                if sym not in defensive:
                    wt = val / eq if eq else 0
                    day_ret = 0.0
                    if sym in daily_returns:
                        sr = daily_returns[sym]
                        if date in sr.index:
                            day_ret = float(sr.loc[date])
                    sym_daily.setdefault(sym, []).append((date, wt, day_ret))

        # Calculate average weight and contribution for each symbol
        symbol_stats = []
        for sym, entries in sym_daily.items():
            n_sessions = len(entries)
            pct_of_month = n_sessions / len(mdf)
            avg_weight = sum(e[1] for e in entries) / n_sessions

            # Cumulative return for the symbol during the month
            sym_returns = [e[2] for e in entries if e[2] != 0]
            if sym_returns:
                cum_ret = 1.0
                for r in sym_returns:
                    cum_ret *= (1 + r)
                sym_return = cum_ret - 1
            else:
                sym_return = 0.0

            # Weighted contribution: avg_weight * symbol_return
            contribution = avg_weight * sym_return

            # Start and end weights
            start_wt = entries[0][1]
            end_wt = entries[-1][1]

            symbol_stats.append({
                "sym": sym,
                "sessions": n_sessions,
                "pct_month": pct_of_month,
                "avg_wt": avg_weight,
                "sym_return": sym_return,
                "contribution": contribution,
                "start_wt": start_wt,
                "end_wt": end_wt,
            })

        symbol_stats.sort(key=lambda x: -x["avg_wt"])

        total_contrib = sum(s["contribution"] for s in symbol_stats)
        total_avg_wt = sum(s["avg_wt"] for s in symbol_stats)

        print(f"\n{'─'*110}")
        print(f"  {month}  |  Portfolio Return: {m_return:+.2%}  |  Sum of Contributions: {total_contrib:+.2%}  |  Avg Invested: {total_avg_wt:.0%}")
        print(f"{'─'*110}")
        print(f"  {'Symbol':>6s}  {'Sessions':>8s}  {'% Month':>8s}  {'Avg Wt':>7s}  {'Start Wt':>8s}  {'End Wt':>8s}  {'Sym Return':>10s}  {'Contribution':>12s}  {'Weight x Return'}")
        print(f"  {'─'*105}")

        for s in symbol_stats:
            wr_str = f"{s['avg_wt']:.1%} x {s['sym_return']:+.1%} = {s['contribution']:+.2%}"
            print(f"  {s['sym']:>6s}  {s['sessions']:>8d}  {s['pct_month']:>8.0%}  {s['avg_wt']:>7.1%}  {s['start_wt']:>8.1%}  {s['end_wt']:>8.1%}  {s['sym_return']:>+10.2%}  {s['contribution']:>+12.2%}  {wr_str}")

        print(f"  {'─'*105}")
        print(f"  {'TOTAL':>6s}  {'':>8s}  {'':>8s}  {total_avg_wt:>7.1%}  {'':>8s}  {'':>8s}  {'':>10s}  {total_contrib:>+12.2%}")

        # Explain any difference between portfolio return and sum of contributions
        diff = m_return - total_contrib
        if abs(diff) > 0.001:
            print(f"\n  Note: Portfolio return ({m_return:+.2%}) vs sum of contributions ({total_contrib:+.2%})")
            print(f"  Difference ({diff:+.2%}) comes from: intra-month weight changes (compounding), cash drag, and trade timing")

        # Day-by-day snapshot
        print(f"\n  {'Date':>10s}  {'Equity':>10s}  {'Positions'}")
        for _, row in mdf.iterrows():
            date_str = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d")
            eq = float(row["equity"])
            pos = parse_json(row.get("positions", {}))
            risk = {k: v for k, v in pos.items() if k not in defensive}
            if risk:
                parts = [f"{k}:{v/eq:.0%}" for k, v in sorted(risk.items(), key=lambda x: -x[1])]
                print(f"  {date_str:>10s}  ${eq:>10,.0f}  {' / '.join(parts)}")
            else:
                print(f"  {date_str:>10s}  ${eq:>10,.0f}  [cash]")

    print(f"\n{'='*110}")


if __name__ == "__main__":
    run()
