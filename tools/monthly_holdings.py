"""Month-by-month holdings detail for 2026."""

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

    # Scale
    history_df = history_df.copy()
    scale = starting_equity / float(history_df["equity"].iloc[0]) if float(history_df["equity"].iloc[0]) else 1.0
    for col in ["equity", "cash", "invested", "turnover"]:
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

    # Also load signals if available -- we need the scored ranks
    # The signals are stored per-run in the backtest; let's re-run analyze to get them
    from src.data.duckdb_store import pooled_connections as pc
    import src.data.cache_warmup  # noqa: F401

    # Compute per-symbol daily returns from the bars
    daily_returns = {}
    for sym in daily_history:
        df = daily_history[sym]
        daily_returns[sym] = df["close"].pct_change()

    hist = history_df.reset_index()
    hist["month"] = pd.to_datetime(hist["timestamp"]).dt.to_period("M")

    defensive = {"SGOV", "BIL", "IEF", "AGG"}

    print(f"\n{'='*100}")
    print(f"  MONTH-BY-MONTH HOLDINGS DETAIL  |  2026 YTD")
    print(f"{'='*100}")

    for month in hist["month"].unique():
        mdf = hist[hist["month"] == month]
        m_start_eq = float(mdf.iloc[0]["equity"])
        m_end_eq = float(mdf.iloc[-1]["equity"])
        m_return = m_end_eq / m_start_eq - 1.0 if m_start_eq else 0.0
        m_turnover = float(pd.to_numeric(mdf.get("turnover", pd.Series([0])), errors="coerce").fillna(0).abs().sum())

        print(f"\n{'─'*100}")
        print(f"  {month}  |  Return: {m_return:+.2%}  |  Turnover: ${m_turnover:,.0f}")
        print(f"{'─'*100}")

        # Collect all positions held during the month with their weights
        held_symbols = {}  # sym -> list of (date, weight, equity)
        for _, row in mdf.iterrows():
            pos = parse_json(row.get("positions", {}))
            eq = float(row["equity"])
            date_str = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d")
            for sym, val in pos.items():
                if sym not in defensive:
                    held_symbols.setdefault(sym, []).append((date_str, val / eq if eq else 0, eq))

        # For each held symbol, compute its return during the month
        month_dates = [pd.to_datetime(row["timestamp"]) for _, row in mdf.iterrows()]
        first_date = month_dates[0]
        last_date = month_dates[-1]

        print(f"\n  {'Symbol':>6s}  {'Sessions':>8s}  {'% of Month':>10s}  {'Avg Wt':>7s}  {'Month Return':>12s}  {'Contribution':>12s}  {'Start Wt':>8s}  {'End Wt':>8s}")
        print(f"  {'─'*95}")

        symbol_summary = []
        for sym, entries in sorted(held_symbols.items(), key=lambda x: -len(x[1])):
            sessions = len(entries)
            pct_month = sessions / len(mdf)
            avg_weight = sum(e[1] for e in entries) / len(entries)
            start_wt = entries[0][1]
            end_wt = entries[-1][1]

            # Symbol return over the month
            sym_ret = 0.0
            if sym in daily_returns:
                sr = daily_returns[sym]
                in_month = sr[(sr.index >= first_date) & (sr.index <= last_date)]
                if len(in_month) > 0:
                    sym_ret = float((1 + in_month).prod() - 1)

            contribution = avg_weight * sym_ret

            symbol_summary.append({
                "sym": sym, "sessions": sessions, "pct": pct_month,
                "avg_wt": avg_weight, "ret": sym_ret, "contrib": contribution,
                "start_wt": start_wt, "end_wt": end_wt,
            })

            print(f"  {sym:>6s}  {sessions:>8d}  {pct_month:>10.0%}  {avg_weight:>7.1%}  {sym_ret:>+12.2%}  {contribution:>+12.2%}  {start_wt:>8.1%}  {end_wt:>8.1%}")

        total_contrib = sum(s["contrib"] for s in symbol_summary)
        print(f"  {'─'*95}")
        print(f"  {'TOTAL':>6s}  {'':>8s}  {'':>10s}  {'':>7s}  {'':>12s}  {total_contrib:>+12.2%}")
        print()

        # Day-by-day detail
        print(f"  {'Date':>10s}  {'Equity':>10s}  {'Positions'}")
        for _, row in mdf.iterrows():
            date_str = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d")
            eq = float(row["equity"])
            pos = parse_json(row.get("positions", {}))
            risk = {k: v for k, v in pos.items() if k not in defensive}
            if risk:
                parts = [f"{k}:{v/eq:.0%}" for k, v in sorted(risk.items(), key=lambda x: -x[1])]
                print(f"  {date_str:>10s}  ${eq:>10,.0f}  {' / '.join(parts)}")
            else:
                cash = float(row.get("cash", 0))
                print(f"  {date_str:>10s}  ${eq:>10,.0f}  [cash {cash/eq:.0%}]")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    run()
