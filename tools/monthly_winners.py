"""Monthly winners for 2026 -- what the score picked vs what actually won."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.payloads.backtest import _fetch_backtest_history
from src.core.config import get_config
from src.data.duckdb_store import pooled_connections

import logging
logging.basicConfig(level=logging.WARNING)


def run():
    strategy = "rally_rotation"
    period = "2026-01-01:2026-12-31"
    config = get_config(strategy_id=strategy)

    with pooled_connections(read_only=True):
        daily_history = _fetch_backtest_history(strategy, period, config)

    # Compute monthly returns for each symbol
    risk_on = ["QQQM", "XSD", "XBI", "XOP", "KRE", "XRT", "IEMG", "EWJ", "VGK", "GLD", "USO", "SPY", "IWM", "SCHD", "SLV"]

    monthly_returns = {}
    for sym in risk_on:
        if sym not in daily_history:
            continue
        df = daily_history[sym].copy()
        df.index = pd.to_datetime(df.index)
        monthly = df["close"].resample("ME").last().pct_change().dropna()
        monthly_returns[sym] = monthly

    # Build monthly return matrix
    ret_df = pd.DataFrame(monthly_returns)
    ret_df.index = ret_df.index.to_period("M")

    # Filter to 2026
    ret_df = ret_df[ret_df.index >= "2026-01"]

    print(f"\n{'='*100}")
    print(f"  MONTHLY ETF RETURNS  |  2026 YTD")
    print(f"{'='*100}")
    print()

    # Print header
    syms = sorted(ret_df.columns)
    print(f"  {'Month':>8s}", end="")
    for s in syms:
        print(f"  {s:>6s}", end="")
    print()
    print(f"  {'─' * (8 + len(syms) * 8)}")

    for month in ret_df.index:
        print(f"  {str(month):>8s}", end="")
        for s in syms:
            val = ret_df.loc[month, s]
            if pd.notna(val):
                marker = " *" if val == ret_df.loc[month].max() else ""
                print(f"  {val:>+5.1%}{marker}", end="")
            else:
                print(f"  {'--':>6s}", end="")
        print()

    # Now show top 5 per month
    print(f"\n{'='*100}")
    print(f"  TOP 5 WINNERS EACH MONTH  |  2026 YTD")
    print(f"{'='*100}")

    for month in ret_df.index:
        row = ret_df.loc[month].dropna().sort_values(ascending=False)
        top5 = row.head(5)
        print(f"\n  {month}")
        print(f"  {'─'*50}")
        for rank, (sym, ret) in enumerate(top5.items(), 1):
            bar = "█" * max(1, int(ret * 200)) if ret > 0 else "░" * max(1, int(abs(ret) * 200))
            print(f"  {rank}. {sym:>5s}  {ret:>+7.2%}  {bar}")

    # What the portfolio held vs what won
    print(f"\n{'='*100}")
    print(f"  PORTFOLIO HOLDINGS vs MONTHLY WINNERS")
    print(f"{'='*100}")

    # Load backtest results
    from src.algorithms.registry import get_algorithm_class
    from src.api.payloads.backtest import (
        _backtest_starting_equity,
        _configured_history_providers,
    )
    from src.execution.replay import replay

    strategy = "rally_rotation"
    starting_equity = _backtest_starting_equity()
    algorithm = get_algorithm_class(strategy).from_config(config)
    schedule = algorithm.schedule

    trade_dates_all = sorted(set.intersection(*(set(df.index) for df in daily_history.values())))
    trade_dates = [d for d in trade_dates_all if d >= pd.Timestamp("2026-01-01", tz="UTC")]

    with pooled_connections(read_only=True):
        history_df, _ = replay(
            algorithm, config, daily_history=daily_history, trade_dates=trade_dates,
            should_run=lambda date: int(date.dayofweek) in schedule.weekdays,
            starting_equity=starting_equity,
            history_providers=_configured_history_providers(config),
        )

    history_df = history_df.copy()
    scale = starting_equity / float(history_df["equity"].iloc[0]) if float(history_df["equity"].iloc[0]) else 1.0
    if "positions" in history_df.columns:
        history_df["positions"] = history_df["positions"].apply(
            lambda p: {s: v * scale for s, v in p.items()} if isinstance(p, dict) else {}
        )

    hist = history_df.reset_index()
    hist["month"] = pd.to_datetime(hist["timestamp"]).dt.to_period("M")

    defensive = {"SGOV", "BIL", "IEF", "AGG"}

    for month in hist["month"].unique():
        if str(month) not in [str(p) for p in ret_df.index]:
            continue

        mdf = hist[hist["month"] == month]

        # Get all symbols held during the month
        held = set()
        for _, row in mdf.iterrows():
            pos = row.get("positions", {})
            if isinstance(pos, str):
                try:
                    pos = json.loads(pos)
                except:
                    pos = {}
            for sym in pos:
                if sym not in defensive:
                    held.add(sym)

        # Get month returns for held symbols
        month_row = ret_df.loc[month]
        held_returns = {s: month_row[s] for s in held if s in month_row and pd.notna(month_row[s])}

        # Get actual month winner
        actual_winner = month_row.dropna().idxmax()
        actual_winner_ret = month_row.dropna().max()

        # Did we hold the winner?
        held_winner = actual_winner in held

        print(f"\n  {month}")
        print(f"  {'─'*60}")
        print(f"  Actual winner: {actual_winner} ({actual_winner_ret:+.2%})")
        print(f"  Held winner:   {'YES' if held_winner else 'NO'}")
        print(f"  Symbols held:  {', '.join(sorted(held))}")
        if held_returns:
            print(f"  Held returns:  ", end="")
            for s, r in sorted(held_returns.items(), key=lambda x: -x[1]):
                print(f"{s}:{r:+.1%} ", end="")
            print()

    print(f"\n{'='*100}")


if __name__ == "__main__":
    run()
