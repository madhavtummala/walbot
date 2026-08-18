"""Full daily table for a month: SPY, algo, positions with weights, trades.

    python -m tools.full_daily_table --period 2026-01-01:2026-01-31
    python -m tools.full_daily_table --period 2026-02-01:2026-02-28
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.config_sweep import Sweep, deployed_tuning
from tools.attribution import _parse_overrides


def parse_json(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return val if isinstance(val, dict) else {}


def run(period: str):
    tuning = deployed_tuning("rally_rotation")
    defensive = {s.upper() for s in (tuning.get("defensive_universe") or [])}

    sweep = Sweep([period])
    result = sweep.run("rally_rotation", "deployed", tuning, period)
    curve, _ = sweep.last_curve

    if curve.empty:
        print("ERROR: No history.")
        return

    # SPY bars aligned to the same dates as the curve
    from src.core.config import get_config
    from src.api.payloads.backtest import _fetch_backtest_history
    from src.data.duckdb_store import pooled_connections

    base_config = get_config()
    with pooled_connections(read_only=True):
        daily_history = _fetch_backtest_history("rally_rotation", period, base_config)
    spy_bars = daily_history["SPY"]["close"].astype(float)
    # Align spy to the curve's timestamps
    curve_dates = [pd.to_datetime(ts) for ts in curve.index]
    spy_aligned = pd.Series(
        [float(spy_bars.asof(d)) for d in curve_dates],
        index=curve.index,
    )

    hist = curve.reset_index()

    # ---- Header ----
    first_date = pd.to_datetime(hist.iloc[0]["timestamp"]).strftime("%Y-%m-%d")
    last_date = pd.to_datetime(hist.iloc[-1]["timestamp"]).strftime("%Y-%m-%d")
    starting_eq = float(hist.iloc[0]["equity"])
    ending_eq = float(hist.iloc[-1]["equity"])
    spy_start = float(spy_aligned.iloc[0])

    print(f"\n  DUAL MOMENTUM DAILY TABLE  |  {first_date} to {last_date}  |  Starting: ${starting_eq:,.0f}")
    print(f"  {'':->96s}")

    # Column headers
    print(f"  {'Date':10s} {'Dow':4s} {'Equity':>11s} {'Algo Ret':>9s} {'SPY Ret':>9s} {'Book'}")

    prev_eq = starting_eq
    prev_spy = spy_start
    prev_positions = {}

    for idx, (_, row) in enumerate(hist.iterrows()):
        date = pd.to_datetime(row["timestamp"])
        date_str = date.strftime("%Y-%m-%d")
        dow = date.strftime("%a")
        eq = float(row["equity"])
        pos = parse_json(row.get("positions", {}))
        risk = {k: v for k, v in pos.items() if k not in defensive}

        # Returns
        algo_ret = eq / prev_eq - 1.0 if prev_eq else 0
        spy_close = float(spy_aligned.iloc[idx]) if idx < len(spy_aligned) else float(spy_bars.asof(date))
        spy_ret = spy_close / prev_spy - 1.0 if prev_spy else 0

        # Book string
        if risk:
            parts = []
            for sym, val in sorted(risk.items(), key=lambda x: -x[1]):
                pct = val / eq if eq else 0
                parts.append(f"{sym} {pct:.0%}")
            book = " / ".join(parts)
        else:
            book = "[cash]"

        # Change marker
        all_syms = set(list(prev_positions.keys()) + list(risk.keys()))
        changed = any(
            abs(prev_positions.get(s, 0) - risk.get(s, 0)) > 0.001
            for s in all_syms
        ) if prev_positions else True
        marker = " *" if changed else "  "

        print(f"  {date_str} {dow:3s}  ${eq:>10,.0f} {algo_ret:>+9.2%} {spy_ret:>+9.2%}  {book}{marker}")

        prev_eq = eq
        prev_spy = spy_close
        prev_positions = risk

    # Summary
    total_algo = ending_eq / starting_eq - 1.0
    total_spy = float(spy_aligned.iloc[-1]) / spy_start - 1.0
    print(f"  {'':->96s}")
    print(f"  {'TOTAL':10s} {'':4s}  ${ending_eq:>10,.0f} {total_algo:>+9.2%} {total_spy:>+9.2%}")
    print(f"\n  * = holdings changed from previous day")
    print(f"  Reranks: {len(hist)} (every session)")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="2026-01-01:2026-01-31")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    run(args.period)


if __name__ == "__main__":
    main()
