"""Monthly backtest report: SPY return, algo return, reranks, and holdings with weightages.

    python -m tools.backtest_monthly_report
    python -m tools.backtest_monthly_report --period 2026-01-01:2026-02-28
    python -m tools.backtest_monthly_report --period 2026-02-01:2026-02-28 --set rerank_interval_days=3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.rally_rotation.config import RallyRotationConfig
from tools.attribution import _parse_overrides
from tools.config_sweep import Sweep, deployed_tuning

logger = logging.getLogger(__name__)


def parse_json(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return val if isinstance(val, dict) else {}


def run(period: str, overrides: dict[str, Any]):
    tuning = {**deployed_tuning("rally_rotation"), **overrides}
    rerank_interval = tuning.get("rerank_interval_days", 0)

    sweep = Sweep([period])
    run_result = sweep.run("rally_rotation", "deployed", tuning, period)
    curve, _coverage = sweep.last_curve

    if curve.empty:
        print("ERROR: No history.")
        return

    # Load SPY bars directly from DuckDB (avoids lookback_days limitation)
    from src.data.duckdb_store import pooled_connections, read_bars

    with pooled_connections(read_only=True):
        spy_raw = read_bars("SPY", interval_minutes=1440, limit=10000)

    spy_bars = spy_raw.set_index("timestamp")["close"].astype(float)
    spy_bars.index = pd.to_datetime(spy_bars.index, utc=True)
    spy_bars = spy_bars.sort_index()

    hist = curve.reset_index()
    hist["month"] = pd.to_datetime(hist["timestamp"]).dt.to_period("M")

    defensive = {s.upper() for s in (tuning.get("defensive_universe") or [])}

    # Identify rerank days based on the interval
    run_index = 0
    last_rerank_run = -999
    rerank_dates = set()
    for _, row in hist.iterrows():
        run_index += 1
        if rerank_interval == 0 or run_index - last_rerank_run >= rerank_interval:
            rerank_dates.add(pd.to_datetime(row["timestamp"]))
            last_rerank_run = run_index

    # ---- Header ----
    first_date = pd.to_datetime(hist.iloc[0]["timestamp"]).strftime("%Y-%m-%d")
    last_date = pd.to_datetime(hist.iloc[-1]["timestamp"]).strftime("%Y-%m-%d")
    starting_eq = float(hist.iloc[0]["equity"])
    ending_eq = float(hist.iloc[-1]["equity"])

    print(f"\n{'='*110}")
    print(f"  MONTHLY BACKTEST REPORT  |  Rally Rotation  |  {first_date} to {last_date}")
    print(f"  Starting equity: ${starting_eq:,.0f}  |  Ending equity: ${ending_eq:,.0f}  |  Total return: {ending_eq/starting_eq - 1:+.2%}")
    print(f"  Rerank interval: {'every session' if rerank_interval == 0 else f'every {rerank_interval} sessions'}")
    if overrides:
        print(f"  Overrides: {', '.join(f'{k}={v}' for k, v in overrides.items())}")
    print(f"{'='*110}")

    months = hist["month"].unique()

    for month in months:
        mdf = hist[hist["month"] == month]
        m_start_eq = float(mdf.iloc[0]["equity"])
        m_end_eq = float(mdf.iloc[-1]["equity"])
        m_return = m_end_eq / m_start_eq - 1.0 if m_start_eq else 0.0

        # SPY return for this month
        m_dates = [pd.to_datetime(row["timestamp"]) for _, row in mdf.iterrows()]
        spy_first = float(spy_bars.loc[m_dates[0]] if m_dates[0] in spy_bars.index else spy_bars.asof(m_dates[0]))
        spy_last = float(spy_bars.loc[m_dates[-1]] if m_dates[-1] in spy_bars.index else spy_bars.asof(m_dates[-1]))
        spy_return = spy_last / spy_first - 1.0 if spy_first else 0.0

        # Reranks in this month
        m_reranks = [d for d in m_dates if d in rerank_dates]

        # Collect all positions with weights during the month
        held = {}  # sym -> [(date, weight)]
        for _, row in mdf.iterrows():
            pos = parse_json(row.get("positions", {}))
            eq = float(row["equity"])
            date_str = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d")
            for sym, val in pos.items():
                if sym not in defensive:
                    held.setdefault(sym, []).append((date_str, val / eq if eq else 0))

        # Month-end positions with weights
        last_row = mdf.iloc[-1]
        end_pos = parse_json(last_row.get("positions", {}))
        end_eq = float(last_row["equity"])
        end_weights = {s: v / end_eq for s, v in end_pos.items() if s not in defensive and end_eq}
        cash_pct = 1.0 - sum(v for v in end_weights.values())

        # ---- Month block ----
        print(f"\n{'─'*110}")
        print(f"  {month}  |  SPY: {spy_return:+.2%}  |  Algo: {m_return:+.2%}  |  Alpha: {m_return - spy_return:+.2%}  |  Sessions: {len(mdf)}  |  Reranks: {len(m_reranks)}")
        print(f"{'─'*110}")

        # Rerank dates
        if m_reranks:
            rerank_strs = [d.strftime("%m-%d") for d in m_reranks]
            print(f"  Rerank days: {', '.join(rerank_strs)}")

        # Holdings table
        if held:
            print(f"\n  {'Symbol':>6s}  {'Sessions':>8s}  {'% Month':>8s}  {'Avg Wt':>7s}  {'Start Wt':>9s}  {'End Wt':>9s}")
            print(f"  {'─'*52}")
            for sym, entries in sorted(held.items(), key=lambda x: -len(x[1])):
                sessions = len(entries)
                pct_month = sessions / len(mdf)
                avg_wt = sum(e[1] for e in entries) / len(entries)
                start_wt = entries[0][1]
                end_wt = entries[-1][1]
                print(f"  {sym:>6s}  {sessions:>8d}  {pct_month:>8.0%}  {avg_wt:>7.1%}  {start_wt:>9.1%}  {end_wt:>9.1%}")
        else:
            print(f"\n  [no risk-on holdings this month]")

        # Month-end book
        print(f"\n  Month-end book:")
        if end_weights:
            for sym, w in sorted(end_weights.items(), key=lambda x: -x[1]):
                print(f"    {sym:>6s}  {w:>7.1%}")
        print(f"    {'Cash':>6s}  {cash_pct:>7.1%}")

    # ---- YTD summary ----
    all_dates = [pd.to_datetime(row["timestamp"]) for _, row in hist.iterrows()]
    total_spy_first = float(spy_bars.loc[all_dates[0]] if all_dates[0] in spy_bars.index else spy_bars.asof(all_dates[0]))
    total_spy_last = float(spy_bars.loc[all_dates[-1]] if all_dates[-1] in spy_bars.index else spy_bars.asof(all_dates[-1]))
    total_spy = total_spy_last / total_spy_first - 1.0
    total_algo = ending_eq / starting_eq - 1.0

    print(f"\n{'='*110}")
    print(f"  YTD SUMMARY")
    print(f"{'='*110}")
    print(f"  SPY:    {total_spy:+.2%}")
    print(f"  Algo:   {total_algo:+.2%}")
    print(f"  Alpha:  {total_algo - total_spy:+.2%}")
    print(f"  Total return (net): {run_result.metrics['net_return_5bps']:+.2%}  (at 5bps cost)")
    print(f"  Turnover: {run_result.metrics['turnover_x_stake']:.1f}x  |  Max drawdown: {run_result.metrics['max_drawdown']:+.2%}")
    print(f"{'='*110}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="2026-01-01:2026-12-31")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("src.core.orders", "src.brokerages.paper.brokerage",
                  "src.algorithms.rally_rotation.algorithm", "src.data.provider_cache",
                  "src.connectors"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    overrides = _parse_overrides(args.overrides)
    run(args.period, overrides)


if __name__ == "__main__":
    main()
