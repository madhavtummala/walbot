"""Replay one configuration and say what it actually held, and when.

A sweep ranks configurations; it does not tell you *why* one won. A twenty-point gap that turns
out to be one symbol held for six weeks is a different fact from one spread across the book, and
only the second is a reason to change the deployed config.

    python -m tools.explain_run --algorithm rally_rotation --universe wide --period 12m
    python -m tools.explain_run --algorithm rally_rotation --from data/config_sweep_12m.csv --label universe=wide
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import pandas as pd

from src.data.duckdb_store import pooled_connections
from tools.config_sweep import UNIVERSES, Sweep, deployed_tuning

logger = logging.getLogger(__name__)


def tuning_from(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.axes_results:
        with open(args.axes_results.replace(".csv", ".json"), encoding="utf-8") as handle:
            rows = json.load(handle)
        for row in rows:
            if row["algorithm"] == args.algorithm and row["label"] == args.label:
                return row["label"], row["tuning"]
        raise SystemExit(f"no {args.algorithm} row labelled {args.label!r} in {args.axes_results}")

    tuning = deployed_tuning(args.algorithm)
    if args.universe:
        tuning = {**tuning, "risk_on_universe": list(UNIVERSES[args.universe])}
    return (args.universe or "deployed"), tuning


def report(curve: pd.DataFrame, starting_equity: float) -> None:
    """Time in the book and dollar-days per symbol, plus the equity path's shape."""
    held: dict[str, list[float]] = {}
    for positions in curve["positions"]:
        row = positions if isinstance(positions, dict) else {}
        for symbol in set(row) | set(held):
            held.setdefault(symbol, []).append(float(row.get(symbol, 0.0)))

    dates = len(curve)
    rows = []
    for symbol, values in held.items():
        series = pd.Series(values)
        days_held = int((series.abs() > 1e-6).sum())
        if not days_held:
            continue
        rows.append({
            "symbol": symbol,
            "days_held": days_held,
            "pct_of_window": days_held / dates,
            "avg_weight_when_held": float(series[series.abs() > 1e-6].mean()) / starting_equity,
            "peak_weight": float(series.max()) / starting_equity,
        })
    rows.sort(key=lambda item: -item["days_held"])

    print(f"\n{'symbol':8s} {'days held':>10s} {'% of window':>12s} {'avg wt':>8s} {'peak wt':>8s}")
    for row in rows:
        print(f"{row['symbol']:8s} {row['days_held']:10d} {row['pct_of_window']:11.0%} "
              f"{row['avg_weight_when_held']:8.1%} {row['peak_weight']:8.1%}")

    equity = pd.to_numeric(curve["equity"], errors="coerce")
    best = equity.pct_change().nlargest(5)
    worst = equity.pct_change().nsmallest(5)
    print(f"\nheld {len(rows)} distinct symbols across {dates} dates")
    print("best days :", ", ".join(f"{curve.index[i].date()} {v:+.2%}" for i, v in best.items()))
    print("worst days:", ", ".join(f"{curve.index[i].date()} {v:+.2%}" for i, v in worst.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default="rally_rotation")
    parser.add_argument("--period", default="12m")
    parser.add_argument("--universe", choices=sorted(UNIVERSES), default=None)
    parser.add_argument("--from", dest="axes_results", default=None)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    label, tuning = tuning_from(args)
    sweep = Sweep([args.period])
    with pooled_connections(read_only=True):
        run = sweep.run(args.algorithm, label, tuning, args.period)
        curve, _ = sweep.last_curve

    print(f"\n{args.algorithm} / {label} / {args.period}: "
          f"ret={run.metrics['total_return']:+.2%} net@5bps={run.metrics['net_return_5bps']:+.2%} "
          f"dd={run.metrics['max_drawdown']:+.2%} turnover={run.metrics['turnover_x_stake']:.0f}x")
    report(curve, sweep.starting_equity)
    return 0


if __name__ == "__main__":
    sys.exit(main())
