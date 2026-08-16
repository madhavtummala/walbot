"""Rank a sweep's output and say which axes actually moved anything.

Reads the CSV ``config_sweep`` writes. Kept separate from the sweep itself so a ranking can be
re-cut -- by a different metric, against a different cost assumption -- without paying for the
replays again.

    python -m tools.sweep_report data/config_sweep_12m.csv
    python -m tools.sweep_report data/config_sweep_12m.csv --sort sharpe
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Any


NUMERIC = {
    "total_return", "net_return_1bps", "net_return_5bps", "cagr", "max_drawdown",
    "sharpe", "calmar", "turnover_x_stake", "cost_drag_5bps", "orders",
    "dividend_income", "coverage", "seconds",
}


def load(path: str) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in NUMERIC & set(row):
            try:
                row[key] = float(row[key])
            except (TypeError, ValueError):
                row[key] = 0.0
    return rows


def breakeven_bps(row: dict[str, Any]) -> float:
    """The one-way trading cost, in basis points, that would erase this config's whole return.

    Assumption-free, unlike ``net_return_5bps``: rather than guessing a cost and subtracting it,
    it reports how much headroom the configuration has. A config returning 15% on 161x turnover
    dies at 9bps; one returning 42% on 201x survives to 21bps. Whether that is comfortable is a
    question about the spreads of the symbols it actually trades, which this repo does not
    store -- ``market_bars`` holds OHLCV, never a quote.
    """
    turnover = row.get("turnover_x_stake", 0.0)
    return (row.get("total_return", 0.0) / turnover) * 10_000 if turnover else float("inf")


def table(rows: list[dict[str, Any]], sort_key: str, baseline: dict[str, Any] | None) -> str:
    ordered = sorted(rows, key=lambda row: row.get(sort_key, 0.0), reverse=True)
    lines = [
        f"{'':>3}  {'variant':30s} {'return':>8s} {'net@5bps':>9s} {'vs base':>8s} "
        f"{'maxDD':>7s} {'sharpe':>7s} {'turnover':>9s} {'breakeven':>10s}"
    ]
    for rank, row in enumerate(ordered, start=1):
        delta = ""
        if baseline is not None:
            delta = f"{row['net_return_5bps'] - baseline['net_return_5bps']:+7.2%}"
        marker = " *" if row["label"] == "baseline" else "  "
        lines.append(
            f"{rank:>3}{marker}{row['label']:28s} {row['total_return']:+8.2%} "
            f"{row['net_return_5bps']:+9.2%} {delta:>8s} {row['max_drawdown']:+7.2%} "
            f"{row['sharpe']:7.2f} {row['turnover_x_stake']:8.1f}x {breakeven_bps(row):9.1f}bp"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--sort", default="net_return_5bps",
                        help="net_return_5bps (default), total_return, sharpe, calmar")
    parser.add_argument("--top", type=int, default=0, help="show only the best N per algorithm")
    args = parser.parse_args(argv)

    rows = load(args.path)
    for algorithm in sorted({row["algorithm"] for row in rows}):
        for period in sorted({row["period"] for row in rows if row["algorithm"] == algorithm}):
            subset = [r for r in rows if r["algorithm"] == algorithm and r["period"] == period]
            baseline = next((r for r in subset if r["label"] == "baseline"), None)
            print(f"\n{'=' * 108}\n{algorithm}  /  {period}  ({len(subset)} configurations, "
                  f"ranked by {args.sort}; * = currently deployed)\n{'=' * 108}")
            shown = subset
            if args.top:
                ranked = sorted(subset, key=lambda r: r.get(args.sort, 0.0), reverse=True)[: args.top]
                shown = ranked + ([baseline] if baseline and baseline not in ranked else [])
            print(table(shown, args.sort, baseline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
