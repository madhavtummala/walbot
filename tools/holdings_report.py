"""What the book actually held, month by month.

The sweep says how a configuration scored and ``tools/attribution`` says where the turnover
went. Neither says what was *owned*, which is the question asked when a result looks wrong --
a year of +2% has a completely different explanation if it was spent in T-bills than if it was
spent fully invested in names that went nowhere.

Reported as average weight over each month rather than as a month-end snapshot. A book that
rotates every few sessions has no meaningful month-end position: whatever it happened to hold
on the 31st is one draw from the distribution, and reading it as "what the strategy owned in
March" is how a rotation gets mistaken for a conviction.

    python -m tools.holdings_report --period 2026-01-01:2026-08-14
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import pandas as pd

from src.algorithms.rally_rotation.config import RallyRotationConfig
from tools.attribution import _parse_overrides
from tools.config_sweep import Sweep, deployed_tuning

logger = logging.getLogger(__name__)


def monthly_weights(curve: pd.DataFrame) -> pd.DataFrame:
    """Average weight per symbol per month, as a fraction of that day's equity.

    Cash is whatever the holdings do not account for, so it appears as a column rather than
    being inferred by the reader from a row that does not sum to 1.
    """
    rows: list[dict[str, Any]] = []
    for record in curve.itertuples():
        equity = float(getattr(record, "equity", 0.0) or 0.0)
        if equity <= 0:
            continue
        held = {str(symbol): float(value) / equity
                for symbol, value in (getattr(record, "positions", {}) or {}).items()}
        rows.append({"month": pd.Timestamp(record.Index).strftime("%Y-%m"),
                     **held,
                     "cash": max(1.0 - sum(held.values()), 0.0)})
    frame = pd.DataFrame(rows).fillna(0.0)
    return frame.groupby("month").mean(numeric_only=True)


def monthly_performance(curve: pd.DataFrame) -> pd.DataFrame:
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    monthly = equity.resample("ME").last()
    opening = equity.resample("ME").first()
    # First month measured from the opening stake rather than from its own first close, or the
    # month the book was funded silently loses whatever it earned on day one.
    previous = monthly.shift(1)
    previous.iloc[0] = float(equity.iloc[0])
    returns = (monthly / previous - 1.0).dropna()
    turnover = pd.to_numeric(curve["turnover"], errors="coerce").fillna(0.0).resample("ME").sum()
    orders = pd.to_numeric(curve["order_count"], errors="coerce").fillna(0.0).resample("ME").sum()
    return pd.DataFrame({
        "return": returns,
        "equity": monthly,
        "turnover": turnover.reindex(returns.index),
        "orders": orders.reindex(returns.index),
        "opening": opening.reindex(returns.index),
    })


def report(curve: pd.DataFrame, tuning: dict[str, Any], label: str, top: int = 8) -> None:
    weights = monthly_weights(curve)
    performance = monthly_performance(curve)
    defensive = {str(s).upper() for s in (tuning.get("defensive_universe") or [])}

    print(f"\n{'=' * 92}\n{label}\n{'=' * 92}")

    print("\n-- month by month ------------------------------------------------------------------")
    print(f"{'month':8s} {'return':>8s} {'equity':>10s} {'orders':>7s}  average weights")
    for month, row in performance.iterrows():
        stamp = pd.Timestamp(month).strftime("%Y-%m")
        if stamp not in weights.index:
            continue
        book = weights.loc[stamp].sort_values(ascending=False)
        book = book[book > 0.005]
        parts = []
        for symbol, weight in list(book.items())[:top]:
            mark = "*" if symbol in defensive else ""
            parts.append(f"{symbol}{mark} {weight:.0%}")
        print(f"{stamp:8s} {row['return']:+7.2%} {row['equity']:10,.0f} "
              f"{int(row['orders']):7d}  {'  '.join(parts)}")
    print("  (* = defensive sleeve; weights are the month's daily average, not a month-end snapshot)")

    print("\n-- most-held names -----------------------------------------------------------------")
    overall = weights.mean().sort_values(ascending=False)
    for symbol, weight in overall[overall > 0.005].items():
        months = int((weights[symbol] > 0.005).sum())
        kind = "cash" if symbol == "cash" else ("defensive" if symbol in defensive else "risk")
        print(f"  {symbol:6s} {weight:6.1%} average   held in {months}/{len(weights)} months"
              f"   [{kind}]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="2026-01-01:2026-08-14")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--top", type=int, default=8, help="names to print per month")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("src.core.orders", "src.brokerages.paper.brokerage",
                  "src.algorithms.rally_rotation.algorithm", "src.data.provider_cache",
                  "src.connectors"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    tuning = {**deployed_tuning("rally_rotation"), **_parse_overrides(args.overrides)}
    sweep = Sweep([args.period])
    run = sweep.run("rally_rotation", "deployed", tuning, args.period)
    curve, _coverage = sweep.last_curve
    report(curve, tuning, f"rally_rotation  /  {args.period}", top=args.top)
    print(f"\ntotal return {run.metrics['total_return']:+.2%}   "
          f"net@5bps {run.metrics['net_return_5bps']:+.2%}   "
          f"max drawdown {run.metrics['max_drawdown']:+.2%}   "
          f"turnover {run.metrics['turnover_x_stake']:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
