"""Day-by-day holdings with change detection for any month.

Every session's book is shown. Days where the holdings changed (entry, exit,
weight shift) are flagged so you can see exactly when the algo acted vs held steady.

    python -m tools.daily_holdings --period 2026-01-01:2026-01-31
    python -m tools.daily_holdings --period 2026-02-01:2026-02-28
    python -m tools.daily_holdings --period 2026-01-01:2026-03-31 --set rerank_interval_days=3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.rally_rotation.config import RallyRotationConfig
from tools.attribution import _parse_overrides
from tools.config_sweep import Sweep, deployed_tuning


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
    defensive = {s.upper() for s in (tuning.get("defensive_universe") or [])}

    sweep = Sweep([period])
    result = sweep.run("rally_rotation", "deployed", tuning, period)
    curve, _ = sweep.last_curve

    if curve.empty:
        print("ERROR: No history.")
        return

    hist = curve.reset_index()
    hist["month"] = pd.to_datetime(hist["timestamp"]).dt.to_period("M")

    first_date = pd.to_datetime(hist.iloc[0]["timestamp"]).strftime("%Y-%m-%d")
    last_date = pd.to_datetime(hist.iloc[-1]["timestamp"]).strftime("%Y-%m-%d")

    print(f"\n{'='*100}")
    print(f"  DAILY HOLDINGS  |  {first_date} to {last_date}  |  {len(hist)} sessions")
    print(f"  Rerank interval: {'every session' if rerank_interval == 0 else f'every {rerank_interval} sessions'}")
    if overrides:
        print(f"  Overrides: {', '.join(f'{k}={v}' for k, v in overrides.items())}")
    print(f"{'='*100}")

    prev_positions = {}

    for _, row in hist.iterrows():
        date = pd.to_datetime(row["timestamp"])
        date_str = date.strftime("%Y-%m-%d")
        dow = date.strftime("%a")
        eq = float(row["equity"])
        pos = parse_json(row.get("positions", {}))
        risk = {k: v for k, v in pos.items() if k not in defensive}
        cash_val = float(row.get("cash", 0))
        cash_pct = cash_val / eq if eq else 0

        # Detect what changed
        all_syms = set(list(prev_positions.keys()) + list(risk.keys()))
        added = []
        removed = []
        resized = []
        for sym in sorted(all_syms):
            old_w = prev_positions.get(sym, 0)
            new_w = risk.get(sym, 0)
            if old_w == 0 and new_w > 0:
                added.append((sym, new_w))
            elif old_w > 0 and new_w == 0:
                removed.append(sym)
            elif old_w > 0 and new_w > 0 and abs(new_w / eq - old_w / (prev_eq if prev_eq else 1)) > 0.001:
                resized.append((sym, old_w / (prev_eq if prev_eq else 1), new_w / eq))

        changed = bool(added or removed or resized)
        prev_eq = eq

        # Build the book string
        if risk:
            book = " / ".join(f"{k} {v/eq:.0%}" for k, v in sorted(risk.items(), key=lambda x: -x[1]))
        else:
            book = f"[cash {cash_pct:.0%}]"

        # Build change annotation
        annotations = []
        if added:
            for sym, w in added:
                annotations.append(f"+{sym} {w/eq:.0%}")
        if removed:
            for sym in removed:
                annotations.append(f"-{sym}")
        if resized:
            for sym, old_w_pct, new_w_pct in resized:
                delta = new_w_pct - old_w_pct
                annotations.append(f"~{sym} {delta:+.0%}")

        # Print
        if not prev_positions:
            # First row - no comparison
            print(f"\n  {date_str} {dow}  ${eq:>10,.0f}  {book}")
        elif changed:
            change_str = "  ".join(annotations)
            print(f"  {date_str} {dow}  ${eq:>10,.0f}  {book}   <-- {change_str}")
        else:
            print(f"  {date_str} {dow}  ${eq:>10,.0f}  {book}")

        prev_positions = risk

    # Summary
    starting_eq = float(hist.iloc[0]["equity"])
    ending_eq = float(hist.iloc[-1]["equity"])
    total_return = ending_eq / starting_eq - 1.0

    # Count change days vs hold days
    change_count = 0
    hold_count = 0
    prev_pos_check = {}
    for _, row in hist.iterrows():
        eq = float(row["equity"])
        pos = parse_json(row.get("positions", {}))
        risk = {k: v for k, v in pos.items() if k not in defensive}
        if prev_pos_check:
            syms = set(list(prev_pos_check.keys()) + list(risk.keys()))
            any_change = any(
                abs(prev_pos_check.get(s, 0) - risk.get(s, 0)) > 0.001
                for s in syms
            )
            if any_change:
                change_count += 1
            else:
                hold_count += 1
        prev_pos_check = risk

    print(f"\n{'─'*100}")
    print(f"  Return: {total_return:+.2%}  |  Days with changes: {change_count}  |  Days held steady: {hold_count}")
    print(f"{'─'*100}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="2026-01-01:2026-01-31")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("src.core.orders", "src.brokerages.providers.paper",
                  "src.algorithms.rally_rotation.algorithm", "src.data.provider_cache",
                  "src.connectors"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    overrides = _parse_overrides(args.overrides)
    run(args.period, overrides)


if __name__ == "__main__":
    main()
