"""Sweep the bear/put side of Options Flip through the real walk-forward harness.

The sibling ``options_flip_config_sweep`` answers "is this knob dead weight" on the long side,
against defaults that a year of walk-forward work already tuned. This one answers a different and
earlier question: **does the short side have an edge at all, and at what threshold does it start
to exist?** The long side's defaults are not a prior here -- they were fitted to a
dip-then-rebound pattern, and the put side is that pattern read in a mirror on an asset class
that does not behave symmetrically.

So the first axis is always ``min_bear_trend_strength``. At the shipped default (1.50) a name has
to be decisively broken before a put is proposed, and on most windows that arms nothing at all --
a sweep that only varied delta and the reach quantiles would report zeros in every row and look
like a broken harness rather than a working gate.

Every variant re-runs :func:`tools.options_flip_walk_forward.walk_forward` with one field changed
via ``dataclasses.replace`` on the real ``OptionsFlipConfig`` -- the production code path, not a
re-implementation.

Run (needs the put-side cache; see ``tools/_optcache/fetch.py --type put``):

    STATE_DUCKDB_PATH=data/walbot.duckdb python -m tools.options_flip_bear_sweep \\
        --symbols SMH --start 2026-08-15 --end 2026-08-31
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from datetime import date
from typing import Any
from unittest import mock

import pandas as pd

from src.algorithms.options_flip.algorithm import OptionsFlipAlgorithm
from src.core.options import PUT

from .options_flip_config_sweep import MULTIPLIER, _summarize
from .options_flip_walk_forward import walk_forward

logger = logging.getLogger("optflip_bear_sweep")

#: The threshold ladder, run first and on its own. Zero means "any negative reading proposes a
#: put", which is not a setting anyone should deploy -- it is the control that shows what the
#: gate is actually buying, since every stricter row is a subset of the trades this one takes.
BEAR_FLOORS = [0.0, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50]

#: The second pass, run at whichever floor the first pass showed actually trades. Each is one
#: field against the bear baseline, the same isolation discipline the long-side sweep uses.
KNOBS: list[tuple[str, dict[str, Any]]] = [
    ("baseline", {}),
    ("delta 0.50 (further OTM, cheaper)", {"target_delta": 0.50}),
    ("delta 0.62", {"target_delta": 0.62}),
    ("delta 0.90 (deep ITM, less skew)", {"target_delta": 0.90}),
    ("entry_reach 0.35 (shallower pop)", {"entry_reach": 0.35}),
    ("entry_reach 0.75 (deeper pop)", {"entry_reach": 0.75}),
    ("exit_reach 0.30 (easier target)", {"exit_reach": 0.30}),
    ("exit_reach 0.70 (harder target)", {"exit_reach": 0.70}),
    ("stop disabled", {"stop_loss_pct": 0.0}),
    ("stop 0.25 (tighter)", {"stop_loss_pct": 0.25}),
    ("stop 0.75 (loose, rides bounces)", {"stop_loss_pct": 0.75}),
    ("max_hold 2 sessions", {"max_hold_sessions": 2}),
    ("max_hold 8 sessions", {"max_hold_sessions": 8}),
    ("exit_patience 0.3 (concedes fast)", {"exit_patience": 0.3}),
    ("exit_patience 1.5 (holds firm)", {"exit_patience": 1.5}),
    ("profit floor $5 (admits more)", {"min_profit_per_contract": 5.0}),
    ("profit floor $40 (stricter)", {"min_profit_per_contract": 40.0}),
    ("adverse gap ceiling 0.5 ATR", {"gap_atr": 0.5}),
    # These exist because of one measured trade: SMH's 2026-08-24 put was forced out at
    # -41% when price poked 0.30 ATR back over its 20d average on a single bounce session, and
    # the same contract was -18% the next day. These ask whether that was one bad trade or the
    # gate being too twitchy to hold a downtrend, which bounces over its average by nature.
    ("exit band 0.25 ATR", {"exit_trend_band_atr": 0.25}),
    ("exit band 0.50 ATR", {"exit_trend_band_atr": 0.50}),
    ("exit band 0.75 ATR", {"exit_trend_band_atr": 0.75}),
    ("exit band 1.50 ATR (very slack)", {"exit_trend_band_atr": 1.50}),
]


def run_variant(
    config: Any, label: str, symbols: list[str], *, budget: float,
    start: date | None, end: date | None, **overrides: Any,
) -> pd.DataFrame:
    """One config variant, across every symbol, on the put side of the real ``plan()``."""
    base_cfg = OptionsFlipAlgorithm(config).tuning(config)
    varied = dataclasses.replace(base_cfg, **overrides)
    rows = []
    with mock.patch.object(OptionsFlipAlgorithm, "tuning", lambda self, cfg: varied):
        for symbol in symbols:
            log, daily, _ticks = walk_forward(
                symbol, config, option_type=PUT, budget=budget, start=start, end=end,
            )
            row = _summarize(symbol, log, daily)
            row["variant"] = label
            rows.append(row)
    return pd.DataFrame(rows)


def _show(out: pd.DataFrame, order: list[str], title: str) -> None:
    print(f"\n{title}")
    agg = out.groupby("variant", sort=False).agg(
        entries=("entries", "sum"), exits=("exits", "sum"), wins=("wins", "sum"),
        losses=("losses", "sum"), total_pl=("total_pl", "sum"),
        open_pl=("open_pl", "sum"), still_held=("still_held", "sum"),
        max_gain=("max_gain", "max"), max_loss=("max_loss", "min"),
    ).reindex(order)
    agg["win_rate"] = agg["wins"] / agg["exits"].replace(0, pd.NA)
    agg["avg_pl"] = agg["total_pl"] / agg["exits"].replace(0, pd.NA)
    money = lambda v: f"${v:,.0f}" if pd.notna(v) else "--"  # noqa: E731
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(agg[["entries", "exits", "wins", "losses", "win_rate", "avg_pl",
                   "max_gain", "max_loss", "total_pl", "still_held", "open_pl"]].to_string(
            formatters={"win_rate": lambda v: f"{v:.0%}" if pd.notna(v) else "--",
                        "avg_pl": money, "max_gain": money, "max_loss": money,
                        "total_pl": money, "open_pl": money}))


def _run_pass(
    config: Any, args: Any, variants: list[tuple[str, dict[str, Any]]], title: str,
    fixed: dict[str, Any], out_csv: str,
) -> pd.DataFrame:
    """One pass, written to disk a row at a time.

    A sweep is tens of minutes of walk-forwards, and the first version of this held every result
    in memory and printed once at the end -- so a run that died partway through (or had its
    output swallowed by a buffering pipe) lost all of it and had to start over. Each variant now
    appends to ``out_csv`` as it finishes, and progress goes to stderr unbuffered, so an
    interrupted sweep still leaves everything it had already measured.
    """
    common = dict(budget=args.budget, start=args.start, end=args.end)
    frames = []
    for index, (label, overrides) in enumerate(variants, start=1):
        began = time.time()
        frame = run_variant(config, label, args.symbols, **{**fixed, **overrides}, **common)
        frames.append(frame)
        header = not os.path.exists(out_csv)
        frame.to_csv(out_csv, mode="a", header=header, index=False)
        done = frame[["entries", "exits", "total_pl", "open_pl"]].sum()
        print(f"  [{index:2d}/{len(variants)}] {label:38s} "
              f"{int(done['entries'])} entries, {int(done['exits'])} exits, "
              f"${done['total_pl']:+,.0f} realized"
              + (f" {done['open_pl']:+,.0f} open" if done["open_pl"] else "")
              + f"   ({time.time() - began:.0f}s)",
              file=sys.stderr, flush=True)
    out = pd.concat(frames, ignore_index=True)
    _show(out, [label for label, _ in variants], title)
    sys.stdout.flush()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="+", default=["SMH"])
    parser.add_argument("--budget", type=float, default=5000.0)
    parser.add_argument("--start", type=date.fromisoformat, default=None)
    parser.add_argument("--end", type=date.fromisoformat, default=None)
    parser.add_argument("--out", default="data/bear_sweep")
    parser.add_argument("--passes", default="1,2", help="which passes to run, e.g. '1' or '2'")
    parser.add_argument(
        "--floor", type=float, default=None,
        help="Bear floor to hold fixed for the knob pass. Omit to use the loosest floor that "
             "actually opened a position in the threshold pass.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    wanted = {part.strip() for part in args.passes.split(",")}

    from src.core.config import get_config
    config = get_config()
    floor = args.floor

    # ── pass 1: where does the short side start trading at all? ──────────────────
    if "1" in wanted:
        print("Pass 1 -- min_bear_trend_strength ladder", file=sys.stderr, flush=True)
        labels = [f"bear floor {value:.2f}σ" for value in BEAR_FLOORS]
        variants = [
            (label, {"min_bear_trend_strength": value})
            for label, value in zip(labels, BEAR_FLOORS)
        ]
        threshold = _run_pass(
            config, args, variants,
            "Pass 1 -- min_bear_trend_strength, everything else at default:",
            {}, f"{args.out}_pass1.csv",
        )
        if floor is None:
            traded = threshold.groupby("variant", sort=False)["entries"].sum()
            hit = [f for label, f in zip(labels, BEAR_FLOORS) if traded.get(label, 0) > 0]
            floor = max(hit) if hit else 0.0

    # ── pass 2: the rest of the knobs, at a floor that trades ────────────────────
    if "2" in wanted:
        if floor is None:
            floor = 0.35
        print(f"\nPass 2 -- knobs at bear floor {floor:.2f}σ", file=sys.stderr, flush=True)
        _run_pass(
            config, args, KNOBS,
            f"Pass 2 -- one knob at a time, bear floor pinned at {floor:.2f}σ:",
            {"min_bear_trend_strength": floor}, f"{args.out}_pass2.csv",
        )

    print(f"\n(1 contract per trade, ${MULTIPLIER:.0f} multiplier; synthetic 2% spread, "
          f"no historical bid/ask exists to read. Per-variant rows in {args.out}_pass*.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
