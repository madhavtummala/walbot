"""Combinations of Options Flip config knobs, not one knob at a time.

``options_flip_config_sweep.py`` isolates each knob against the baseline, which is the right way
to find which single lever matters most -- but it can't say whether two individually-good changes
still add up once combined, since patience, delta and hold-length interact (a longer hold only
pays if the target has room to wait for it; a deeper delta only pays if the exit patience does not
give it back early). This sweeps the combinations that single-knob sweep's own results pointed at:
``exit_patience`` up (1.5 beat 0.7 on its own), ``max_hold_sessions`` up modestly (8 beat 4, this
tries a smaller step first), ``entry_patience`` up (3.0 beat 1.5), and ``target_delta`` up, per
what was asked -- even though the single-knob sweep actually found *lower* delta (0.62) the
single biggest lever on its own; one combo revisits that tension directly (see ``delta down,
everything else up``) rather than silently dropping it.

``entry_reach``/``exit_reach`` are chosen rather than swept here: the single-knob results favoured
shallower entries (0.35 beat 0.75) and easier targets (0.30 beat 0.70), so every combo below uses
0.40 / 0.35 -- moderate steps in the direction that won, not the extremes, since combining several
knobs each pushed to their individual extreme is how a sweep finds a config that overfits this one
month rather than one that is probably still good next month.

Run:

    STATE_DUCKDB_PATH=data/walbot.duckdb python -m tools.options_flip_config_combo_sweep
"""

from __future__ import annotations

import logging

import pandas as pd

from .options_flip_config_sweep import run_variant

logger = logging.getLogger("optflip_combo_sweep")

#: Chosen from the single-knob sweep's winners: entry_reach 0.35 and exit_reach 0.30 each beat
#: their harder-to-clear counterparts, so every combo below uses a moderate step toward each
#: (0.40 / 0.35) rather than the baseline (0.55 / 0.55) or the single-knob winning extreme.
CHOSEN_REACH = {"entry_reach": 0.40, "exit_reach": 0.35}

COMBOS: list[tuple[str, dict]] = [
    ("baseline", {}),
    ("moderate lift: delta .85, patience up, hold 5", {
        "target_delta": 0.85, "entry_patience": 2.0, "exit_patience": 1.2,
        "max_hold_sessions": 5, **CHOSEN_REACH,
    }),
    ("stronger lift: delta .90, patience up more, hold 6", {
        "target_delta": 0.90, "entry_patience": 2.5, "exit_patience": 1.5,
        "max_hold_sessions": 6, **CHOSEN_REACH,
    }),
    ("deep ITM: delta .95, patience up, hold 6", {
        "target_delta": 0.95, "entry_patience": 2.0, "exit_patience": 1.5,
        "max_hold_sessions": 6, **CHOSEN_REACH,
    }),
    ("reach-only: baseline delta/patience/hold, reach moved", dict(CHOSEN_REACH)),
    ("patience+hold only, delta unchanged (0.8)", {
        "entry_patience": 2.0, "exit_patience": 1.5, "max_hold_sessions": 6, **CHOSEN_REACH,
    }),
    ("counterpoint: delta DOWN to .62, patience+hold up anyway", {
        "target_delta": 0.62, "entry_patience": 2.0, "exit_patience": 1.5,
        "max_hold_sessions": 6, **CHOSEN_REACH,
    }),
]


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    from src.core.config import get_config
    config = get_config()

    frames = [run_variant(config, label, **overrides) for label, overrides in COMBOS]
    out = pd.concat(frames, ignore_index=True)

    print("Per symbol, per combo:")
    cols = ["variant", "symbol", "entries", "exits", "wins", "losses", "win_rate",
            "avg_pl", "max_gain", "max_loss", "total_pl"]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(out[cols].to_string(index=False,
              formatters={"win_rate": "{:.0%}".format,
                          "avg_pl": "${:,.0f}".format, "max_gain": "${:,.0f}".format,
                          "max_loss": "${:,.0f}".format, "total_pl": "${:,.0f}".format}))

    print("\nAcross both symbols, per combo:")
    agg = out.groupby("variant", sort=False).agg(
        entries=("entries", "sum"), exits=("exits", "sum"), wins=("wins", "sum"),
        losses=("losses", "sum"), total_pl=("total_pl", "sum"),
        max_gain=("max_gain", "max"), max_loss=("max_loss", "min"),
    ).reindex([label for label, _ in COMBOS])
    agg["win_rate"] = agg["wins"] / agg["exits"].replace(0, pd.NA)
    agg["avg_pl"] = agg["total_pl"] / agg["exits"].replace(0, pd.NA)
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(agg[["entries", "exits", "wins", "losses", "win_rate", "avg_pl",
                   "max_gain", "max_loss", "total_pl"]].to_string(
              formatters={"win_rate": lambda v: f"{v:.0%}" if pd.notna(v) else "--",
                          "avg_pl": lambda v: f"${v:,.0f}" if pd.notna(v) else "--",
                          "max_gain": "${:,.0f}".format, "max_loss": "${:,.0f}".format,
                          "total_pl": "${:,.0f}".format}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
