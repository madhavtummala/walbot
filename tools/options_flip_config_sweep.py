"""Sweep Options Flip config knobs through the real walk-forward harness.

Before cutting a tunable knob, this asks the same question ``bull_regime.py``'s own docstring
already answers for the two gates it removed: does changing this value change anything real on
the data we have, or is it dead weight the strategy carries for no measured reason? Each variant
here re-runs :func:`tools.options_flip_walk_forward.walk_forward` with one field changed via
``dataclasses.replace`` on the real ``OptionsFlipConfig`` -- the actual production code path,
not a re-implementation -- and reports trade count and realized dollar P/L against the baseline.

Run (needs the same cache as its sibling tools):

    STATE_DUCKDB_PATH=data/walbot.duckdb python -m tools.options_flip_config_sweep
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any
from unittest import mock

import pandas as pd

from src.algorithms.options_flip.algorithm import OptionsFlipAlgorithm
from src.algorithms.options_flip.config import OptionsFlipConfig

from .options_flip_walk_forward import walk_forward

logger = logging.getLogger("optflip_sweep")

#: SMH never arms in the baseline (trend_strength never reaches the floor, see the walk-forward
#: write-up) and none of the variants below touch the trend gate, so it would report zero in
#: every column -- dropped here to keep the sweep to the symbols that can actually show a
#: difference. Add it back for a variant that touches ``min_trend_strength`` or the regime gate.
SYMBOLS = ["IBIT", "GLD"]

#: Contract multiplier: every trade here is 1 contract, so a premium delta of $1.00 is $100.
MULTIPLIER = 100.0


def _summarize(symbol: str, log: pd.DataFrame, daily: pd.DataFrame) -> dict[str, Any]:
    """One row of real stats for one symbol under one config variant.

    ``dollar_pl`` is computed from the log's own ``entry_price``/``price`` pair on each exit --
    the same fill prices the harness resolved fills against, not a re-derivation.
    """
    exits = log[log["event"].isin(["FILL_TARGET", "FILL_STOP"])].copy() if not log.empty else log
    fills = log[log["event"] == "FILL_ENTRY"] if not log.empty else log
    row: dict[str, Any] = {
        "symbol": symbol,
        "entries": len(fills),
        "exits": len(exits),
        "wins": 0, "losses": 0, "win_rate": 0.0,
        "avg_pl": 0.0, "max_gain": 0.0, "max_loss": 0.0, "total_pl": 0.0,
        "still_held": int(((daily["state_at_close"] == "held")).iloc[-1]) if not daily.empty else 0,
    }
    if exits.empty:
        return row
    dollar_pl = (exits["price"] - exits["entry_price"]) * MULTIPLIER
    wins = dollar_pl[dollar_pl > 0]
    row.update({
        "wins": int(len(wins)),
        "losses": int(len(exits) - len(wins)),
        "win_rate": float(len(wins) / len(exits)),
        "avg_pl": float(dollar_pl.mean()),
        "max_gain": float(dollar_pl.max()),
        "max_loss": float(dollar_pl.min()),
        "total_pl": float(dollar_pl.sum()),
    })
    return row


def run_variant(config: Any, label: str, **overrides: Any) -> pd.DataFrame:
    """One config variant, across every symbol, via the real ``plan()`` walk-forward."""
    base_cfg = OptionsFlipAlgorithm(config).tuning(config)
    varied = dataclasses.replace(base_cfg, **overrides)
    rows = []
    with mock.patch.object(OptionsFlipAlgorithm, "tuning", lambda self, cfg: varied):
        for symbol in SYMBOLS:
            log, daily, _ticks = walk_forward(symbol, config)
            row = _summarize(symbol, log, daily)
            row["variant"] = label
            rows.append(row)
    return pd.DataFrame(rows)


#: (label, overrides) -- delta, the two reach probabilities, both patience curves, and the stop,
#: each isolated against the current defaults (target_delta 0.8, entry_reach 0.55,
#: exit_reach 0.55, entry_patience 1.5, exit_patience 0.7, stop_loss_pct 0.5).
VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("baseline", {}),
    ("delta 0.5 (further OTM, cheaper)", {"target_delta": 0.5}),
    ("delta 0.62 (old default)", {"target_delta": 0.62}),
    ("delta 0.95 (deep ITM)", {"target_delta": 0.95}),
    ("entry_reach 0.35 (shallower dip)", {"entry_reach": 0.35}),
    ("entry_reach 0.75 (deeper dip)", {"entry_reach": 0.75}),
    ("exit_reach 0.30 (easier target)", {"exit_reach": 0.30}),
    ("exit_reach 0.70 (harder target)", {"exit_reach": 0.70}),
    ("entry_patience 0.5 (chases fast)", {"entry_patience": 0.5}),
    ("entry_patience 3.0 (very patient)", {"entry_patience": 3.0}),
    ("exit_patience 0.3 (concedes fast)", {"exit_patience": 0.3}),
    ("exit_patience 1.5 (holds firm)", {"exit_patience": 1.5}),
    ("stop disabled", {"stop_loss_pct": 0.0}),
    ("stop 0.25 (tighter)", {"stop_loss_pct": 0.25}),
    ("max_hold 2 sessions", {"max_hold_sessions": 2}),
    ("max_hold 8 sessions", {"max_hold_sessions": 8}),
    ("no OI floor", {"min_open_interest": 0}),
    ("no spread cap", {"max_spread_pct": 0.0}),
]


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    from src.core.config import get_config
    config = get_config()

    frames = [run_variant(config, label, **overrides) for label, overrides in VARIANTS]
    out = pd.concat(frames, ignore_index=True)

    print("Per symbol, per variant:")
    cols = ["variant", "symbol", "entries", "exits", "wins", "losses", "win_rate",
            "avg_pl", "max_gain", "max_loss", "total_pl"]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(out[cols].to_string(index=False,
              formatters={"win_rate": "{:.0%}".format,
                          "avg_pl": "${:,.0f}".format, "max_gain": "${:,.0f}".format,
                          "max_loss": "${:,.0f}".format, "total_pl": "${:,.0f}".format}))

    print("\nAcross both symbols, per variant:")
    agg = out.groupby("variant", sort=False).agg(
        entries=("entries", "sum"), exits=("exits", "sum"), wins=("wins", "sum"),
        losses=("losses", "sum"), total_pl=("total_pl", "sum"),
        max_gain=("max_gain", "max"), max_loss=("max_loss", "min"),
    ).reindex([label for label, _ in VARIANTS])
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
