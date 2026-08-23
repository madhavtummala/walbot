"""Explore vol gates, climax exits, and exit gate configurations for rally_rotation.

Tests the gaps identified in the algorithm review:
1. Volatility gates (vol_ceiling, vol_rising_threshold) -- entry-only, currently deployed at 0.5/0.3
2. Climax exit gates (climax_ma_distance_min, range_expansion_limit, climax_volume_ratio_min)
3. Exit gate tightening (exit_threshold_slack, exit_rank_max, exit_max_eligible_days)
4. Combined best configurations

Uses the deployed config as baseline and sweeps one axis at a time.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time
from typing import Any

import pandas as pd

# Point to the sweep copy of DuckDB so we don't lock the production file.
os.environ.setdefault("STATE_DUCKDB_PATH", "data/walbot_sweep.duckdb")

from tools.config_sweep import (
    Sweep,
    _axis,
    _measure,
    deployed_tuning,
    execute,
    write_results,
)
from src.data.duckdb_store import pooled_connections


def vol_ceiling_axes(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Vol ceiling: reject names with annualized vol above this level."""
    variants = [("baseline", dict(base))]
    for ceiling in (0.30, 0.40, 0.50, 0.60, 0.80, 1.0):
        variants.append(_axis(base, f"vol_ceiling={ceiling}", vol_ceiling=ceiling))
    # Also test with vol_ceiling OFF (0.0) to see if the gate helps at all
    variants.append(_axis(base, "vol_ceiling=OFF", vol_ceiling=0.0))
    return variants


def vol_rising_axes(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Rising vol gate: reject if 5d vol exceeds 20d vol by more than this ratio."""
    variants = [("baseline", dict(base))]
    for threshold in (0.10, 0.20, 0.30, 0.50, 0.75, 1.0):
        variants.append(_axis(base, f"vol_rising={threshold}", vol_rising_threshold=threshold))
    variants.append(_axis(base, "vol_rising=OFF", vol_rising_threshold=0.0))
    return variants


def climax_exit_axes(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Climax exit: MA distance + range expansion + volume spike."""
    variants = [("baseline", dict(base))]

    # Individual axes -- each condition in isolation
    variants.append(_axis(base, "climax_ma=0.10", climax_ma_distance_min=0.10))
    variants.append(_axis(base, "climax_ma=0.15", climax_ma_distance_min=0.15))
    variants.append(_axis(base, "climax_ma=0.20", climax_ma_distance_min=0.20))
    variants.append(_axis(base, "climax_ma=0.25", climax_ma_distance_min=0.25))

    variants.append(_axis(base, "climax_range=1.5", range_expansion_limit=1.5))
    variants.append(_axis(base, "climax_range=2.0", range_expansion_limit=2.0))
    variants.append(_axis(base, "climax_range=2.5", range_expansion_limit=2.5))
    variants.append(_axis(base, "climax_range=3.0", range_expansion_limit=3.0))

    variants.append(_axis(base, "climax_vol=1.2", climax_volume_ratio_min=1.2))
    variants.append(_axis(base, "climax_vol=1.5", climax_volume_ratio_min=1.5))
    variants.append(_axis(base, "climax_vol=2.0", climax_volume_ratio_min=2.0))

    # Combined climax configurations
    variants.append(_axis(base, "climax_tight",
                          climax_ma_distance_min=0.10, range_expansion_limit=1.5,
                          climax_volume_ratio_min=1.2))
    variants.append(_axis(base, "climax_moderate",
                          climax_ma_distance_min=0.15, range_expansion_limit=2.0,
                          climax_volume_ratio_min=1.5))
    variants.append(_axis(base, "climax_loose",
                          climax_ma_distance_min=0.25, range_expansion_limit=3.0,
                          climax_volume_ratio_min=2.0))

    # Climax OFF -- all three at 0
    variants.append(_axis(base, "climax=OFF",
                          climax_ma_distance_min=0.0, range_expansion_limit=0.0,
                          climax_volume_ratio_min=0.0))
    return variants


def exit_gate_axes(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Exit gate tightening: slack, rank, eligibility count."""
    variants = [("baseline", dict(base))]

    # Tighter exit threshold slack (less forgiving before exit)
    for slack in (0.0, 0.02, 0.03, 0.05, 0.08, 0.10):
        variants.append(_axis(base, f"slack={slack}", exit_threshold_slack=slack))

    # Tighter exit rank (force exit sooner)
    for rank in (4, 5, 6, 7, 8):
        variants.append(_axis(base, f"exit_rank={rank}", exit_rank_max=rank))

    # Tighter eligibility persistence (exit after fewer qualifying runs)
    for days in (1, 2, 3, 4, 5):
        variants.append(_axis(base, f"exit_eligible={days}d", exit_max_eligible_days=days))

    # Combined tight exit
    variants.append(_axis(base, "exit_tight",
                          exit_threshold_slack=0.02, exit_rank_max=5,
                          exit_max_eligible_days=2))
    variants.append(_axis(base, "exit_very_tight",
                          exit_threshold_slack=0.0, exit_rank_max=4,
                          exit_max_eligible_days=1))
    return variants


def combined_best_axes(base: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Combine the best-performing settings from each axis."""
    variants = [("baseline", dict(base))]

    # Conservative combined: moderate vol gate + moderate climax + moderate exit
    variants.append(_axis(base, "combined_conservative",
                          vol_ceiling=0.50, vol_rising_threshold=0.30,
                          climax_ma_distance_min=0.15, range_expansion_limit=2.0,
                          climax_volume_ratio_min=1.5,
                          exit_threshold_slack=0.03, exit_rank_max=5,
                          exit_max_eligible_days=2))

    # Aggressive combined: tight vol + tight climax + tight exit
    variants.append(_axis(base, "combined_aggressive",
                          vol_ceiling=0.30, vol_rising_threshold=0.20,
                          climax_ma_distance_min=0.10, range_expansion_limit=1.5,
                          climax_volume_ratio_min=1.2,
                          exit_threshold_slack=0.0, exit_rank_max=4,
                          exit_max_eligible_days=1))

    # Vol-focused: vol gates ON, everything else at baseline
    variants.append(_axis(base, "vol_only",
                          vol_ceiling=0.50, vol_rising_threshold=0.30,
                          climax_ma_distance_min=0.0, range_expansion_limit=0.0,
                          climax_volume_ratio_min=0.0))

    # Exit-focused: vol gates OFF, tight exits
    variants.append(_axis(base, "exit_only",
                          vol_ceiling=0.0, vol_rising_threshold=0.0,
                          climax_ma_distance_min=0.0, range_expansion_limit=0.0,
                          climax_volume_ratio_min=0.0,
                          exit_threshold_slack=0.02, exit_rank_max=5,
                          exit_max_eligible_days=2))

    # No gates at all: everything OFF
    variants.append(_axis(base, "all_gates_OFF",
                          vol_ceiling=0.0, vol_rising_threshold=0.0,
                          climax_ma_distance_min=0.0, range_expansion_limit=0.0,
                          climax_volume_ratio_min=0.0,
                          exit_threshold_slack=0.05))

    # All gates at deployed defaults (should match baseline)
    variants.append(_axis(base, "all_gates_deployed",
                          vol_ceiling=0.50, vol_rising_threshold=0.30,
                          climax_ma_distance_min=0.15, range_expansion_limit=2.0,
                          climax_volume_ratio_min=1.5))
    return variants


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all",
                        choices=["vol_ceiling", "vol_rising", "climax", "exit", "combined", "all"])
    parser.add_argument("--period", default="12m")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--starting-equity", type=float, default=10000.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("src.core.orders", "src.brokerages.paper.brokerage",
                  "src.algorithms.rally_rotation.algorithm",
                  "src.data.provider_cache", "src.connectors"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    periods = [p.strip() for p in args.period.split(",") if p.strip()]
    base = deployed_tuning("rally_rotation")

    stages = {
        "vol_ceiling": vol_ceiling_axes(base),
        "vol_rising": vol_rising_axes(base),
        "climax": climax_exit_axes(base),
        "exit": exit_gate_axes(base),
        "combined": combined_best_axes(base),
    }

    if args.stage == "all":
        selected = []
        for name, variants in stages.items():
            selected.extend([(f"{name}/{label}", tuning) for label, tuning in variants])
    else:
        selected = stages[args.stage]

    print(f"Running {len(selected)} variants across {', '.join(periods)} "
          f"(cost_bps={args.cost_bps}, equity=${args.starting_equity:,.0f})")
    print("=" * 100)

    sweep = Sweep(periods, starting_equity=args.starting_equity, cost_bps=args.cost_bps)
    runs = execute(sweep, "rally_rotation", selected, periods)

    out = "data/gate_exploration.csv"
    write_results(runs, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
