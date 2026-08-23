"""Run config_sweep variants in parallel using copies of the DuckDB file.

Each worker gets its own copy of walbot.duckdb so there is no lock contention,
and the main database file stays free for the dashboard.

Usage:
    python -m tools.parallel_sweep --algorithm rally_rotation --stage axes --period 12m --workers 4
    python -m tools.parallel_sweep --algorithm rally_rotation --stage sticky --period 12m,6m --workers 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DUCKDB = PROJECT_ROOT / "data" / "walbot.duckdb"


def _copy_duckdb(dest: Path) -> Path:
    """Copy the main DuckDB to dest, returning the copy path."""
    dest.mkdir(parents=True, exist_ok=True)
    copy_path = dest / "walbot.duckdb"
    shutil.copy2(DEFAULT_DUCKDB, copy_path)
    for suffix in (".wal", ".tmp"):
        src = Path(str(DEFAULT_DUCKDB) + suffix)
        if src.exists():
            shutil.copy2(src, Path(str(copy_path) + suffix))
    return copy_path


def _chunk(items: list, n: int) -> list[list]:
    """Split items into n roughly equal chunks."""
    k, extra = divmod(len(items), n)
    chunks = []
    start = 0
    for i in range(n):
        end = start + k + (1 if i < extra else 0)
        chunks.append(items[start:end])
        start = end
    return chunks


WORKER_SCRIPT_TEMPLATE = '''
import json, os, sys
os.environ["STATE_DUCKDB_PATH"] = {duckdb_path!r}

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
for noisy in ("src.core.orders", "src.brokerages.paper.brokerage",
              "src.algorithms.rally_rotation.algorithm",
              "src.data.provider_cache", "src.connectors"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from tools.config_sweep import Sweep, execute, write_results

with open({variants_path!r}) as f:
    variants = json.load(f)

periods = {periods!r}
algorithm_id = {algorithm_id!r}
starting_equity = {starting_equity}
open_in = {open_in!r}
cost_bps = {cost_bps}
out_path = {out_path!r}

sweep = Sweep(periods, starting_equity=starting_equity, open_in=open_in, cost_bps=cost_bps)
runs = execute(sweep, algorithm_id, variants, periods)

results = [{{**run.row(), "tuning": run.tuning}} for run in runs]
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, sort_keys=True)
print(f"worker done: {{len(runs)}} runs -> {{out_path}}")
'''


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default="rally_rotation",
                        choices=["rally_rotation", "bursty_dca", "both"])
    parser.add_argument("--starting-equity", type=float, default=None)
    parser.add_argument("--cost-bps", type=float, default=0.0)
    parser.add_argument("--open-in", default="")
    parser.add_argument("--stage", default="axes",
                        choices=["axes", "wide", "wide_churn", "y2023", "confirm", "sticky", "finalists"])
    parser.add_argument("--from", dest="axes_results", default="data/config_sweep_12m.csv")
    parser.add_argument("--period", default="12m")
    parser.add_argument("--out", default="data/config_sweep.csv")
    parser.add_argument("--workers", type=int, default=4,
                        help="number of parallel workers (each gets a DuckDB copy)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    periods = [item.strip() for item in args.period.split(",") if item.strip()]
    algorithms = ["rally_rotation"] if args.algorithm == "both" else [args.algorithm]

    # Build all variants
    from tools.config_sweep import GRIDS, finalists

    all_variants: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = []
    for algorithm_id in algorithms:
        if args.stage == "finalists":
            variants = finalists(algorithm_id, args.axes_results)
        else:
            variants = GRIDS[algorithm_id][args.stage]()
        all_variants.append((algorithm_id, variants))

    # Flatten into (algorithm_id, label, tuning)
    flat: list[tuple[str, str, dict[str, Any]]] = []
    for algorithm_id, variants in all_variants:
        for label, tuning in variants:
            flat.append((algorithm_id, label, tuning))

    total_variants = len(flat)
    workers = max(1, min(args.workers, total_variants))
    chunks = _chunk(flat, workers)

    # Create temp dir for DuckDB copies
    tmp_dir = Path(tempfile.mkdtemp(prefix="walbot_sweep_"))
    logger.info("Creating %d DuckDB copies in %s", workers, tmp_dir)

    copies = []
    for i in range(workers):
        if not chunks[i]:
            continue
        copy_path = _copy_duckdb(tmp_dir / f"worker_{i}")
        copies.append(copy_path)
        logger.info("  worker %d: %s (%.1f MB)", i, copy_path, copy_path.stat().st_size / 1e6)

    # Launch subprocesses
    procs = []
    out_files = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        # Group by algorithm_id
        algo_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for algorithm_id, label, tuning in chunk:
            algo_groups.setdefault(algorithm_id, []).append((label, tuning))

        for algorithm_id, variants in algo_groups.items():
            out_path = tmp_dir / f"worker_{i}_{algorithm_id}.json"
            out_files.append(out_path)

            # Write variants to a JSON file so we avoid Python repr issues with booleans
            variants_path = tmp_dir / f"worker_{i}_{algorithm_id}_variants.json"
            with open(variants_path, "w") as f:
                json.dump(variants, f)

            script = WORKER_SCRIPT_TEMPLATE.format(
                duckdb_path=str(copies[i]),
                periods=periods,
                algorithm_id=algorithm_id,
                variants_path=str(variants_path),
                starting_equity=args.starting_equity,
                open_in=args.open_in,
                cost_bps=args.cost_bps,
                out_path=str(out_path),
            )

            proc = subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            procs.append((proc, algorithm_id, i))

    logger.info("Launched %d subprocesses (%d variants total)", len(procs), total_variants)

    # Wait for all to finish
    all_runs = []
    for proc, algorithm_id, worker_id in procs:
        stdout, stderr = proc.communicate(timeout=600)
        if proc.returncode != 0:
            logger.error("worker %d (%s) failed (rc=%d):\n%s", worker_id, algorithm_id,
                         proc.returncode, stderr.decode(errors="replace")[-500:])
            continue
        out_path = tmp_dir / f"worker_{worker_id}_{algorithm_id}.json"
        if out_path.exists():
            with open(out_path) as f:
                rows = json.load(f)
            all_runs.extend(rows)
            logger.info("worker %d (%s): %d runs", worker_id, algorithm_id, len(rows))

    # Write merged results
    if all_runs:
        # Write CSV
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_runs[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_runs)
        # Write JSON
        json_path = str(out_path).replace(".csv", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2, sort_keys=True)
        print(f"\nwrote {len(all_runs)} rows to {out_path}")

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Cleaned up %s", tmp_dir)

    # Print summary
    if all_runs:
        print(f"\n{'='*100}")
        print(f"  PARALLEL SWEEP COMPLETE  |  {len(all_runs)} runs  |  {workers} workers")
        print(f"{'='*100}")
        ranked = sorted(all_runs, key=lambda r: r.get("net_return_5bps", 0), reverse=True)
        print(f"\n  Top 10 by net return (5bps cost):")
        print(f"  {'Label':30s} {'Return':>8s} {'Net5':>8s} {'DD':>8s} {'Sharpe':>7s} {'Turn':>6s}")
        print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")
        for run in ranked[:10]:
            print(f"  {run['label']:30s} {run['total_return']:+8.2%} {run['net_return_5bps']:+8.2%} "
                  f"{run['max_drawdown']:+8.2%} {run['sharpe']:7.2f} {run['turnover_x_stake']:5.1f}x")

    return 0


if __name__ == "__main__":
    sys.exit(main())
