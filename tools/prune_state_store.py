"""Bring the DuckDB store back in line with the code that reads it.

The refactor that split ``src/algorithms`` and ``src/brokerages`` into packages also changed
where algorithm memory lives: every algorithm now writes ``algorithm_state:<id>:<account>``
through :func:`algorithm_state_key`, where before each one invented its own key. Nothing
migrated the old rows, so the state was not corrupt -- it was *unreachable*, which is worse,
because a cold read looks exactly like a first run. Rally Rotation had been restarting its
eligibility window on every run for that reason, and reporting "1 of the last 1 runs" forever.

Three kinds of row, and the distinction is the whole point of doing this with a script rather
than a DELETE:

*Renamed* -- the algorithm still exists and the value still means something, so it is copied to
the key the code now reads. Only Bursty DCA's accrual qualifies, and it is real banked money.

*Orphaned* -- written by an algorithm that no longer exists (``dca``, ``defensive_momentum``),
or by a key format nothing reads. Dropped.

*Cache* -- regenerable. Dropped when the thing that would regenerate it is gone: a provider no
longer in the config, a symbol no longer in the universe, a backtest whose config fingerprint
can no longer be produced.

Idempotent: running it twice is a no-op, because every step is expressed as "make the store
look like this" rather than "apply this change". ``--dry-run`` prints the plan and writes
nothing. Take a copy of the file first regardless; this deletes rows.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import duckdb

from src.algorithms.bursty_dca.algorithm import SymbolState
from src.algorithms.bursty_dca.config import plan_budgets, raw_plan, sanitize_plan
from src.algorithms.registry import ALGORITHMS
from src.core.config import get_config
from src.data.state_store import STATE_DUCKDB_PATH, algorithm_state_key
from src.data.universe import tradable_symbols

#: Old key -> the algorithm whose state it held. Only algorithms still registered are carried
#: over; the rest are listed here so the report can say *why* a row is being dropped rather
#: than leaving it to the reader to notice the id is gone.
LEGACY_STATE_KEYS = {
    "dca_accrual:bursty_dca:paper": ("bursty_dca", "paper"),
    "dca_accrual:bursty_dca:schwab_individual": ("bursty_dca", "schwab_individual"),
    "dca_accrual:bursty_dca:local_paper": ("bursty_dca", "local_paper"),
    "dca_accrual:dca:paper": ("dca", "paper"),
    "dca_accrual:dca:local_paper": ("dca", "local_paper"),
    "rally_rotation_runtime:paper": ("rally_rotation", "paper"),
}

#: Keys written by code that no longer exists, and which no rename can rescue.
DEAD_STATE_KEYS = (
    # An intraday risk breaker belonging to the deleted ``defensive_momentum`` algorithm.
    "defensive_momentum_intraday_risk",
    # A symbol list with no reader anywhere in src/, web/ or tests/. The tradable universe in
    # walbot.yaml is what decides what gets priced now.
    "watchlist",
)


def _plan_symbols(config: Any) -> set[str]:
    return set(plan_budgets(sanitize_plan(raw_plan(config, "bursty_dca"), tradable_symbols(config))))


def _carry_over(algorithm_id: str, value: str, plan: set[str]) -> dict[str, Any] | None:
    """The part of a legacy value worth keeping, or ``None`` to drop the row.

    Rally Rotation's window is dropped even though the algorithm survives: it is a bare list of
    per-*run* observations with no day attached, and :mod:`..algorithms.rally_rotation.memory`
    now keys on market days. Carrying it would let the old five-fires-a-session inflation
    survive as if it were five days of evidence.
    """
    if algorithm_id not in ALGORITHMS or algorithm_id != "bursty_dca":
        return None
    accrual = json.loads(value)
    if not isinstance(accrual, dict):
        return None
    # Symbols the plan no longer names are never read -- ``plan`` iterates budgets, not state --
    # but they would come back to life the day one is added to a bucket again, carrying an
    # accrued balance earned under a plan nobody remembers writing.
    return {symbol: state for symbol, state in accrual.items() if symbol in plan}


def prune(connection: Any, *, dry_run: bool) -> list[str]:
    config = get_config()
    universe = sorted(set(config.symbols))
    plan = _plan_symbols(config)
    report: list[str] = []

    def run(sql: str, params: list[Any] | None = None) -> int:
        if dry_run:
            return 0
        connection.execute(sql, params or [])
        return 0

    # -- app_state: renames and orphans ------------------------------------------------
    for key, (algorithm_id, account_id) in LEGACY_STATE_KEYS.items():
        row = connection.execute("select value from app_state where key = ?", [key]).fetchone()
        if row is None:
            continue
        carried = _carry_over(algorithm_id, row[0], plan)
        if carried:
            target = algorithm_state_key(algorithm_id, account_id)
            dropped = sorted(set(json.loads(row[0])) - set(carried))
            report.append(
                f"  rename  {key}\n"
                f"       -> {target}  keeping {sorted(carried)}"
                + (f", dropping {dropped}" if dropped else "")
            )
            run(
                "insert into app_state (key, value, updated_at) values (?, ?, now()) "
                "on conflict (key) do update set value = excluded.value, updated_at = excluded.updated_at",
                [target, json.dumps(carried, sort_keys=True)],
            )
        else:
            why = "algorithm no longer registered" if algorithm_id not in ALGORITHMS else "run-indexed window, unreadable as days"
            report.append(f"  drop    {key}  ({why})")
        run("delete from app_state where key = ?", [key])

    for key in DEAD_STATE_KEYS:
        if connection.execute("select 1 from app_state where key = ?", [key]).fetchone():
            report.append(f"  drop    {key}  (no reader in the current code)")
            run("delete from app_state where key = ?", [key])

    # -- algorithm_state: normalise to the shape the dataclass actually round-trips -------
    # Separate from the rename above so it also reaches rows a previous run already moved.
    # The pre-refactor accrual carried a ``path_started_at`` that :class:`SymbolState` has no
    # field for, so it survives every read as a value nothing can ever act on. Rewriting through
    # the dataclass is what makes "matches the current code" true of the contents and not just
    # of the key.
    for key, value in connection.execute(
        "select key, value from app_state where key like 'algorithm_state:bursty_dca:%' order by key"
    ).fetchall():
        stored = json.loads(value)
        clean = {symbol: SymbolState.from_dict(state).as_dict() for symbol, state in stored.items()}
        if clean == stored:
            continue
        extra = sorted({field for state in stored.values() for field in state} - set(next(iter(clean.values()), {})))
        report.append(f"  normalise {key}  (dropping {extra})")
        run("update app_state set value = ?, updated_at = now() where key = ?",
            [json.dumps(clean, sort_keys=True), key])

    # -- backtest_cache: a cache with no eviction ---------------------------------------
    # Every entry is keyed by a hash of the config it was computed under, so an entry whose
    # config has changed is not stale -- it is unreachable, and will sit there forever.
    row = connection.execute("select value from app_state where key = 'backtest_cache'").fetchone()
    if row:
        cache = json.loads(row[0])
        items = cache.get("items") or {}
        if items:
            report.append(f"  clear   backtest_cache  ({len(items)} items, {len(row[0]) / 1e6:.1f} MB)")
            cache["items"] = {}
            run(
                "update app_state set value = ?, updated_at = now() where key = 'backtest_cache'",
                [json.dumps(cache, sort_keys=True)],
            )

    # -- market_bars: symbols outside the tradable universe ------------------------------
    stale = connection.execute(
        "select symbol, count(*) from market_bars where symbol not in (select unnest(?)) "
        "group by 1 order by 1", [universe]).fetchall()
    if stale:
        total = sum(count for _, count in stale)
        report.append(f"  delete  market_bars  {total:,} rows for {[s for s, _ in stale]}")
        run("delete from market_bars where symbol not in (select unnest(?))", [universe])

    # -- api_cache: providers the config no longer names ---------------------------------
    # The provider order is the authority on who gets asked; a cached payload from anyone else
    # can never be served, because the lookup is keyed by provider.
    live = {p.lower() for p in (
        list(config.eod_market_data_provider_order or [])
        + list(config.intraday_market_data_provider_order or [])
    )}
    dead = connection.execute(
        "select provider, count(*) from api_cache where lower(provider) not in (select unnest(?)) "
        "group by 1 order by 1", [sorted(live)]).fetchall()
    if dead:
        report.append(f"  delete  api_cache  {sum(n for _, n in dead):,} rows from {[p for p, _ in dead]} "
                      f"(configured: {sorted(live)})")
        run("delete from api_cache where lower(provider) not in (select unnest(?))", [sorted(live)])

    # -- sentiment_records: no configured provider to refresh them -----------------------
    if not (config.sentiment_data_provider_order or []):
        count = connection.execute("select count(*) from sentiment_records").fetchone()[0]
        if count:
            report.append(f"  delete  sentiment_records  {count} rows (no sentiment provider configured)")
            run("delete from sentiment_records")

    # -- dividends: symbols outside the universe -----------------------------------------
    count = connection.execute(
        "select count(*) from dividends where symbol not in (select unnest(?))", [universe]).fetchone()[0]
    if count:
        report.append(f"  delete  dividends  {count} rows outside the universe")
        run("delete from dividends where symbol not in (select unnest(?))", [universe])

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and write nothing.")
    parser.add_argument("--no-compact", action="store_true",
                        help="Skip the CHECKPOINT that reclaims freed pages on disk.")
    args = parser.parse_args()

    connection = duckdb.connect(str(STATE_DUCKDB_PATH), read_only=args.dry_run)
    report = prune(connection, dry_run=args.dry_run)

    print(f"{'PLAN (nothing written)' if args.dry_run else 'APPLIED'} -- {STATE_DUCKDB_PATH}")
    print("\n".join(report) if report else "  nothing to do")

    if not args.dry_run and not args.no_compact:
        # DuckDB does not return freed pages to the filesystem on DELETE alone.
        connection.execute("CHECKPOINT")
    connection.close()


if __name__ == "__main__":
    main()
