"""Prune the market-data cache down to one provider and two resolutions.

The cache accumulated across a period when three providers were in rotation and the intraday
grid was whatever yfinance could serve. That left four kinds of dead weight: providers no
longer used, a 15-minute tier superseded by 5-minute bars covering the same span, symbols
dropped from the tradable universe, and bars for nothing anyone trades.

Pruning is worth doing rather than ignoring because this cache is the only source of intraday
history beyond Schwab's 259-day window -- every row kept is a row that has to stay correct
through future migrations, and every row dropped is one that never can go wrong again.

Deliberately a separate command, never automatic: deleting cached bars is irreversible for
anything older than what the API will re-serve.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from src.core.config import get_config
from src.core.options import is_osi_symbol, parse_osi
from src.data.duckdb_store import (
    DAILY_INTERVAL_MINUTES,
    DUCKDB_STATE_PATH,
    _connect,
    market_bars_summary,
)

logger = logging.getLogger(__name__)

#: What the cache is for: Schwab is the only feed in use, at one intraday grid plus dailies.
KEEP_PROVIDER = "schwab"
KEEP_INTERVALS = (5, DAILY_INTERVAL_MINUTES)


def _universe() -> set[str]:
    return {symbol.upper() for symbol in get_config().symbols}


def _contract_is_live(symbol: str, as_of: date) -> bool:
    """Whether an OSI symbol names a contract that has not expired yet.

    Option-contract bars are kept while the contract trades: Options Flip predicts its exit
    band from the contract's *own* price history, so deleting it would silently drop that
    algorithm back to the underlying-delta translation. Once the contract expires the rows can
    never be relevant again -- nobody can trade it, and no future run will ask for it -- so
    they are dead weight that only grows, one contract at a time, every expiry.

    An unparseable symbol is treated as live: refusing to delete what we cannot identify is the
    recoverable direction.
    """
    try:
        return parse_osi(symbol)["expiry"] >= as_of
    except Exception:  # noqa: BLE001 - an unreadable symbol is not a licence to delete it
        return True


def _keep_symbols(db_path: str | None = None, as_of: date | None = None) -> set[str]:
    """Every symbol worth keeping: the tradable universe, plus unexpired option contracts."""
    as_of = as_of or datetime.now(timezone.utc).date()
    keep = _universe()
    for row in market_bars_summary(db_path=db_path):
        symbol = str(row["symbol"]).upper()
        if is_osi_symbol(symbol) and _contract_is_live(symbol, as_of):
            keep.add(symbol)
    return keep


def plan(db_path: str | None = None, as_of: date | None = None) -> dict[str, Any]:
    """What would be deleted, and what would remain, without touching anything."""
    as_of = as_of or datetime.now(timezone.utc).date()
    wanted = _universe()
    keep = _keep_symbols(db_path=db_path, as_of=as_of)
    rows = market_bars_summary(db_path=db_path)
    buckets: dict[str, int] = {
        "other_providers": 0,
        "other_intervals": 0,
        "expired_contracts": 0,
        "outside_universe": 0,
        "kept": 0,
    }
    expired: set[str] = set()
    for row in rows:
        symbol = str(row["symbol"]).upper()
        if row["provider"] != KEEP_PROVIDER:
            buckets["other_providers"] += row["rows"]
        elif row["interval_minutes"] not in KEEP_INTERVALS:
            buckets["other_intervals"] += row["rows"]
        elif is_osi_symbol(symbol):
            # Split out from "outside the universe" because it is a different fact: a contract
            # is not a symbol someone stopped trading, it is one that stopped existing.
            if symbol in keep:
                buckets["kept"] += row["rows"]
            else:
                buckets["expired_contracts"] += row["rows"]
                expired.add(symbol)
        elif symbol not in wanted:
            buckets["outside_universe"] += row["rows"]
        else:
            buckets["kept"] += row["rows"]
    return {
        "universe": sorted(wanted),
        "keep_provider": KEEP_PROVIDER,
        "keep_intervals": list(KEEP_INTERVALS),
        "live_contracts": sorted(symbol for symbol in keep if is_osi_symbol(symbol)),
        "expired_contracts": sorted(expired),
        "rows": buckets,
    }


def prune(*, db_path: str | None = None, apply: bool = False) -> dict[str, Any]:
    """Delete everything outside the kept provider, intervals, and universe."""
    outcome = plan(db_path=db_path)
    if not apply:
        outcome["applied"] = False
        return outcome

    # Live option contracts are kept alongside the universe -- see ``_contract_is_live``.
    wanted = sorted(_keep_symbols(db_path=db_path))
    placeholders = ",".join(["?"] * len(wanted))
    intervals = ",".join(str(int(value)) for value in KEEP_INTERVALS)
    with _connect(db_path) as connection:
        before = int(connection.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0] or 0)
        connection.execute(
            f"""
            DELETE FROM market_bars
            WHERE provider <> ?
               OR interval_minutes NOT IN ({intervals})
               OR symbol NOT IN ({placeholders})
            """,
            [KEEP_PROVIDER, *wanted],
        )
        after = int(connection.execute("SELECT COUNT(*) FROM market_bars").fetchone()[0] or 0)
        # Reclaims the pages the delete freed; without it the file keeps its old size.
        connection.execute("CHECKPOINT")
    outcome.update({"applied": True, "rows_before": before, "rows_after": after, "deleted": before - after})
    logger.info("Pruned %s rows; %s remain", before - after, after)
    return outcome


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually delete. Omit for a dry run.")
    parser.add_argument("--db-path", default=DUCKDB_STATE_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    print(json.dumps(prune(db_path=args.db_path, apply=args.apply), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
