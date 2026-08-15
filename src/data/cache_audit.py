"""Check that the daily and intraday tiers of the cache tell the same story.

Both now come from Schwab, both are stamped when their close occurred, and a 5-minute bar
ending at the session close describes exactly the same moment as that session's daily bar.
So the two must agree there -- and where they do not, the gap says something specific about
which of them is wrong.

That check matters more than it used to. Reads blend resolutions into one series, so a
systematic offset between the tiers is not a curiosity: it is a step discontinuity in the
middle of a price history, at whatever point the fine bars run out.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import pandas as pd

from src.core.config import get_config
from src.data.duckdb_store import DAILY_INTERVAL_MINUTES, DUCKDB_STATE_PATH, _connect

logger = logging.getLogger(__name__)

#: Below this the two tiers are the same number to within rounding.
MATCH_TOLERANCE = 0.01


def audit(
    *,
    intraday_interval: int = 5,
    provider: str = "schwab",
    db_path: str = DUCKDB_STATE_PATH,
) -> dict[str, Any]:
    """Compare each session's daily close against the intraday bar covering the same instant."""
    wanted = sorted({symbol.upper() for symbol in get_config().symbols})
    if not wanted:
        return {"symbols": [], "rows": []}
    placeholders = ",".join(["?"] * len(wanted))
    query = f"""
        SELECT d.symbol,
               COUNT(*) AS shared_sessions,
               SUM(CASE WHEN abs(d.close - i.close) <= {MATCH_TOLERANCE} THEN 1 ELSE 0 END) AS matching,
               MEDIAN(abs(d.close - i.close) / nullif(d.close, 0)) AS median_rel_gap,
               MAX(abs(d.close - i.close) / nullif(d.close, 0)) AS max_rel_gap,
               MIN(d.timestamp) AS first_session,
               MAX(d.timestamp) AS last_session
        FROM market_bars d
        JOIN market_bars i
          ON d.symbol = i.symbol AND d.provider = i.provider AND d.timestamp = i.timestamp
        WHERE d.provider = ? AND d.interval_minutes = ? AND i.interval_minutes = ?
          AND d.symbol IN ({placeholders})
        GROUP BY d.symbol
        ORDER BY d.symbol
    """
    with _connect(db_path) as connection:
        frame = connection.execute(
            query, [provider, DAILY_INTERVAL_MINUTES, int(intraday_interval), *wanted]
        ).fetchdf()

    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        shared = int(record["shared_sessions"])
        matching = int(record["matching"] or 0)
        rows.append(
            {
                "symbol": str(record["symbol"]),
                "shared_sessions": shared,
                "matching": matching,
                "match_rate": round(matching / shared, 4) if shared else 0.0,
                "median_rel_gap": round(float(record["median_rel_gap"] or 0.0), 6),
                "max_rel_gap": round(float(record["max_rel_gap"] or 0.0), 6),
                "first_session": pd.Timestamp(record["first_session"]).isoformat(),
                "last_session": pd.Timestamp(record["last_session"]).isoformat(),
            }
        )
    covered = {row["symbol"] for row in rows}
    return {
        "provider": provider,
        "intraday_interval": int(intraday_interval),
        "symbols_checked": len(rows),
        "symbols_without_overlap": sorted(set(wanted) - covered),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intraday-interval", type=int, default=5)
    parser.add_argument("--provider", default="schwab")
    parser.add_argument("--db-path", default=DUCKDB_STATE_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(json.dumps(audit(intraday_interval=args.intraday_interval, provider=args.provider,
                           db_path=args.db_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
