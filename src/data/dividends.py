"""The dividend ledger: discrete cash events, stored apart from prices.

This table is the whole reason ``market_bars`` can stay clean. A distribution used to be
folded into ``adjusted_close`` by back-adjusting the price series, which had three costs: it
rewrote history every time a new payment landed, it made the same instant carry two different
prices depending on which resolution you asked for, and it hid real cash inside what looked
like price appreciation -- so a backtest that marked positions at ``close`` booked no income
at all. Holding the events here instead lets each consumer take what it actually needs: the
ledger credits cash, the signal layer derives a total-return series on demand, and neither
one has to mutate a price.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Iterable

import pandas as pd

from src.core.interfaces import CashDividend
from src.data.duckdb_store import (
    _connect,
    connection_is_read_only,
    create_dividends_table,
)

logger = logging.getLogger(__name__)

DIVIDEND_COLUMNS = [
    "symbol",
    "ex_date",
    "record_date",
    "payable_date",
    "amount",
    "special",
    "source",
    "fetched_at",
]


def _collapse(dividends: Iterable[CashDividend]) -> list[CashDividend]:
    """One row per ``(symbol, ex_date)``, which is what a ledger credits against.

    Two distinct things arrive looking alike, and they need opposite treatment:

    * The same payment reported twice. Alpaca files GPIX's 2025-01-03 distribution under two
      corporate-action ids with an identical 0.34964 rate; yfinance and the 0.35 print both
      say it was paid once. Keeping both would pay the holder twice.
    * Two real payments sharing an ex-date -- ordinary income plus a capital-gains
      distribution, as XYLD did on 2021-12-30 with 0.343335 and 0.114365. Both are cash that
      actually arrives, so dropping either understates the return.

    Equal amounts separate the cases: identical rates on one ex-date are a duplicated record,
    different rates are different payments. Two genuinely equal payments on one date would be
    collapsed, which has not been observed and is the safer way to be wrong -- understating
    income is recoverable, inventing it silently is not.
    """
    seen: dict[tuple[str, Any], dict[float, CashDividend]] = {}
    for item in dividends:
        if not item.symbol or item.amount <= 0 or item.ex_date is None:
            continue
        key = (item.symbol.upper(), item.ex_date)
        # Rounded so float noise across providers cannot masquerade as a second payment.
        seen.setdefault(key, {})[round(float(item.amount), 6)] = item

    out: list[CashDividend] = []
    for (symbol, ex_date), by_amount in seen.items():
        parts = list(by_amount.values())
        first = parts[0]
        out.append(
            CashDividend(
                symbol=symbol,
                ex_date=ex_date,
                amount=float(sum(p.amount for p in parts)),
                payable_date=first.payable_date or ex_date,
                record_date=first.record_date,
                special=any(p.special for p in parts),
                source=first.source,
            )
        )
    return out


def write_dividends(
    dividends: Iterable[CashDividend],
    *,
    db_path: str | None = None,
) -> int:
    """Upsert distributions keyed by ``(symbol, ex_date)``.

    A published dividend does not change, so a re-fetch should be a no-op rather than a
    duplicate. The key enforces that at the storage layer instead of trusting every caller to
    de-duplicate -- which is exactly the trust that left 4,783 duplicated SGOV bars behind.
    Because :func:`_collapse` runs first, the stored amount is already the day's total, so
    re-writing it replaces rather than accumulates.
    """
    rows = []
    now = datetime.now(timezone.utc)
    for item in _collapse(dividends):
        rows.append(
            (
                item.symbol.upper(),
                item.ex_date,
                item.record_date,
                item.payable_date or item.ex_date,
                float(item.amount),
                bool(item.special),
                item.source or "",
                now,
            )
        )
    if not rows:
        return 0
    with _connect(db_path) as connection:
        create_dividends_table(connection)
        connection.executemany(
            """
            INSERT OR REPLACE INTO dividends
                (symbol, ex_date, record_date, payable_date, amount, special, source, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def read_dividends(
    symbols: list[str] | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """Distributions as a frame, oldest ex-date first."""
    clauses: list[str] = []
    params: list[Any] = []
    if symbols:
        wanted = sorted({str(s).upper() for s in symbols})
        clauses.append("symbol IN (" + ",".join(["?"] * len(wanted)) + ")")
        params.extend(wanted)
    if start is not None:
        clauses.append("ex_date >= ?")
        params.append(start)
    if end is not None:
        clauses.append("ex_date <= ?")
        params.append(end)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(db_path) as connection:
        # A read-only connection (inside a read-only batch pool) cannot run the CREATE, but
        # then it does not need to: the table was created by whichever read-write connection
        # opened the file first. See ``connection_is_read_only``.
        if not connection_is_read_only(connection):
            create_dividends_table(connection)
        frame = connection.execute(
            f"SELECT {', '.join(DIVIDEND_COLUMNS)} FROM dividends {where} ORDER BY symbol, ex_date",
            params,
        ).fetchdf()
    if frame.empty:
        return pd.DataFrame(columns=DIVIDEND_COLUMNS)
    return frame


def dividends_by_symbol(
    symbols: list[str] | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
    db_path: str | None = None,
) -> dict[str, pd.Series]:
    """``{symbol: Series(amount, indexed by ex-date)}`` -- the shape both consumers want."""
    frame = read_dividends(symbols, start=start, end=end, db_path=db_path)
    out: dict[str, pd.Series] = {}
    if frame.empty:
        return out
    for symbol, group in frame.groupby("symbol"):
        series = pd.Series(
            group["amount"].astype(float).to_numpy(),
            index=pd.to_datetime(group["ex_date"]),
        )
        out[str(symbol)] = series.sort_index()
    return out


