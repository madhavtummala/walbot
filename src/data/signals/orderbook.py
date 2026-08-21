"""Microstructure signals from streaming data: spread and hidden-liquidity evidence.

These read the shapes :class:`~src.connectors.market.schwab_stream.StreamStore` produces --
quote dicts with ``bid``/``ask``, book dicts with ``bids``/``asks`` levels, and trade
prints. They are heuristics over partial information: the store keeps the *latest* book and
a bounded window of prints, not full depth history, so iceberg detection here is evidence
gathering for a human or an agent, not a market-making grade estimator.
"""

from __future__ import annotations

from typing import Any


def quote_spread(quote: dict[str, Any]) -> dict[str, float] | None:
    """Absolute and relative spread of one quote, in price points and basis points.

    ``None`` when the quote carries no two-sided market -- the field is absent pre-open and
    for symbols the stream has only seen trade on.
    """
    try:
        bid = float(quote.get("bid") or 0.0)
        ask = float(quote.get("ask") or 0.0)
    except (TypeError, ValueError):
        return None
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2
    absolute = ask - bid
    return {"spread": round(absolute, 6), "spread_bps": round(absolute / mid * 10_000, 2) if mid else 0.0}


def iceberg_evidence(
    book: dict[str, Any],
    trades: list[dict[str, Any]],
    *,
    size_multiple: float = 3.0,
    min_prints: int = 3,
) -> list[dict[str, Any]]:
    """Price levels where executed volume outran displayed size -- possible iceberg orders.

    An iceberg shows a small slice and refills as it trades, so its fingerprint is repeated
    prints at one price while the book keeps displaying far less than has already executed
    there. For every traded price this compares total printed size against the size currently
    displayed at that level; a multiple of ``size_multiple`` across at least ``min_prints``
    prints is reported, strongest first.

    Deliberately conservative: without historical book snapshots a refill that finished
    before we looked is invisible, so absence of evidence here is weak evidence of absence.
    """
    if not book or not trades:
        return []
    displayed: dict[float, float] = {}
    for side in ("bids", "asks"):
        for level in book.get(side) or []:
            price = round(float(level.get("price") or 0.0), 4)
            displayed[price] = displayed.get(price, 0.0) + float(level.get("size") or 0.0)

    executed: dict[float, dict[str, float]] = {}
    for trade in trades:
        price = round(float(trade.get("price") or 0.0), 4)
        entry = executed.setdefault(price, {"volume": 0.0, "prints": 0})
        entry["volume"] += float(trade.get("size") or 0.0)
        entry["prints"] += 1

    candidates: list[dict[str, Any]] = []
    for price, stats in executed.items():
        shown = displayed.get(price, 0.0)
        volume = stats["volume"]
        if stats["prints"] < min_prints or shown <= 0:
            continue
        multiple = volume / shown
        if multiple < size_multiple:
            continue
        candidates.append(
            {
                "price": price,
                "executed_volume": round(volume, 4),
                "prints": int(stats["prints"]),
                "displayed_size": round(shown, 4),
                "hidden_multiple": round(multiple, 2),
            }
        )
    candidates.sort(key=lambda item: -item["hidden_multiple"])
    return candidates
