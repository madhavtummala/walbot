"""Realized P/L, reconstructed from a broker's own fill record.

No venue reports this as a field. Schwab and Alpaca both answer "what do you hold and what is
it worth", and neither answers "what did you make on what you already sold" -- so the number
has to be built by matching sells against the buys that opened them.

Average cost rather than FIFO. The two disagree only on *which* lot a partial sell closed,
which matters for tax and not for the question this answers ("has this account made money"),
and average cost needs no lot identifiers that a broker may not expose.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

#: One fill: ``{symbol, action, quantity, price, multiplier, date}``. ``multiplier`` is 100 for
#: an option contract and 1 for shares, because a broker quotes an option per share while
#: selling it a hundred at a time.
Fill = Dict[str, Any]


def realized_from_fills(fills: Iterable[Fill]) -> Dict[str, Any]:
    """Realized P/L across a sequence of fills, oldest first.

    Returns ``{realized_pl, closes, unmatched}``. ``unmatched`` counts sells this could find no
    open position for, which happens for a position opened before the window began -- its cost
    is genuinely unknown, so it is left out of the total and counted instead. A caller that
    reports the total without the count would be publishing a number that is quietly too small.
    """
    #: Open position per symbol, as running shares and running average cost per share.
    open_lots: Dict[str, List[float]] = {}
    realized = 0.0
    closes = 0
    unmatched = 0

    for fill in _chronological(fills):
        symbol = str(fill.get("symbol") or "").upper()
        quantity = abs(float(fill.get("quantity") or 0.0))
        price = float(fill.get("price") or 0.0)
        multiplier = float(fill.get("multiplier") or 1.0)
        if not symbol or quantity <= 0 or price <= 0:
            continue

        shares, average = open_lots.get(symbol, [0.0, 0.0])
        if str(fill.get("action") or "").lower() == "buy":
            total = shares + quantity
            # Weighted, so a second buy at a different price moves the basis rather than
            # replacing it.
            average = ((shares * average) + (quantity * price)) / total if total else 0.0
            open_lots[symbol] = [total, average]
            continue

        if shares <= 0:
            # A sell with nothing open: either a short, or a close of something bought before
            # the window. Both make the basis unknowable from this feed alone.
            unmatched += 1
            continue

        # A sell larger than what is open closes what it can; the rest is unmatched.
        closed = min(quantity, shares)
        realized += (price - average) * closed * multiplier
        closes += 1
        if quantity > shares:
            unmatched += 1
        open_lots[symbol] = [shares - closed, average]

    return {"realized_pl": realized, "closes": closes, "unmatched": unmatched}


def _chronological(fills: Iterable[Fill]) -> List[Fill]:
    """Oldest first, because a sell can only be matched against a buy that preceded it.

    Sorted here rather than trusted from the caller: every broker feed in this codebase hands
    back newest-first, which would match each sell against buys that had not happened yet.
    """
    return sorted(fills or [], key=lambda fill: str(fill.get("date") or ""))
