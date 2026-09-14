"""Explain an account's realized P/L, fill by fill, and name every unmatched sell.

The account page reports one number and a count of what it could not match. This prints the
working: every fill the broker reported, the running position each one left behind, and the
symbol and date of each sell that had no opening buy to match against.

Run it where the broker's credentials already live, so only one machine touches them:

    docker exec walbot python -m tools.realized_check schwab_main
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone

from src.api.payloads.accounts import REALIZED_ACTIVITY_DAYS
from src.core.config import get_config
from src.core.pipeline import resolve_brokerage


def main(account_id: str = "") -> int:
    config = get_config(account_id=account_id) if account_id else get_config()
    brokerage = resolve_brokerage(config)
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=REALIZED_ACTIVITY_DAYS)
    fills = brokerage.get_fills(start, end)

    if fills is None:
        print(f"{config.account_id}: this broker reports no fill feed at all.")
        return 0
    print(f"{config.account_id}: {len(fills)} fills from {start} to {end}\n")
    if not fills:
        print("No fills in the window. Realized P/L is genuinely zero, not missing.")
        return 0

    # The same walk the shared matcher does, printed rather than summed, so a wrong total can
    # be traced to the fill that caused it.
    open_lots: dict[str, list[float]] = {}
    realized = 0.0
    unmatched: list[dict] = []
    for fill in sorted(fills, key=lambda row: str(row.get("date") or "")):
        symbol = str(fill.get("symbol") or "").upper()
        quantity = abs(float(fill.get("quantity") or 0.0))
        price = float(fill.get("price") or 0.0)
        multiplier = float(fill.get("multiplier") or 1.0)
        action = str(fill.get("action") or "").lower()
        stamp = str(fill.get("date") or "")[:10]
        shares, average = open_lots.get(symbol, [0.0, 0.0])

        if action == "buy":
            total = shares + quantity
            average = ((shares * average) + (quantity * price)) / total if total else 0.0
            open_lots[symbol] = [total, average]
            print(f"  {stamp}  BUY   {symbol:24} {quantity:>8.2f} @ {price:>9.2f}  -> holding {total:g} @ {average:.2f}")
            continue

        if shares <= 0:
            unmatched.append(fill)
            print(f"  {stamp}  SELL  {symbol:24} {quantity:>8.2f} @ {price:>9.2f}  -- UNMATCHED (nothing open)")
            continue

        closed = min(quantity, shares)
        gain = (price - average) * closed * multiplier
        realized += gain
        open_lots[symbol] = [shares - closed, average]
        note = "" if quantity <= shares else f"  -- {quantity - shares:g} of it UNMATCHED"
        if quantity > shares:
            unmatched.append(fill)
        print(f"  {stamp}  SELL  {symbol:24} {closed:>8.2f} @ {price:>9.2f}  -> {gain:+.2f}{note}")

    print(f"\nRealized P/L: {realized:,.2f} across the window.")
    still_open = {s: lot for s, lot in open_lots.items() if lot[0] > 0}
    if still_open:
        print(f"Still open (not counted): {', '.join(f'{s} {lot[0]:g}' for s, lot in still_open.items())}")
    if unmatched:
        print(f"\n{len(unmatched)} unmatched sell(s) -- no opening buy inside the window:")
        for fill in unmatched:
            print(f"  {str(fill.get('date') or '')[:10]}  {fill.get('symbol')}  qty {fill.get('quantity')}")
        print(
            "\nEach is a position opened before the window began, or a fill whose opening leg\n"
            "the broker files under a different transaction type (an exercise or assignment is\n"
            "not a trade). Their cost is unknown, so they are excluded from the total above --\n"
            "which means the real realized figure is larger than what the page reports."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
