"""The day-by-day story: every gate that blocked, every order placed/repriced/cancelled, every
fill -- for a symbol, across the walk-forward month, in one chronological narrative.

``walk_forward()`` returns every tick's full check list in memory, which answers "what did the
algorithm see" but leaves "what did it *do*" to be read back out of raw order snapshots by eye.
This tool does that read: it diffs each tick's desired orders against the previous tick's for the
same key (``{symbol}:entry`` / ``:target`` / ``:stop``) to say PLACED, REPRICED or CANCELLED,
interleaves the fills ``walk_forward``'s own log recorded, and -- on a tick with nothing resting
and nothing filled -- names the first blocking gate rather than printing the same "not trending"
reading every five minutes.

Run:

    STATE_DUCKDB_PATH=data/walbot.duckdb python -m tools.options_flip_day_narrative
    STATE_DUCKDB_PATH=data/walbot.duckdb python -m tools.options_flip_day_narrative --symbol GLD
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import pandas as pd

from .options_flip_contract_backtest import SYMBOLS
from .options_flip_walk_forward import walk_forward

logger = logging.getLogger("optflip_narrative")


def _fmt_minute(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _order_price(order: dict[str, Any]) -> float | None:
    return order.get("limit_price") if order.get("limit_price") is not None else order.get("stop_price")


def _diff_orders(prev: dict[str, dict], cur: dict[str, dict]) -> list[str]:
    """Order lifecycle events between two consecutive ticks, keyed by ``{symbol}:{role}``."""
    events = []
    for key, order in cur.items():
        price = _order_price(order)
        before = prev.get(key)
        if before is None:
            events.append(
                f"PLACED   {key:<14s} {order['action']:5s} {order['order_type']:6s} "
                f"{order['symbol']} @ {price}"
            )
        elif _order_price(before) != price:
            events.append(
                f"REPRICED {key:<14s} {_order_price(before)} -> {price}"
            )
    for key, order in prev.items():
        if key not in cur:
            events.append(f"CANCELLED {key:<13s} {order['symbol']} (was @ {_order_price(order)})")
    return events


#: Sentinel minute for the end-of-day commentary (resting orders, day summary) -- past every
#: real tick minute (600-960), so it always sorts last within its day regardless of symbol.
_DAY_END = 10_000

#: The role suffix on a ``day`` order's key -- see ``lifecycle._flat_or_bidding``, the only order
#: this algorithm places with ``time_in_force="day"``. Every other key (``:target``, ``:stop``) is
#: ``gtc`` and genuinely survives the broker's own end-of-day expiry.
_DAY_ORDER_SUFFIX = ":entry"


def symbol_events(
    symbol: str, log: pd.DataFrame, daily: pd.DataFrame, ticks: list[dict]
) -> dict[str, list[tuple[int, str]]]:
    """``{day: [(minute, text), ...]}`` for one symbol's whole replay.

    ``prev_orders`` -- the baseline the diff compares each tick against -- is carried *across*
    day boundaries, not reset per day. It has to be: the target and stop are ``gtc`` and genuinely
    still rest at the broker when a new session opens, so the first tick of a new day should read
    REPRICED against yesterday's close, not PLACED against an empty baseline -- resetting per day
    was this function's own bug, caught by a first-tick-of-the-day "PLACED" that should have been
    a "REPRICED".

    The entry is the one key that really is gone every morning -- it is a ``day`` order and the
    exchange expires it at the close regardless of what the algorithm still wants -- so ``:entry``
    is dropped from the carried baseline at each day boundary, and a still-open bid correctly
    reads PLACED again the next morning rather than a REPRICED the broker never actually did.
    """
    events: dict[str, list[tuple[int, str]]] = {}
    fills_by_day: dict[str, list[dict]] = {}
    if not log.empty:
        for _, row in log.iterrows():
            fills_by_day.setdefault(row["day"], []).append(row.to_dict())

    ticks_by_day: dict[str, list[dict]] = {}
    for tick in ticks:
        ticks_by_day.setdefault(tick["day"], []).append(tick)

    prev_orders: dict[str, dict] = {}
    for day in sorted(ticks_by_day):
        # The broker's own end-of-day expiry for the ``day``-TIF entry -- not ours to diff, since
        # nothing we did cancelled it and printing "CANCELLED" here would blame the algorithm for
        # what the exchange did on its own.
        prev_orders = {k: v for k, v in prev_orders.items() if not k.endswith(_DAY_ORDER_SUFFIX)}

        day_ticks = sorted(ticks_by_day[day], key=lambda t: t["minute"])
        day_fills = sorted(fills_by_day.get(day, []), key=lambda f: f["minute"])
        day_events: list[tuple[int, str]] = []
        prev_blocking_label: str | None = None
        fill_idx = 0
        for tick in day_ticks:
            minute = tick["minute"]
            # Fills that landed at or before this tick's minute, in order -- resolved against
            # the *previous* tick's resting order (see walk_forward's own no-lookahead comment),
            # so printing them here, before this tick's fresh orders, keeps that causality visible.
            while fill_idx < len(day_fills) and day_fills[fill_idx]["minute"] <= minute:
                fill = day_fills[fill_idx]
                ret = fill.get("ret")
                ret_text = f", {ret:+.1%}" if ret is not None and not pd.isna(ret) else ""
                day_events.append((fill["minute"], (
                    f"FILL  {fill['event']:12s} {fill['osi']} @ {fill['price']:.2f}{ret_text}"
                )))
                fill_idx += 1

            orders = json.loads(tick["orders_json"] or "[]")
            cur_orders = {o["key"]: o for o in orders}
            for event in _diff_orders(prev_orders, cur_orders):
                day_events.append((minute, event))
            prev_orders = cur_orders

            checks = json.loads(tick["checks_json"] or "[]")
            blocking = next((c for c in checks if c.get("blocking")), None)
            blocking_label = blocking.get("label") if blocking else None
            # Only when the *reason* changes -- a gate blocked for the same reason five minutes
            # running is not new information, but "Trend strength" giving way to "Holding VWAP"
            # partway through the morning is.
            if blocking_label and blocking_label != prev_blocking_label:
                day_events.append((minute, f"GATE  {blocking_label}: {blocking.get('value')}"))
            prev_blocking_label = blocking_label

        if prev_orders:
            names = ", ".join(sorted(prev_orders))
            day_events.append((_DAY_END, f"(resting into the close: {names})"))

        day_row = daily[daily["day"] == day]
        if not day_row.empty:
            r = day_row.iloc[0]
            day_events.append((_DAY_END, (
                f"day summary: {r['state_at_open']} -> {r['state_at_close']}"
                f"{'  [ENTRY BID PLACED]' if r['bid_placed'] else ''}"
                f"{'  [FILLED TODAY]' if r['filled_today'] else ''}"
            )))
        events[day] = day_events
    return events


def build_merged_narrative(all_events: dict[str, dict[str, list[tuple[int, str]]]]) -> list[str]:
    """One chronological log across every symbol, day by day -- the union of each symbol's
    ``symbol_events``, re-sorted by minute (then symbol, for ties) instead of grouped by symbol."""
    lines: list[str] = []
    days = sorted({day for events in all_events.values() for day in events})
    for day in days:
        lines.append(f"\n--- {day} ---")
        merged: list[tuple[int, str, str]] = []
        for symbol, events in all_events.items():
            for minute, text in events.get(day, []):
                merged.append((minute, symbol, text))
        merged.sort(key=lambda item: (item[0], item[1]))
        for minute, symbol, text in merged:
            label = "day end" if minute == _DAY_END else _fmt_minute(minute)
            lines.append(f"  {label:>8s}  {symbol:5s} {text}")
    return lines


def build_narrative(symbol: str, log: pd.DataFrame, daily: pd.DataFrame, ticks: list[dict]) -> list[str]:
    """One symbol's whole replay as pre-joined lines -- a thin formatter over ``symbol_events``,
    so the day-boundary fix documented there only has to exist in one place."""
    lines: list[str] = []
    for day, day_events in symbol_events(symbol, log, daily, ticks).items():
        lines.append(f"\n--- {day} ---")
        for minute, text in day_events:
            label = "        " if minute == _DAY_END else f"  {_fmt_minute(minute)} "
            lines.append(f"{label} {text}")
    return lines


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=None, help="Defaults to every symbol.")
    parser.add_argument(
        "--merged", action="store_true",
        help="One chronological log across every symbol, interleaved by minute, instead of a "
             "section per symbol.",
    )
    args = parser.parse_args()

    from src.core.config import get_config
    config = get_config()
    symbols = [args.symbol.upper()] if args.symbol else SYMBOLS

    if args.merged:
        all_events = {}
        for symbol in symbols:
            log, daily, ticks = walk_forward(symbol, config)
            if not ticks:
                print(f"{symbol}: no sessions replayed (missing cache?)")
                continue
            all_events[symbol] = symbol_events(symbol, log, daily, ticks)
        for line in build_merged_narrative(all_events):
            print(line)
        return 0

    for symbol in symbols:
        log, daily, ticks = walk_forward(symbol, config)
        if not ticks:
            print(f"\n{'=' * 100}\n{symbol}: no sessions replayed (missing cache?)\n{'=' * 100}")
            continue
        print(f"\n{'=' * 100}\n{symbol}\n{'=' * 100}")
        for line in build_narrative(symbol, log, daily, ticks):
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
