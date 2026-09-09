"""Walk Options Flip forward through real history by calling the *actual* ``plan()``.

``options_flip_contract_backtest.py`` re-implements pieces of the lifecycle -- a static or
ratcheted fill check, a hand-rolled target/deadline resolver -- against levels computed by calling
the real
``conditional_levels``/``choose_band``, but never through ``OptionsFlipAlgorithm.plan()`` itself.
That leaves two things no re-implementation can get right: whether a held position correctly
blocks a new entry into the same symbol (this algorithm holds *one* contract at a time), and
whether the entry and exit ratchets, band prediction, and gates all agree with each other the way
they do in production, because they are one function there and several approximations here.

This tool removes the approximation. It drives ``OptionsFlipAlgorithm.plan()`` at the same 5-minute
cadence the live cron uses, across real Sep-18 2026 option history, with:

* a synthetic option chain per tick, priced off the contract's own real close (bid/ask collapsed
  to that price -- no historical spread exists to read), delta from Black-Scholes on the
  underlying's realised vol, IV backed out of the contract's own real price for the rest of the
  greeks -- the same proxy ``options_flip_contract_backtest.py`` already uses, just re-derived
  every tick instead of once at 10:00;
* real option and underlying bars for the band and the ratchets, at the resolution the algorithm
  itself asks for;
* state threaded through ``context.state`` -> ``plan.state`` -> next tick's ``context.state``, a
  plain dict carried by the loop -- the same pattern Rally Rotation's own tests use
  (``tests/test_rally_rotation.py``'s ``context_for``), not the state store: nothing here calls
  ``execute()``, so there is no ``save_state()`` to redirect with ``ephemeral_state()``.

**What is still a proxy, stated plainly:** the option chain (no historical chains exist to read);
zero bid/ask spread (only a close price is cached, so ``max_spread_pct`` never binds); open
interest is the current-day snapshot, not historical. Everything past contract selection --
gates, band prediction, both ratchets, one-contract-per-symbol state -- is the real code.

Run (needs the same cache ``tools/_optcache/fetch.py`` + ``fetch_options.py`` build):

    STATE_DUCKDB_PATH=data/walbot.duckdb python -m tools.options_flip_walk_forward
"""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

from src.algorithms.options_flip.algorithm import OptionsFlipAlgorithm
from src.core.interfaces import AlgorithmContext
from src.core.options import CALL, OptionContract, osi_symbol, parse_osi

from .options_flip_contract_backtest import (
    EXPIRY,
    SYMBOLS,
    _bs_delta,
    _bs_greeks,
    _bs_price_iv,
    _load_daily,
    _load_intraday,
    _load_option_strikes,
    _market_daily_through,
    _realized_ann_vol,
)

logger = logging.getLogger("optflip_walk_forward")

#: Every 5 minutes from the first fire to the close, matching ``OptionsFlipAlgorithm.cron``.
TICKS = list(range(10 * 60, 16 * 60 + 1, 5))


def _tick_key(day: date, minute: int) -> int:
    """A monotonic int for ``(day, minute)`` -- sortable and hashable, and free of timezone
    handling: the tick loop only ever needs *ordering* and *exact match*, never a real clock."""
    return day.toordinal() * 10_000 + minute


def _prepare_strikes(opt_all: pd.DataFrame) -> dict[float, dict[str, Any]]:
    """Index each cached strike's bars once, so every tick is an O(log n) lookup instead of a
    fresh ``to_datetime``/``tz_convert``/filter over the whole cache -- the tick loop calls this
    tens of thousands of times, and re-deriving the same index that often is what made the first
    version of this tool time out.
    """
    prepared: dict[float, dict[str, Any]] = {}
    for strike, sub in opt_all.groupby("strike"):
        local = sub.copy()
        ts = pd.to_datetime(local["timestamp"], utc=True).dt.tz_convert("America/New_York")
        local["day"] = ts.dt.date
        local["minute"] = ts.dt.hour * 60 + ts.dt.minute
        local["key"] = [_tick_key(d, m) for d, m in zip(local["day"], local["minute"])]
        local = local.sort_values("key").reset_index(drop=True)
        prepared[float(strike)] = {
            "keys": local["key"].to_numpy(),
            "rows": local[["timestamp", "open", "high", "low", "close", "oi", "key"]]
                    .to_dict("records"),
        }
    return prepared


def _latest_row(strike_index: dict[str, Any], key: int) -> dict[str, Any] | None:
    """The most recent bar at or before ``key`` for one strike, via binary search."""
    idx = int(np.searchsorted(strike_index["keys"], key, side="right")) - 1
    return strike_index["rows"][idx] if idx >= 0 else None


def _exact_row(strike_index: dict[str, Any], key: int) -> dict[str, Any] | None:
    """The bar for exactly this tick, or ``None`` if this strike did not print then."""
    row = _latest_row(strike_index, key)
    return row if row is not None and row["key"] == key else None


#: No historical bid/ask exists to read, only a close. Splitting it symmetrically into a small
#: assumed spread rather than collapsing bid=ask=mark matters for more than realism:
#: ``select_contract``'s liquidity gate requires ``0 < spread_pct`` -- a real two-sided market --
#: precisely to exclude a degenerate quote, and a *literal* zero spread is exactly that
#: degenerate case. Bid=ask=mark made every synthetic contract fail liquidity regardless of open
#: interest, which read as "nothing tradable" for a reason that had nothing to do with the day.
SYNTHETIC_SPREAD_PCT = 0.02


def _synthetic_chain(
    strikes: dict[float, dict[str, Any]], daily_through: pd.DataFrame, spot: float,
    day: date, minute: int, symbol: str,
) -> list[OptionContract]:
    """One ``OptionContract`` per cached Sep-18 strike, priced as of this tick.

    Mark is the strike's own real close nearest this tick; bid/ask are that mark split around a
    small assumed spread (see ``SYNTHETIC_SPREAD_PCT``) since no historical spread exists to
    read. Delta comes from Black-Scholes on the underlying's realised vol; the rest of the
    greeks come from IV backed out of the contract's own real price, the same fallback
    ``fill_missing_deltas`` uses live.
    """
    key = _tick_key(day, minute)
    vol = _realized_ann_vol(daily_through, day)
    years = max((EXPIRY - day).days, 0) / 365.0
    asof_ms = pd.Timestamp(f"{day} {minute // 60:02d}:{minute % 60:02d}", tz="America/New_York").timestamp() * 1000.0
    contracts: list[OptionContract] = []
    for strike, index in strikes.items():
        row = _latest_row(index, key)
        if row is None:
            continue
        mark = float(row["close"])
        if mark <= 0:
            continue
        oi = int(row.get("oi", 0) or 0)
        delta = _bs_delta(spot, strike, day, vol)
        iv = _bs_price_iv(spot, strike, years, mark) if years > 0 else 0.0
        greeks = _bs_greeks(spot, strike, years, iv, mark) if iv > 0 else {}
        osi = osi_symbol(symbol, EXPIRY, CALL, strike, padded=True)
        half_spread = mark * SYNTHETIC_SPREAD_PCT / 2.0
        contracts.append(OptionContract(
            osi_symbol=osi, underlying=symbol, option_type=CALL, strike=strike,
            expiry=EXPIRY, bid=round(mark - half_spread, 2), ask=round(mark + half_spread, 2), mark=mark,
            delta=float(greeks.get("delta", delta)), gamma=float(greeks.get("gamma", 0.0)),
            theta=float(greeks.get("theta", 0.0)), vega=float(greeks.get("vega", 0.0)),
            open_interest=oi, volume=0, implied_volatility=float(iv),
            quote_time_ms=asof_ms,
        ))
    return contracts


def _option_bars_upto(
    strikes: dict[float, dict[str, Any]], osi: str, day: date, minute: int,
) -> pd.DataFrame:
    """One contract's raw bars up to this tick -- the shape ``option_history`` returns live."""
    try:
        strike = parse_osi(osi)["strike"]
    except ValueError:
        return pd.DataFrame()
    index = strikes.get(strike)
    if index is None:
        return pd.DataFrame()
    key = _tick_key(day, minute)
    n = int(np.searchsorted(index["keys"], key, side="right"))
    rows = index["rows"][:n]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)[["timestamp", "open", "high", "low", "close"]]


def _bar_at(strikes: dict[float, dict[str, Any]], strike: float, day: date, minute: int) -> dict[str, float] | None:
    """This contract's own 5m bar for exactly this tick, or ``None`` if it did not print."""
    index = strikes.get(strike)
    if index is None:
        return None
    row = _exact_row(index, _tick_key(day, minute))
    if row is None:
        return None
    return {"low": float(row["low"]), "high": float(row["high"]), "close": float(row["close"])}


#: Every intraday run this harness replays, in full -- not a summary. One row per (symbol, day,
#: minute) tick, carrying the *entire* signal ``_plan_one`` produced that tick: every check
#: (label/ok/value/limit/blocking), the estimate (band prediction, direction, greeks-priced
#: profit), the desired orders, and what was held coming in. This is the actual output of the
#: production ``plan()`` call, not a re-derivation -- kept in memory and handed to the caller
#: (``options_flip_day_narrative.py`` reads it directly) rather than round-tripped through a
#: database: nothing here needs a fact from a run that already finished, and duckdb persistence
#: was cut once that stopped being true -- see the day-narrative tool for the reader this fed.
def _order_to_dict(order) -> dict[str, Any]:
    req = order.request
    return {
        "key": order.key, "symbol": req.symbol, "action": req.action,
        "quantity": req.quantity, "order_type": req.order_type,
        "limit_price": req.limit_price, "stop_price": req.stop_price,
        "time_in_force": req.time_in_force,
    }


def _order_underlying(order: Any) -> str:
    """Which symbol an order belongs to -- from the option contract itself, not a tag.

    ``extra["underlying"]`` is set on the entry leg (see ``lifecycle._flat_or_bidding``) but
    never on the sell legs a held position rests (``lifecycle._sell_leg`` only carries
    ``position_intent``) -- production never needs it there, since reconciliation matches by
    the option contract's own symbol. Filtering on the tag alone silently dropped every exit
    order from this harness: ``mine`` came back empty for every held position, ``pending``
    never became an exit, and a filled position rode to the end of the backtest window
    regardless of its target, its stop, or ``max_hold_sessions`` -- the bug this function
    fixes. Parsed from the option symbol itself so both legs resolve the same way.
    """
    tagged = order.request.extra.get("underlying") if order.request.extra else None
    if tagged:
        return str(tagged)
    try:
        return str(parse_osi(order.request.symbol)["underlying"])
    except Exception:
        return ""


def walk_forward(symbol: str, config: Any) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Replay ``OptionsFlipAlgorithm.plan()`` itself, tick by tick, one contract at a time.

    Fills are checked against the *following* tick's bar, never the tick that set the price --
    a bar's low is always <= its own close, so checking a limit priced off a bar against that
    same bar is a tautology, not a simulation (the bug this tool's sibling caught and fixed).
    The order resting at tick T was priced off what was known at T; whether the market reached
    it is only knowable from the bar spanning T to T+5, i.e. the *next* tick's bar.
    """
    algorithm = OptionsFlipAlgorithm(config)
    # ``_symbols()`` resolves the universe from the *real* account's configured symbol list
    # first, independent of what bars this harness fed in -- a symbol this backtest wants to
    # replay but the live account never configured (or a stale universe) would otherwise never
    # reach ``_plan_one`` at all, and every day would read as "nothing happened" rather than as
    # a gate that actually fired. Pinned to exactly this symbol for the run: the whole point of
    # calling ``plan()`` in a loop is to answer "what would production do for this symbol",
    # not to replay the live account's other, unrelated bindings.
    symbols_patch = mock.patch.object(
        OptionsFlipAlgorithm, "_symbols", staticmethod(lambda cfg, config: [symbol])
    )
    daily = _load_daily(symbol)
    intraday = _load_intraday(symbol)
    opt_all = _load_option_strikes(symbol)
    if opt_all.empty or intraday.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    # Indexed once, up front -- this is what keeps 70-odd ticks/day x ~20 days x ~17 strikes
    # from re-deriving the same timezone conversion and filter on every single lookup.
    strikes = _prepare_strikes(opt_all)

    stamps = pd.to_datetime(intraday["timestamp"], utc=True).dt.tz_convert("America/New_York")
    sf = intraday.copy()
    sf["ts"] = stamps
    sf["day"] = stamps.dt.date
    sf["minute"] = stamps.dt.hour * 60 + stamps.dt.minute
    counts = sf.groupby("day").size()
    sf = sf[sf["day"].isin(counts[counts >= 60].index)].reset_index(drop=True)
    sessions = sorted(sf["day"].unique())
    opt_days = set(
        pd.to_datetime(opt_all["timestamp"], utc=True).dt.tz_convert("America/New_York").dt.date.unique()
    )
    sessions = [d for d in sessions if d in opt_days]
    if len(sessions) < 3:
        return pd.DataFrame(), pd.DataFrame(), []

    symbols_patch.start()
    try:
        return _walk_days(
            symbol, algorithm, config, sessions, sf, daily, strikes,
        )
    finally:
        symbols_patch.stop()


def _walk_days(
    symbol: str, algorithm: OptionsFlipAlgorithm, config: Any, sessions: list[date],
    sf: pd.DataFrame, daily: pd.DataFrame, strikes: dict[float, dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    state: dict[str, Any] = {}
    positions: dict[str, int] = {}
    pending: dict[str, Any] | None = None
    entry_prices: dict[str, float] = {}
    log: list[dict[str, Any]] = []
    daily_log: list[dict[str, Any]] = []
    #: Every tick, in full -- see :data:`TICKS_TABLE`.
    ticks: list[dict[str, Any]] = []

    for day in sessions[1:-1]:
        history = sf[sf["day"] < day]
        today = sf[sf["day"] == day]
        day_row: dict[str, Any] = {
            "symbol": symbol, "day": day.isoformat(), "state_at_open": None,
            "state_at_close": None, "headline": None,
            # ``blocked_by`` is the *last* tick's blocking check -- near-universally "Time to
            # work" once past ~13:15, since that fires on any day with nothing resting,
            # regardless of why. ``first_blocked_by`` is what actually explains the day.
            "first_blocked_by": None, "blocked_by": None,
            "bid_placed": False, "filled_today": False,
            # End-of-day snapshots of the checks that carry the algorithm's own numbers --
            # overwritten every tick, so the value left standing is what was true at the close
            # (or at the last tick before a fill/exit stopped needing one).
            "band_check": None, "entry_bid_check": None,
            "exit_target_check": None, "stop_check": None,
        }

        for minute in TICKS:
            bar = today[today["minute"] == minute]
            if bar.empty:
                continue
            price = float(bar.iloc[0]["close"])

            # ── resolve the order that was resting coming into this tick ──────────────────
            if pending is not None:
                strike = parse_osi(pending["osi"])["strike"]
                opt_bar = _bar_at(strikes, strike, day, minute)
                if opt_bar is not None:
                    if pending["side"] == "entry" and opt_bar["low"] <= pending["limit"]:
                        positions = {pending["osi"]: pending["quantity"]}
                        entry_prices[pending["osi"]] = pending["limit"]
                        log.append({
                            "symbol": symbol, "day": day.isoformat(), "event": "FILL_ENTRY",
                            "minute": minute, "osi": pending["osi"], "price": pending["limit"],
                        })
                        day_row["filled_today"] = True
                        pending = None
                    elif pending["side"] == "exit":
                        hit_target = pending.get("target") and opt_bar["high"] >= pending["target"]
                        hit_stop = pending.get("stop") and opt_bar["low"] <= pending["stop"]
                        if hit_target or hit_stop:
                            fill_px = pending["target"] if hit_target else pending["stop"]
                            positions = {}
                            ep = entry_prices.pop(pending["osi"], 0.0)
                            log.append({
                                "symbol": symbol, "day": day.isoformat(),
                                "event": "FILL_TARGET" if hit_target else "FILL_STOP",
                                "minute": minute, "osi": pending["osi"], "price": fill_px,
                                "entry_price": ep,
                                "ret": (fill_px / ep - 1.0) if ep else None,
                            })
                            day_row["filled_today"] = True
                            pending = None

            # ── build this tick's context and ask the real algorithm what it wants ────────
            seen_today = today[today["minute"] <= minute]
            daily_through = _market_daily_through(daily, day, seen_today)
            ts = pd.Timestamp(
                f"{day} {minute // 60:02d}:{minute % 60:02d}", tz="America/New_York"
            ).to_pydatetime()
            intraday_seen = pd.concat([history, seen_today], ignore_index=True)[
                ["timestamp", "open", "high", "low", "close", "volume"]
            ]
            latest_prices = {symbol: price}
            for osi in positions:
                held_strike = parse_osi(osi)["strike"]
                opt_bar = _bar_at(strikes, held_strike, day, minute)
                if opt_bar is not None:
                    latest_prices[osi] = opt_bar["close"]

            def _chain_reader(sym, option_type="", min_dte=0, max_dte=120, *,
                               _d=day, _m=minute, _p=price, _dt=daily_through):
                return _synthetic_chain(strikes, _dt, _p, _d, _m, symbol)

            def _history_reader(osi, *, _d=day, _m=minute):
                return _option_bars_upto(strikes, osi, _d, _m)

            context = AlgorithmContext(
                config=config,
                daily_bars_by_symbol={symbol: daily_through},
                intraday_bars_by_symbol={symbol: intraday_seen},
                positions=dict(positions),
                latest_prices=latest_prices,
                state=state,
                timestamp=ts,
                extra={"option_chain": _chain_reader, "option_history": _history_reader},
            )
            plan = algorithm.plan(context)
            state = dict(plan.state)

            signal = plan.signals.get(symbol) or {}
            # The full record: everything ``_plan_one`` produced this tick, not a summary of it.
            ticks.append({
                "symbol": symbol, "day": day.isoformat(), "minute": minute, "ts": ts,
                "state": signal.get("state"), "headline": signal.get("headline"),
                "direction": (signal.get("estimate") or {}).get("direction"),
                "contract": signal.get("contract"),
                "checks_json": json.dumps(signal.get("checks") or []),
                "estimate_json": json.dumps(signal.get("estimate") or {}, default=str),
                "orders_json": json.dumps(
                    [_order_to_dict(o) for o in plan.desired_orders
                     if _order_underlying(o) == symbol]
                ),
                "positions_json": json.dumps(positions),
            })
            if day_row["state_at_open"] is None:
                day_row["state_at_open"] = signal.get("state")
            day_row["state_at_close"] = signal.get("state")
            day_row["headline"] = signal.get("headline") or day_row["headline"]
            checks = signal.get("checks") or []
            blocking = next((c for c in checks if c.get("blocking")), None)
            if blocking is not None:
                text = f"{blocking.get('label')}: {blocking.get('value')}"
                day_row["blocked_by"] = text
                if day_row["first_blocked_by"] is None:
                    day_row["first_blocked_by"] = text
            for label, key in (
                ("Option band", "band_check"), ("Entry bid", "entry_bid_check"),
                ("Profit target", "exit_target_check"), ("Protective stop", "stop_check"),
            ):
                found = next((c for c in checks if c.get("label") == label), None)
                if found is not None:
                    day_row[key] = f"{found.get('value')} ({found.get('limit')})"

            mine = [o for o in plan.desired_orders if _order_underlying(o) == symbol]
            if any(o.key == f"{symbol}:entry" for o in mine):
                day_row["bid_placed"] = True
            if not positions:
                entry = next((o for o in mine if o.key == f"{symbol}:entry"), None)
                pending = (
                    {"side": "entry", "osi": entry.request.symbol, "limit": float(entry.request.limit_price),
                     "quantity": int(entry.request.quantity)}
                    if entry is not None else None
                )
            else:
                held_osi = next(iter(positions))
                entry_price = entry_prices.get(held_osi, 0.0)
                bracket = next((o for o in mine if o.key == f"{symbol}:bracket"), None)
                target_o = next((o for o in mine if o.key == f"{symbol}:target"), None)
                stop_o = next((o for o in mine if o.key == f"{symbol}:stop"), None)
                target = float((bracket or target_o).request.limit_price) if (bracket or target_o) else None
                stop = float((bracket.request.children[1].stop_price if bracket and bracket.request.children
                              else (stop_o.request.stop_price if stop_o else 0.0)) or 0.0) or None
                pending = ({"side": "exit", "osi": held_osi, "target": target, "stop": stop,
                            "entry_price": entry_price} if (target or stop) else None)

        # abandon an unfilled bid overnight -- ``plan_symbol``'s own day-changed guard never
        # fires in production (nothing sets ``session["day_changed"]``; see the write-up), so
        # a bid otherwise carries into tomorrow on its own memory. Left as-is here deliberately,
        # to replay what production actually does rather than what its comment claims it does.
        daily_log.append(day_row)

    return pd.DataFrame(log), pd.DataFrame(daily_log), ticks


def report(symbol: str, log: pd.DataFrame, daily: pd.DataFrame) -> None:
    print(f"\n{symbol}: {len(log)} events")
    if not log.empty:
        print(log.to_string(index=False))
        fills = log[log["event"] == "FILL_ENTRY"]
        exits = log[log["event"].isin(["FILL_TARGET", "FILL_STOP"])]
        print(f"  entries filled: {len(fills)}   exits resolved: {len(exits)}")
        if len(exits):
            # By realized return, not by which order filled: the exit ratchet concedes the
            # target down toward the mark as the deadline nears (see lifecycle.py), so a
            # "FILL_TARGET" late in a losing hold can print a negative return -- the target
            # itself moved below entry, not the market recovering. Counting event type as the
            # win/loss label undercounted losses that filled through a conceded target rather
            # than a stop.
            wins = exits[exits["ret"] > 0]
            print(f"  WIN {len(wins)}  LOSS {len(exits) - len(wins)}   "
                  f"mean return {exits['ret'].mean():+.1%}")

    print(f"\n{symbol}: day by day")
    if daily.empty:
        print("  no sessions.")
        return
    header = (f"  {'day':11s} {'open':6s} {'close':6s} {'bid':4s} {'fill':5s}  reason it stood down / headline")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in daily.iterrows():
        why = r["first_blocked_by"] or r["headline"] or ""
        print(f"  {r['day']:11s} {str(r['state_at_open'] or '-'):6s} "
              f"{str(r['state_at_close'] or '-'):6s} {('Y' if r['bid_placed'] else '-'):4s} "
              f"{('Y' if r['filled_today'] else '-'):5s}  {why}")
        if r["band_check"]:
            print(f"      band:   {r['band_check']}")
        if r["entry_bid_check"]:
            print(f"      entry:  {r['entry_bid_check']}")
        if r["exit_target_check"]:
            print(f"      target: {r['exit_target_check']}")
        if r["stop_check"]:
            print(f"      stop:   {r['stop_check']}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    from src.core.config import get_config
    config = get_config()
    for symbol in SYMBOLS:
        log, daily, _ticks = walk_forward(symbol, config)
        report(symbol, log, daily)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
