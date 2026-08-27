"""One symbol's position through its life: what should be resting at the broker, and why.

A pure function of the run's inputs. It receives what was measured and what is held, and returns
the orders that should exist plus the state to remember -- it places nothing, reads no clock of
its own, and calls no broker. That is what lets the same code be reasoned about in a test with a
handful of dictionaries.

The three states, and the single question each one answers:

``FLAT``     Is there a reason to be in this name today? Direction, contract, and the bid.
``BIDDING``  Is the bid still in the right place? Re-price it as the walk-in progresses.
``HELD``     Can the target be raised? The stop never moves; the limit only ever ratchets up.

Transitions are never decided here from a fill notification -- there isn't one. They are read
from the broker's positions each run: a contract we bid for and now hold has filled, and one we
held and no longer do has closed. A cron-driven strategy cannot observe the moment something
happens, so it must be able to infer it from the state of the world, and that is strictly better
anyway -- it is the same code path whether the fill happened five minutes ago or during an hour
the process was down.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ...core.interfaces import Check, DesiredOrder, OrderRequest
from ...core.options import CALL, OptionContract
from .config import REPRICE_MIN_PRICE_FRACTION
from .excursion import option_price_for

logger = logging.getLogger(__name__)

FLAT = "flat"
BIDDING = "bidding"
HELD = "held"

#: Suffixes for the reconciler's order keys. The key names the *role* an order plays for a
#: symbol, so re-pricing across a session is one order rather than a dozen.
ENTRY = "entry"
BRACKET = "bracket"
#: Used only where the broker cannot hold an OCO, so the two legs are separate orders.
TARGET = "target"
STOP = "stop"


@dataclass(frozen=True)
class SymbolPlan:
    """What one symbol wants this run: its orders, its memory, and the reasoning behind both."""

    symbol: str
    state: str
    orders: list[DesiredOrder]
    memory: dict[str, Any]
    checks: list[Check]
    headline: str = ""
    #: What the deck needs to judge the trade rather than just watch it: the contract, its size
    #: and cost, and the price band this run expects to transact in. Carried separately from
    #: ``memory`` because none of it is remembered -- it is recomputed every run, and it is
    #: reported even on the runs that place no orders, which are the ones worth explaining.
    estimate: dict[str, Any] = field(default_factory=dict)


def plan_symbol(
    symbol: str,
    *,
    memory: dict[str, Any],
    held_contract: str,
    direction: str,
    contract: OptionContract | None,
    contracts: int,
    underlying_now: float,
    entry_target: float,
    exit_target: float,
    checks: list[Check],
    config: Any,
    session: dict[str, Any],
    oco: bool = True,
) -> SymbolPlan:
    """The orders that should be resting for ``symbol`` right now.

    ``held_contract`` is the OSI symbol this account actually holds, read from the broker and
    empty when flat -- the authority on which state we are in. ``session`` carries the run's
    market-time facts (``fraction_remaining``, ``market_day``) so
    this function needs no clock.
    """
    if held_contract:
        return _held(
            symbol, memory, held_contract, contracts, underlying_now, exit_target,
            checks, config, session, oco,
        )
    if memory.get("state") == BIDDING and not held_contract and session.get("day_changed"):
        # A bid that survived the night is not re-priced, it is abandoned: the excursion budget
        # it was built from described a session that has ended.
        logger.info("[%s] Options Flip abandoning yesterday's unfilled bid", symbol)
        memory = {}
    return _flat_or_bidding(
        symbol, memory, direction, contract, contracts, underlying_now, entry_target, checks, config, session
    )


def _flat_or_bidding(
    symbol: str,
    memory: dict[str, Any],
    direction: str,
    contract: OptionContract | None,
    contracts: int,
    underlying_now: float,
    entry_target: float,
    checks: list[Check],
    config: Any,
    session: dict[str, Any],
) -> SymbolPlan:
    """No position: bid for one, or stand down."""
    if not direction or contract is None or contracts <= 0:
        return SymbolPlan(symbol, FLAT, [], {}, checks, "No trade today")

    # At the pullback level, translated into premium through delta -- not at the mid.
    #
    # Resting at the mid fills on almost every armed session and pays whatever the market asks;
    # this waits for the dip the level model predicts, and is refused when that dip is unlikely
    # to arrive in time. The two are different strategies and the backtest measures this one: a
    # static limit at ``entry_target`` filled 4 of 11 armed sessions on real option bars, and the
    # seven misses sat 5-6.5% below the mark and simply never traded there.
    #
    # The adverse selection a fixed offset suffers -- filling on days that fell, absent on days
    # that rose -- is answered by ``entry_reach`` conditioning the depth on the day in
    # front of it, rather than by abandoning the pullback.
    floor_price = (
        option_price_for(entry_target, underlying_now=underlying_now,
                         option_mark=contract.midpoint, delta=contract.delta)
        if entry_target > 0 else contract.midpoint
    )
    # Soft ratchet. The limit starts at the pullback level and gives ground toward the mark as
    # the session runs out, on a curve ``entry_patience`` shapes. It is deliberately the patient
    # side: raising a bid to meet a rising ask is how a pullback strategy quietly becomes a
    # momentum-chasing one, and an entry that never fills costs only the opportunity.
    #
    # It never goes above the mark, so the spread is still not crossed.
    given_up = (1.0 - max(float(session.get("fraction_remaining", 0.0)), 0.0)) ** max(
        float(getattr(config, "entry_patience", 1.0)), 0.01
    )
    limit = min(floor_price + (contract.midpoint - floor_price) * given_up, contract.midpoint)

    request = OrderRequest(
        symbol=contract.osi_symbol,
        action="buy",
        quantity=contracts,
        order_type="limit",
        limit_price=round(limit, 2),
        asset_type="option",
        time_in_force="day",
        extra={"position_intent": "buy_to_open", "underlying": symbol},
    )
    # Zero when the stop is disabled, not ``limit x 1.0``, which is the entry price itself.
    # The held path re-checks the setting so nothing acts on it today -- but a position carrying
    # a recorded stop equal to its own fill would be closed the instant the stop was turned on.
    stop_pct = float(config.stop_loss_pct)
    stop = round(max(limit * (1.0 - stop_pct), 0.01), 2) if stop_pct > 0 else 0.0
    checks = checks + [Check(
        label="Entry bid",
        ok=True,
        value=(
            f"${limit:.2f} for {contract.osi_symbol}, waiting for the underlying at "
            f"${entry_target:,.2f} (now ${underlying_now:,.2f})"
            + (f" — ratcheted {given_up:.0%} toward the mark" if given_up > 0.01 else "")
            if entry_target > 0 else
            f"${limit:.2f} for {contract.osi_symbol}, at the mid (no level)"
        ),
        limit=f"pullback limit, patience {float(getattr(config, 'entry_patience', 1.0)):.1f}; "
              f"never above the mark, abandoned unfilled at the close",
    )]
    memory = {
        "state": BIDDING,
        "contract": contract.osi_symbol,
        "direction": direction,
        "contracts": contracts,
        "bid": round(limit, 2),
        # The stop travels with the entry rather than waiting for the fill. A limit buy can only
        # fill at or below its price, so a stop struck off the limit is never looser than the cap
        # -- and being known now is what lets the protective pair go up attached to the entry
        # instead of a run later, leaving the position naked in between.
        "stop": stop,
        "market_day": session.get("market_day", ""),
    }
    return SymbolPlan(
        symbol, BIDDING,
        [DesiredOrder(
            key=f"{symbol}:{ENTRY}", request=request,
            # Spread-relative, so an illiquid contract is not re-priced for a move the market
            # cannot distinguish -- see ``REPRICE_MIN_SPREAD_FRACTION``.
            replace_tolerance=_reprice_tolerance(contract, limit, config),
        )],
        memory, checks,
        f"Bidding ${limit:.2f} for the {contract.strike:g} {contract.option_type}",
    )


def _held(
    symbol: str,
    memory: dict[str, Any],
    held_contract: str,
    contracts: int,
    underlying_now: float,
    exit_target: float,
    checks: list[Check],
    config: Any,
    session: dict[str, Any],
    oco: bool,
) -> SymbolPlan:
    """Holding a contract: maintain the bracket, ratchet the target, honour the deadline."""
    quantity = max(int(memory.get("contracts", contracts) or contracts), 1)
    fill_price = float(memory.get("fill_price", 0.0) or 0.0)
    mark = float(memory.get("mark", 0.0) or 0.0)
    direction = str(memory.get("direction") or CALL)

    # Struck off the entry *limit*, recorded when the order was placed, and never recomputed.
    # Not off the fill: the limit is known at submission time, which is what lets the stop ride
    # up attached to the entry as one bracket. Not off the current mark either -- that would be a
    # trailing stop, a different strategy, and one that ratchets the risk floor upward on exactly
    # the noise this stop exists to sit beneath.
    recorded = float(memory.get("stop", 0.0) or 0.0)
    anchor = float(memory.get("bid", 0.0) or 0.0) or fill_price
    stop_pct = float(config.stop_loss_pct)
    # Zero disables the stop. The bracket then rests the profit target alone and the deadline is
    # the only exit that forces the issue -- which is the intended shape for a bounded-loss long
    # call, not an oversight.
    stop = (recorded or round(max(anchor * (1.0 - stop_pct), 0.01), 2)) if stop_pct > 0 else 0.0

    held_days = int(memory.get("sessions_held", 0) or 0)
    deadline = held_days >= max(int(config.max_hold_sessions), 1)

    # The exit target steps *down* with the days, and that is the opposite of the ratchet it
    # replaces. A target that only ever rose asked more of a position the longer it failed to
    # deliver, which is how a winner becomes a deadline exit at the bid. Seeking a fraction of
    # the modelled gain -- 70% on the day of entry, giving up a step a session -- is what makes
    # the order executable rather than theoretical, and it is the reference design's schedule.
    #
    #   day 0: entry + 0.70 x gain     day 1: + 0.50     day 2: + 0.30
    #
    # The floor is the entry itself: the schedule gives up profit, never principal. Getting out
    # at cost is the deadline exit's job, and it is a different decision.
    # Nothing to price a bracket from. Guarded before the schedule rather than after it: the
    # schedule floors at a penny, so an unpriceable contract would otherwise rest a one-cent ask
    # against a position whose value is unknown -- an order that is certain to fill and certain
    # to be wrong.
    if mark <= 0 and fill_price <= 0:
        return SymbolPlan(
            symbol, HELD, [],
            {**memory, "state": HELD, "contract": held_contract},
            checks + [Check(
                label="Priceable",
                ok=False,
                value="no current mark for the contract, and no recorded fill",
                limit="a quote to size the bracket from",
                blocking=True,
            )],
            f"Holding {held_contract} — unpriced this run",
        )

    modelled = _target_premium(mark, underlying_now, exit_target, direction, memory)
    entry_price = float(memory.get("bid", 0.0) or 0.0) or fill_price or mark
    gain = max(modelled - entry_price, 0.0)
    # Soft ratchet on the sell side, mirroring the entry's. ``exit_patience`` below 1 concedes
    # early, which is the intended default: a position that reaches its deadline unsold is sold
    # at whatever the market offers, and with the stop disabled the deadline is the only thing
    # that ends a losing trade. Conceding early is cheaper than conceding at gunpoint.
    elapsed = min(held_days / max(int(config.max_hold_sessions), 1), 1.0)
    conceded = elapsed ** max(float(getattr(config, "exit_patience", 1.0)), 0.01)
    asked = max(float(config.exit_gain_share) * (1.0 - conceded), 0.0)
    target = round(max(entry_price + asked * gain, 0.01), 2)

    if deadline:
        # Out of time. Converge on the market across what is left of the session, so the ask is
        # at the bid by the close rather than resting at a price the position may no longer wait
        # for. This is the one step allowed to price below the entry: the deadline outranks the
        # profit target, and a position still open past it is a worse risk than a small loss.
        decay = float(session.get("fraction_remaining", 0.0))
        target = round(max(mark + (target - mark) * decay, 0.01), 2)

    if target <= 0:
        # A quote the feed missed -- and no recorded fill to anchor on either -- leaves nothing
        # to price a bracket from. Resting nothing says so rather than crashing the run: with no
        # fill recorded there is no book of ours at the broker for an empty plan to cancel.
        return SymbolPlan(
            symbol, HELD, [],
            {**memory, "state": HELD, "contract": held_contract},
            checks + [Check(
                label="Priceable",
                ok=False,
                value="no current mark for the contract",
                limit="a quote to size the bracket from",
                blocking=True,
            )],
            f"Holding {held_contract} — unpriced this run",
        )

    orders = _bracket_orders(symbol, held_contract, quantity, target, stop, config, oco=oco)

    unrealised = (mark / fill_price - 1.0) if fill_price > 0 and mark > 0 else 0.0
    checks = checks + [
        Check(
            label="Profit target",
            ok=True,
            value=(
                f"${target:.2f} ({(target / fill_price - 1.0):+.0%} on the fill)"
                if fill_price > 0 else f"${target:.2f}"
            ),
            limit=(
                f"asking {asked:.0%} of the modelled gain, session {held_days + 1} of "
                f"{int(config.max_hold_sessions)} (patience "
                f"{float(getattr(config, 'exit_patience', 1.0)):.1f})"
                if not deadline else "deadline — converging on the market"
            ),
        ),
        Check(
            label="Protective stop",
            ok=True,
            value=(
                (f"${stop:.2f} at the exchange" if oco
                 else f"${stop:.2f} at the exchange, as a separate order")
                if stop > 0 else
                f"none — the {quantity}-contract premium is the loss cap"
            ),
            limit=(
                f"{float(config.stop_loss_pct):.0%} below the ${anchor:.2f} entry" if stop > 0
                else "disabled; the deadline is the exit that forces the issue"
            ),
        ),
        Check(
            label="Hold deadline",
            ok=not deadline,
            value=f"session {int(memory.get('sessions_held', 0) or 0) + 1} of {config.max_hold_sessions}",
            limit=f"flatten after {config.max_hold_sessions}",
            blocking=deadline,
        ),
    ]

    return SymbolPlan(
        symbol, HELD, orders,
        {**memory, "state": HELD, "contract": held_contract, "target": target, "stop": stop},
        checks,
        f"Holding {held_contract} — {unrealised:+.0%}, target ${target:.2f}",
    )


def _reprice_tolerance(contract: OptionContract, price: float, config: Any) -> float:
    """How far the wanted price must move before the resting order is re-placed, as a fraction.

    The larger of a fraction of the price and a fraction of the *spread*. On a tight market the
    price term governs and this is the old behaviour; on a wide one the spread term takes over,
    because two cents on a market quoted 1.20/1.35 is a sixth of the spread -- inside the noise
    of the quote, and re-placing for it is a round trip that buys nothing.
    """
    from .config import REPRICE_MIN_PRICE_FRACTION, REPRICE_MIN_SPREAD_FRACTION

    if price <= 0:
        return REPRICE_MIN_PRICE_FRACTION
    spread = max(contract.ask - contract.bid, 0.0)
    return max(REPRICE_MIN_PRICE_FRACTION, (spread * REPRICE_MIN_SPREAD_FRACTION) / price)


def _target_premium(mark: float, underlying_now: float, exit_target: float, direction: str, memory: dict) -> float:
    """The profit target in premium, from the target *level* the model produced.

    ``exit_target`` is an absolute underlying price, not a fraction. It used to be a fraction and
    the caller passed a price into it -- ``underlying_now * (1 + 403.75)`` -- which put the target
    four hundred times the spot, so it never filled and every position ran to its deadline. The
    levels model speaks in prices; so does this now, and the ambiguity is gone rather than
    documented.
    """
    if exit_target <= 0:
        return 0.0
    return option_price_for(
        exit_target, underlying_now=underlying_now, option_mark=mark,
        delta=float(memory.get("delta", 0.0) or 0.0),
    )


def _sell_leg(contract: str, quantity: int, **kwargs: Any) -> OrderRequest:
    return OrderRequest(
        symbol=contract, action="sell", quantity=quantity, asset_type="option",
        time_in_force="gtc", extra={"position_intent": "sell_to_close"}, **kwargs,
    )


def _bracket_orders(
    symbol: str, contract: str, quantity: int, target: float, stop: float, config: Any, *, oco: bool
) -> list[DesiredOrder]:
    """What should be resting against an open position -- a pair, or a lone target.

    **A ``stop`` of zero means no stop order reaches the exchange at all.** Not a stop at a
    distant price, not an OCO with one live leg: a single resting sell limit, and nothing else.
    The strategy is then a resting buy limit followed by a resting sell limit, with the premium
    of ``contracts_per_trade`` as the loss cap and the deadline as the exit that forces the
    issue. ``oco`` is not consulted in that case, because there is no pair for a venue to hold.

    With a live stop and OCO the venue owns the invariant that only one side can fill, and the
    bracket is one order. Without OCO -- Alpaca refuses any complex order class on options --
    the same two legs go up independently, and that invariant becomes ours.

    **The exposure that creates, stated plainly:** when one leg fills, the other is briefly live
    against a position that no longer exists. The next reconciliation cancels it, because a flat
    symbol wants no bracket, so the window is one run of the cadence rather than open-ended. Two
    things keep it survivable in the meantime: the broker rejects a ``sell_to_close`` with
    nothing to close, and the remaining leg is a *sell* of a contract we no longer hold rather
    than anything that could open new exposure.
    """
    tolerance = REPRICE_MIN_PRICE_FRACTION
    limit_leg = _sell_leg(contract, quantity, order_type="limit", limit_price=target)
    # Checked before anything else, so no code path below can construct a stop leg. Note the key
    # is ``:target`` rather than ``:bracket``: turning the stop off on a position that already
    # has a bracket resting therefore reads to the reconciler as "the bracket is no longer
    # wanted, this target is", and it cancels the one and places the other. That is the intended
    # transition and it is why the two shapes do not share a key.
    if stop <= 0:
        return [DesiredOrder(key=f"{symbol}:{TARGET}", request=limit_leg,
                             replace_tolerance=tolerance)]
    stop_leg = _sell_leg(contract, quantity, order_type="stop", stop_price=stop)
    if not oco:
        return [
            DesiredOrder(key=f"{symbol}:{TARGET}", request=limit_leg, replace_tolerance=tolerance),
            # No tolerance: the stop never moves, so any difference from what is resting means
            # the resting order is not the one this position wants.
            DesiredOrder(key=f"{symbol}:{STOP}", request=stop_leg),
        ]
    return [DesiredOrder(
        key=f"{symbol}:{BRACKET}",
        request=OrderRequest(
            symbol=contract, action="sell", quantity=quantity, order_type="limit",
            limit_price=target, asset_type="option", time_in_force="gtc", strategy="oco",
            extra={"position_intent": "sell_to_close", "underlying": symbol},
            children=(limit_leg, stop_leg),
        ),
        replace_tolerance=tolerance,
    )]
