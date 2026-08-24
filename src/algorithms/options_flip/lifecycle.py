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
    exit_budget: float,
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
            symbol, memory, held_contract, contracts, underlying_now, exit_budget,
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

    # The bid walks in toward the contract's midpoint and stops there -- it never crosses the
    # spread. That is structural rather than a clamp: ``entry_underlying_target`` never returns a
    # level on the wrong side of the market, so the delta translation can only ever land at or
    # below the mark. Reaching here at all means the direction still confirms, since it is
    # re-derived every run and a symbol that lost its confirmation returned above with no orders.
    limit = option_price_for(
        entry_target or underlying_now,
        underlying_now=underlying_now,
        option_mark=contract.midpoint,
        delta=contract.delta,
    )

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
    checks = checks + [Check(
        label="Entry bid",
        ok=True,
        value=(
            f"${limit:.2f} for {contract.osi_symbol} "
            f"(waiting for the underlying at ${entry_target or underlying_now:,.2f}, "
            f"now ${underlying_now:,.2f})"
        ),
        limit=f"{config.target_fill_probability:.0%} target fill probability",
    )]
    memory = {
        "state": BIDDING,
        "contract": contract.osi_symbol,
        "direction": direction,
        "contracts": contracts,
        "bid": round(limit, 2),
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
    exit_budget: float,
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

    # The stop is anchored to the fill and never recomputed. Recomputing it from the current mark
    # would be a trailing stop, which is a different strategy -- and one that ratchets the risk
    # floor upward on exactly the noise this stop exists to sit beneath.
    stop = round(max(fill_price * (1.0 - float(config.stop_loss_pct)), 0.01), 2)

    deadline = int(memory.get("sessions_held", 0) or 0) >= max(int(config.max_hold_sessions), 1)
    if deadline:
        # Out of time: converge on the mark along the same curve the entry bid walks, so the ask
        # is at the market by the close rather than resting at an ambitious price the position is
        # no longer allowed to wait for.
        decay = float(session.get("fraction_remaining", 0.0)) ** max(float(config.entry_decay_power), 0.0)
        ambitious = _exit_target(mark, underlying_now, exit_budget, direction, memory)
        target = round(max(mark + (ambitious - mark) * decay, 0.01), 2)
    else:
        target = round(_exit_target(mark, underlying_now, exit_budget, direction, memory), 2)

    # Ratchet: the target only ever rises. A target that fell would be chasing a losing position
    # down, which is what the stop is for.
    previous = float(memory.get("target", 0.0) or 0.0)
    if previous > 0 and target < previous * (1.0 + float(config.ratchet_min_improvement)):
        target = previous

    orders = _bracket_orders(symbol, held_contract, quantity, target, stop, config, oco=oco)

    unrealised = (mark / fill_price - 1.0) if fill_price > 0 and mark > 0 else 0.0
    checks = checks + [
        Check(
            label="Profit target",
            ok=True,
            value=f"${target:.2f} ({(target / fill_price - 1.0):+.0%} on the fill)" if fill_price > 0 else f"${target:.2f}",
            limit="raised only, never lowered",
        ),
        Check(
            label="Protective stop",
            ok=True,
            value=(
                f"${stop:.2f} at the exchange"
                if oco else f"${stop:.2f} at the exchange, as a separate order"
            ),
            limit=f"{float(config.stop_loss_pct):.0%} below the ${fill_price:.2f} fill",
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


def _exit_target(mark: float, underlying_now: float, exit_budget: float, direction: str, memory: dict) -> float:
    """The profit target, as the mirror of the entry: a call is sold into a rise, a put a fall.

    Reasoned from the current mark rather than the session open, unlike the entry. A profit target
    is a move to be captured *from here*, and a position may be held across sessions, which makes
    "today's open" meaningless for it.
    """
    underlying_target = (
        underlying_now * (1.0 + exit_budget) if direction == CALL
        else underlying_now * (1.0 - exit_budget)
    )
    return option_price_for(
        underlying_target, underlying_now=underlying_now, option_mark=mark,
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
    """The protective pair, in whichever shape this broker will hold.

    With OCO the venue owns the invariant that only one side can fill, and the bracket is one
    order. Without it -- Alpaca refuses any complex order class on options -- the same two legs
    go up independently, and that invariant becomes ours.

    **The exposure that creates, stated plainly:** when one leg fills, the other is briefly live
    against a position that no longer exists. The next reconciliation cancels it, because a flat
    symbol wants no bracket, so the window is one run of the cadence rather than open-ended. Two
    things keep it survivable in the meantime: the broker rejects a ``sell_to_close`` with
    nothing to close, and the remaining leg is a *sell* of a contract we no longer hold rather
    than anything that could open new exposure.
    """
    tolerance = float(config.ratchet_min_improvement)
    limit_leg = _sell_leg(contract, quantity, order_type="limit", limit_price=target)
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
