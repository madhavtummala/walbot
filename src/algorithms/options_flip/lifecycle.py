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
from .config import ENTRY_MAX_REPRICE_PCT, REPRICE_MIN_PRICE_FRACTION, SELL_GATE_CONCESSION_RATE
from .excursion import option_price_for

logger = logging.getLogger(__name__)

FLAT = "flat"
BIDDING = "bidding"
HELD = "held"

#: Suffixes for the reconciler's order keys. The key names the *role* an order plays for a
#: symbol, so re-pricing across a session is one order rather than a dozen. Target and stop are
#: always two independent orders -- see ``_bracket_orders`` -- never one broker-side bracket.
ENTRY = "entry"
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
    entry_premium: float | None = None,
    target_premium: float | None = None,
    sell_ok: bool = True,
) -> SymbolPlan:
    """The orders that should be resting for ``symbol`` right now.

    ``held_contract`` is the OSI symbol this account actually holds, read from the broker and
    empty when flat -- the authority on which state we are in. ``session`` carries the run's
    market-time facts (``fraction_remaining``, ``market_day``) so
    this function needs no clock.

    ``entry_premium`` and ``target_premium`` are the option-band prediction in premium dollars
    (see :mod:`.option_band`), supplied by the caller when the chosen contract has a usable price
    history of its own. When they are ``None`` the caller has not yet collected that history, and
    ``entry_target``/``exit_target`` (underlying levels) are translated through delta as the
    cold-start fallback. ``sell_ok`` is the bull-run gate, re-read on the sales side: when it has
    closed, a held position is sold at the mark rather than asked to keep waiting for a target.
    """
    if held_contract:
        return _held(
            symbol, memory, held_contract, contracts, underlying_now, exit_target,
            checks, config, session, target_premium=target_premium, sell_ok=sell_ok,
        )
    if memory.get("state") == BIDDING and not held_contract and session.get("day_changed"):
        # A bid that survived the night is not re-priced, it is abandoned: the excursion budget
        # it was built from described a session that has ended.
        logger.info("[%s] Options Flip abandoning yesterday's unfilled bid", symbol)
        memory = {}
    return _flat_or_bidding(
        symbol, memory, direction, contract, contracts, underlying_now, entry_target, checks, config, session,
        entry_premium=entry_premium,
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
    entry_premium: float | None = None,
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
        if (entry_target > 0 and entry_premium is None)
        else (entry_premium if (entry_premium and entry_premium > 0) else contract.midpoint)
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

    # A step cap independent of the ratchet curve: entry_patience bounds how the *formula*
    # moves across a session, but the ceiling it walks toward is the contract's own quoted
    # mid, and on a thin contract that quote can itself jump between runs rather than move
    # smoothly -- a stale print catching up, not a real repricing. Re-asserting the mid cap
    # after the clamp keeps the older invariant (never above the mark) intact regardless of
    # which direction the clamp moved the price.
    previous_bid = float(memory.get("bid", 0.0) or 0.0)
    if previous_bid > 0:
        max_step = previous_bid * ENTRY_MAX_REPRICE_PCT
        limit = min(max(limit, previous_bid - max_step), previous_bid + max_step)
        limit = min(limit, contract.midpoint)

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
        )
        + f" — pullback limit, patience {float(getattr(config, 'entry_patience', 1.0)):.1f}; "
          f"never above the mark, abandoned unfilled at the close",
        gate=False,
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
    target_premium: float | None = None,
    sell_ok: bool = True,
) -> SymbolPlan:
    """Holding a contract: maintain the bracket, ratchet the target, honour the deadline."""
    # The caller's count first: it is the broker's own position size. Memory is the intent this
    # algorithm had at entry, which a partial fill or a hand-trimmed position makes wrong -- and
    # an exit sized above what is held is rejected outright, leaving the position unprotected.
    quantity = max(int(contracts or memory.get("contracts", 0) or 1), 1)
    fill_price = float(memory.get("fill_price", 0.0) or 0.0)
    mark = float(memory.get("mark", 0.0) or 0.0)
    direction = str(memory.get("direction") or CALL)

    # Struck off what the position actually cost, and not off the current mark -- that would be
    # a trailing stop, a different strategy, and one that ratchets the risk floor upward on
    # exactly the noise this stop exists to sit beneath.
    #
    # It used to anchor to the entry *limit* instead, because the limit is known at submission
    # time and the stop once rode up attached to the entry as one bracket. That is no longer how
    # it is placed -- ``_bracket_orders`` rests two independent orders after the fill -- and a
    # limit buy fills at or below its price, so anchoring there set the floor above where the
    # configured percentage puts it and cut positions short of their stated loss cap.
    recorded = float(memory.get("stop", 0.0) or 0.0)
    anchor = fill_price or float(memory.get("bid", 0.0) or 0.0)
    # A recorded stop is kept only while there is no fill to do better with.
    if fill_price > 0:
        recorded = 0.0
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

    # What the position cost, best evidence first: the broker's average entry price (recorded
    # into ``fill_price`` by ``_refresh_held``), then the limit we bid, then the mark.
    #
    # The bid used to come first. It is an upper bound rather than a cost -- a limit buy fills
    # at or below its price -- so anchoring to it asked the exit for a gain measured from a
    # price the account never paid, and reported "+X% on the fill" against the same wrong
    # number. It stays as the second choice because an upper bound is still a far better
    # anchor than the current mark, which carries no relationship to cost at all.
    entry_price = fill_price or float(memory.get("bid", 0.0) or 0.0) or mark
    # The bull-run gate is re-read on the sales side, and the sell band is re-predicted every run.
    # When the freshly predicted target lies at or below the current mark, there is nothing left
    # to ratchet toward and the position is priced at the mark outright -- this is a read of the
    # model, not of the regime, so it is not debounced.
    band_exhausted = target_premium is not None and (target_premium <= 0 or target_premium <= mark)
    if band_exhausted:
        modelled = target = mark
        asked = 0.0
        gain = 0.0
    else:
        modelled = (
            target_premium if target_premium and target_premium > 0
            else _target_premium(mark, underlying_now, exit_target, direction, memory)
        )
        gain = max(modelled - entry_price, 0.0)
        # Soft ratchet on the sell side, mirroring the entry's. ``exit_patience`` below 1 concedes
        # early, which is the intended default: a position that reaches its deadline unsold is sold
        # at whatever the market offers, and with the stop disabled the deadline is the only thing
        # that ends a losing trade. Conceding early is cheaper than conceding at gunpoint. The
        # "max_hold factor" the user asked for is exactly this: the prediction is dialled toward
        # the mark as the sessions run out.
        elapsed = min(held_days / max(int(config.max_hold_sessions), 1), 1.0)
        conceded = elapsed ** max(float(getattr(config, "exit_patience", 1.0)), 0.01)
        asked = max(float(config.exit_gain_share) * (1.0 - conceded), 0.0)
        target = round(max(entry_price + asked * gain, 0.01), 2)

    # Sold outright, ahead of the schedule above, whenever either clock runs out: the deadline
    # (out of *time*) or the bull-regime gate (out of *thesis*). Both converge the same way --
    # toward the mark, across what is left of the relevant clock -- and both are allowed to price
    # below the entry, since a position no longer worth waiting on is a worse risk than a small
    # loss. The gate's clock is a streak of consecutive closed reads rather than a session
    # fraction, so one flicker barely moves the target -- see ``SELL_GATE_CONCESSION_RATE``.
    gate_streak = 0 if sell_ok else int(memory.get("gate_failed_streak", 0) or 0) + 1
    gate_decay = (1.0 - SELL_GATE_CONCESSION_RATE) ** gate_streak
    day_decay = float(session.get("fraction_remaining", 0.0)) if deadline else 1.0
    decay = min(gate_decay, day_decay)
    if decay < 1.0:
        target = round(max(mark + (target - mark) * decay, 0.01), 2)

    # A missing mark makes every price below degenerate -- the modelled gain collapses to zero
    # and the schedule floors at the entry -- so the last known ask is re-asserted instead of a
    # breakeven one computed from nothing.
    if mark <= 0 and float(memory.get("target", 0.0) or 0.0) > 0:
        target = float(memory["target"])
    if target <= 0:
        # Fall back to the price this position was last asking, rather than resting nothing.
        #
        # Resting nothing does not mean "leave things as they are": the reconciler cancels every
        # recorded order that a run stops wanting, so a single missed quote withdrew the live
        # profit target *and the protective stop* from an open position, and re-placed them on
        # the next fire. A transient feed gap should not open a hole in the protection -- so the
        # last known prices are re-asserted, which the reconciler sees as unchanged and leaves
        # alone. Only a position that has never had a target rests nothing, and that one has no
        # orders at the broker to withdraw.
        target = float(memory.get("target", 0.0) or 0.0)
    if target <= 0:
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

    orders = _bracket_orders(symbol, held_contract, quantity, target, stop, config)

    unrealised = (mark / fill_price - 1.0) if fill_price > 0 and mark > 0 else 0.0
    checks = checks + [
        Check(
            label="Profit target",
            ok=True,
            # A reading: it reports the order that is resting, and refuses nothing. So the
            # schedule behind it belongs in the value -- ``limit`` means "what it had to be",
            # and the deck prefixes that with "needs", which turns a note into a false rule.
            value=(
                (f"${target:.2f} ({(target / fill_price - 1.0):+.0%} on the fill)"
                 if fill_price > 0 else f"${target:.2f}")
                + f" — asking {asked:.0%} of the modelled gain, session {held_days + 1} of "
                + f"{int(config.max_hold_sessions)} (patience "
                + f"{float(getattr(config, 'exit_patience', 1.0)):.1f})"
                + ("" if not band_exhausted else " — band target at/below the mark, sold outright")
                + ("" if gate_streak <= 0 else f" — bull gate closed {gate_streak} run(s), "
                   f"{1.0 - gate_decay:.0%} converged to the mark")
                + ("" if not deadline else f" — deadline, {1.0 - day_decay:.0%} converged to the market")
            ),
            gate=False,
        ),
        Check(
            label="Protective stop",
            ok=True,
            value=(
                f"${stop:.2f} at the exchange, as a separate order — "
                f"{float(config.stop_loss_pct):.0%} below the ${anchor:.2f} fill"
                if stop > 0 else
                f"none — the {quantity}-contract premium is the loss cap, and the deadline is "
                f"the exit that forces the issue"
            ),
            gate=False,
        ),
        Check(
            label="Hold deadline",
            ok=not deadline,
            value=f"session {int(memory.get('sessions_held', 0) or 0) + 1} of {config.max_hold_sessions}",
            limit=f"≤ {config.max_hold_sessions} sessions held",
            blocking=deadline,
        ),
    ]

    return SymbolPlan(
        symbol, HELD, orders,
        {
            **memory, "state": HELD, "contract": held_contract, "target": target, "stop": stop,
            "gate_failed_streak": gate_streak,
        },
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
    symbol: str, contract: str, quantity: int, target: float, stop: float, config: Any,
) -> list[DesiredOrder]:
    """What should be resting against an open position -- a target, or a target and a stop.

    **Always two independent orders, never a broker-side OCO/bracket.** The invariant a bracket
    exists to hold -- that only one side can ever fill -- is already this module's job: the
    lifecycle is re-derived from ``context.positions`` every run regardless, so a flat symbol
    already means "no sell orders wanted" whether the fill came from the target, the stop, or
    (before this) a bracket's OCO leg. Paying for a broker-side invariant this code already
    enforces itself was complexity bought twice, and Alpaca refuses any complex order class on
    options anyway -- the OCO path only ever ran on the brokers that could take it.

    **A ``stop`` of zero means no stop order reaches the exchange at all.** Not a stop at a
    distant price: a single resting sell limit, and nothing else. The strategy is then a
    resting buy limit followed by a resting sell limit, with the premium of
    ``contracts_per_trade`` as the loss cap and the deadline as the exit that forces the issue.

    **The exposure two independent legs create, stated plainly:** when one fills, the other is
    briefly live against a position that no longer exists. The next reconciliation cancels it,
    because a flat symbol wants no sell orders, so the window is one run of the cadence rather
    than open-ended. Two things keep it survivable in the meantime: the broker rejects a
    ``sell_to_close`` with nothing to close, and the remaining leg is a *sell* of a contract we
    no longer hold rather than anything that could open new exposure.
    """
    tolerance = REPRICE_MIN_PRICE_FRACTION
    limit_leg = _sell_leg(contract, quantity, order_type="limit", limit_price=target)
    # Checked before anything else, so no code path below can construct a stop leg.
    if stop <= 0:
        return [DesiredOrder(key=f"{symbol}:{TARGET}", request=limit_leg,
                             replace_tolerance=tolerance)]
    stop_leg = _sell_leg(contract, quantity, order_type="stop", stop_price=stop)
    return [
        DesiredOrder(key=f"{symbol}:{TARGET}", request=limit_leg, replace_tolerance=tolerance),
        # No tolerance: the stop never moves, so any difference from what is resting means
        # the resting order is not the one this position wants.
        DesiredOrder(key=f"{symbol}:{STOP}", request=stop_leg),
    ]
