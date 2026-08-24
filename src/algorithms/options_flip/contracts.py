"""Picking the contract to trade, from a chain of a few hundred.

Three filters in a fixed order -- expiry, then delta, then liquidity -- and the order matters
because each one is cheaper and more decisive than the next.

**There is deliberately no spread ceiling.** A resting limit buy and a resting limit sell never
cross the spread -- a patient limit rests *inside* it and is paid by whoever is impatient, so a
wide market is if anything an opportunity for this strategy rather than a cost. The one place it
does bite is the stop, which is a market order once triggered and sells into the bid; that
slippage is an accepted risk of the strategy rather than something gated here.

**Open interest is the liquidity test**, and it is a different question: whether a resting order
finds a counterparty at all, which no spread figure answers. It varies by more than an order of
magnitude with the *expiry* rather than the symbol -- SPY's monthly carries a median above 2,000
across in-band strikes while its next weekly carries 78, and some weeklies sit at zero.

The spread is still measured and reported, because it is worth seeing: a wide quote makes the
midpoint a fiction, and every price estimate here is computed from it. At 12% wide, "fair value"
is a guess with a 12% error bar and the delta translation inherits it. That is information for
the reader, not a veto.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ...core.interfaces import Check
import dataclasses

from ...core.options import CALL, OptionContract, black_scholes_delta
from .config import DELTA_BUCKET, DELTA_TOLERANCE


def select_contract(
    chain: list[OptionContract], *, direction: str, as_of: date, config: Any
) -> tuple[OptionContract | None, OptionContract | None, list[Check]]:
    """``(chosen, candidate, checks)``: what will be traded, what was closest, and why.

    ``chosen`` is ``None`` when nothing survives, which is a normal outcome rather than an error:
    a thin chain on a quiet name should mean no trade, not a bad trade.

    ``candidate`` is the best contract inside the delta band *whether or not it cleared the
    liquidity floors*, and it exists purely so the deck can be evaluated. "No trade -- tightest
    spread 11.8%" says a gate closed; the same row carrying that contract's price, estimated
    range and spread says whether the name is 1% away from tradable or hopeless. That distinction
    is the whole question when deciding if a strategy is worth running, and it cannot be answered
    from a rejection alone.
    """
    wanted_type = CALL if direction == CALL else "put"
    candidates = [contract for contract in chain if contract.option_type == wanted_type]
    checks: list[Check] = []

    min_dte = max(int(config.min_dte), 0)
    dated = [contract for contract in candidates if contract.dte(as_of) >= min_dte]
    checks.append(Check(
        label=f"Contracts at least {min_dte} days out",
        ok=bool(dated),
        value=f"{len(dated)} of {len(candidates)} {wanted_type}s",
        limit=f"DTE ≥ {min_dte}",
        blocking=not dated,
    ))
    if not dated:
        return None, None, checks

    # The *nearest expiry that actually yields a tradable contract*, rather than the nearest
    # expiry full stop. Committing to the first one and giving up if it is thin throws away
    # trades for no reason: measured live, IBIT's 2 Sep weekly carries 92 open interest across
    # its in-band strikes while 4 Sep carries 13,252. Two days later is a rounding error against
    # a one-to-two-day hold; no trade at all is not.
    #
    # Still ascending, so the preference for less premium is intact -- a later expiry is only
    # reached when every nearer one failed.
    # One target, negated for puts. Candidates sit within DELTA_TOLERANCE of it.
    target = float(config.target_delta) * (1.0 if direction == CALL else -1.0)
    min_interest = int(config.min_open_interest)

    considered = 0
    best_effort: OptionContract | None = None
    for expiry in sorted({contract.expiry for contract in dated}):
        at_expiry = [contract for contract in dated if contract.expiry == expiry]
        in_band = [c for c in at_expiry if abs(c.delta - target) <= DELTA_TOLERANCE]
        if not in_band:
            continue
        considered += 1
        # Remembered from the first expiry that had any in-band strike, so a run that ends up
        # trading nothing can still show what it was looking at.
        candidate = min(in_band, key=lambda c: _rank(c, target))
        best_effort = best_effort or candidate
        liquid = [contract for contract in in_band if contract.open_interest >= min_interest]
        if liquid:
            best = min(liquid, key=lambda c: _rank(c, target))
            checks.append(Check(
                label="Strike inside the delta band",
                ok=True,
                value=f"{len(in_band)} of {len(at_expiry)} at {expiry:%d %b}",
                limit=f"delta within {DELTA_TOLERANCE:.2f} of {target:+.2f}",
            ))
            checks.append(Check(
                label="Liquid enough to trade",
                ok=True,
                value=f"{len(liquid)} of {len(in_band)} tradable at {expiry:%d %b}",
                limit=f"OI ≥ {min_interest}",
            ))
            checks.append(Check(
                label="Contract chosen",
                ok=True,
                value=(
                    f"{best.osi_symbol} — ${best.strike:g} {best.option_type}, "
                    f"delta {best.delta:+.2f}, {best.dte(as_of)}d, "
                    f"vol {best.volume}, OI {best.open_interest}, "
                    f"{best.bid:.2f}/{best.ask:.2f} ({best.spread_pct:.1%} wide)"
                ),
                limit=f"nearest delta {target:+.2f}, then volume",
            ))
            return best, candidate, checks

    # Nothing anywhere in the window cleared both filters.
    if not considered:
        checks.append(Check(
            label="Strike inside the delta band",
            ok=False,
            value=f"no strike in band across {len({c.expiry for c in dated})} expiries",
            limit=f"delta within {DELTA_TOLERANCE:.2f} of {target:+.2f}",
            blocking=True,
        ))
        return None, None, checks

    deepest = max((c.open_interest for c in dated if abs(c.delta - target) <= DELTA_TOLERANCE), default=0)
    checks.append(Check(
        label="Liquid enough to trade",
        ok=False,
        value=f"best open interest {deepest} across {considered} expiries",
        limit=f"OI ≥ {min_interest}",
        blocking=True,
    ))
    return None, best_effort, checks


def affordable_contracts(contract: OptionContract, config: Any) -> int:
    """How many contracts the per-trade notional cap allows, at this contract's ask.

    Priced at the ask rather than the mid because the cap is a statement about money that could
    actually leave the account, and a marketable order pays the offer.
    """
    wanted = max(int(config.contracts_per_position), 1)
    cost = (contract.ask or contract.midpoint) * 100.0
    if cost <= 0:
        return 0
    return max(min(wanted, int(float(config.max_notional_per_trade) // cost)), 0)


def fill_missing_deltas(
    chain: list[OptionContract], *, spot: float, annual_volatility: float, as_of: date
) -> tuple[list[OptionContract], bool]:
    """``(chain, estimated)`` -- greeks from Black-Scholes where the provider supplied none.

    Schwab returns ``-999`` for every greek outside the hours it computes them. This strategy
    selects entirely on delta, so without this the algorithm stops choosing contracts whenever
    that happens, and reports it as though the delta band were misconfigured.

    Only applied when the chain carries *no* usable delta at all. A partially-populated chain is
    left alone: mixing a provider's deltas with estimated ones would rank contracts against each
    other on numbers that came from different models.

    The estimate is worse than the real thing -- European exercise, no dividends, and the
    underlying's realised volatility standing in for the option's implied vol, which is missing
    for the same reason the greeks are. Measured against a Schwab chain it ran about 0.1 high on
    in-the-money strikes. The *ordering* is what selection needs and that is preserved, but the
    absolute level can shift the chosen strike by one or two, so the caller says on the deck that
    the greeks were estimated.
    """
    if not chain or any(c.delta for c in chain):
        return chain, False
    if spot <= 0 or annual_volatility <= 0:
        return chain, False
    filled = [
        dataclasses.replace(c, delta=black_scholes_delta(
            spot, c.strike, max((c.expiry - as_of).days, 0) / 365.0,
            annual_volatility, c.option_type,
        ))
        for c in chain
    ]
    return filled, True


def _rank(contract: OptionContract, target: float) -> tuple[int, int, int]:
    """Sort key: nearest the target delta, then the most traded, then the deepest inventory.

    Delta distance is compared in buckets rather than exactly, and that is the whole reason the
    liquidity keys ever get read. Delta is continuous, so a strict comparison is decided on a
    third decimal place every time and nothing after it is ever reached -- which is how a chain
    ends up choosing a contract 0.01 closer to target that trades forty times a week over one
    that trades ten thousand.

    Expected profit is deliberately *not* a key. It works out to exactly ``delta × band × 100``
    -- the contract's own price cancels out of the delta translation -- so ranking by it is
    ranking by the highest delta in the band, which contradicts aiming at a target rather than
    reinforcing it. Where the strike sits is ``target_delta``'s decision to make.

    Volume before open interest, because they answer different questions and only one of them is
    about whether a resting order will fill. Open interest counts positions already held: on
    IAU's chain the highest open interest of any strike, 1,798, belonged to a contract that
    traded forty times in six sessions, while a strike with 160 traded ninety-six times.
    """
    return (
        round(abs(contract.delta - target) / DELTA_BUCKET),
        -int(contract.volume),
        -int(contract.open_interest),
    )
