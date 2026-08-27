"""Translating a predicted underlying price into a contract price.

What remains of a module that used to predict the underlying as well. The prediction moved to
:mod:`.band`; the *translation* is still needed and is still the right shape, so it stays here
under the name the rest of the package imports.

The excursion-quantile machinery that used to sit above this -- ``excursions``,
``agreeing_mask``, ``expected_excursion``, ``predicted_extreme``, ``entry_underlying_target`` --
is gone. It priced a pullback bid, and measured over 183 sessions of IBIT and GLD that bid was
adversely selected: it filled on the days whose high came first (-0.54% bid to close) and was
absent on the days that ran (+1.14%). Deleted rather than left unused, so nothing can quietly
call it again.
"""

from __future__ import annotations

import pandas as pd


def session_fraction_remaining(now: pd.Timestamp, *, open_time: pd.Timestamp, close_time: pd.Timestamp) -> float:
    """How much of the regular session is left, as a fraction in ``[0, 1]``."""
    total = (close_time - open_time).total_seconds()
    if total <= 0:
        return 0.0
    remaining = (close_time - now).total_seconds()
    return min(max(remaining / total, 0.0), 1.0)


def target_price(current_price: float, budget: float, *, direction: str) -> float:
    """A price offset from the current one, by a fraction. Used by the *exit* only.

    The entry does not use this -- see :func:`entry_underlying_target`, which anchors to the
    session open instead. The exit legitimately reasons from the current mark: a profit target is
    a move to be captured from here, and a position may be held across sessions, which makes
    "today's open" meaningless for it.

    A buy waits below the market for a call and above it for a put, because the adverse move for
    a long put is the underlying rising. The *option* order is a buy either way -- the direction
    lives in the contract, not in the side.
    """
    if current_price <= 0:
        return 0.0
    move = current_price * max(budget, 0.0)
    return current_price + move if direction == "put" else current_price - move


def option_price_for(
    underlying_target: float,
    *,
    underlying_now: float,
    option_mark: float,
    delta: float,
) -> float:
    """Translate an underlying price into a contract price, first order, through delta.

    Predicting the underlying and translating is deliberate. A contract's own intraday series is
    thin, wide, has no history before today, and drifts down all day on theta for reasons that
    have nothing to do with direction -- so its "low" is not the thing being predicted. The
    underlying has decades of clean daily bars, and the chain hands us the delta to convert with.

    ``delta`` must keep its sign -- negative for a put. That is what makes one formula serve both
    sides: the underlying rising is a gain for a call and a loss for a put, and the sign of delta
    already says so. Taking its magnitude would price a put's adverse move as a gain.

    Gamma is ignored. Over the fraction of a percent this budget spans, the second-order term is
    smaller than the tick, and ignoring it errs toward bidding slightly low -- which for a buy is
    the safe side.
    """
    if underlying_now <= 0 or option_mark <= 0:
        return 0.0
    move = (underlying_target - underlying_now) * float(delta)
    return max(option_mark + move, 0.01)
