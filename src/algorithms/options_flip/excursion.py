"""Translate a predicted underlying price into a contract price."""

from __future__ import annotations

import pandas as pd


def session_fraction_remaining(now: pd.Timestamp, *, open_time: pd.Timestamp, close_time: pd.Timestamp) -> float:
    """Share of the regular session left, in ``[0, 1]``."""
    total = (close_time - open_time).total_seconds()
    if total <= 0:
        return 0.0
    remaining = (close_time - now).total_seconds()
    return min(max(remaining / total, 0.0), 1.0)


def target_price(current_price: float, budget: float, *, direction: str) -> float:
    """A price offset from ``current_price`` by a fraction. Exit side only."""
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

    ``delta`` keeps its sign (negative for a put) so one formula serves both sides. Gamma is
    ignored -- negligible over this budget's size, and erring toward a slightly low bid is the
    safe side for a buy.
    """
    if underlying_now <= 0 or option_mark <= 0:
        return 0.0
    move = (underlying_target - underlying_now) * float(delta)
    return max(option_mark + move, 0.01)
