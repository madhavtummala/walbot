"""How much of each name, and how much of that is worth trading today.

Everything here maps a set of chosen symbols onto weights, and then damps the move from the
current book to those weights. The two halves were split across separate modules -- one called
"layers", one called "stateful" -- although neither reads any state and both are pure functions
of a weight vector. They are one subject: sizing.
"""

from __future__ import annotations

from typing import Any

from .config import EPSILON, RallyRotationConfig


def score_to_weights(rows: list[dict[str, Any]], config: RallyRotationConfig) -> dict[str, float]:
    """Split ``risk_on_gross_max`` between the selected names, in proportion to score.

    The score enters as its positive part, so a name barely above the universe median gets a
    small position rather than an equal one, tilted by ``volatility_tilt``.

    No per-name cap. A cap on top of a two-name split does not diversify anything -- it forces
    the pair toward equal weight and throws away the ranking the whole algorithm exists to
    produce. It also came with a water-filling routine to re-spread the overflow, because the
    first version dropped it and left the book quietly under-invested: one selected name produced
    a 50% position and 50% idle cash rather than the gross the configuration asked for. Both are
    gone; a proportional split cannot leave a residual.
    """
    raw: dict[str, float] = {}
    for row in rows:
        excess = max(float(row.get("base_score", 0.0)), 0.0)
        volatility = float(row.get("annual_volatility", 0.0))
        # sigma ** tilt: negative divides (risk parity), zero ignores it, positive leans in.
        scale = (volatility + EPSILON) ** config.volatility_tilt if config.volatility_tilt else 1.0
        raw[str(row["symbol"])] = excess * scale

    gross = max(config.risk_on_gross_max, 0.0)
    total = sum(raw.values())
    if total <= EPSILON:
        # Every candidate sits exactly at the floor: equal-weight rather than divide by zero.
        return {symbol: gross / len(raw) for symbol in raw} if raw else {}
    return {symbol: gross * value / total for symbol, value in raw.items()}


def defensive_weights(scored: dict[str, dict[str, Any]], config: RallyRotationConfig) -> dict[str, float]:
    """Where the book sits when risk-on is not permitted.

    Ranked by medium-term absolute return, so the defensive sleeve is itself chosen rather than
    fixed. No per-name cap: the point of this sleeve is to be in T-bills, and a cap would force
    idle cash for no reason.
    """
    candidates = [scored[symbol] for symbol in config.defensive_universe if symbol in scored]
    if not candidates:
        return {}
    candidates.sort(key=lambda row: float(row.get("abs_return", 0.0)), reverse=True)
    chosen = candidates[: max(config.defensive_max_positions, 1)]
    share = max(config.risk_on_gross_max, 0.0) / len(chosen)
    return {str(row["symbol"]): share for row in chosen}


def park_residual(
    weights: dict[str, float],
    defensive_book: dict[str, float],
    config: RallyRotationConfig,
) -> dict[str, float]:
    """Put whatever the risk sleeve could not deploy into the defensive sleeve, not into cash.

    A real funded account never holds idle cash: the balance sits in T-bills until something
    needs it, which is the same reason the backtester opens the book in a cash equivalent rather
    than in cash. Without this the book was a three-way split -- risk assets, T-bills, and a raw
    cash slice earning nothing -- when the intent is binary. Over 2023 that slice averaged 10.3%
    of equity and reached 15% in some months.
    """
    gross = max(config.risk_on_gross_max, 0.0)
    residual = gross - sum(value for value in weights.values() if value > 0)
    total = sum(defensive_book.values())
    if residual <= EPSILON or total <= EPSILON:
        return dict(weights)

    # Spread across the defensive sleeve in its own proportions, so a multi-name sleeve keeps the
    # ranking ``defensive_weights`` gave it.
    combined = dict(weights)
    for symbol, share in defensive_book.items():
        combined[symbol] = combined.get(symbol, 0.0) + residual * share / total
    return combined


# --------------------------------------------------------------------------------------
# Damping: how much of the gap to the target is worth crossing today.
# --------------------------------------------------------------------------------------


def apply_turnover_filters(
    target: dict[str, float],
    current: dict[str, float],
    equity: float,
    config: RallyRotationConfig,
) -> dict[str, float]:
    """Drop trades too small to be worth their costs, keeping the current weight instead.

    A full exit is never "too small". The thresholds here exist to suppress small *adjustments*
    to a position the algorithm still wants; applied to a close they instead trap it, because a
    holding below ``rebalance_weight_threshold`` can never move far enough to clear the bar and
    is therefore held forever. That is how an 8-position book came to carry a mean of 12 names,
    5 of them frozen below the threshold and holding a fifth of the equity.
    """
    minimum_notional = _minimum_notional(equity, config)
    filtered: dict[str, float] = {}
    for symbol, weight in target.items():
        held = float(current.get(symbol, 0.0))
        move = abs(weight - held)
        # Leaving the book entirely: only the absolute notional floor applies, so a position the
        # algorithm has decided to exit is actually exited.
        closing = weight <= 0 < held
        too_small = (not closing and move < max(config.rebalance_weight_threshold, 0.0)) or (
            move * max(equity, 0.0) < minimum_notional
        )
        filtered[symbol] = held if too_small else weight
    return _fit_to_budget(filtered, target, current, equity, config)


def _fit_to_budget(
    filtered: dict[str, float],
    target: dict[str, float],
    current: dict[str, float],
    equity: float,
    config: RallyRotationConfig,
) -> dict[str, float]:
    """Give back whatever holding a name at its current weight borrowed from the budget.

    The band suppresses moves in both directions, but only the *trims* were funding anything.
    Holding two incumbents above target because their trims were too small to trade, while
    letting a new entry through at full size because its move was large enough, produces a target
    vector that no longer sums to the gross the algorithm budgeted for.

    Measured on 2026-01-08: the plan proposed exactly 100.0%; the band held IEMG at 12.2% against
    a 6.6% target and XSD at 36.0% against 29.1%, passed XBI's 12.1% entry untouched, and handed
    planning a 106.9% book. That is unfundable by construction on an account already 98%
    invested, so ``fund_planned_orders`` pro-rata shrank the buys and warned -- 16 of 24 sessions
    that January, and on one of them the entry was dropped outright.

    The repair shrinks the *increases* rather than forcing the trims through: a suppressed trim
    stays suppressed, and the entry simply arrives smaller, filling in over later runs as the
    incumbents drift far enough to trade.

    Only the notional floor is re-applied to a shrunken leg, not ``rebalance_weight_threshold``.
    The band exists to suppress *drift* -- small adjustments to a position already held -- and an
    opening is not drift. Re-applying it here killed the entry outright whenever the freed budget
    came to less than the band, which on the 2026-01-08 book meant a 12.1% entry shrinking to
    5.2% and then being dropped: the opposite of the intent.
    """
    budget = sum(weight for weight in target.values() if weight > 0)
    excess = sum(weight for weight in filtered.values() if weight > 0) - budget
    if excess <= EPSILON:
        return filtered

    increases = {
        symbol: weight - float(current.get(symbol, 0.0))
        for symbol, weight in filtered.items()
        if weight - float(current.get(symbol, 0.0)) > EPSILON
    }
    total = sum(increases.values())
    if total <= EPSILON:
        # Nothing is being added, so the overshoot is entirely incumbents held above target.
        # Forcing those trims through is the one thing the band exists to prevent.
        return filtered

    keep = max(1.0 - excess / total, 0.0)
    minimum_notional = _minimum_notional(equity, config)
    repaired = dict(filtered)
    for symbol, increase in increases.items():
        held = float(current.get(symbol, 0.0))
        shrunk = held + increase * keep
        repaired[symbol] = held if (shrunk - held) * max(equity, 0.0) < minimum_notional else shrunk
    return repaired


def _minimum_notional(equity: float, config: RallyRotationConfig) -> float:
    return max(config.minimum_trade_notional, config.minimum_trade_nav_fraction * max(equity, 0.0))
