"""How much of each name, and how much of that is worth trading today.

Maps a set of chosen symbols onto weights, then damps the move from the current book to those
weights -- both pure functions of a weight vector.
"""

from __future__ import annotations

from typing import Any

from .config import EPSILON, RallyRotationConfig


def score_to_weights(rows: list[dict[str, Any]], config: RallyRotationConfig) -> dict[str, float]:
    """Split ``risk_on_gross_max`` between the selected names, in proportion to score (positive
    part only, tilted by ``volatility_tilt``). No per-name cap -- one on top of a two-name split
    forces equal weight and throws away the ranking the algorithm exists to produce; a
    proportional split cannot leave a residual, so no water-filling is needed either.
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
    """Put whatever the risk sleeve could not deploy into the defensive sleeve, not into cash --
    a real funded account never holds idle cash; the balance sits in T-bills."""
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

    A full exit is never "too small": these thresholds suppress small *adjustments* to a
    position still wanted, but applied to a close they'd trap a position below the threshold
    forever, since it can never move far enough on its own to clear the bar.
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

    The band suppresses moves both directions, but only the *trims* were funding anything: a
    suppressed trim plus a full-size new entry sums to more than the gross budgeted. This
    shrinks the *increases* instead of forcing trims through -- a suppressed trim stays
    suppressed, and the entry arrives smaller, filling in as incumbents later drift far enough
    to trade on their own. Only the notional floor is re-applied to a shrunken leg, not
    ``rebalance_weight_threshold``: that band suppresses drift on a position already held, and
    an opening is not drift -- re-applying it here could shrink an entry below the band and
    drop it outright, the opposite of the intent.
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
