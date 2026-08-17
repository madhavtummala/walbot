"""Step 2: the decisions that need memory -- cooldowns, eligibility runs, theme rotation.
"""

from __future__ import annotations

from .config import EPSILON, DualMomentumConfig
from .layers import theme_of



import logging
from datetime import datetime
from typing import Any

from ...common.timeutils import minutes_between as _minutes_between, parse_iso_utc as _parse_time

logger = logging.getLogger(__name__)


def _run_facts(signals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Read the run-level fields ``analyze`` denormalised onto every row.

    The timestamp used to be one of them. It is a parameter of ``refine`` now, so it no longer
    has to be copied onto every signal row and parsed back out -- and a caller cannot lose it
    by handing over signals it built itself.
    """
    for row in signals.values():
        return {
            "data_ok": bool(row.get("data_ok")),
            "data_detail": str(row.get("data_detail", "")),
            "vol_scale": float(row.get("vol_scale", 1.0) or 1.0),
        }
    return {"data_ok": False, "data_detail": "no signals", "vol_scale": 1.0}


def _defensive_book(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    """The defensive allocation ``analyze`` computed, whatever it ended up proposing."""
    return {
        symbol: float(row.get("defensive_weight", 0.0) or 0.0)
        for symbol, row in rows.items()
        if float(row.get("defensive_weight", 0.0) or 0.0) > 0
    }


def apply_turnover_filters(
    target: dict[str, float],
    current: dict[str, float],
    equity: float,
    config: DualMomentumConfig,
) -> dict[str, float]:
    """Drop trades too small to be worth their costs, keeping the current weight instead.

    A full exit is never "too small". The thresholds here exist to suppress small
    *adjustments* to a position the algorithm still wants; applied to a close they instead
    trap it, because a holding below ``rebalance_weight_threshold`` can never move far enough
    to clear the bar and is therefore held forever. That is how an 8-position book came to
    carry a mean of 12 names, 5 of them frozen below the threshold and holding a fifth of the
    equity, with ``max_positions`` exceeded on half of all days.
    """
    minimum_notional = max(config.minimum_trade_notional, config.minimum_trade_nav_fraction * max(equity, 0.0))
    filtered: dict[str, float] = {}
    for symbol, weight in target.items():
        held = float(current.get(symbol, 0.0))
        move = abs(weight - held)
        # Leaving the book entirely: only the absolute notional floor applies, so a position
        # the algorithm has decided to exit is actually exited.
        closing = weight <= 0 < held
        if not closing and move < max(config.rebalance_weight_threshold, 0.0):
            filtered[symbol] = held
            continue
        if move * max(equity, 0.0) < minimum_notional:
            filtered[symbol] = held
            continue
        filtered[symbol] = weight
    return _fit_to_budget(filtered, target, current, equity, config)


def _fit_to_budget(
    filtered: dict[str, float],
    target: dict[str, float],
    current: dict[str, float],
    equity: float,
    config: DualMomentumConfig,
) -> dict[str, float]:
    """Give back whatever holding a name at its current weight borrowed from the budget.

    The band suppresses moves in both directions, but only the *trims* were funding anything.
    Holding two incumbents above target because their trims were too small to trade, while
    letting a new entry through at full size because its move was large enough, produces a
    target vector that no longer sums to the gross the algorithm budgeted for.

    Measured on 2026-01-08: ``refine`` proposed exactly 100.0%; the band held IEMG at 12.2%
    against a 6.6% target and XSD at 36.0% against 29.1%, passed XBI's 12.1% entry untouched,
    and handed planning a 106.9% book. That is unfundable by construction on an account
    already 98% invested, so ``fund_planned_orders`` pro-rata shrank the buys and warned --
    16 of 24 sessions that January, and on one of them the entry was dropped outright.

    The repair shrinks the *increases* rather than forcing the trims through: a suppressed
    trim stays suppressed, and the entry simply arrives smaller, filling in over later runs as
    the incumbents drift far enough to trade.

    Only the notional floor is re-applied to a shrunken leg, not
    ``rebalance_weight_threshold``. The band exists to suppress *drift* -- small adjustments to
    a position the algorithm already holds -- and an opening is not drift. Re-applying it here
    killed the entry outright whenever the freed budget came to less than the band, which on
    the 2026-01-08 book meant a 12.1% entry shrinking to 5.2% and then being dropped: the
    opposite of the intent, since the whole repair exists so that entries still happen.
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
    minimum_notional = max(config.minimum_trade_notional, config.minimum_trade_nav_fraction * max(equity, 0.0))
    repaired = dict(filtered)
    for symbol, increase in increases.items():
        held = float(current.get(symbol, 0.0))
        shrunk = held + increase * keep
        move = shrunk - held
        repaired[symbol] = held if move * max(equity, 0.0) < minimum_notional else shrunk
    return repaired


def partial_adjustment(
    target: dict[str, float],
    current: dict[str, float],
    config: DualMomentumConfig,
) -> dict[str, float]:
    """Move ``config.rebalance_step`` of the way from ``current`` to ``target``.

    See ``rebalance_step``. Applied before :func:`apply_turnover_filters`, so a step small
    enough to leave the move under the no-trade band results in no trade at all rather than in
    a token one -- the two brakes compose instead of fighting.

    A target of zero is honoured in full. Decaying an exit would keep a name the strategy has
    already rejected on the book for several more runs, which is the one kind of turnover
    saving that is not worth having.
    """
    step = max(min(config.rebalance_step, 1.0), 0.0)
    if step >= 1.0:
        return dict(target)
    adjusted: dict[str, float] = {}
    for symbol, weight in target.items():
        held = float(current.get(symbol, 0.0))
        adjusted[symbol] = weight if weight <= 0 else held + step * (weight - held)
    return adjusted


def action_due(
    state: dict[str, Any],
    action: str,
    as_of: datetime,
    interval_days: int,
) -> bool:
    """Whether ``action`` may be taken now, given its own interval in days.

    One clock per *kind* of decision rather than one clock for the whole algorithm. The run
    cadence and the decision cadence had collapsed into each other: the algorithm runs once a
    session, so it re-ranked, re-sized and rotated themes once a session, against a score whose
    slowest horizon is twelve sessions.

    ``signal_refresh_minutes`` used to be the single knob for this and was read by nothing at
    all -- declared in the config, described in the dashboard's explainer as "how often
    membership may change", and never compared against the ``last_selection_at`` the algorithm
    faithfully wrote on every run. It is replaced by these, because "how often may membership
    change" turned out to be four different questions with four different right answers.

    Measured against the timestamp ``analyze`` reasoned about rather than the wall clock, so a
    replay throttles exactly as the live runner would. ``state`` is mutated by
    :func:`record_action`, not here, so a caller that decides not to act does not restart the
    clock.
    """
    days = max(interval_days, 0)
    if days <= 0:
        return True
    last = _parse_time(str(state.get(f"last_{action}_at", "")))
    if last is None:
        return True
    return _minutes_between(as_of, last) >= days * 1440


def record_action(state: dict[str, Any], action: str, as_of: str) -> None:
    """Start ``action``'s clock. Called only when the action was actually available."""
    state[f"last_{action}_at"] = as_of


def track_eligibility(
    state: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
) -> dict[str, list[int]]:
    """Record today's eligibility per symbol and return each one's recent window.

    Eligibility is otherwise a stateless per-day test, which makes every floor a coin-flip
    boundary for anything sitting near it. Counting instead asks "has this been qualifying?",
    so an entry needs a run of agreement and an exit needs a run of disagreement.

    Returns the series rather than the count so callers can tell a name that has failed the
    test from one that has simply not been watched long enough yet. That distinction matters:
    a cold state store would otherwise read every holding as ineligible and liquidate the book
    on the first run after a restart.

    ``state`` is mutated in place; persisting it is the caller's business.
    """
    window = max(config.eligibility_window, 1)
    history = state.get("eligible_history")
    if not isinstance(history, dict):
        history = {}
    updated: dict[str, list[int]] = {}
    for symbol, row in rows.items():
        past = history.get(symbol) or []
        if not isinstance(past, list):
            past = []
        updated[symbol] = [*past, 1 if int(row.get("eligible", 0)) else 0][-window:]
    state["eligible_history"] = updated
    return updated


def resolve_themes(
    held: set[str],
    entrants: set[str],
    theme_score: dict[str, float],
    config: DualMomentumConfig,
) -> set[str]:
    """Which themes the book holds: keep what is held, fill free slots, then displace.

    Selection operates on themes rather than names because a theme is the unit the strategy
    actually has a view about. Swapping QQQM for XSD inside ``us_growth`` is handled by
    ``theme_allocation`` at full speed; getting in or out of ``us_growth`` at all is this
    function's decision, and it is deliberately reluctant.
    """
    slots = max(config.max_positions, 0)
    delta = max(config.min_score_delta_to_replace, 0.0)

    def score(theme: str) -> float:
        return float(theme_score.get(theme, 0.0))

    selection = set(held)
    contenders = sorted(entrants - selection, key=score, reverse=True)

    for theme in contenders:
        if len(selection) >= slots:
            break
        selection.add(theme)

    for theme in contenders:
        if theme in selection or not selection:
            continue
        weakest = min(selection, key=score)
        if score(theme) <= score(weakest) + delta:
            continue
        logger.info(
            "Dual Momentum rotating theme %s (%.2f) out for %s (%.2f)",
            weakest, score(weakest), theme, score(theme),
        )
        selection.discard(weakest)
        selection.add(theme)

    if len(selection) > slots:
        selection = set(sorted(selection, key=score, reverse=True)[:slots])
    return selection


def _record_exits(state: dict[str, Any], held: set[str], final: set[str], as_of: str) -> None:
    """Remember when a symbol left the book, so re-entry can be held off for a cooldown."""
    exits = state.setdefault("exited_at", {})
    if not isinstance(exits, dict):
        exits = {}
        state["exited_at"] = exits
    for symbol in held - final:
        exits[str(symbol)] = as_of
    for symbol in final:
        exits.pop(str(symbol), None)


def _in_cooldown(
    state: dict[str, Any],
    symbol: str,
    now: datetime | None,
    config: DualMomentumConfig,
) -> bool:
    """Whether ``symbol`` exited too recently to be re-entered.

    Elapsed wall-clock minutes, not market ones: ``cooldown_after_exit`` counts algorithm
    *runs*, so the window is that many times ``risk_refresh_minutes``. Compared against the
    timestamp ``analyze`` recorded -- which the backtester sets to the historical bar, so a
    replay applies the same cooldown the live runner would have.
    """
    exits = state.get("exited_at") if isinstance(state.get("exited_at"), dict) else {}
    exited_at = _parse_time(str(exits.get(symbol, "")))
    if exited_at is None:
        return False
    window = max(config.cooldown_after_exit, 0) * max(config.risk_refresh_minutes, 1)
    if _minutes_between(now, exited_at) >= window:
        return False
    logger.info("Dual Momentum holding off %s: inside the %d-minute re-entry cooldown", symbol, window)
    return True
