"""Step 2: the decisions that need memory -- regime confirmation, cooldowns, eligibility runs.
"""

from __future__ import annotations

from .config import DualMomentumConfig
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
            "regime_risk_on": bool(row.get("regime_risk_on")),
            "regime_detail": str(row.get("regime_detail", "")),
            "vol_scale": float(row.get("vol_scale", 1.0) or 1.0),
        }
    return {"regime_risk_on": False, "regime_detail": "no signals", "vol_scale": 1.0}


def _defensive_book(rows: dict[str, dict[str, Any]]) -> dict[str, float]:
    """The defensive allocation ``analyze`` computed, whatever it ended up proposing."""
    return {
        symbol: float(row.get("defensive_weight", 0.0) or 0.0)
        for symbol, row in rows.items()
        if float(row.get("defensive_weight", 0.0) or 0.0) > 0
    }


def confirm_regime(state: dict[str, Any], raw_risk_on: bool, config: DualMomentumConfig) -> dict[str, Any]:
    """Turn the raw gate into a state, requiring consecutive agreement in both directions.

    Without this the book flips on a single reading that straddles the threshold, and pays
    the spread twice for the privilege.
    """
    confirmed = bool(state.get("regime_risk_on", False))
    agree = int(state.get("regime_agree", 0) or 0)
    disagree = int(state.get("regime_disagree", 0) or 0)

    if raw_risk_on:
        agree, disagree = agree + 1, 0
        if not confirmed and agree >= max(config.regime_confirm_bars, 1):
            confirmed = True
    else:
        disagree, agree = disagree + 1, 0
        if confirmed and disagree >= max(config.regime_exit_confirm_bars, 1):
            confirmed = False

    return {"regime_risk_on": confirmed, "regime_agree": agree, "regime_disagree": disagree}


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
    return filtered


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
