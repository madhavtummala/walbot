"""Step 2: the decisions that need memory -- regime confirmation, cooldowns, drawdown breaker.
"""

from __future__ import annotations

from .config import DualMomentumConfig



import logging
from datetime import datetime
from typing import Any

from ...common.timeutils import minutes_between as _minutes_between, parse_iso_utc as _parse_time
from ..risk import session_drawdown_breached

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


def intraday_drawdown_breached(
    state: dict[str, Any],
    equity: float,
    config: DualMomentumConfig,
    as_of: datetime,
) -> bool:
    """This algorithm's binding of the shared session breaker -- see ``algorithms/risk.py``."""
    return session_drawdown_breached(state, equity, config.intraday_drawdown_limit, as_of)


def apply_turnover_filters(
    target: dict[str, float],
    current: dict[str, float],
    equity: float,
    config: DualMomentumConfig,
) -> dict[str, float]:
    """Drop trades too small to be worth their costs, keeping the current weight instead."""
    minimum_notional = max(config.minimum_trade_notional, config.minimum_trade_nav_fraction * max(equity, 0.0))
    filtered: dict[str, float] = {}
    for symbol, weight in target.items():
        held = float(current.get(symbol, 0.0))
        move = abs(weight - held)
        if move < max(config.rebalance_weight_threshold, 0.0):
            filtered[symbol] = held
            continue
        if move * max(equity, 0.0) < minimum_notional:
            filtered[symbol] = held
            continue
        filtered[symbol] = weight
    return filtered


def _resolve_replacements(
    incumbents: set[str],
    entrants: set[str],
    rows: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
) -> set[str]:
    """Fill free slots first, then let a challenger displace the weakest incumbent.

    The challenger has to win by ``min_score_delta_to_replace``; a hair's-breadth improvement
    is noise, and trading on it costs the spread every time.
    """
    def score(symbol: str) -> float:
        return float(rows.get(symbol, {}).get("base_score", 0.0))

    selection = set(incumbents)
    free = max(config.max_positions, 0) - len(selection)
    ordered = sorted(entrants, key=score, reverse=True)

    for symbol in ordered[: max(free, 0)]:
        selection.add(symbol)

    for symbol in ordered[max(free, 0):]:
        if not selection:
            break
        weakest = min(selection, key=score)
        if score(symbol) <= score(weakest) + max(config.min_score_delta_to_replace, 0.0):
            continue
        logger.info(
            "Dual Momentum replacing %s (%.2f) with %s (%.2f)",
            weakest, score(weakest), symbol, score(symbol),
        )
        selection.discard(weakest)
        selection.add(symbol)
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
