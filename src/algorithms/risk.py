"""Risk controls shared by more than one algorithm.

Everything here is a pure function of its arguments. In particular nothing reads a clock: the
session drawdown breaker used to live twice, once keyed on the algorithm's own timestamp and
once on ``date.today()``, and only the first was correct under replay. A shared helper that
*can* fall back to the wall clock is the same bug with an extra step, because the fallback is
reached by forgetting rather than by deciding -- so ``as_of`` is required.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def session_key(as_of: datetime) -> str:
    """The trading day ``as_of`` falls in, as the breaker's state keys it."""
    return as_of.date().isoformat()


def session_drawdown_breached(
    state: dict[str, Any],
    equity: float,
    drawdown_limit: float,
    as_of: datetime,
) -> bool:
    """Session circuit breaker: once tripped it stays tripped until the next session.

    ``as_of`` is the moment the algorithm is reasoning about -- live the wall clock, in a
    replay the historical bar. It decides two things: which session's opening equity the
    drawdown is measured from, and when the breaker resets. Reading a clock here instead would
    make every replay step "today", so the breaker would trip once on the backtest's first bad
    day and stay latched for the rest of the run.

    ``state`` is mutated in place; persisting it is the caller's business, because where the
    state lives differs per algorithm.
    """
    session = session_key(as_of)
    if state.get("session") != session:
        state.update({"session": session, "session_start_equity": equity, "halted": False})
    start_equity = float(state.get("session_start_equity") or equity)
    drawdown = (equity / start_equity - 1.0) if start_equity > 0 else 0.0
    if drawdown <= drawdown_limit:
        state["halted"] = True
    state["session_drawdown"] = drawdown
    return bool(state.get("halted"))
