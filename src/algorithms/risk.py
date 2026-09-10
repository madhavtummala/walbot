"""Risk controls shared by more than one algorithm.

Pure functions of their arguments -- nothing reads a clock. ``as_of`` is required rather than
defaulted to ``date.today()`` so a replay's historical bar is never silently read as "today".
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

    ``as_of`` decides which session's opening equity the drawdown is measured from and when the
    breaker resets. ``state`` is mutated in place; persisting it is the caller's business.
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
