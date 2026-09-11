"""The session drawdown breaker. Currently wired to no algorithm, deliberately.

It measures the fall from the equity it first saw *this session*, so it only means anything to
an algorithm that runs more than once inside one. At a daily cadence every run opens a new
session and rebases that reference to current equity -- the drawdown it reports is identically
zero however far the book fell overnight, and a knob that reads as crash protection while
providing none is worse than no knob. That is why Rally Rotation's ``intraday_drawdown_limit``
was removed rather than switched off; ``max_daily_drop``, which reads close-to-close returns,
is the stop that survived.

Kept rather than deleted because the logic is correct for an intraday algorithm and the trap is
in the interaction, not in the code. ``test_a_session_breaker_cannot_fire_at_this_algorithms_cadence``
in ``tests/test_rally_rotation.py`` is the record of it, and the only caller.

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
