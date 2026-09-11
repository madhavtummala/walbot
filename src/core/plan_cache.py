"""Plans held between the call that proposes one and the call that acts on it.

An agent reviews a proposal and then decides whether to submit it, which is two round trips
with a gap in between. Something has to hold the plan across that gap. It used to be the agent
itself -- ``get_algorithm_plan`` serialised a plan out and ``place_orders`` rebuilt one from
whatever came back -- which made the agent's copy authoritative: a hand-authored payload that
no algorithm ever produced submitted just as readily as a real one, and ``latest_prices`` and
``state`` were committed exactly as sent.

Holding it here instead inverts that. The agent gets a token, and the plan it reviewed is the
plan that executes, because the agent never had the ability to change it. What it may still do
is named explicitly and applied on this side -- see :mod:`src.core.plan_edits`.

Deliberately in-process and deliberately forgetful. A stored plan carries the prices it was
built with, and ``place_orders`` commits those rather than re-reading the market, so an old
plan submits stale limit prices. :data:`DEFAULT_TTL_SECONDS` is what keeps "what the agent
reviewed" and "what is true now" close enough to each other to be the same thing; expiry is a
normal outcome, not a failure. That also settles the storage question -- a restart takes longer
than the TTL, so a plan surviving one would be too old to trade anyway. Revisit only if the MCP
server is ever run with more than one worker, where two processes would not share this dict.
"""

from __future__ import annotations

import logging
import secrets
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .interfaces import AlgorithmPlan

logger = logging.getLogger(__name__)

#: Short enough that the prices riding on a claimed plan are still roughly the market's, long
#: enough for an agent to read the signals and run a sentiment check against the news.
DEFAULT_TTL_SECONDS = 90


class PlanUnavailable(Exception):
    """A token named no plan this cache can still hand back.

    Carries ``reason`` in the shape the MCP tools already report refusals in, so the agent is
    told whether to re-plan or to stop rather than being handed a bare failure.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PendingPlan:
    """A proposal waiting on a decision, with the binding it was sized against."""

    plan: AlgorithmPlan
    binding_id: str
    account_id: str
    stashed_at: datetime
    expires_at: datetime


_lock = threading.Lock()
_pending: dict[str, PendingPlan] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _evict_expired(now: datetime) -> None:
    """Drop what can no longer be claimed. Called under ``_lock``."""
    for token in [token for token, entry in _pending.items() if entry.expires_at <= now]:
        del _pending[token]


def stash(
    plan: AlgorithmPlan,
    *,
    binding_id: str,
    account_id: str,
    ttl_seconds: int = DEFAULT_TTL_SECONDS,
) -> str:
    """Hold ``plan`` and return the token that claims it.

    The account is recorded alongside because a plan is sized against one specific book; a
    token claimed against a different account would submit quantities that were never
    computed for it.
    """
    now = _now()
    token = secrets.token_urlsafe(16)
    entry = PendingPlan(
        plan=plan,
        binding_id=binding_id,
        account_id=account_id,
        stashed_at=now,
        expires_at=now + timedelta(seconds=max(1, int(ttl_seconds))),
    )
    with _lock:
        _evict_expired(now)
        _pending[token] = entry
    return token


def claim(token: str) -> PendingPlan:
    """Take a plan out of the cache. Single use: a second claim of the same token fails.

    Consuming the token rather than reading it is what stops a retry -- an agent that did not
    see the response, a scheduler nudge, a duplicated tool call -- from submitting the same
    batch of orders twice.

    Spent even when a later gate refuses the submission, which costs a wasted token on a
    binding that turned out to be switched off. That is the right way round: re-planning is a
    cheap read, and an agent that reaches a gate, waits for it to open, and then submits should
    be acting on prices from after the wait rather than from before it.
    """
    now = _now()
    with _lock:
        _evict_expired(now)
        entry = _pending.pop(str(token or ""), None)
    if entry is None:
        raise PlanUnavailable(
            "No pending plan for that token. It may have expired, or already been used: "
            "call get_algorithm_plan again and review the fresh plan."
        )
    # Belt and braces: eviction above already removed it, but claiming is the one path where
    # acting on a stale plan costs money rather than a wasted read.
    if entry.expires_at <= now:
        age = (now - entry.stashed_at).total_seconds()
        raise PlanUnavailable(
            f"That plan is {age:.0f}s old and its prices are stale. Call get_algorithm_plan again."
        )
    return entry


def clear() -> None:
    """Forget everything pending. For tests and for a clean shutdown."""
    with _lock:
        _pending.clear()
