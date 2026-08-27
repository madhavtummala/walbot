"""What this algorithm remembers between runs, and the decisions that need it.

Everything here reads or writes ``AlgorithmContext.state``. Nothing else in the package does --
which is the point of gathering it: the pure market maths and the path-dependent decisions were
previously interleaved under a module named "stateful" that was mostly stateless.

The memory is small and deliberately so: one clock per kind of decision, and a short window of
per-symbol history. All of it is mutated in place and handed back on the plan; none of it is
persisted here. That happens in ``execute``, and only if orders actually went out.

**Everything here is measured in market days, never in runs.** That is the one invariant this
module exists to hold. Counting runs made every interval below a function of the binding's cron:
under ``15 10-14 * * 1-5`` -- five fires a session -- "ranked in the top 3 for 3 runs" was
satisfied by 36 minutes of one morning, and "re-rank every 5 days" re-ranked daily. Worse, the
count only advances when the algorithm actually runs, so pausing a binding froze every clock
mid-stride and editing a cron silently redefined all of them at once. Bursty DCA had already
settled this for accrual: the schedule controls the *opportunity* to act, never the rate.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from ...core.interfaces import MARKET_TZ
from .config import RallyRotationConfig

logger = logging.getLogger(__name__)


def market_day(as_of: datetime) -> str:
    """The trading day a run belongs to, as an ISO date in market-local time.

    Market-local rather than UTC, and the same :data:`MARKET_TZ` the cron scheduler fires
    against, so a run and the day it is filed under can never disagree. Under UTC a 16:00 ET
    fire in summer lands on the next calendar day and would open a second bucket for a session
    that had already been recorded.

    A naive timestamp is read as market time rather than as UTC. Replays hand this method bar
    stamps, which are market-local by construction.
    """
    moment = as_of.astimezone(MARKET_TZ) if as_of.tzinfo else as_of.replace(tzinfo=MARKET_TZ)
    return moment.date().isoformat()


def _weekdays_inclusive(first: date, last: date) -> int:
    """Weekdays from ``first`` to ``last``, both ends included."""
    if last < first:
        return 0
    total = (last - first).days + 1
    weeks, remainder = divmod(total, 7)
    count = weeks * 5
    for offset in range(remainder):
        if (first + timedelta(days=offset)).weekday() < 5:
            count += 1
    return count


def sessions_since(last_day: str, as_of: datetime) -> int:
    """Sessions elapsed since ``last_day``, not counting the day itself.

    Weekdays stand in for sessions. Elapsed time is what a throttle should measure -- a binding
    that was paused for a fortnight has to re-rank the moment it comes back, and a count of runs
    says it is not due yet -- but *calendar* days are the wrong unit for it: a 3-day interval set
    on a Friday would be satisfied by Monday, throttling Tuesday-to-Thursday decisions while
    waving every Friday-to-Monday one straight through.

    Weekdays rather than a real exchange calendar, so the arithmetic stays a pure function of two
    dates and a replay throttles identically to the live path. The cost is that a holiday inside
    the window counts as a session, so an interval can come due one day early a handful of times
    a year. That is a rounding error against a knob measured in weeks, and it is bounded: it can
    only ever make the throttle *shorter*, never let it run past its interval.
    """
    start = date.fromisoformat(last_day)
    end = date.fromisoformat(market_day(as_of))
    return _weekdays_inclusive(start + timedelta(days=1), end)


def action_due(state: dict[str, Any], action: str, interval_days: int, as_of: datetime) -> bool:
    """Whether ``action`` may be taken now, given its own interval in trading days.

    One clock per *kind* of decision rather than one for the whole algorithm. The run cadence and
    the decision cadence had collapsed into each other: the algorithm re-ranked, re-sized and
    rotated on every fire, against a score whose slowest horizon is twelve sessions.

    A cold clock -- no record, or the ``last_<action>_run`` counter written by the run-indexed
    predecessor -- reads as due. Re-ranking once on the first run after an upgrade is the safe
    direction: the alternative is inventing a start date for a clock nobody started.

    ``state`` is mutated by :func:`record_action`, not here, so a caller that asks and then
    decides not to act does not restart the clock.
    """
    interval = max(interval_days, 0)
    if interval <= 0:
        return True
    last = state.get(f"last_{action}_day")
    if not isinstance(last, str) or not last:
        return True
    try:
        return sessions_since(last, as_of) >= interval
    except ValueError:
        logger.warning("Rally Rotation ignoring an unreadable %s clock: %r", action, last)
        return True


def record_action(state: dict[str, Any], action: str, as_of: datetime) -> None:
    """Start ``action``'s clock on this run's market day. Only when the action was taken."""
    state[f"last_{action}_day"] = market_day(as_of)


def track_ranking(
    state: dict[str, Any],
    rows: dict[str, dict[str, Any]],
    config: RallyRotationConfig,
    as_of: datetime,
) -> dict[str, dict[str, int]]:
    """The same window, for whether the name ranked inside ``entry_rank_max``.

    A tighter gate than eligibility: the name must not only pass absolute momentum but also be
    among the best relative performers, and have been for long enough to mean it.
    """
    limit = max(config.entry_rank_max, 0)
    return _track(
        state, "ranked_top_history", rows, config, as_of,
        lambda row: int(0 < int(row.get("rank") or 0) <= limit),
    )


def qualifying_days(series: dict[str, int] | None) -> int:
    """Days in the window on which the name met the test."""
    return sum(1 for value in (series or {}).values() if value)


def observed_days(series: dict[str, int] | None) -> int:
    """Days in the window the algorithm actually looked at this name.

    Distinct from :func:`qualifying_days` by design, and the difference is what tells a name that
    failed the test apart from one nobody has watched long enough yet.
    """
    return len(series or {})


def _track(
    state: dict[str, Any],
    key: str,
    rows: dict[str, dict[str, Any]],
    config: RallyRotationConfig,
    as_of: datetime,
    measure: Any,
) -> dict[str, dict[str, int]]:
    """Fold this run into a per-symbol window of the last ``eligibility_window`` market days.

    Keyed by day rather than appended per run, which is the whole point: under a cron that fires
    five times a session the old list counted one morning as five days of evidence, so the
    settling period a name had to serve was set by the schedule rather than by the config.

    Within a day the **last** run wins. A day is one observation of a daily-bar test, and the
    latest look is the best-informed one -- rather than the first, which would freeze the day at
    the open, or "eligible at any point today", which would let a name that led for one bar and
    faded bank a full day of credit.

    Only days this algorithm actually ran are recorded. Absent days are not filled in as failures
    because they are not evidence of anything: a paused binding should make a name take longer to
    qualify, never disqualify it retroactively.

    A window written by the run-indexed predecessor is a plain list with no days attached, so it
    is dropped rather than guessed at. Both gates read an empty window as "not watched long
    enough", which holds the book still and makes new entries wait out the settling period -- the
    safe direction on both sides, and it refills itself within ``eligibility_window`` sessions.
    """
    window = max(config.eligibility_window, 1)
    today = market_day(as_of)
    history = state.get(key)
    history = history if isinstance(history, dict) else {}
    updated: dict[str, dict[str, int]] = {}
    for symbol, row in rows.items():
        past = history.get(symbol)
        past = {str(day): int(value) for day, value in past.items()} if isinstance(past, dict) else {}
        past[today] = measure(row)
        updated[symbol] = {day: past[day] for day in sorted(past)[-window:]}
    state[key] = updated
    return updated


def resolve_positions(
    held: set[str],
    ranked: list[dict[str, Any]],
    config: RallyRotationConfig,
    as_of: str = "",
) -> set[str]:
    """Which ETFs the book holds: keep what still ranks, fill free slots, then displace.

    Two kinds of hesitance, doing different jobs.

    *Rank* is the cheap one. A name enters only from inside ``entry_rank_max`` but is kept while
    it stays inside ``exit_rank_max``, so an incumbent that slips a place or two is not sold for
    it. Ejecting on the entry rank instead meant a holding that drifted to 7th of 14 on scores
    separated by 0.15 was sold and the freed slot refilled from the top with no margin at all.

    *Score* is the one that costs something to cross. Displacing an incumbent -- as opposed to
    filling an empty slot -- needs ``min_score_delta_to_replace`` of daylight, so a challenger
    that is ahead by a rounding error waits.
    """
    slots = max(config.max_positions, 0)
    delta = max(config.min_score_delta_to_replace, 0.0)
    score = {str(row["symbol"]): float(row.get("base_score", 0.0)) for row in ranked}
    rank = {str(row["symbol"]): int(row.get("rank") or 0) for row in ranked}

    selection = {
        symbol for symbol in held
        if rank.get(symbol) and rank[symbol] <= max(config.exit_rank_max, 0)
    }
    contenders = [
        str(row["symbol"]) for row in ranked
        if str(row["symbol"]) not in selection
        and 0 < int(row.get("rank") or 0) <= max(config.entry_rank_max, 0)
    ]

    for symbol in contenders:
        if len(selection) >= slots:
            break
        selection.add(symbol)

    for symbol in contenders:
        if symbol in selection or not selection:
            continue
        weakest = min(selection, key=lambda name: score.get(name, 0.0))
        if score.get(symbol, 0.0) <= score.get(weakest, 0.0) + delta:
            continue
        logger.info(
            "[%s] Rally Rotation replacing %s (%.2f) with %s (%.2f)",
            as_of[:10] if as_of else "--------", weakest, score.get(weakest, 0.0), symbol, score.get(symbol, 0.0),
        )
        selection.discard(weakest)
        selection.add(symbol)

    if len(selection) > slots:
        selection = set(sorted(selection, key=lambda name: -score.get(name, 0.0))[:slots])
    return selection
