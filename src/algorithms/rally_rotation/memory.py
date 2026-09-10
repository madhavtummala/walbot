"""What this algorithm remembers between runs, and the decisions that need it.

Everything here reads or writes ``AlgorithmContext.state``; nothing else in the package does.
Mutated in place and handed back on the plan -- persisted by ``execute``, only if orders went
out.

**Everything here is measured in market days, never in runs.** Counting runs makes every
interval a function of the binding's cron cadence rather than the config: the schedule controls
the *opportunity* to act, never the rate.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

from ...core.interfaces import MARKET_TZ
from .config import RallyRotationConfig

logger = logging.getLogger(__name__)


def market_day(as_of: datetime) -> str:
    """The trading day a run belongs to, as an ISO date in market-local time -- not UTC, so a
    late-session fire can't land on the next calendar day. A naive timestamp is read as market
    time; replays hand this bar stamps, which are market-local by construction.
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

    Weekdays stand in for sessions rather than a real exchange calendar, so the arithmetic stays
    a pure function of two dates and a replay throttles identically to the live path. A holiday
    inside the window counts as a session, so an interval can come due up to a day early -- a
    rounding error against a knob measured in weeks, and bounded to only ever shorten a wait.
    """
    start = date.fromisoformat(last_day)
    end = date.fromisoformat(market_day(as_of))
    return _weekdays_inclusive(start + timedelta(days=1), end)


def action_due(state: dict[str, Any], action: str, interval_days: int, as_of: datetime) -> bool:
    """Whether ``action`` may be taken now, given its own interval in trading days.

    One clock per *kind* of decision, not one for the whole algorithm. A cold clock (no record)
    reads as due -- acting once on the first run after an upgrade is the safe direction, rather
    than inventing a start date for a clock nobody started. Mutated by :func:`record_action`,
    not here, so asking and then deciding not to act doesn't restart the clock.
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

    Keyed by day, not appended per run, so a cron that fires several times a session doesn't
    count one morning as several days of evidence. Within a day the **last** run wins, as the
    best-informed look. Only days this algorithm actually ran are recorded -- an absent day is
    not evidence of failure, so a paused binding makes a name take longer to qualify rather than
    disqualifying it retroactively.
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

    Two kinds of hesitance. *Rank*: a name enters only from inside ``entry_rank_max`` but is
    kept while it stays inside the wider ``exit_rank_max``, so slipping a place or two doesn't
    sell it. *Score*: displacing an incumbent (not filling an empty slot) needs
    ``min_score_delta_to_replace`` of daylight, so a challenger ahead by a rounding error waits.
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
