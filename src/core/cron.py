"""Cron expressions: the one statement of when an algorithm runs.

Five standard fields -- ``minute hour day-of-month month day-of-week`` -- evaluated in
:data:`MARKET_TZ`, because everything else that decides *when* in this codebase is market-local
and a schedule that drifted an hour against the trading session twice a year would be a
different schedule in summer than in winter.

This replaces a pair of overlapping controls. A binding carried a ``frequency``
(``15m``/``1hr``/``1d``/...) and the algorithm class carried a ``Schedule`` of start time,
end time and refresh interval -- but only the second ever timed anything. ``frequency_minutes``
was consumed in exactly one way, ``is None``, to sort scheduled bindings from agent-driven
ones, so choosing ``15m`` over ``2hr`` for Bursty DCA changed nothing at all: it fired at 11:00
and 15:00 either way. The dropdown was a control that could not control its own subject.

Hand-rolled rather than taken from ``croniter``, which is not a dependency here. The subset
below is the whole of what a trading schedule needs, and it is small enough to read.

Two deliberate departures from crontab(5), both toward predictability:

* **Day-of-month and day-of-week are ANDed**, never ORed. Standard cron ORs them when both are
  restricted, so ``0 11 1 * 1-5`` means "the 1st, *or* any weekday" -- which fires every
  weekday and is essentially never what someone writing it intended.
* **No implicit catch-up.** A fire time is only honoured within :data:`GRACE_MINUTES` of it, so
  a process that was down all morning resumes at the next scheduled time rather than firing
  every slot it missed the moment it comes back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from .interfaces import MARKET_TZ

#: How late a fire time may still be honoured. The runtime polls every 300s, so this is several
#: polls of slack for a slow run or a restart, without reaching back far enough to replay a
#: schedule that was missed while the process was down.
GRACE_MINUTES = 15

#: ``name, low, high`` per field, in expression order.
_FIELDS: tuple[tuple[str, int, int], ...] = (
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day of month", 1, 31),
    ("month", 1, 12),
    ("day of week", 0, 6),
)

_WEEKDAY_NAMES = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
_STEP = re.compile(r"^(?P<range>[^/]+)(?:/(?P<step>\d+))?$")


class CronError(ValueError):
    """An expression that cannot be parsed, with a message meant for the person who typed it."""


@dataclass(frozen=True)
class CronSpec:
    """A parsed expression: the set of allowed values for each of the five fields."""

    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    #: The text this was parsed from, so a round trip through config is lossless.
    expression: str = ""

    def matches(self, moment: datetime) -> bool:
        """Whether ``moment`` -- to the minute, in market time -- is a fire time."""
        local = moment.astimezone(MARKET_TZ) if moment.tzinfo else moment.replace(tzinfo=MARKET_TZ)
        return (
            local.minute in self.minutes
            and local.hour in self.hours
            and self.matches_date(local)
        )

    def matches_date(self, moment: datetime) -> bool:
        """Whether the *date* can fire at all, ignoring the time of day.

        Split out for the backtest, which steps one bar per day and needs to know whether a
        date runs without inventing a clock time for it.
        """
        # ``isoweekday`` is Mon=1..Sun=7; cron counts Sun=0..Sat=6.
        return (
            moment.day in self.days_of_month
            and moment.month in self.months
            and (moment.isoweekday() % 7) in self.days_of_week
        )


def parse_cron(expression: str) -> CronSpec:
    """Parse a five-field expression, or raise :class:`CronError` naming what is wrong."""
    text = " ".join(str(expression or "").split())
    if not text:
        raise CronError("Empty schedule. Use five fields, for example '0 11 * * 1-5'.")

    parts = text.split(" ")
    if len(parts) != 5:
        raise CronError(
            f"Expected 5 fields (minute hour day-of-month month day-of-week), got {len(parts)}: "
            f"'{text}'. For example '0 11 * * 1-5' is 11:00 on weekdays."
        )

    values = [_parse_field(part, *spec) for part, spec in zip(parts, _FIELDS)]
    return CronSpec(
        minutes=values[0],
        hours=values[1],
        days_of_month=values[2],
        months=values[3],
        days_of_week=values[4],
        expression=text,
    )


def _parse_field(field: str, name: str, low: int, high: int) -> frozenset[int]:
    allowed: set[int] = set()
    for item in field.split(","):
        item = item.strip()
        if not item:
            raise CronError(f"Empty entry in the {name} field: '{field}'.")
        match = _STEP.match(item)
        if not match:
            raise CronError(f"Cannot read '{item}' in the {name} field.")

        step_text = match.group("step")
        step = int(step_text) if step_text else 1
        if step < 1:
            raise CronError(f"Step must be 1 or more in the {name} field: '{item}'.")

        allowed.update(_expand_range(match.group("range").strip(), item, name, low, high)[::step])

    return frozenset(allowed)


def _expand_range(text: str, item: str, name: str, low: int, high: int) -> list[int]:
    if text == "*":
        return list(range(low, high + 1))

    bounds = text.split("-")
    if len(bounds) > 2:
        raise CronError(f"Cannot read the range '{text}' in the {name} field.")
    try:
        start = _bounded(int(bounds[0]), name, low, high)
        end = _bounded(int(bounds[1]), name, low, high) if len(bounds) == 2 else start
    except ValueError as error:
        if isinstance(error, CronError):
            raise
        raise CronError(f"Cannot read '{item}' in the {name} field; expected a number.") from None

    if start > end:
        raise CronError(f"Range runs backwards in the {name} field: '{text}'.")
    return list(range(start, end + 1))


def _bounded(value: int, name: str, low: int, high: int) -> int:
    # Cron lets 7 mean Sunday as well as 0, and people write it both ways.
    if name == "day of week" and value == 7:
        return 0
    if not low <= value <= high:
        raise CronError(f"{value} is out of range for the {name} field ({low}-{high}).")
    return value


def cron_fire_key(spec: CronSpec, now: datetime, grace_minutes: int = GRACE_MINUTES) -> str | None:
    """The fire time this run belongs to, or ``None`` if ``now`` is not near one.

    Returns the *scheduled* minute rather than the current one, so the value is stable across
    every poll inside a fire's grace window -- which is what lets the runtime use it to run a
    schedule exactly once. Scanning backwards rather than forwards is deliberate: the runtime
    asks "should I be running right now", not "when is the next one".
    """
    local = now.astimezone(MARKET_TZ) if now.tzinfo else now.replace(tzinfo=MARKET_TZ)
    local = local.replace(second=0, microsecond=0)
    for offset in range(max(int(grace_minutes), 0) + 1):
        candidate = local - timedelta(minutes=offset)
        if spec.matches(candidate):
            return candidate.isoformat(timespec="minutes")
    return None


def describe_cron(spec: CronSpec) -> str:
    """A one-line reading of the expression, for the dashboard.

    Only the common shapes get prose; anything more elaborate falls back to the expression
    itself, which is more honest than a summary that quietly drops a field.
    """
    days = _describe_days(spec)
    if days is None:
        # A day pattern prose cannot carry. Returning the raw expression beats appending a time
        # to a half-description and printing "0 0 1 * * at 00:00".
        return spec.expression
    hours = sorted(spec.hours)
    minutes = sorted(spec.minutes)
    every_hour = len(hours) == 24
    every_minute = len(minutes) == 60

    if every_minute or (len(minutes) > 1 and not _is_even_step(minutes)):
        return spec.expression
    if len(minutes) > 1:
        step = minutes[1] - minutes[0]
        span = "" if every_hour else f" between {hours[0]:02d}:00 and {hours[-1]:02d}:59"
        return f"{days}, every {step}m{span}"
    if every_hour:
        return f"{days}, hourly at :{minutes[0]:02d}"
    times = ", ".join(f"{hour:02d}:{minutes[0]:02d}" for hour in hours[:4])
    if len(hours) > 4:
        times += f" and {len(hours) - 4} more"
    return f"{days} at {times}"


def _is_even_step(minutes: list[int]) -> bool:
    step = minutes[1] - minutes[0]
    return all(later - earlier == step for earlier, later in zip(minutes, minutes[1:]))


def _describe_days(spec: CronSpec) -> str | None:
    """Prose for the day pattern, or ``None`` when it has none worth trusting."""
    if len(spec.days_of_month) < 31 or len(spec.months) < 12:
        return None
    weekdays = spec.days_of_week
    if weekdays == frozenset(range(7)):
        return "Every day"
    if weekdays == frozenset({1, 2, 3, 4, 5}):
        return "Weekdays"
    return ", ".join(f"{_WEEKDAY_NAMES[day]}s" for day in sorted(weekdays))
