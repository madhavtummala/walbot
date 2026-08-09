from __future__ import annotations

from datetime import datetime

# Field bounds for the five standard cron positions.
MINUTE_RANGE = (0, 59)
HOUR_RANGE = (0, 23)
DAY_OF_MONTH_RANGE = (1, 31)
MONTH_RANGE = (1, 12)
DAY_OF_WEEK_RANGE = (0, 7)


def cron_field_matches(field: str, value: int, *, min_value: int, max_value: int) -> bool:
    """Return whether ``value`` satisfies a single cron field.

    Supports ``*``, comma lists, ``a-b`` ranges, and ``*/step`` / ``a-b/step`` increments.
    """
    field = str(field or "*").strip()
    if field == "*":
        return True
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, raw_step = part.split("/", 1)
            try:
                step = max(int(raw_step), 1)
            except ValueError:
                return False
        if part == "*":
            start, end = min_value, max_value
        elif "-" in part:
            raw_start, raw_end = part.split("-", 1)
            try:
                start, end = int(raw_start), int(raw_end)
            except ValueError:
                return False
        else:
            try:
                start = end = int(part)
            except ValueError:
                return False
        if start <= value <= end and (value - start) % step == 0:
            return True
    return False


def cron_matches(pattern: str, now: datetime) -> bool:
    """Return whether a 5-field cron ``pattern`` fires at ``now``."""
    fields = str(pattern or "").split()
    if len(fields) != 5:
        return False
    minute, hour, day_of_month, month, day_of_week = fields
    return (
        cron_field_matches(minute, now.minute, min_value=0, max_value=59)
        and cron_field_matches(hour, now.hour, min_value=0, max_value=23)
        and cron_field_matches(day_of_month, now.day, min_value=1, max_value=31)
        and cron_field_matches(month, now.month, min_value=1, max_value=12)
        and cron_day_of_week_matches(day_of_week, now.weekday())
    )


def cron_day_of_week_matches(field: str, python_weekday: int) -> bool:
    """Match a cron day-of-week field against a ``datetime.weekday()`` value (Mon=0..Sun=6).

    Cron uses Sun=0..Sat=6 (with 7 also meaning Sunday), so Sunday is normalized to match either 0 or 7.
    """
    cron_dow = 0 if python_weekday == 6 else python_weekday + 1
    return cron_field_matches(field, cron_dow, min_value=0, max_value=7) or (
        cron_dow == 0 and cron_field_matches(field, 7, min_value=0, max_value=7)
    )
