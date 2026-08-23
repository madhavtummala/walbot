"""Scheduling: which minute a binding fires on, and how it says it wants no clock at all.

Timing used to be split between a binding ``frequency`` and the algorithm class's ``Schedule``,
and only the second one ever timed anything -- ``frequency_minutes`` was read exclusively as
``is None``. These now exercise one mechanism: the binding's cron expression, in market time.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.core import bot_runtime as bot_runtime
from src.core.cron import GRACE_MINUTES, CronError, cron_fire_key, describe_cron, parse_cron

MARKET_TZ = bot_runtime.MARKET_TZ

# 2026-08-12 is a Wednesday, 2026-08-16 a Sunday, 2026-05-18 a Monday.
WEDNESDAY = (2026, 8, 12)
SUNDAY = (2026, 8, 16)


class _FrozenClock:
    """Stands in for the datetime module so `datetime.now(tz)` is deterministic."""

    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self, tz=None):
        return self._moment.astimezone(tz) if tz else self._moment


def _binding_controls(cron: str = "*/30 9-15 * * 1-5", strategy: str = "rally_rotation") -> dict:
    return {
        "bindings": [{"id": "b1", "strategy": strategy, "account_id": "paper",
                      "enabled": True, "cron": cron}],
        "trading_account_id": "paper",
    }


def _at(monkeypatch, date: tuple[int, int, int], hour: int, minute: int) -> None:
    moment = datetime(*date, hour, minute, tzinfo=MARKET_TZ)
    monkeypatch.setattr(bot_runtime, "datetime", _FrozenClock(moment))


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def test_the_five_fields_are_read_in_order() -> None:
    spec = parse_cron("5 11 3 7 2")

    assert spec.minutes == frozenset({5})
    assert spec.hours == frozenset({11})
    assert spec.days_of_month == frozenset({3})
    assert spec.months == frozenset({7})
    assert spec.days_of_week == frozenset({2})


def test_lists_ranges_and_steps_all_expand() -> None:
    assert parse_cron("0,30 * * * *").minutes == frozenset({0, 30})
    assert parse_cron("0 9-11 * * *").hours == frozenset({9, 10, 11})
    assert parse_cron("*/15 * * * *").minutes == frozenset({0, 15, 30, 45})
    assert parse_cron("0 9-15/3 * * *").hours == frozenset({9, 12, 15})


def test_sunday_may_be_written_as_seven() -> None:
    """crontab(5) accepts both, and people reach for whichever they learned first."""
    assert parse_cron("0 11 * * 7").days_of_week == parse_cron("0 11 * * 0").days_of_week


@pytest.mark.parametrize(
    "expression",
    ["", "0 11 * *", "0 11 * * * *", "0 25 * * 1-5", "abc 11 * * 1", "0 11 * * 8", "5-1 11 * * 1"],
)
def test_an_unusable_expression_is_refused_rather_than_guessed_at(expression: str) -> None:
    with pytest.raises(CronError):
        parse_cron(expression)


def test_the_message_names_the_field_that_is_wrong() -> None:
    """The string is typed by hand into a text box, so the error is the whole of the UI."""
    with pytest.raises(CronError, match="hour"):
        parse_cron("0 25 * * 1-5")


# --------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------


def test_day_of_month_and_day_of_week_are_anded_not_ored() -> None:
    """The one place this departs from crontab(5), deliberately.

    Standard cron ORs them when both are restricted, so "the 1st, on weekdays" fires every
    weekday of the month -- which is never what the person writing it meant.
    """
    spec = parse_cron("0 11 1 * 1-5")

    assert spec.matches(datetime(2026, 6, 1, 11, 0, tzinfo=MARKET_TZ))       # 1st, a Monday
    assert not spec.matches(datetime(2026, 6, 2, 11, 0, tzinfo=MARKET_TZ))   # a weekday, not the 1st
    assert not spec.matches(datetime(2026, 3, 1, 11, 0, tzinfo=MARKET_TZ))   # the 1st, a Sunday


def test_matches_date_ignores_the_time_of_day() -> None:
    """What the backtest gates on: it steps one bar per day and has no clock time to offer."""
    spec = parse_cron("0 11 * * 1-5")

    assert spec.matches_date(datetime(*WEDNESDAY))
    assert not spec.matches_date(datetime(*SUNDAY))


def test_expressions_are_read_in_market_time() -> None:
    """A schedule stated in UTC would slide an hour against the session twice a year."""
    spec = parse_cron("0 11 * * 1-5")
    from datetime import timezone

    # 15:00 UTC is 11:00 EDT in August.
    assert spec.matches(datetime(2026, 8, 12, 15, 0, tzinfo=timezone.utc))
    assert not spec.matches(datetime(2026, 8, 12, 11, 0, tzinfo=timezone.utc))


# --------------------------------------------------------------------------------------
# Fire keys -- the value the runtime dedupes a run on
# --------------------------------------------------------------------------------------


def test_a_fire_key_is_the_scheduled_minute_not_the_current_one() -> None:
    """Every poll inside the grace window must agree, or the loop runs the schedule twice."""
    spec = parse_cron("0 11 * * 1-5")
    on_the_minute = cron_fire_key(spec, datetime(*WEDNESDAY, 11, 0, tzinfo=MARKET_TZ))
    seven_minutes_late = cron_fire_key(spec, datetime(*WEDNESDAY, 11, 7, tzinfo=MARKET_TZ))

    assert on_the_minute == seven_minutes_late
    assert on_the_minute is not None and on_the_minute.startswith("2026-08-12T11:00")


def test_a_missed_schedule_is_not_replayed_once_the_grace_window_passes() -> None:
    """A process down all morning resumes at the next slot rather than firing every one it
    missed the moment it comes back."""
    spec = parse_cron("0 11 * * 1-5")

    assert cron_fire_key(spec, datetime(*WEDNESDAY, 11, GRACE_MINUTES, tzinfo=MARKET_TZ)) is not None
    assert cron_fire_key(spec, datetime(*WEDNESDAY, 11, GRACE_MINUTES + 1, tzinfo=MARKET_TZ)) is None


def test_no_fire_key_before_the_scheduled_minute() -> None:
    spec = parse_cron("0 11 * * 1-5")

    assert cron_fire_key(spec, datetime(*WEDNESDAY, 10, 59, tzinfo=MARKET_TZ)) is None


def test_separate_times_produce_separate_keys() -> None:
    spec = parse_cron("0 11,15 * * 1-5")
    morning = cron_fire_key(spec, datetime(*WEDNESDAY, 11, 2, tzinfo=MARKET_TZ))
    afternoon = cron_fire_key(spec, datetime(*WEDNESDAY, 15, 2, tzinfo=MARKET_TZ))

    assert morning is not None and afternoon is not None and morning != afternoon


# --------------------------------------------------------------------------------------
# The binding wiring
# --------------------------------------------------------------------------------------


def test_a_binding_fires_on_its_own_cron(monkeypatch) -> None:
    monkeypatch.setattr(bot_runtime, "load_controls", lambda: _binding_controls("0 11 * * 1-5"))
    _at(monkeypatch, WEDNESDAY, 11, 0)

    key = bot_runtime._binding_run_key("b1")()

    assert key is not None and key.startswith("b1:")


def test_a_binding_does_not_fire_off_its_cron(monkeypatch) -> None:
    monkeypatch.setattr(bot_runtime, "load_controls", lambda: _binding_controls("0 11 * * 1-5"))
    _at(monkeypatch, SUNDAY, 11, 0)

    assert bot_runtime._binding_run_key("b1")() is None


def test_the_bindings_cron_overrides_the_algorithm_default(monkeypatch) -> None:
    """The whole point of moving the schedule onto the binding: a deployment can choose.

    Bursty DCA's class default is 11:00, and this binding says 14:00 -- so 14:00 is when it
    runs, and 11:00 is not.
    """
    monkeypatch.setattr(bot_runtime, "load_controls", lambda: _binding_controls("0 14 * * 1-5", "bursty_dca"))

    _at(monkeypatch, WEDNESDAY, 11, 0)
    assert bot_runtime._binding_run_key("b1")() is None
    _at(monkeypatch, WEDNESDAY, 14, 0)
    assert bot_runtime._binding_run_key("b1")() is not None


def test_an_empty_cron_is_never_scheduled(monkeypatch) -> None:
    """How a binding says an agent drives it, replacing the ``mcp`` frequency."""
    monkeypatch.setattr(bot_runtime, "load_controls", lambda: _binding_controls(""))
    _at(monkeypatch, WEDNESDAY, 12, 0)

    assert bot_runtime._binding_run_key("b1")() is None
    assert bot_runtime._binding_enabled("b1")(_binding_controls("")) is False


def test_a_hand_edited_unusable_cron_refuses_to_fire(monkeypatch) -> None:
    """Saving is validated, so this is a config edited outside the dashboard. Not firing is
    the safe reading -- a schedule nobody can parse must not be guessed at."""
    controls = {
        "bindings": [{"id": "b1", "strategy": "rally_rotation", "account_id": "paper",
                      "enabled": True, "cron": "0 99 * * *"}],
        "trading_account_id": "paper",
    }
    monkeypatch.setattr(bot_runtime, "load_controls", lambda: controls)
    monkeypatch.setattr(bot_runtime, "normalize_cron", lambda value, strategy=None: str(value))
    _at(monkeypatch, WEDNESDAY, 12, 0)

    assert bot_runtime._binding_run_key("b1")() is None


# --------------------------------------------------------------------------------------
# Per-algorithm defaults
# --------------------------------------------------------------------------------------


def test_each_algorithm_states_a_runnable_default() -> None:
    """A default nobody can parse would be worse than none: it reaches bindings silently."""
    from src.algorithms.registry import get_algorithm_class

    for algorithm_id in ("bursty_dca", "rally_rotation", "options_flip"):
        assert parse_cron(get_algorithm_class(algorithm_id).cron).expression, algorithm_id


def test_dca_defaults_to_one_weekday_run() -> None:
    from src.algorithms.bursty_dca.algorithm import BurstyDCAAlgorithm

    spec = parse_cron(BurstyDCAAlgorithm.cron)

    assert spec.hours == frozenset({11})
    assert spec.days_of_week == frozenset({1, 2, 3, 4, 5})


def test_runtime_has_no_dca_loop_of_its_own() -> None:
    """Two schedulers driving one accrual state was the hazard this collapse removes."""
    assert not hasattr(bot_runtime.bot_runtime, "dca")
    # "algorithm" mirrors the first binding for callers that predate the binding list.
    assert set(bot_runtime.bot_runtime.snapshot()) == {"bindings", "algorithm"}


# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------


def test_describe_cron_reads_the_common_shapes_back_in_words() -> None:
    assert describe_cron(parse_cron("0 11 * * 1-5")) == "Weekdays at 11:00"
    assert describe_cron(parse_cron("0 11,15 * * 1-5")) == "Weekdays at 11:00, 15:00"
    assert describe_cron(parse_cron("*/15 9-15 * * 1-5")).startswith("Weekdays, every 15m")
    assert describe_cron(parse_cron("30 9 * * 1")) == "Mondays at 09:30"


def test_describe_cron_falls_back_to_the_expression_it_cannot_summarise() -> None:
    """Better than prose that quietly drops a field it did not know how to say."""
    assert describe_cron(parse_cron("0 0 1 * *")) == "0 0 1 * *"
