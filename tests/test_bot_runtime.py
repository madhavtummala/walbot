from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from src.core import bot_runtime as bot_runtime
from src.core.interfaces import DAILY_AT_OPEN, Schedule, describe_schedule

HOURLY = Schedule()
HALF_HOURLY = Schedule(refresh_minutes=30, jitter_minutes=0)


def test_regular_market_hours_are_central_weekdays() -> None:
    assert bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 8, 30), HOURLY)
    assert bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 14, 59), HOURLY)
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 8, 29), HOURLY)
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 15, 0), HOURLY)
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 23, 10, 0), HOURLY)


def test_regular_market_hours_honor_the_schedule_window() -> None:
    window = Schedule(refresh_minutes=30, jitter_minutes=0, start_time="09:30", end_time="14:00")
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 9, 29), window)
    assert bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 9, 30), window)
    assert bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 13, 59), window)
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 14, 0), window)


def test_schedule_weekdays_gate_the_run() -> None:
    """The part ``refresh_minutes`` cannot express: a cadence coarser than daily."""
    mondays = replace(DAILY_AT_OPEN, jitter_minutes=0, weekdays=(0,))
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 18, 9, 0), mondays)  # Monday
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 19, 9, 0), mondays) is None  # Tuesday
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 0), mondays) is None  # Friday


def test_daily_at_open_yields_one_bucket_per_session() -> None:
    """A refresh at or above the session length collapses to a single run per day."""
    daily = replace(DAILY_AT_OPEN, jitter_minutes=0)
    morning = bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 0), daily)
    afternoon = bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 14, 30), daily)
    assert morning == afternoon == "algorithm:2026-05-22T08:30-05:00"


def test_algorithm_bucket_key_uses_refresh_window() -> None:
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 8, 31), HALF_HOURLY) == (
        "algorithm:2026-05-22T08:30-05:00"
    )
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 8, 59), HALF_HOURLY) == (
        "algorithm:2026-05-22T08:30-05:00"
    )
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 0), HALF_HOURLY) == (
        "algorithm:2026-05-22T09:00-05:00"
    )
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 23, 9, 0), HALF_HOURLY) is None


def test_algorithm_bucket_key_anchors_to_market_open_for_hourly_runs() -> None:
    hourly = Schedule(jitter_minutes=0)
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 0), hourly) == (
        "algorithm:2026-05-22T08:30-05:00"
    )
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 30), hourly) == (
        "algorithm:2026-05-22T09:30-05:00"
    )


def test_algorithm_bucket_key_waits_for_jitter_offset(monkeypatch) -> None:
    monkeypatch.setattr(bot_runtime, "_algorithm_jitter_offset_minutes", lambda *_args: 4)

    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 8, 33), HOURLY) is None
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 8, 34), HOURLY) == (
        "algorithm:2026-05-22T08:30-05:00"
    )


def test_algorithm_bucket_key_anchors_to_the_scheduled_start() -> None:
    window = Schedule(refresh_minutes=30, jitter_minutes=0, start_time="09:30", end_time="14:00")
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 45), window) == (
        "algorithm:2026-05-22T09:30-05:00"
    )
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 29), window) is None


# --------------------------------------------------------------------------------------
# Cadence comes from the selected algorithm's class, not from config.
# --------------------------------------------------------------------------------------


def test_active_schedule_reads_the_selected_algorithms_class() -> None:
    from src.algorithms.dca.bursty import BurstyDCAAlgorithm

    assert bot_runtime._active_schedule({"active_strategy": "dca"}).weekdays == (0, 1, 2, 3, 4)
    assert bot_runtime._active_schedule({"active_strategy": "bursty_dca"}) == BurstyDCAAlgorithm.schedule
    assert bot_runtime._active_schedule({"active_strategy": "rally_rotation"}) == DAILY_AT_OPEN


def test_active_schedule_falls_back_for_an_unknown_strategy() -> None:
    """Only affects how often the idle loop re-reads controls, so a default is safe here."""
    assert bot_runtime._active_schedule({"active_strategy": "retired_thing"}) == Schedule()


def test_active_schedule_treats_a_saved_none_as_dca() -> None:
    assert bot_runtime._active_schedule({"active_strategy": "none"}).weekdays == (0, 1, 2, 3, 4)


def test_dca_is_rarer_than_bursty_dca() -> None:
    """Same budget either way -- accrual is wall-clock -- so this only changes trade size."""
    from src.algorithms.dca.bot import DCAAlgorithm
    from src.algorithms.dca.bursty import BurstyDCAAlgorithm

    assert DCAAlgorithm.schedule.weekdays == (0,)
    assert BurstyDCAAlgorithm.schedule.weekdays == (0, 1, 2, 3, 4)


def test_runtime_has_no_dca_loop_of_its_own() -> None:
    """Two schedulers driving one accrual state was the hazard this collapse removes."""
    assert not hasattr(bot_runtime.bot_runtime, "dca")
    # "algorithm" mirrors the first binding for callers that predate the binding list.
    assert set(bot_runtime.bot_runtime.snapshot()) == {"bindings", "algorithm"}


def test_describe_schedule_renders_the_cadence_for_the_dashboard() -> None:
    assert describe_schedule(replace(DAILY_AT_OPEN, weekdays=(0,))) == "Mondays at 08:30"
    assert describe_schedule(DAILY_AT_OPEN) == "Weekdays at 08:30"
    assert describe_schedule(Schedule()) == "Weekdays, every 60m from 08:30 to 15:00"


def _binding_controls(frequency: str = "1hr", strategy: str = "rally_rotation") -> dict:
    return {
        "bindings": [{"id": "b1", "strategy": strategy, "account_id": "paper",
                      "enabled": True, "frequency": frequency}],
        "trading_account_id": "paper",
    }


def test_a_binding_never_fires_outside_the_algorithms_session(monkeypatch) -> None:
    """The binding chooses the cadence; the algorithm still owns the session window.

    Without this an hourly binding wakes at 03:00 on a Sunday, fetches a full universe of
    market data, and only then discovers the market is closed.
    """
    monkeypatch.setattr(bot_runtime, "load_controls", lambda: _binding_controls())
    sunday_3am = datetime(2026, 8, 16, 3, 0, tzinfo=bot_runtime.MARKET_TZ)
    monkeypatch.setattr(bot_runtime, "datetime", _FrozenClock(sunday_3am))

    assert bot_runtime._binding_run_key("b1")() is None


def test_a_binding_fires_inside_the_session(monkeypatch) -> None:
    monkeypatch.setattr(bot_runtime, "load_controls", lambda: _binding_controls())
    wednesday_noon = datetime(2026, 8, 12, 12, 0, tzinfo=bot_runtime.MARKET_TZ)
    monkeypatch.setattr(bot_runtime, "datetime", _FrozenClock(wednesday_noon))

    key = bot_runtime._binding_run_key("b1")()

    assert key is not None and key.startswith("b1:")


def test_the_frequency_sets_the_bucket_size(monkeypatch) -> None:
    """Two runs inside one bucket dedupe; the next bucket is a new key."""
    monkeypatch.setattr(bot_runtime, "load_controls", lambda: _binding_controls("15m"))
    first = datetime(2026, 8, 12, 12, 0, tzinfo=bot_runtime.MARKET_TZ)
    monkeypatch.setattr(bot_runtime, "datetime", _FrozenClock(first))
    key_at_noon = bot_runtime._binding_run_key("b1")()
    monkeypatch.setattr(bot_runtime, "datetime", _FrozenClock(first.replace(minute=7)))
    key_at_seven_past = bot_runtime._binding_run_key("b1")()
    monkeypatch.setattr(bot_runtime, "datetime", _FrozenClock(first.replace(minute=15)))
    key_at_quarter_past = bot_runtime._binding_run_key("b1")()

    assert key_at_noon == key_at_seven_past
    assert key_at_quarter_past != key_at_noon


def test_an_mcp_binding_is_never_scheduled(monkeypatch) -> None:
    monkeypatch.setattr(bot_runtime, "load_controls", lambda: _binding_controls("mcp"))
    monkeypatch.setattr(bot_runtime, "datetime", _FrozenClock(datetime(2026, 8, 12, 12, 0, tzinfo=bot_runtime.MARKET_TZ)))

    assert bot_runtime._binding_run_key("b1")() is None
    assert bot_runtime._binding_enabled("b1")(_binding_controls("mcp")) is False


class _FrozenClock:
    """Stands in for the datetime module so `datetime.now(tz)` is deterministic."""

    def __init__(self, moment: datetime) -> None:
        self._moment = moment

    def now(self, tz=None):
        return self._moment.astimezone(tz) if tz else self._moment
