from __future__ import annotations

from datetime import datetime

from src import bot_runtime


def test_cron_matches_multiple_hours_in_same_day() -> None:
    assert bot_runtime._cron_matches("0 9-11 * * 1-5", datetime(2026, 5, 21, 9, 0))
    assert bot_runtime._cron_matches("0 9-11 * * 1-5", datetime(2026, 5, 21, 10, 0))
    assert not bot_runtime._cron_matches("0 9-11 * * 1-5", datetime(2026, 5, 21, 10, 30))


def test_cron_matches_steps_and_lists() -> None:
    assert bot_runtime._cron_matches("*/15 9,10 * * 1-5", datetime(2026, 5, 21, 10, 45))
    assert not bot_runtime._cron_matches("*/15 9,10 * * 1-5", datetime(2026, 5, 21, 11, 45))


def test_regular_market_hours_are_central_weekdays() -> None:
    assert bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 8, 30))
    assert bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 14, 59))
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 8, 29))
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 15, 0))
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 23, 10, 0))


def test_algorithm_bucket_key_uses_refresh_window() -> None:
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 8, 31), 30) == (
        "algorithm:2026-05-22T08:30-05:00"
    )
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 8, 59), 30) == (
        "algorithm:2026-05-22T08:30-05:00"
    )
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 0), 30) == (
        "algorithm:2026-05-22T09:00-05:00"
    )
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 23, 9, 0), 30) is None


def test_algorithm_bucket_key_anchors_to_market_open_for_hourly_runs() -> None:
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 0), 60) == (
        "algorithm:2026-05-22T08:30-05:00"
    )
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 30), 60) == (
        "algorithm:2026-05-22T09:30-05:00"
    )


def test_algorithm_bucket_key_waits_for_jitter_offset(monkeypatch) -> None:
    monkeypatch.setattr(bot_runtime, "_algorithm_jitter_offset_minutes", lambda *_args: 4)

    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 8, 33), 60, 5) is None
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 8, 34), 60, 5) == (
        "algorithm:2026-05-22T08:30-05:00"
    )


def test_options_enabled_requires_strategy_and_kill_switch_off(monkeypatch) -> None:
    class Config:
        kill_switch = False

    monkeypatch.setattr(bot_runtime, "get_config", lambda: Config())

    assert bot_runtime._options_enabled({"options_trading_enabled": True, "options_strategy": "covered_call"})
    assert not bot_runtime._options_enabled({"options_trading_enabled": True, "options_strategy": "none"})

    Config.kill_switch = True

    assert not bot_runtime._options_enabled({"options_trading_enabled": True, "options_strategy": "covered_call"})
