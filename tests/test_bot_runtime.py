from __future__ import annotations

from datetime import datetime
from typing import Any

from src.core import bot_runtime as bot_runtime
from src.core import pipeline as pipeline
from src.core.cron import cron_matches


def test_cron_matches_multiple_hours_in_same_day() -> None:
    assert cron_matches("0 9-11 * * 1-5", datetime(2026, 5, 21, 9, 0))
    assert cron_matches("0 9-11 * * 1-5", datetime(2026, 5, 21, 10, 0))
    assert not cron_matches("0 9-11 * * 1-5", datetime(2026, 5, 21, 10, 30))


def test_cron_matches_steps_and_lists() -> None:
    assert cron_matches("*/15 9,10 * * 1-5", datetime(2026, 5, 21, 10, 45))
    assert not cron_matches("*/15 9,10 * * 1-5", datetime(2026, 5, 21, 11, 45))


def test_regular_market_hours_are_central_weekdays() -> None:
    assert bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 8, 30))
    assert bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 14, 59))
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 8, 29))
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 15, 0))
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 23, 10, 0))


def test_regular_market_hours_honor_configured_window() -> None:
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 9, 29), "09:30", "14:00")
    assert bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 9, 30), "09:30", "14:00")
    assert bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 13, 59), "09:30", "14:00")
    assert not bot_runtime._is_regular_market_hours(datetime(2026, 5, 22, 14, 0), "09:30", "14:00")


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


def test_algorithm_bucket_key_anchors_to_configured_start() -> None:
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 45), 30, 0, "09:30", "14:00") == (
        "algorithm:2026-05-22T09:30-05:00"
    )
    assert bot_runtime._algorithm_bucket_key(datetime(2026, 5, 22, 9, 29), 30, 0, "09:30", "14:00") is None


def test_run_dca_submits_shares_through_shared_order_path(monkeypatch) -> None:
    class Config:
        kill_switch = False
        require_trade_approval = False
        alpaca_data_feed = "iex"
        trade_approval_timeout_seconds = 300
        trade_approval_poll_seconds = 5

    class FakeBrokerage:
        def is_market_open(self) -> bool:
            return True

        def submit_order(self, request) -> dict[str, Any]:
            return {"order_id": f"dca-{request.symbol}"}

    captured: dict[str, Any] = {}

    monkeypatch.setattr(bot_runtime, "get_config", lambda **kw: Config())
    monkeypatch.setattr("src.api.api_payloads.universe_payload", lambda: {"rows": []})
    monkeypatch.setattr(bot_runtime, "load_dca_plan", lambda rows: {"enabled": True})
    monkeypatch.setattr(
        bot_runtime, "allocation_preview", lambda plan: [{"symbol": "AAA", "action": "buy", "notional": 300.0}]
    )
    monkeypatch.setattr(pipeline, "resolve_brokerage", lambda config: FakeBrokerage())
    monkeypatch.setattr(bot_runtime, "create_data_client", lambda config: object())
    monkeypatch.setattr(bot_runtime, "get_latest_price", lambda symbol, client, data_feed=None: 100.0)
    monkeypatch.setattr(bot_runtime, "log_position_changes", lambda results: captured.update({"results": results}))

    bot_runtime._run_dca(account_id=None)

    assert len(captured["results"]) == 1
    order = captured["results"][0]
    assert order["symbol"] == "AAA"
    assert order["quantity"] == 3  # floor(300 / 100)
    assert order["order_id"] == "dca-AAA"


def test_options_enabled_requires_strategy_and_kill_switch_off(monkeypatch) -> None:
    class Config:
        kill_switch = False

    monkeypatch.setattr(bot_runtime, "get_config", lambda: Config())

    assert bot_runtime._options_enabled({"options_trading_enabled": True, "options_strategy": "covered_call"})
    assert not bot_runtime._options_enabled({"options_trading_enabled": True, "options_strategy": "none"})

    Config.kill_switch = True

    assert not bot_runtime._options_enabled({"options_trading_enabled": True, "options_strategy": "covered_call"})
