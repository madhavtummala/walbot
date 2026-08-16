from __future__ import annotations

import logging

from src.common.logging_utils import (
    configure_logging,
    demote_uvicorn_access_logs_to_debug,
    log_position_changes,
)


def test_configure_logging_uses_console_only(monkeypatch) -> None:
    monkeypatch.delenv("TRADING_LOG_LEVEL", raising=False)
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    try:
        configure_logging()

        handlers = root.handlers

        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)
        assert root.level == logging.WARNING
        assert handlers[0].level == logging.WARNING
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_configure_logging_uses_env_level(monkeypatch) -> None:
    monkeypatch.setenv("TRADING_LOG_LEVEL", "WARN")
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    try:
        configure_logging()

        assert root.level == logging.WARNING
        assert root.handlers[0].level == logging.WARNING
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_uvicorn_access_logs_are_hidden_above_debug() -> None:
    logger = logging.getLogger("uvicorn.access")
    original_filters = logger.filters[:]
    original_level = logger.level
    try:
        logger.filters = []
        logger.setLevel(logging.INFO)
        demote_uvicorn_access_logs_to_debug()
        record = logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1, "GET /api/status", (), None
        )

        assert logger.filter(record) is False
    finally:
        logger.filters = original_filters
        logger.setLevel(original_level)


def test_uvicorn_access_logs_are_debug_when_debug_enabled() -> None:
    logger = logging.getLogger("uvicorn.access")
    original_filters = logger.filters[:]
    original_level = logger.level
    try:
        logger.filters = []
        logger.setLevel(logging.DEBUG)
        demote_uvicorn_access_logs_to_debug()
        record = logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1, "GET /api/status", (), None
        )

        assert bool(logger.filter(record)) is True

        assert record.levelno == logging.DEBUG
        assert record.levelname == "DEBUG"
    finally:
        logger.filters = original_filters
        logger.setLevel(original_level)


def test_log_position_changes_reports_submitted_orders(caplog, monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with caplog.at_level(logging.WARNING):
        log_position_changes(
            [
                {"symbol": "AAA", "action": "hold", "quantity": 0},
                {
                    "symbol": "BBB",
                    "action": "buy",
                    "quantity": 3,
                    "current_shares": 0,
                    "target_shares": 3,
                    "trade_dollars": 300.0,
                    "order_id": "order-1",
                },
            ]
        )

    assert "Position changes submitted: 1 order(s)" in caplog.text
    assert "BUY BBB qty=3 current=0 target=3 trade_dollars=300.00 order_id=order-1" in caplog.text


def test_log_position_changes_reports_no_submitted_orders(caplog, monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with caplog.at_level(logging.WARNING):
        log_position_changes([{"symbol": "AAA", "action": "hold", "quantity": 0}])

    assert "No position changes were submitted." in caplog.text
