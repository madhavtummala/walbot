from __future__ import annotations

import logging

from src.logging_utils import configure_logging, log_position_changes


def test_configure_logging_uses_console_only() -> None:
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    try:
        configure_logging()

        handlers = root.handlers

        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)
    finally:
        root.handlers = original_handlers


def test_log_position_changes_reports_submitted_orders(caplog) -> None:
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


def test_log_position_changes_reports_no_submitted_orders(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        log_position_changes([{"symbol": "AAA", "action": "hold", "quantity": 0}])

    assert "No position changes were submitted." in caplog.text
