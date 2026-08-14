from __future__ import annotations

from src.api.api_payloads import algorithm_activity_payload
from src.data.order_journal import JOURNAL_LIMIT, load_order_journal, record_orders
from src.data.state_store import ephemeral_state


def test_orders_are_recorded_with_the_algorithm_that_placed_them() -> None:
    with ephemeral_state():
        record_orders("dca", "paper", [{"symbol": "SPY", "action": "buy", "quantity": 2, "status": "submitted", "order_id": "abc"}])
        record_orders("fast_momentum", "paper", [{"symbol": "QQQ", "action": "sell", "quantity": 1, "status": "submitted"}])

        rows = load_order_journal(strategy="dca")

        assert [row["symbol"] for row in rows] == ["SPY"]
        assert rows[0]["side"] == "buy"
        assert rows[0]["account_id"] == "paper"
        assert rows[0]["submitted_at"]


def test_journal_is_newest_first_and_capped() -> None:
    with ephemeral_state():
        for index in range(JOURNAL_LIMIT + 20):
            record_orders("dca", "paper", [{"symbol": f"S{index}", "action": "buy", "quantity": 1, "status": "submitted"}])

        assert len(load_order_journal(strategy="dca", limit=JOURNAL_LIMIT * 2)) == JOURNAL_LIMIT
        assert load_order_journal(strategy="dca", limit=1)[0]["symbol"] == f"S{JOURNAL_LIMIT + 19}"


def test_a_skipped_order_still_records_why() -> None:
    """A short sale the broker refuses never reaches the brokerage and carries no status."""
    with ephemeral_state():
        record_orders("fast_momentum", "paper", [
            {"symbol": "TSLA", "action": "skip", "quantity": 0, "reason": "not shortable"},
        ])

        row = load_order_journal()[0]

        assert row["status"] == "skipped"
        assert row["reason"] == "not shortable"


def test_journalling_never_raises_on_a_bad_payload() -> None:
    with ephemeral_state():
        assert record_orders("dca", "paper", []) == []
        assert record_orders("dca", "paper", ["not-a-dict"]) == []  # type: ignore[list-item]


def test_activity_payload_is_scoped_to_one_algorithm() -> None:
    with ephemeral_state():
        record_orders("dca", "paper", [{"symbol": "SPY", "action": "buy", "quantity": 1, "status": "submitted"}])
        record_orders("fast_momentum", "paper", [{"symbol": "QQQ", "action": "buy", "quantity": 1, "status": "submitted"}])

        payload = algorithm_activity_payload(strategy="dca", limit=10)

        assert payload["strategy"] == "dca"
        assert [row["symbol"] for row in payload["rows"]] == ["SPY"]


def test_a_backtest_cannot_pollute_the_live_journal() -> None:
    """Replays run inside ephemeral_state, so their orders are discarded with the block."""
    with ephemeral_state() as store:
        record_orders("dca", "paper", [{"symbol": "SPY", "action": "buy", "quantity": 1, "status": "submitted"}])
        assert store  # the write landed in the throwaway dict, not DuckDB

    with ephemeral_state():
        assert load_order_journal() == []
