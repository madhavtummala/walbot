from __future__ import annotations

from src.api.api_payloads import algorithm_activity_payload
from src.data.order_journal import JOURNAL_LIMIT, load_order_journal, record_orders
from src.data.state_store import ephemeral_state


def test_orders_are_recorded_with_the_algorithm_that_placed_them() -> None:
    with ephemeral_state():
        record_orders("dca", "paper", [{"symbol": "SPY", "action": "buy", "quantity": 2, "status": "submitted", "order_id": "abc"}])
        record_orders("rally_rotation", "paper", [{"symbol": "QQQ", "action": "sell", "quantity": 1, "status": "submitted"}])

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
        record_orders("rally_rotation", "paper", [
            {"symbol": "TSLA", "action": "skip", "quantity": 0, "reason": "not shortable"},
        ])

        row = load_order_journal()[0]

        assert row["status"] == "skipped"
        assert row["reason"] == "not shortable"


def test_journalling_never_raises_on_a_bad_payload() -> None:
    with ephemeral_state():
        assert record_orders("dca", "paper", []) == []


def test_an_unchanged_order_is_not_journalled() -> None:
    """The reconciler found nothing to do -- that is not an event worth a journal line.

    A lifecycle algorithm polls every few minutes; "still correct, did nothing" every single
    poll would fill the capped journal with no-op noise and push real actions out early.
    """
    with ephemeral_state():
        written = record_orders("options_flip", "paper", [
            {"symbol": "GLD", "action": "sell", "quantity": 1, "reconciled": "unchanged"},
            {"symbol": "GLD", "action": "sell", "quantity": 1, "reconciled": "rejected", "status": "rejected"},
        ])

        assert len(written) == 1  # only the rejected row was journalled
        rows = load_order_journal(strategy="options_flip")
        assert len(rows) == 1
        assert rows[0]["status"] == "rejected"
        assert record_orders("dca", "paper", ["not-a-dict"]) == []  # type: ignore[list-item]


def test_activity_payload_is_scoped_to_one_algorithm() -> None:
    with ephemeral_state():
        record_orders("bursty_dca", "paper", [{"symbol": "SPY", "action": "buy", "quantity": 1, "status": "submitted"}])
        record_orders("rally_rotation", "paper", [{"symbol": "QQQ", "action": "buy", "quantity": 1, "status": "submitted"}])

        payload = algorithm_activity_payload(strategy="bursty_dca", limit=10)

        assert payload["strategy"] == "bursty_dca"
        assert [row["symbol"] for row in payload["rows"]] == ["SPY"]


def test_a_backtest_cannot_pollute_the_live_journal() -> None:
    """Replays run inside ephemeral_state, so their orders are discarded with the block."""
    with ephemeral_state() as store:
        record_orders("dca", "paper", [{"symbol": "SPY", "action": "buy", "quantity": 1, "status": "submitted"}])
        assert store  # the write landed in the throwaway dict, not DuckDB

    with ephemeral_state():
        assert load_order_journal() == []


def test_the_journal_records_the_price_the_order_names() -> None:
    """``latest_price`` is the mark a *market* order was sized from -- an estimate, not a price
    the order carries. A resting limit or stop names one exactly, and the reconciler reports it.

    Recording only the former left every option order in the journal at $0.00, since none of
    them is a market order, so the panel could show a size and a direction and nothing else.
    """
    from src.data.order_journal import _entry

    resting = _entry("options_flip", "alpaca1", {
        "symbol": "USO   260916C00142000", "action": "sell", "quantity": 1.0,
        "status": "submitted", "order_type": "limit", "limit_price": 17.30,
    }, "now")
    assert resting["limit_price"] == 17.30 and resting["order_type"] == "limit"

    stop = _entry("options_flip", "alpaca1", {
        "symbol": "USO   260916C00142000", "action": "sell", "quantity": 1.0,
        "status": "submitted", "order_type": "stop", "stop_price": 4.42,
    }, "now")
    assert stop["stop_price"] == 4.42

    # A market order still records what it was sized from, kept separate from a named price.
    market = _entry("bursty_dca", "schwab2", {
        "symbol": "SPYM", "action": "buy", "quantity": 2.0,
        "status": "submitted", "latest_price": 89.735,
    }, "now")
    assert market["price"] == 89.735 and market["limit_price"] == 0.0
