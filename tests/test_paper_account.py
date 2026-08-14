from __future__ import annotations

from src.api.api_payloads import account_activity_payload, accounts_payload, positions_payload
from src.brokerages.providers.paper import PaperBrokerage
from src.core.config import get_config
from src.core.interfaces import OrderRequest
from src.data.order_journal import record_orders
from src.data.state_store import ephemeral_state

LOCAL = "local_paper"


def _order(symbol: str, action: str, quantity: float, price: float) -> OrderRequest:
    return OrderRequest(symbol=symbol, action=action, quantity=quantity, extra={"latest_price": price})


def test_local_paper_is_a_configured_account_needing_no_credentials() -> None:
    row = next(row for row in accounts_payload()["rows"] if row["id"] == LOCAL)

    assert row["label"] == "Local Paper"
    assert row["broker"] == "paper"
    # Nothing to set in .env, so it is always ready to be pointed at.
    assert row["credentials_ready"] is True
    assert row["missing_env"] == []


def test_the_book_tracks_average_entry_so_it_can_report_pl() -> None:
    with ephemeral_state():
        brokerage = PaperBrokerage(get_config(account_id=LOCAL))
        brokerage.submit_order(_order("SPY", "buy", 10, 100.0))
        brokerage.submit_order(_order("SPY", "buy", 10, 120.0))

        row = next(row for row in brokerage.book()["rows"] if row["symbol"] == "SPY")

        assert row["qty"] == 20
        assert row["avg_entry_price"] == 110.0
        # Still marked at the last fill until a run marks it.
        assert row["unrealized_pl"] == 200.0

        brokerage.mark_prices({"SPY": 130.0})
        row = next(row for row in brokerage.book()["rows"] if row["symbol"] == "SPY")
        assert row["unrealized_pl"] == 400.0
        assert round(row["unrealized_plpc"], 4) == 0.1818


def test_selling_part_of_a_position_leaves_the_basis_alone() -> None:
    """A partial sale does not change what the remaining shares cost."""
    with ephemeral_state():
        brokerage = PaperBrokerage(get_config(account_id=LOCAL))
        brokerage.submit_order(_order("QQQ", "buy", 10, 100.0))
        brokerage.submit_order(_order("QQQ", "sell", 4, 150.0))

        row = next(row for row in brokerage.book()["rows"] if row["symbol"] == "QQQ")

        assert row["qty"] == 6
        assert row["avg_entry_price"] == 100.0


def test_closing_a_position_forgets_its_basis() -> None:
    with ephemeral_state():
        brokerage = PaperBrokerage(get_config(account_id=LOCAL))
        brokerage.submit_order(_order("GLD", "buy", 5, 200.0))
        brokerage.submit_order(_order("GLD", "sell", 5, 250.0))

        assert brokerage.book()["rows"] == []
        # Cash keeps the realised gain: paid 1000, received 1250.
        assert brokerage.get_account_state()["cash"] == 100_250.0


def test_two_local_accounts_do_not_share_one_book() -> None:
    with ephemeral_state():
        PaperBrokerage(get_config(account_id=LOCAL)).submit_order(_order("SPY", "buy", 3, 100.0))

        other = PaperBrokerage(get_config(account_id="another_paper"))

        assert other.get_positions() == {}


def test_positions_payload_reads_the_book_instead_of_calling_a_broker() -> None:
    with ephemeral_state():
        brokerage = PaperBrokerage(get_config(account_id=LOCAL))
        brokerage.submit_order(_order("SPY", "buy", 2, 500.0))

        payload = positions_payload(LOCAL)

        assert payload["error"] == ""
        assert payload["equity"] == 100_000.0
        assert [row["symbol"] for row in payload["rows"]] == ["SPY"]
        # No broker means no notion of yesterday's close, so the day figure stays absent
        # rather than being invented.
        assert payload["day_pl"] is None


def test_activity_for_a_local_book_comes_from_the_bot_journal() -> None:
    """The paper brokerage fills immediately and keeps no order log of its own."""
    with ephemeral_state():
        record_orders("dca", LOCAL, [{"symbol": "SPY", "action": "buy", "quantity": 2,
                                      "status": "submitted", "latest_price": 500.0}])
        record_orders("dca", "paper", [{"symbol": "QQQ", "action": "buy", "quantity": 1, "status": "submitted"}])

        rows = account_activity_payload(account_id=LOCAL)["rows"]

        assert [row["symbol"] for row in rows] == ["SPY"]
        assert rows[0]["filled_avg_price"] == 500.0
