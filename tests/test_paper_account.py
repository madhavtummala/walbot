from __future__ import annotations

import pytest

from src.api.api_payloads import account_activity_payload, accounts_payload, positions_payload
from src.brokerages.paper.brokerage import PaperBrokerage
from src.core.config import (
    UNNAMED_ACCOUNT_ID,
    Config,
    UnknownAccountError,
    get_account_broker_type,
    get_config,
)
from src.core.interfaces import OrderRequest
from src.data.order_journal import record_orders
from src.data.state_store import ephemeral_state

LOCAL = "local_paper"


def _order(symbol: str, action: str, quantity: float, price: float) -> OrderRequest:
    return OrderRequest(symbol=symbol, action=action, quantity=quantity, extra={"latest_price": price})



def _patch_payloads(monkeypatch, name, value):
    """Patch ``name`` on every payload module that resolves it.

    ``api_payloads`` is a facade now: the implementations live in ``src/api/payloads/`` and
    each module resolves its imports in its own namespace, so setting an attribute on the
    facade has no effect. Patching wherever the name actually exists keeps these tests stating
    an intent ("this dependency returns X") rather than a location.
    """
    import importlib
    import pkgutil

    import src.api.payloads as payloads_package

    patched = 0
    for info in pkgutil.iter_modules(payloads_package.__path__):
        module = importlib.import_module(f"src.api.payloads.{info.name}")
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            patched += 1
    assert patched, f"no payload module defines {name!r}"


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


def test_the_paper_book_is_whole_share_only() -> None:
    """Sizing reads whole-share off the class, and a fractional order is refused outright.

    Backtests execute through this same class, so the flag moves both paths at once -- and
    refusing in ``submit_order`` is what stops anything bypassing the sizer from filling.
    """
    assert PaperBrokerage.supports_fractional_shares is False

    with ephemeral_state():
        brokerage = PaperBrokerage(get_config(account_id=LOCAL))
        with pytest.raises(ValueError, match="whole shares only"):
            brokerage.submit_order(_order("SPY", "buy", 10.5, 100.0))
        assert brokerage.get_positions() == {}


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
    """Books are keyed by account, so one account's fill is invisible to another.

    The second account is built directly rather than looked up: ``get_config`` now refuses an
    account that is not configured instead of quietly handing back the default one, and the
    property under test is about the book key, not about config resolution.
    """
    with ephemeral_state():
        PaperBrokerage(get_config(account_id=LOCAL)).submit_order(_order("SPY", "buy", 3, 100.0))

        other = PaperBrokerage(Config(account_id="another_paper"))

        assert other.get_positions() == {}


def test_asking_for_an_account_that_does_not_exist_is_an_error() -> None:
    """It must never resolve to a different account.

    Silently substituting the default meant an account page could show another account's
    money under the requested name, and -- because ``live_runner.run_once(account_id=...)``
    resolves the same way -- a binding naming a renamed or deleted account would have sent its
    orders to the default book.
    """
    with pytest.raises(UnknownAccountError):
        get_config(account_id="no_such_account")

    with pytest.raises(UnknownAccountError):
        get_account_broker_type("no_such_account")

    # A bare config carries the "no account named" sentinel, which is not a lookup failure.
    assert get_account_broker_type(UNNAMED_ACCOUNT_ID)


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


def test_a_non_alpaca_account_is_not_reported_from_alpaca(monkeypatch) -> None:
    """Only the Alpaca branch may use the Alpaca client.

    A Schwab account fell through to it and displayed the *Alpaca* account's equity and P/L
    under the Schwab account's name -- two accounts showing one balance, with nothing saying
    which was real.
    """
    from src.api import api_payloads

    class FakeBrokerage:
        def get_account_state(self):
            return {"equity": 4321.0, "cash": 321.0}

        def get_positions(self):
            return {"SPY": 3.0}

        def get_position_details(self):
            return [
                {
                    "symbol": "SPY",
                    "qty": 3.0,
                    "avg_entry_price": 0.0,
                    "market_value": 300.0,
                    "unrealized_pl": 0.0,
                    "unrealized_plpc": 0.0,
                }
            ]

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: FakeBrokerage())

    def fail(*_args, **_kwargs):
        raise AssertionError("the Alpaca client must not be used for a Schwab account")

    _patch_payloads(monkeypatch, "create_trading_client", fail)

    payload = api_payloads.positions_payload("schwab_individual")

    assert payload["equity"] == 4321.0
    assert payload["cash"] == 321.0
    assert [row["symbol"] for row in payload["rows"]] == ["SPY"]
    assert payload["rows"][0]["market_value"] == 300.0
    assert payload["total_pl"] == 0.0


def test_config_resolves_real_files_from_the_repo_root() -> None:
    """Guards a failure mode that type checking cannot see.

    ``_project_root`` counts parent directories from ``__file__``, so moving the config module
    changes what it resolves to. When ``config.py`` became a package the count went stale, the
    root resolved to ``src/``, every config file read as missing, and ``get_config()`` quietly
    returned the unnamed-account sentinel with an empty universe -- no error anywhere.
    """
    from src.core.config import config_file_path, get_config

    assert config_file_path().exists(), "walbot.yaml must resolve from the repo root"

    config = get_config()
    assert config.account_id != UNNAMED_ACCOUNT_ID, "a configured default account must be found"
    assert config.symbols, "the tradable universe must load"


def test_the_book_values_its_cash_equivalents_for_order_funding() -> None:
    """Funding has to value holdings the running algorithm may never price.

    DCA buys VTI and has no reason to look up SGOV, yet SGOV is exactly what would fund the
    buy -- so the mark comes from the broker's own book rather than from the plan's prices.
    """
    with ephemeral_state():
        config = Config(account_id="funding_paper", cash_equivalents=["SGOV", "BIL"])
        book = PaperBrokerage(config)
        book.submit_order(_order("SGOV", "buy", 10, 100.0))
        book.submit_order(_order("SPY", "buy", 1, 50.0))

        holdings = book.get_cash_equivalents()

        assert holdings == {"SGOV": {"shares": 10.0, "price": 100.0, "value": 1_000.0}}


def test_cash_equivalents_are_empty_when_the_account_configures_none() -> None:
    with ephemeral_state():
        book = PaperBrokerage(Config(account_id="funding_paper", cash_equivalents=[]))
        book.submit_order(_order("SGOV", "buy", 10, 100.0))

        assert book.get_cash_equivalents() == {}


def test_paper_refuses_option_orders_rather_than_buying_the_underlying(tmp_path) -> None:
    """It fills at the mark with no resting orders, so it cannot represent a contract.

    Before this refusal existed the order went through as shares of whatever the symbol string
    happened to be, and was reported back as a filled contract.
    """
    from src.core.interfaces import OrderRequest

    with ephemeral_state():
        brokerage = PaperBrokerage(Config(account_id="local_paper"))
        with pytest.raises(NotImplementedError, match="option"):
            brokerage.submit_order(OrderRequest(
                symbol="QQQM  260220C00100000", action="buy", quantity=1,
                order_type="limit", limit_price=1.15, asset_type="option",
                extra={"latest_price": 1.15},
            ))


def test_paper_refuses_a_bracket_it_cannot_hold(tmp_path) -> None:
    from src.core.interfaces import OrderRequest

    child = OrderRequest(symbol="AAA", action="sell", quantity=1, order_type="limit", limit_price=2.0)
    with ephemeral_state():
        brokerage = PaperBrokerage(Config(account_id="local_paper"))
        with pytest.raises(NotImplementedError, match="OCO"):
            brokerage.submit_order(OrderRequest(
                symbol="AAA", action="sell", quantity=1, order_type="limit", limit_price=2.0,
                strategy="oco", children=(child, child), extra={"latest_price": 2.0},
            ))
