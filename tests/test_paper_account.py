from __future__ import annotations

from datetime import datetime, timezone

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


def _starting_cash() -> float:
    return float(get_config(account_id=LOCAL).paper_starting_cash)



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
        # Cash keeps the realised gain: paid 1000, received 1250. Stated against the
        # account's own starting cash so resizing the book in config does not read as
        # a regression here.
        assert brokerage.get_account_state()["cash"] == _starting_cash() + 250.0


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
    resolves the same way -- a deployment naming a renamed or deleted account would have sent its
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
        # A buy moves cash into stock, so equity is unchanged from the opening balance.
        assert payload["equity"] == _starting_cash()
        assert [row["symbol"] for row in payload["rows"]] == ["SPY"]
        # The book stamps its own opening value on the session's first read, so the day's move
        # is zero until something in it moves -- not unknown, which is what None meant.
        assert payload["day_pl"] == 0.0


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

    payload = api_payloads.positions_payload("schwab2")

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


def test_two_deployments_on_one_paper_account_do_not_lose_each_others_fills() -> None:
    """The dashboard permits several algorithms on one account, and the scheduler runs a thread
    per deployment -- so two instances of this class can hold the same book at once.

    The book used to be read once at construction and written back whole, so both threads
    started from the same balance and the second write discarded the first's cash and
    positions. Fills vanished from a book whose whole job is to rehearse a live one.
    """
    import threading

    from src.brokerages.paper.brokerage import _state_key
    from src.data.state_store import delete_state, load_state

    account = "concurrency_probe"
    key = _state_key(account)
    config = Config(account_id=account)
    try:
        books = [PaperBrokerage(config), PaperBrokerage(config)]
        # Both read their starting balance before either writes, which is the race exactly.
        starting_cash = books[0].state["cash"]
        assert all(book.state["cash"] == starting_cash for book in books)

        start = threading.Barrier(len(books))

        def fill(book: PaperBrokerage, symbol: str) -> None:
            start.wait()
            book.submit_order(
                OrderRequest(symbol, "buy", 10, extra={"latest_price": 100.0})
            )

        threads = [
            threading.Thread(target=fill, args=(book, symbol))
            for book, symbol in zip(books, ("AAA", "BBB"))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        stored = load_state(key, {})
        # Both fills are in the book, and both were paid for: one lost write would leave a
        # single position and $1,000 of phantom cash.
        assert stored["positions"] == {"AAA": 10.0, "BBB": 10.0}
        assert stored["cash"] == starting_cash - 2000.0
    finally:
        delete_state(key)


def test_paper_refuses_a_short_it_was_not_asked_to_approve() -> None:
    """``submit_planned_orders`` checks short feasibility, but it is one caller of several --
    the reconciler and the MCP tools reach ``submit_order`` directly.

    A sell with nothing held used to be booked as a short that credited cash, with no margin
    requirement and no size limit, so the book could short indefinitely and manufacture its own
    buying power.
    """
    class NoShorts(PaperBrokerage):
        def validate_short_sale_feasibility(self, symbol, quantity, target_shares, latest_price):
            return {"shortable": False, "reason": "not shortable"}

    with ephemeral_state():
        brokerage = NoShorts(Config(account_id=LOCAL))
        with pytest.raises(ValueError, match="Short sale refused"):
            brokerage.submit_order(
                OrderRequest("AAA", "sell", 10, extra={"latest_price": 100.0})
            )


def test_a_schwab_accounts_orders_come_from_the_broker_not_the_bot_journal(monkeypatch) -> None:
    """Including trades placed by hand, which no journal here can ever see.

    The journal records what this bot submitted. Standing it in for the broker's own feed made
    the account page -- and the MCP tool that shares it -- answer "orders this bot placed" to
    the question "what happened in this account", so a buy made in the broker's own app was
    invisible while the position it created sat in the holdings table above it.
    """
    from src.api import api_payloads

    class FakeBrokerage:
        def get_orders(self, status="WORKING", *, days=60):
            assert status == "", "every state, not just the resting ones"
            return [
                {
                    "order_id": "1", "symbol": "XSD", "action": "buy", "status": "FILLED",
                    "quantity": 10.0, "filled_quantity": 10.0, "order_type": "limit",
                    "limit_price": 100.0, "stop_price": 0.0, "filled_avg_price": 99.5,
                    "entered_time": "2026-09-14T14:30:00+0000",
                },
                {
                    "order_id": "2", "symbol": "XBI", "action": "buy", "status": "WORKING",
                    "quantity": 4.0, "filled_quantity": 0.0, "order_type": "limit",
                    "limit_price": 80.0, "stop_price": 0.0, "filled_avg_price": 0.0,
                    "entered_time": "2026-09-14T15:00:00+0000",
                },
            ]

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: FakeBrokerage())

    with ephemeral_state():
        rows = api_payloads.account_activity_payload("schwab2")["rows"]

    # Most recent first, and neither of these was ever written to the bot's journal.
    assert [row["symbol"] for row in rows] == ["XBI", "XSD"]
    assert rows[1]["filled_avg_price"] == 99.5
    # A resting order has no fill, which reads as "--" rather than a $0.00 trade.
    assert rows[0]["filled_avg_price"] is None
    assert rows[0]["status"] == "WORKING"


def test_an_unreachable_broker_reports_itself_rather_than_an_empty_order_list(monkeypatch) -> None:
    """An empty list would read as "this account has not traded", which is a different claim."""
    from src.api import api_payloads

    class FakeBrokerage:
        def get_orders(self, status="WORKING", *, days=60):
            raise RuntimeError("schwab is down")

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: FakeBrokerage())

    payload = api_payloads.account_activity_payload("schwab2")

    assert payload["rows"] == []
    assert "schwab is down" in payload["error"]


def test_the_accounts_open_pl_sums_the_brokers_open_figure(monkeypatch) -> None:
    """Not the day figure, which is what made Day P/L and Open P/L agree on every read."""
    from src.api import api_payloads

    class FakeBrokerage:
        def get_account_state(self):
            return {"equity": 10_060.0, "cash": 60.0, "last_equity": 10_000.0}

        def get_position_details(self):
            return [
                {"symbol": "XSD", "qty": 10.0, "avg_entry_price": 100.0, "market_value": 1_500.0,
                 "unrealized_pl": 500.0, "unrealized_plpc": 0.5, "day_pl": 20.0, "day_pl_percent": 0.013},
                {"symbol": "XBI", "qty": 5.0, "avg_entry_price": 80.0, "market_value": 440.0,
                 "unrealized_pl": 40.0, "unrealized_plpc": 0.1, "day_pl": 40.0, "day_pl_percent": 0.1},
            ]

        def get_dividend_activity(self, start=None, end=None):
            return []

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: FakeBrokerage())

    payload = api_payloads.positions_payload("schwab2")

    # The day's move comes from the account's own opening value; the open figure from the
    # positions. The two are free to disagree, and here they do.
    assert payload["day_pl"] == 60.0
    assert payload["total_pl"] == 540.0
    assert payload["rows"][0]["day_pl"] == 20.0


def _dated_order(symbol: str, stamp: str) -> dict:
    return {
        "order_id": symbol, "symbol": symbol, "action": "buy", "status": "FILLED",
        "quantity": 1.0, "filled_quantity": 1.0, "order_type": "market",
        "limit_price": 0.0, "stop_price": 0.0, "filled_avg_price": 10.0,
        "entered_time": stamp,
    }


def _schwab_stamp(days_ago: float) -> str:
    """Schwab's own spelling: a colonless ``+0000`` offset, which is not what JS or
    ``fromisoformat`` consider the standard form."""
    from datetime import datetime, timedelta, timezone

    moment = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return moment.strftime("%Y-%m-%dT%H:%M:%S+0000")


def _analytics(account_id: str, **kwargs):
    """Analytics with the cache emptied first, so one test cannot answer another's question."""
    from src.api.payloads.accounts import ANALYTICS, account_analytics_payload

    ANALYTICS.invalidate(account_id)
    return account_analytics_payload(account_id, **kwargs)


def _week_ago():
    from datetime import datetime, timedelta, timezone

    return datetime.now(timezone.utc) - timedelta(days=7)


def _fake_schwab(monkeypatch, orders: list[dict]) -> None:
    class FakeBrokerage:
        def get_orders(self, status="WORKING", *, days=60):
            return orders

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: FakeBrokerage())


def test_the_account_page_caps_by_count_and_never_by_date(monkeypatch) -> None:
    """A date cutoff would drop a good-till-cancelled order still resting from last month.

    That order is current exposure, so the page bounds its list by count alone. Only the MCP
    tool, which is answering "what happened today", passes a cutoff.
    """
    from src.api import api_payloads

    _fake_schwab(monkeypatch, [
        _dated_order("XSD", _schwab_stamp(0.2)),
        _dated_order("GTC", _schwab_stamp(35.0)),
    ])

    rows = api_payloads.account_activity_payload("schwab2")["rows"]

    assert [row["symbol"] for row in rows] == ["XSD", "GTC"]


def test_the_mcp_tool_caps_by_count_and_by_the_trading_day(monkeypatch) -> None:
    """Narrower than the page, because the agent is asking "what happened today"."""
    from src.api import api_payloads

    _fake_schwab(monkeypatch, [
        _dated_order("XSD", _schwab_stamp(0.01)),
        _dated_order("XBI", _schwab_stamp(3.0)),
    ])

    rows = api_payloads.account_activity_payload(
        "schwab2", limit=40, since=api_payloads.market_day_start()
    )["rows"]

    assert [row["symbol"] for row in rows] == ["XSD"]


def test_an_order_whose_timestamp_cannot_be_read_is_kept(monkeypatch) -> None:
    """Dropping it would hide a real order because of a format this code did not anticipate."""
    from src.api import api_payloads

    _fake_schwab(monkeypatch, [_dated_order("XSD", "not a timestamp")])

    rows = api_payloads.account_activity_payload("schwab2", since=_week_ago())["rows"]

    assert [row["symbol"] for row in rows] == ["XSD"]


def test_no_cutoff_returns_everything_the_broker_offered(monkeypatch) -> None:
    from src.api import api_payloads

    _fake_schwab(monkeypatch, [
        _dated_order("XSD", _schwab_stamp(0.2)),
        _dated_order("OLD", _schwab_stamp(40.0)),
    ])

    rows = api_payloads.account_activity_payload("schwab2")["rows"]

    assert [row["symbol"] for row in rows] == ["XSD", "OLD"]


def test_the_date_window_is_applied_before_the_count_cap(monkeypatch) -> None:
    """``limit`` caps what survives the window, rather than deciding what the window sees.

    Capping first would let a run of old orders crowd out this week's, so a busy account would
    show "no orders in the last 7 days" while having traded this morning.
    """
    from src.api import api_payloads

    orders = [_dated_order(f"OLD{n}", _schwab_stamp(30.0 + n)) for n in range(5)]
    orders.append(_dated_order("XSD", _schwab_stamp(0.2)))
    _fake_schwab(monkeypatch, orders)

    rows = api_payloads.account_activity_payload("schwab2", limit=3, since=_week_ago())["rows"]

    assert [row["symbol"] for row in rows] == ["XSD"]


def test_a_closed_winning_trade_shows_up_as_realized_not_open(monkeypatch) -> None:
    """The failure this exists to fix.

    An account opened a call, closed it at a profit and went back to cash. It holds nothing, so
    open P/L and day P/L are both correctly zero -- and the page showed nothing else, making a
    profitable account look like it had never traded.
    """
    from src.api import api_payloads

    class FakeBrokerage:
        def get_account_state(self):
            return {"equity": 10_524.84, "cash": 10_524.84, "last_equity": 10_524.84}

        def get_position_details(self):
            return []

        def get_dividend_activity(self, start=None, end=None):
            return []

        def get_fills(self, start=None, end=None):
            return [
                {"symbol": "USO260916C00142000", "action": "buy", "quantity": 1.0,
                 "price": 8.85, "multiplier": 100.0, "date": "2026-09-09T16:23:09Z"},
                {"symbol": "USO260916C00142000", "action": "sell", "quantity": 1.0,
                 "price": 14.10, "multiplier": 100.0, "date": "2026-09-10T16:00:39Z"},
            ]

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: FakeBrokerage())

    assert api_payloads.positions_payload("schwab2")["total_pl"] == 0.0  # nothing held, nothing open

    payload = _analytics("schwab2", refresh=True)

    assert payload["realized_pl"] == 525.0
    assert payload["realized_closes"] == 1
    assert payload["realized_unmatched"] == 0


def test_a_broker_that_cannot_report_fills_says_unknown_rather_than_zero(monkeypatch) -> None:
    """Zero is a claim -- "this account has banked nothing" -- and a different one."""
    from src.api import api_payloads

    class FakeBrokerage:
        def get_account_state(self):
            return {"equity": 100.0, "cash": 100.0}

        def get_position_details(self):
            return []

        def get_dividend_activity(self, start=None, end=None):
            return []

        def get_fills(self, start=None, end=None):
            return None  # the interface default: this venue has no fill feed

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: FakeBrokerage())

    assert _analytics("schwab2", refresh=True)["realized_pl"] is None


def test_an_unreadable_fill_feed_does_not_blank_the_rest_of_the_page(monkeypatch) -> None:
    from src.api import api_payloads

    class FakeBrokerage:
        def get_account_state(self):
            return {"equity": 4_321.0, "cash": 321.0}

        def get_position_details(self):
            return []

        def get_dividend_activity(self, start=None, end=None):
            return []

        def get_fills(self, start=None, end=None):
            raise RuntimeError("transactions endpoint is down")

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: FakeBrokerage())

    assert _analytics("schwab2", refresh=True)["realized_pl"] is None
    # The balances beside it are a separate read and must survive the failure.
    assert api_payloads.positions_payload("schwab2")["equity"] == 4_321.0


def _counting_brokerage(reads: list):
    class FakeBrokerage:
        def get_account_state(self):
            return {"equity": 100.0, "cash": 100.0}

        def get_position_details(self):
            return []

        def get_dividend_activity(self, start=None, end=None):
            reads.append("dividends")
            return []

        def get_fills(self, start=None, end=None):
            reads.append("fills")
            return []

    return FakeBrokerage()


def _inline_analytics(monkeypatch, reads: list):
    """The analytics field with its background work run inline.

    The spawn hook exists for exactly this: a test can assert on what the background pass
    produced instead of sleeping until a thread happens to finish.
    """
    from src.common.lazy_field import LazyField
    from src.api.payloads import accounts

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: _counting_brokerage(reads))
    field = LazyField("test analytics", accounts._compute_analytics, spawn=lambda work: work())
    monkeypatch.setattr(accounts, "ANALYTICS", field)
    return field


def test_a_page_load_answers_instantly_and_fills_itself_in(monkeypatch) -> None:
    """The whole point of making these lazy rather than manual.

    A page load never waits on a year of transactions, and never has to tell the user to press
    a button either -- it answers with what it has and works out the rest in the background.
    """
    from src.api.payloads.accounts import account_analytics_payload

    reads: list[str] = []
    _inline_analytics(monkeypatch, reads)

    # The first read has nothing to answer with, and says so rather than reporting a zero.
    first = account_analytics_payload("schwab2")
    assert first["state"] == "computing"
    assert first["realized_pl"] is None

    # It started the work on its own; by the next render the value is simply there.
    second = account_analytics_payload("schwab2")
    assert second["state"] == "ready"
    assert second["computed_at"], "a computed figure must say how old it is"
    assert reads == ["dividends", "fills"], "computed once, not once per read"


def test_a_warm_value_is_served_without_touching_the_broker(monkeypatch) -> None:
    from src.api.payloads.accounts import account_analytics_payload

    reads: list[str] = []
    _inline_analytics(monkeypatch, reads)

    account_analytics_payload("schwab2")
    assert reads == ["dividends", "fills"]

    for _ in range(5):
        account_analytics_payload("schwab2")
    assert reads == ["dividends", "fills"], "a fresh value is reused until it goes stale"


def test_refresh_recomputes_even_when_a_value_is_already_fresh(monkeypatch) -> None:
    """Which is the point of having a button: the user decides when to insist."""
    from src.api.payloads.accounts import account_analytics_payload

    reads: list[str] = []
    _inline_analytics(monkeypatch, reads)

    account_analytics_payload("schwab2")
    assert reads == ["dividends", "fills"]

    payload = account_analytics_payload("schwab2", refresh=True)

    assert reads == ["dividends", "fills", "dividends", "fills"]
    assert payload["state"] == "ready"


def test_concurrent_reads_start_one_computation_between_them(monkeypatch) -> None:
    """Otherwise every panel on a freshly-opened page starts its own year-long crawl."""
    from src.api.payloads import accounts
    from src.common.lazy_field import LazyField

    reads: list[str] = []
    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: _counting_brokerage(reads))

    pending: list = []
    # Work is queued rather than run, so several reads land while the first is still in flight.
    field = LazyField("test analytics", accounts._compute_analytics, spawn=pending.append)
    monkeypatch.setattr(accounts, "ANALYTICS", field)

    for _ in range(4):
        accounts.account_analytics_payload("schwab2")

    assert len(pending) == 1
    pending[0]()
    assert reads == ["dividends", "fills"]


def test_a_failed_recompute_keeps_the_value_it_had(monkeypatch) -> None:
    """A figure from twenty minutes ago is worth more than a blank."""
    from src.common.lazy_field import LazyField

    attempts = {"n": 0}

    def compute(_key):
        attempts["n"] += 1
        if attempts["n"] > 1:
            raise RuntimeError("schwab is down")
        return {"realized_pl": 525.0}

    field = LazyField("test", compute, spawn=lambda work: work())
    field.get("acct")

    failed = field.get("acct", force=True)

    assert failed["value"] == {"realized_pl": 525.0}
    assert "schwab is down" in failed["error"]


def test_both_figures_come_from_one_resolved_brokerage(monkeypatch) -> None:
    """Each resolution builds a fresh session, and for Schwab that is another OAuth exchange
    and another account lookup -- helpers resolving independently authenticated four times
    for one page."""
    from src.api.payloads import accounts
    from src.common.lazy_field import LazyField

    resolved: list[int] = []
    reads: list[str] = []

    def resolve(_config):
        resolved.append(1)
        return _counting_brokerage(reads)

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", resolve)
    monkeypatch.setattr(
        accounts, "ANALYTICS",
        LazyField("test", accounts._compute_analytics, spawn=lambda work: work()),
    )

    accounts.account_analytics_payload("schwab2", refresh=True)

    assert len(resolved) == 1


def test_the_page_reports_year_to_date_and_carries_the_trailing_year_beside_it(monkeypatch) -> None:
    """Year to date leads because it is what a broker's own statement totals.

    A Schwab account reported a trailing-year profit against the broker's own year-to-date
    loss, and the two were never comparable in the first place.
    """
    from src.api.payloads import accounts
    from src.common.lazy_field import LazyField

    class FakeBrokerage:
        def get_dividend_activity(self, start=None, end=None):
            return []

        def get_fills(self, start=None, end=None):
            year = datetime.now(timezone.utc).year
            return [
                # Last year's loss: inside the trailing year, outside this calendar one.
                {"symbol": "QQQ", "action": "buy", "quantity": 5.0, "price": 50.0,
                 "multiplier": 1.0, "date": f"{year - 1}-10-01"},
                {"symbol": "QQQ", "action": "sell", "quantity": 5.0, "price": 40.0,
                 "multiplier": 1.0, "date": f"{year - 1}-12-01"},
                # This year's gain, opened before January -- the basis still has to be known.
                {"symbol": "SPY", "action": "buy", "quantity": 10.0, "price": 100.0,
                 "multiplier": 1.0, "date": f"{year - 1}-11-01"},
                {"symbol": "SPY", "action": "sell", "quantity": 10.0, "price": 120.0,
                 "multiplier": 1.0, "date": f"{year}-02-01"},
            ]

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: FakeBrokerage())
    monkeypatch.setattr(
        accounts, "ANALYTICS",
        LazyField("test", accounts._compute_analytics, spawn=lambda work: work()),
    )

    payload = accounts.account_analytics_payload("schwab2", refresh=True)

    assert payload["realized_pl"] == 200.0, "year to date: the SPY gain only"
    assert payload["realized_unmatched"] == 0
    # The trailing year is the matching basis, not a reported figure: last year's QQQ loss
    # supplies cost basis for this year's sells and is never totalled on its own.
    assert "realized_pl_1y" not in payload
    assert "activity_year" not in payload


def test_dividends_are_reported_year_to_date_like_realized(monkeypatch) -> None:
    """One window across the page, so two income figures beside each other are comparable."""
    from src.api.payloads import accounts
    from src.common.lazy_field import LazyField

    year = datetime.now(timezone.utc).year

    class FakeBrokerage:
        def get_dividend_activity(self, start=None, end=None):
            return [
                {"symbol": "SGOV", "date": f"{year}-03-01", "amount": 40.0, "description": "div"},
                {"symbol": "SGOV", "date": f"{year}-01-02", "amount": 10.0, "description": "div"},
                # Inside the trailing year, outside the calendar one.
                {"symbol": "SGOV", "date": f"{year - 1}-11-01", "amount": 25.0, "description": "div"},
            ]

        def get_fills(self, start=None, end=None):
            return []

    _patch_payloads(monkeypatch, "get_account_broker_type", lambda _account: "schwab")
    monkeypatch.setattr("src.core.pipeline.resolve_brokerage", lambda _config: FakeBrokerage())
    monkeypatch.setattr(
        accounts, "ANALYTICS",
        LazyField("test", accounts._compute_analytics, spawn=lambda work: work()),
    )

    payload = accounts.account_analytics_payload("schwab2", refresh=True)

    assert payload["dividend_pl"] == 50.0, "year to date: November is outside it"
    assert "dividend_pl_1y" not in payload
    assert len(payload["dividend_rows"]) == 3, "the rows themselves are not windowed away"
