from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from src.brokerages.schwab.brokerage import SchwabBrokerage
from src.brokerages.schwab.client import SchwabAuthError, SchwabSession
from src.connectors.market.schwab import Schwab
from src.core.config import Config
from src.core.interfaces import OrderRequest


class FakeResponse:
    def __init__(self, payload: Any = None, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload) if payload is not None else ""
        self.content = self.text.encode()

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeHTTP:
    """Records requests and replays canned responses keyed by URL suffix."""

    def __init__(self, routes: dict[str, Any]):
        self.routes = routes
        self.calls: list[tuple[str, str, dict]] = []

    def post(self, url, **kwargs):
        return self._resolve("POST", url, kwargs)

    def request(self, method, url, **kwargs):
        return self._resolve(method, url, kwargs)

    def _resolve(self, method, url, kwargs):
        self.calls.append((method, url, kwargs))
        for suffix, response in self.routes.items():
            if url.endswith(suffix) or suffix in url:
                return response
        return FakeResponse({}, status_code=404)


def _config(**overrides) -> Config:
    settings = {
        "schwab_app_key": "key",
        "schwab_app_secret": "secret",
        "schwab_refresh_token": "refresh",
        **overrides,
    }
    return Config(**settings)


def _session(routes: dict[str, Any], config: Config | None = None) -> SchwabSession:
    routes.setdefault("/v1/oauth/token", FakeResponse({"access_token": "token", "expires_in": 1800}))
    return SchwabSession(config or _config(), session=FakeHTTP(routes))


def test_missing_credentials_are_reported_before_any_request() -> None:
    session = SchwabSession(Config(), session=FakeHTTP({}))

    with pytest.raises(SchwabAuthError, match="schwab_app_key"):
        session.access_token()


def test_access_token_is_cached_between_calls() -> None:
    session = _session({})

    assert session.access_token() == "token"
    assert session.access_token() == "token"
    assert sum(1 for method, url, _ in session._session.calls if "oauth/token" in url) == 1


def test_positions_net_long_against_short_quantity() -> None:
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/accounts/HASH": FakeResponse(
                {
                    "securitiesAccount": {
                        "positions": [
                            {"instrument": {"symbol": "AAA"}, "longQuantity": 10, "shortQuantity": 0},
                            {"instrument": {"symbol": "BBB"}, "longQuantity": 0, "shortQuantity": 4},
                            {"instrument": {"symbol": "CCC"}, "longQuantity": 0, "shortQuantity": 0},
                        ]
                    }
                }
            ),
        }
    )
    brokerage = SchwabBrokerage(_config(), session=session)

    assert brokerage.get_positions() == {"AAA": 10.0, "BBB": -4.0}


def test_account_state_prefers_liquidation_value() -> None:
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/accounts/HASH": FakeResponse(
                {
                    "securitiesAccount": {
                        "currentBalances": {
                            "liquidationValue": 5_000.0,
                            "cashBalance": 1_200.0,
                            "buyingPower": 2_400.0,
                        }
                    }
                }
            ),
            "/marketdata/v1/markets": FakeResponse({"equity": {}}),
        }
    )
    brokerage = SchwabBrokerage(_config(), session=session)

    state = brokerage.get_account_state()

    assert state["equity"] == 5_000.0
    assert state["cash"] == 1_200.0
    assert state["buying_power"] == 2_400.0


def test_market_order_payload_matches_schwab_schema() -> None:
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/orders": FakeResponse(None, status_code=201, headers={"Location": "https://x/orders/9988"}),
        }
    )
    brokerage = SchwabBrokerage(_config(), session=session)

    result = brokerage.submit_order(OrderRequest(symbol="aaa", action="buy", quantity=7))

    order_call = next(call for call in session._session.calls if call[1].endswith("/orders"))
    payload = order_call[2]["json"]
    assert payload["orderType"] == "MARKET"
    assert payload["orderStrategyType"] == "SINGLE"
    assert payload["orderLegCollection"] == [
        {"instruction": "BUY", "quantity": 7, "instrument": {"symbol": "AAA", "assetType": "EQUITY"}}
    ]
    assert result["order_id"] == "9988"


def test_fractional_quantities_are_refused() -> None:
    session = _session({"/accounts/accountNumbers": FakeResponse([{"hashValue": "HASH"}])})
    brokerage = SchwabBrokerage(_config(), session=session)

    with pytest.raises(ValueError, match="fractional"):
        brokerage.submit_order(OrderRequest(symbol="AAA", action="buy", quantity=2.5))


def test_schwab_declares_no_fractional_support() -> None:
    assert SchwabBrokerage.supports_fractional_shares is False


def test_quotes_fall_back_from_last_to_mid_to_close(monkeypatch) -> None:
    from src.connectors.market import schwab as market_schwab

    monkeypatch.setattr(market_schwab, "_schwab_token", lambda config, category: "token")
    monkeypatch.setattr(
        market_schwab,
        "_request_json",
        lambda *a, **kw: {
            "AAA": {"quote": {"lastPrice": 10.0}},
            "BBB": {"quote": {"lastPrice": 0, "bidPrice": 4.0, "askPrice": 6.0}},
            "CCC": {"quote": {"lastPrice": 0, "closePrice": 3.0}},
            "DDD": {"quote": {"lastPrice": 0}},
        },
    )

    quotes = Schwab(_config()).fetch_price(["AAA", "BBB", "CCC", "DDD"])

    assert quotes["AAA"]["price"] == 10.0
    assert quotes["BBB"]["price"] == 5.0
    assert quotes["CCC"]["price"] == 3.0
    assert "DDD" not in quotes  # unpriceable symbols are omitted, not zero-priced


# ── option and bracket order payloads ────────────────────────────────────────


def _order_brokerage() -> tuple[SchwabBrokerage, SchwabSession]:
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/orders": FakeResponse(None, status_code=201, headers={"Location": "https://x/orders/4242"}),
        }
    )
    return SchwabBrokerage(_config(), session=session), session


def _sent_payload(session: SchwabSession) -> dict:
    call = next(call for call in session._session.calls if "/orders" in call[1] and call[0] in ("POST", "PUT"))
    return call[2]["json"]


def test_option_limit_buy_opens_the_position() -> None:
    brokerage, session = _order_brokerage()

    brokerage.submit_order(OrderRequest(
        symbol="QQQM  260220C00100000", action="buy", quantity=1,
        order_type="limit", limit_price=1.15, asset_type="option",
        extra={"position_intent": "buy_to_open"},
    ))

    payload = _sent_payload(session)
    assert payload["orderType"] == "LIMIT"
    assert payload["price"] == 1.15
    assert payload["duration"] == "DAY"
    assert payload["orderLegCollection"] == [{
        "instruction": "BUY_TO_OPEN",
        "quantity": 1,
        "instrument": {"symbol": "QQQM  260220C00100000", "assetType": "OPTION"},
    }]


def test_a_bare_option_sell_closes_rather_than_shorting() -> None:
    brokerage, session = _order_brokerage()

    brokerage.submit_order(OrderRequest(
        symbol="QQQM  260220C00100000", action="sell", quantity=1,
        order_type="limit", limit_price=2.80, asset_type="option",
    ))

    assert _sent_payload(session)["orderLegCollection"][0]["instruction"] == "SELL_TO_CLOSE"


def test_stop_order_carries_a_stop_price_and_no_limit() -> None:
    brokerage, session = _order_brokerage()

    brokerage.submit_order(OrderRequest(
        symbol="QQQM  260220C00100000", action="sell", quantity=1,
        order_type="stop", stop_price=1.50, asset_type="option", time_in_force="gtc",
    ))

    payload = _sent_payload(session)
    assert payload["orderType"] == "STOP"
    assert payload["stopPrice"] == 1.50
    assert payload["duration"] == "GOOD_TILL_CANCEL"
    assert "price" not in payload


def test_oco_bracket_nests_both_legs_under_one_order() -> None:
    brokerage, session = _order_brokerage()
    target = OrderRequest(
        symbol="QQQM  260220C00100000", action="sell", quantity=1,
        order_type="limit", limit_price=2.80, asset_type="option", time_in_force="gtc",
    )
    stop = OrderRequest(
        symbol="QQQM  260220C00100000", action="sell", quantity=1,
        order_type="stop", stop_price=1.50, asset_type="option", time_in_force="gtc",
    )

    brokerage.submit_order(OrderRequest(
        symbol="QQQM  260220C00100000", action="sell", quantity=1,
        order_type="limit", limit_price=2.80, asset_type="option",
        strategy="oco", children=(target, stop),
    ))

    payload = _sent_payload(session)
    # The OCO wrapper carries no legs of its own -- only the pair it governs, which is what
    # makes "either fills, never both" the exchange's invariant rather than ours.
    assert payload["orderStrategyType"] == "OCO"
    assert "orderLegCollection" not in payload
    kinds = {child["orderType"] for child in payload["childOrderStrategies"]}
    assert kinds == {"LIMIT", "STOP"}


def test_an_order_type_needing_a_trigger_is_refused_before_it_is_sent() -> None:
    with pytest.raises(ValueError, match="stop_price"):
        OrderRequest(symbol="AAA", action="sell", quantity=1, order_type="stop")


def test_prices_are_rounded_to_the_tick() -> None:
    brokerage, session = _order_brokerage()

    brokerage.submit_order(OrderRequest(
        symbol="QQQM  260220C00100000", action="buy", quantity=1,
        order_type="limit", limit_price=1.1549999, asset_type="option",
    ))

    assert _sent_payload(session)["price"] == 1.15


def test_working_orders_are_flattened_across_brackets() -> None:
    session = _session({
        "/accounts/accountNumbers": FakeResponse([{"hashValue": "HASH"}]),
        "/orders": FakeResponse([{
            "orderId": 1,
            "orderType": "LIMIT",
            "price": 2.80,
            "status": "WORKING",
            "orderStrategyType": "OCO",
            "childOrderStrategies": [
                {
                    "orderId": 2, "orderType": "LIMIT", "price": 2.80, "status": "WORKING",
                    "orderLegCollection": [{
                        "instruction": "SELL_TO_CLOSE", "quantity": 1,
                        "instrument": {"symbol": "QQQM  260220C00100000", "assetType": "OPTION"},
                    }],
                },
                {
                    "orderId": 3, "orderType": "STOP", "stopPrice": 1.50, "status": "WORKING",
                    "orderLegCollection": [{
                        "instruction": "SELL_TO_CLOSE", "quantity": 1,
                        "instrument": {"symbol": "QQQM  260220C00100000", "assetType": "OPTION"},
                    }],
                },
            ],
        }]),
    })
    brokerage = SchwabBrokerage(_config(), session=session)

    orders = brokerage.get_orders()

    # The wrapper contributes no row; both legs do, and each keeps its parent.
    assert [order["order_id"] for order in orders] == ["2", "3"]
    assert all(order["parent_order_id"] == "1" for order in orders)
    assert all(order["asset_type"] == "option" for order in orders)
    assert all(order["action"] == "sell" for order in orders)


def test_cancelling_an_order_that_is_already_gone_is_not_an_error() -> None:
    class Refusing(FakeHTTP):
        def _resolve(self, method, url, kwargs):
            super()._resolve(method, url, kwargs)
            return FakeResponse({"error": "not cancellable"}, status_code=400)

    session = SchwabSession(_config(), session=Refusing({
        "/v1/oauth/token": FakeResponse({"access_token": "t", "expires_in": 1800}),
    }))
    session._access_token = "t"
    session._expires_at = 9_999_999_999
    brokerage = SchwabBrokerage(_config(), session=session)
    brokerage._account_hash = "HASH"

    # A reconciler works from a snapshot seconds old, so racing a fill is routine and must not
    # abort the rest of the pass.
    brokerage.cancel_order("123")


def test_schwab_declares_option_support() -> None:
    assert SchwabBrokerage.supports_options is True


def test_a_trigger_bracket_nests_its_legs_under_an_oco() -> None:
    """Listing the two legs as siblings is *accepted* by Schwab and is wrong.

    Verified against the live API: a TRIGGER carrying two flat children is stored as two
    independent SINGLE orders, so after the target fills the stop stays live and can sell a
    position that no longer exists. Only an OCO wrapper makes "never both" the venue's invariant.
    """
    brokerage, session = _order_brokerage()
    leg = lambda **kw: OrderRequest(
        symbol="QQQM  260220C00100000", action="sell", quantity=1, asset_type="option",
        time_in_force="gtc", extra={"position_intent": "sell_to_close"}, **kw,
    )

    brokerage.submit_order(OrderRequest(
        symbol="QQQM  260220C00100000", action="buy", quantity=1, order_type="limit",
        limit_price=1.15, asset_type="option", strategy="trigger",
        extra={"position_intent": "buy_to_open"},
        children=(leg(order_type="limit", limit_price=2.80), leg(order_type="stop", stop_price=1.50)),
    ))

    payload = _sent_payload(session)
    assert payload["orderStrategyType"] == "TRIGGER"
    assert payload["orderLegCollection"][0]["instruction"] == "BUY_TO_OPEN"
    children = payload["childOrderStrategies"]
    assert len(children) == 1, "the two legs must be one OCO, not two siblings"
    assert children[0]["orderStrategyType"] == "OCO"
    assert {c["orderType"] for c in children[0]["childOrderStrategies"]} == {"LIMIT", "STOP"}


def _orders_route(orders: list) -> dict:
    return {
        "/accounts/accountNumbers": FakeResponse([{"hashValue": "HASH"}]),
        "/orders": FakeResponse(orders),
    }


def test_working_means_every_status_that_is_not_finished() -> None:
    """A fresh order is PENDING_ACTIVATION and an untriggered leg is AWAITING_PARENT_ORDER.

    Filtering on the literal string "WORKING" finds neither, so a reconciler would conclude
    nothing is resting and submit the whole book a second time.
    """
    def order(order_id, status):
        return {
            "orderId": order_id, "orderType": "LIMIT", "price": 1.0, "status": status,
            "orderLegCollection": [{
                "instruction": "BUY_TO_OPEN", "quantity": 1,
                "instrument": {"symbol": "QQQM  260220C00100000", "assetType": "OPTION"},
            }],
        }

    session = _session(_orders_route([
        order(1, "PENDING_ACTIVATION"), order(2, "AWAITING_PARENT_ORDER"),
        order(3, "WORKING"), order(4, "QUEUED"),
        order(5, "FILLED"), order(6, "CANCELED"), order(7, "REJECTED"),
    ]))
    brokerage = SchwabBrokerage(_config(), session=session)

    live = {row["order_id"] for row in brokerage.get_orders("WORKING")}

    assert live == {"1", "2", "3", "4"}


def test_the_orders_request_carries_the_window_schwab_demands() -> None:
    # Schwab answers 400 without fromEnteredTime/toEnteredTime rather than defaulting.
    session = _session(_orders_route([]))
    brokerage = SchwabBrokerage(_config(), session=session)

    brokerage.get_orders("WORKING")

    params = next(c[2]["params"] for c in session._session.calls if "/orders" in c[1])
    assert params["fromEnteredTime"].endswith("Z") and params["toEnteredTime"].endswith("Z")
    assert "T" in params["fromEnteredTime"]


def _rejected_tree(order_id: str = "4242") -> dict:
    """The shape Schwab returns for a bracket refused after it was accepted for processing.

    Taken from a live rejection: the parent carries no reason at all, the offending leg carries
    the real one, and its sibling carries only a pointer back to it.
    """
    return {
        "orderId": order_id,
        "orderStrategyType": "TRIGGER",
        "status": "REJECTED",
        "childOrderStrategies": [{
            "orderStrategyType": "OCO",
            "status": "REJECTED",
            "childOrderStrategies": [
                {"orderId": "4244", "status": "REJECTED",
                 "statusDescription": "Order Rejected due to Order: 4243"},
                {"orderId": "4243", "status": "REJECTED",
                 "statusDescription": "Options orders cannot be entered in sub-penny increments."},
            ],
        }],
    }


def test_a_submission_schwab_rejects_is_not_reported_as_accepted() -> None:
    """201 means "accepted for processing", not "resting".

    Verified live: a bracket carrying a sub-penny stop was answered 201 with a Location header and
    then rejected outright moments later. Reporting the 201 as success claimed a position was
    protected when nothing had been placed at all.
    """
    session = _session({
        # Ordered before the bare "/orders" route so the read-back resolves to the order itself.
        "/orders/4242": FakeResponse(_rejected_tree()),
        "/orders": FakeResponse(None, status_code=201, headers={"Location": "https://x/orders/4242"}),
    })
    brokerage = SchwabBrokerage(_config(), session=session)
    brokerage._account_hash = "HASH"

    result = brokerage.submit_order(OrderRequest(
        symbol="QQQM  260220C00100000", action="buy", quantity=1, order_type="limit",
        limit_price=1.15, asset_type="option", extra={"position_intent": "buy_to_open"},
    ))

    assert result["status"] == "rejected"
    assert result["order_id"] == "4242"


def test_a_rejection_reason_is_read_off_the_leg_that_caused_it() -> None:
    """The parent of a rejected tree carries no ``statusDescription`` -- only the bad leg does."""
    from src.brokerages.schwab.brokerage import _rejection_reason

    reason = _rejection_reason(_rejected_tree())

    assert "sub-penny" in reason
    # Both descriptions are kept: the pointer is what identifies which leg was at fault.
    assert "Order Rejected due to Order: 4243" in reason


def test_an_order_still_pending_reads_as_accepted() -> None:
    """A rejection may not have landed yet, and a pending order must not be called a failure."""
    session = _session({
        "/orders/4242": FakeResponse({"orderId": "4242", "status": "PENDING_ACTIVATION"}),
        "/orders": FakeResponse(None, status_code=201, headers={"Location": "https://x/orders/4242"}),
    })
    brokerage = SchwabBrokerage(_config(), session=session)
    brokerage._account_hash = "HASH"

    result = brokerage.submit_order(OrderRequest(
        symbol="QQQM", action="buy", quantity=1, order_type="market",
    ))

    assert result["status"] == "accepted"


def test_a_replacement_never_carries_the_bracket_it_is_re_pricing() -> None:
    """Schwab answers 400 "Replacing order cannot have child orders." to a PUT that repeats the tree.

    Verified live. The entry of a trigger bracket is re-priced by sending its own leg alone;
    Schwab rebuilds the OCO underneath the replacement from the children's current prices.
    """
    brokerage, session = _order_brokerage()
    leg = lambda **kw: OrderRequest(
        symbol="QQQM  260220C00100000", action="sell", quantity=1, asset_type="option",
        time_in_force="gtc", extra={"position_intent": "sell_to_close"}, **kw,
    )

    brokerage.replace_order("4242", OrderRequest(
        symbol="QQQM  260220C00100000", action="buy", quantity=1, order_type="limit",
        limit_price=1.25, asset_type="option", strategy="trigger",
        extra={"position_intent": "buy_to_open"},
        children=(leg(order_type="limit", limit_price=2.80), leg(order_type="stop", stop_price=1.50)),
    ))

    payload = _sent_payload(session)
    assert payload["price"] == 1.25
    assert "childOrderStrategies" not in payload, "Schwab rejects a replacement carrying children"
    assert payload["orderStrategyType"] == "TRIGGER"


def test_a_position_reports_the_day_and_the_life_of_the_trade_separately() -> None:
    """Two questions, two fields.

    ``currentDayProfitLoss`` was being reported as ``unrealized_pl``, so a position held for
    months read as though it had been opened this morning, and the account page's "Open P/L"
    -- a sum of these -- tracked its own Day P/L so closely the two looked like one number
    printed twice.
    """
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/accounts/HASH": FakeResponse(
                {
                    "securitiesAccount": {
                        "positions": [
                            {
                                "instrument": {"symbol": "XSD", "assetType": "EQUITY"},
                                "longQuantity": 10,
                                "shortQuantity": 0,
                                "averagePrice": 100.0,
                                "marketValue": 1_500.0,
                                "longOpenProfitLoss": 500.0,
                                "currentDayProfitLoss": 20.0,
                            }
                        ]
                    }
                }
            ),
        }
    )
    brokerage = SchwabBrokerage(_config(), session=session)

    row = brokerage.get_position_details()[0]

    assert row["unrealized_pl"] == 500.0
    assert round(row["unrealized_plpc"], 4) == 0.5
    assert row["day_pl"] == 20.0
    # Against where the position started the session -- 1500 now, 20 of it earned today.
    assert round(row["day_pl_percent"], 6) == round(20.0 / 1_480.0, 6)


def _one_position(position: dict) -> SchwabBrokerage:
    return SchwabBrokerage(
        _config(),
        session=_session(
            {
                "/accounts/accountNumbers": FakeResponse(
                    [{"accountNumber": "123", "hashValue": "HASH"}]
                ),
                "/accounts/HASH": FakeResponse(
                    {"securitiesAccount": {"positions": [position]}}
                ),
            }
        ),
    )


def test_a_stale_current_day_cost_is_not_charged_against_the_day() -> None:
    """The live SCHD row, which read ``-647.90`` against the platform's ``$27.00``.

    ``currentDayCost`` is documented as the session's purchases but still carried the previous
    session's -- ``674.90`` here is exactly the prior day's two fills, 10 @ 33.74 and 10 @ 33.75.
    Those shares are inside ``previousSessionLongQuantity``, so they are already priced into the
    opening leg, and Schwab's day figure subtracts them a second time. The position did not
    trade today, so the whole of that cost is spurious.
    """
    brokerage = _one_position(
        {
            "instrument": {"symbol": "SCHD", "assetType": "COLLECTIVE_INVESTMENT"},
            "longQuantity": 270.0,
            "shortQuantity": 0.0,
            "previousSessionLongQuantity": 270.0,
            "averagePrice": 34.290556,
            "marketValue": 9_166.5,
            "longOpenProfitLoss": -91.95012,
            "currentDayProfitLoss": -647.9,
            "currentDayCost": 674.9,
        }
    )

    row = brokerage.get_position_details()[0]

    assert round(row["day_pl"], 2) == 27.0
    # 0.30% on the platform, against the -6.6% Schwab reports in its own percentage field.
    assert round(row["day_pl_percent"], 4) == round(27.0 / 9_139.5, 4)
    # The life of the trade is a separate reading and stays the broker's own.
    assert round(row["unrealized_pl"], 2) == -91.95


def test_shares_bought_today_are_not_counted_as_the_days_move() -> None:
    """The live GS row: 1.875 shares held overnight, 0.125 bought this session.

    Only the stale part of ``currentDayCost`` may be added back. The shares bought today were
    never in the opening leg, so their cost belongs out of the figure -- otherwise a purchase
    reads as a gain of its own size. The payload carries the quantity that moved but not what
    it paid, so the current mark prices it, leaving "what the overnight shares did today".
    """
    brokerage = _one_position(
        {
            "instrument": {"symbol": "GS", "assetType": "EQUITY"},
            "longQuantity": 2.0,
            "shortQuantity": 0.0,
            "previousSessionLongQuantity": 1.875,
            "averagePrice": 1_014.6452375,
            "marketValue": 1_903.52,
            "longOpenProfitLoss": -125.770475,
            "currentDayProfitLoss": -211.5275,
            "currentDayCost": 354.16,
        }
    )

    row = brokerage.get_position_details()[0]

    # 142.6325 once the stale cost is restored, less 0.125 shares at the 951.76 mark.
    assert round(row["day_pl"], 2) == 23.66


def test_a_position_that_did_not_trade_today_keeps_the_brokers_own_figure() -> None:
    """No ``currentDayCost`` means nothing to correct, and the broker's number stands."""
    brokerage = _one_position(
        {
            "instrument": {"symbol": "XBI", "assetType": "COLLECTIVE_INVESTMENT"},
            "longQuantity": 3.0,
            "shortQuantity": 0.0,
            "previousSessionLongQuantity": 3.0,
            "averagePrice": 150.0,
            "marketValue": 514.48,
            "longOpenProfitLoss": 64.48,
            "currentDayProfitLoss": 12.38,
            "currentDayCost": 0.0,
        }
    )

    assert brokerage.get_position_details()[0]["day_pl"] == 12.38


def test_a_position_sold_into_today_is_left_to_the_broker() -> None:
    """A proceeds term has no counterpart in this payload, so a sale invents nothing.

    Reducing a position is the one shape the live account could not exercise, and guessing at
    it would be worse than reporting what Schwab says.
    """
    brokerage = _one_position(
        {
            "instrument": {"symbol": "AIQ", "assetType": "EQUITY"},
            "longQuantity": 5.0,
            "shortQuantity": 0.0,
            "previousSessionLongQuantity": 8.0,
            "averagePrice": 60.0,
            "marketValue": 320.0,
            "longOpenProfitLoss": 20.0,
            "currentDayProfitLoss": 20.0,
            "currentDayCost": 100.0,
        }
    )

    assert brokerage.get_position_details()[0]["day_pl"] == 20.0


def test_one_account_read_serves_balances_and_positions_together() -> None:
    """``fields=positions`` is a superset of the bare body, so asking for both was two calls.

    The account page reads state then positions. That was five round trips per account, of which
    one carried data -- and the sidebar does it for every account on the page.
    """
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/accounts/HASH": FakeResponse(
                {
                    "securitiesAccount": {
                        "currentBalances": {"liquidationValue": 5_000.0, "cashBalance": 1_200.0},
                        "initialBalances": {"liquidationValue": 4_900.0},
                        "positions": [
                            {
                                "instrument": {"symbol": "XSD", "assetType": "EQUITY"},
                                "longQuantity": 10,
                                "shortQuantity": 0,
                                "averagePrice": 100.0,
                                "marketValue": 1_500.0,
                                "currentDayProfitLoss": 20.0,
                            }
                        ],
                    }
                }
            ),
            "/marketdata/v1/markets": FakeResponse({"equity": {}}),
        }
    )
    brokerage = SchwabBrokerage(_config(), session=session)

    state = brokerage.get_account_state()
    rows = brokerage.get_position_details()

    assert state["equity"] == 5_000.0
    assert state["last_equity"] == 4_900.0
    assert rows[0]["symbol"] == "XSD"
    account_reads = [call for call in session._session.calls if call[1].endswith("/accounts/HASH")]
    assert len(account_reads) == 1, "balances and positions must come from one read"
    # The hash is resolved once and the token fetched once, however many reads follow.
    assert sum(1 for _, url, _ in session._session.calls if url.endswith("accountNumbers")) == 1
    assert sum(1 for _, url, _ in session._session.calls if "oauth/token" in url) == 1


def test_market_hours_are_not_re_read_for_every_account_call() -> None:
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/accounts/HASH": FakeResponse({"securitiesAccount": {"currentBalances": {}}}),
            "/marketdata/v1/markets": FakeResponse({"equity": {}}),
        }
    )
    brokerage = SchwabBrokerage(_config(), session=session)

    brokerage.is_market_open()
    brokerage.is_market_open()

    assert sum(1 for _, url, _ in session._session.calls if "markets" in url) == 1


def test_the_order_window_is_the_callers_to_narrow() -> None:
    """A caller that knows what bounds its orders says so; the rest get the wide default."""
    routes = {
        "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
        "/orders": FakeResponse([]),
    }
    session = _session(routes)
    brokerage = SchwabBrokerage(_config(), session=session)

    brokerage.get_orders("WORKING", days=21)
    brokerage.get_orders("WORKING")

    windows = [
        pd.Timestamp(call[2]["params"]["toEnteredTime"])
        - pd.Timestamp(call[2]["params"]["fromEnteredTime"])
        for call in session._session.calls if call[1].endswith("/orders")
    ]
    assert round(windows[0].total_seconds() / 86_400) == 21
    assert round(windows[1].total_seconds() / 86_400) == 60


def test_an_option_position_is_measured_against_the_contract_multiplier() -> None:
    """``marketValue`` carries the 100x, ``averagePrice`` does not.

    Comparing them directly answered a 1% move as a 99% loss, which is exactly the shape of
    wrongness that looks like a real number.
    """
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/accounts/HASH": FakeResponse(
                {
                    "securitiesAccount": {
                        "positions": [
                            {
                                "instrument": {"symbol": "AIQ   260116C00045000", "assetType": "OPTION"},
                                "longQuantity": 2,
                                "shortQuantity": 0,
                                "averagePrice": 8.00,
                                "marketValue": 1_800.0,
                                "currentDayProfitLoss": 50.0,
                            }
                        ]
                    }
                }
            ),
        }
    )
    brokerage = SchwabBrokerage(_config(), session=session)

    row = brokerage.get_position_details()[0]

    assert row["current_price"] == 9.0
    # Schwab sent no open figure for this one, so it is derived: $1,800 now against $1,600 paid.
    assert row["unrealized_pl"] == 200.0
    assert round(row["unrealized_plpc"], 4) == 0.125


def test_a_short_position_reads_its_open_pl_from_schwabs_short_field() -> None:
    """Schwab sends one field or the other, never both, and a missing one is not a flat zero."""
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/accounts/HASH": FakeResponse(
                {
                    "securitiesAccount": {
                        "positions": [
                            {
                                "instrument": {"symbol": "XBI", "assetType": "EQUITY"},
                                "longQuantity": 0,
                                "shortQuantity": 5,
                                "averagePrice": 90.0,
                                "marketValue": -400.0,
                                "shortOpenProfitLoss": 50.0,
                                "currentDayProfitLoss": -10.0,
                            }
                        ]
                    }
                }
            ),
        }
    )
    brokerage = SchwabBrokerage(_config(), session=session)

    row = brokerage.get_position_details()[0]

    assert row["qty"] == -5.0
    assert row["unrealized_pl"] == 50.0
    assert row["day_pl"] == -10.0


def test_fills_are_read_from_the_transactions_feed() -> None:
    """A trade's ``amount`` is signed, and that is what says whether the fill opened or closed.

    Not the order's verb: Schwab's own instruction encodes open/close separately, and the
    transaction record is the one that survives after the order is gone.
    """
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/transactions": FakeResponse(
                [
                    {
                        "type": "TRADE",
                        "tradeDate": "2026-09-09T16:23:09+0000",
                        "transferItems": [
                            {
                                "instrument": {"symbol": "XSD", "assetType": "EQUITY"},
                                "amount": 10.0,
                                "price": 100.0,
                            },
                            # A fee leg: no instrument, no price, and not a trade.
                            {"feeType": "COMMISSION", "amount": -0.65},
                        ],
                    },
                    {
                        "type": "TRADE",
                        "tradeDate": "2026-09-10T16:00:39+0000",
                        "transferItems": [
                            {
                                "instrument": {"symbol": "XSD", "assetType": "EQUITY"},
                                "amount": -10.0,
                                "price": 110.0,
                            }
                        ],
                    },
                ]
            ),
        }
    )
    brokerage = SchwabBrokerage(_config(), session=session)

    fills = brokerage.get_fills()

    assert [(f["symbol"], f["action"], f["quantity"], f["price"]) for f in fills] == [
        ("XSD", "buy", 10.0, 100.0),
        ("XSD", "sell", 10.0, 110.0),
    ]
    assert all(fill["multiplier"] == 1.0 for fill in fills)


def test_an_option_fill_carries_the_hundred_share_multiplier() -> None:
    session = _session(
        {
            "/accounts/accountNumbers": FakeResponse([{"accountNumber": "123", "hashValue": "HASH"}]),
            "/transactions": FakeResponse(
                [
                    {
                        "type": "TRADE",
                        "tradeDate": "2026-09-09T16:23:09+0000",
                        "transferItems": [
                            {
                                "instrument": {"symbol": "AIQ   260116C00045000", "assetType": "OPTION"},
                                "amount": 2.0,
                                "price": 8.61,
                            }
                        ],
                    }
                ]
            ),
        }
    )
    brokerage = SchwabBrokerage(_config(), session=session)

    assert brokerage.get_fills()[0]["multiplier"] == 100.0
