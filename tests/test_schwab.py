from __future__ import annotations

import json
from typing import Any

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
