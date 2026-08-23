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
