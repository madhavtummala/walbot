from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.brokerages.alpaca_client import (
    create_trading_client,
    get_historical_daily_bars,
    get_historical_intraday_bars,
    get_latest_price,
    is_market_open,
    submit_option_limit_order,
    validate_short_sale_feasibility,
)
from src.core.config import Config


class FakeDataClient:
    def __init__(self) -> None:
        self.requests = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        return {"SPY": []}


def test_get_latest_price_builds_current_stock_bars_request() -> None:
    client = FakeDataClient()

    try:
        get_latest_price("SPY", client, end_date=datetime(2026, 5, 16, tzinfo=timezone.utc))
    except RuntimeError:
        pass

    assert client.requests[0].symbol_or_symbols == ["SPY"]
    assert client.requests[0].feed.value == "iex"


def test_get_historical_daily_bars_builds_current_stock_bars_request() -> None:
    client = FakeDataClient()

    get_historical_daily_bars(
        ["SPY"],
        lookback_days=10,
        extra_buffer_days=0,
        data_client=client,
        end_date=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )

    assert client.requests[0].symbol_or_symbols == ["SPY"]
    assert client.requests[0].feed.value == "iex"


def test_get_historical_intraday_bars_builds_30_minute_request() -> None:
    client = FakeDataClient()

    get_historical_intraday_bars(
        ["SPY"],
        lookback_bars=13,
        data_client=client,
        end_date=datetime(2026, 5, 16, tzinfo=timezone.utc),
    )

    assert client.requests[0].symbol_or_symbols == ["SPY"]
    assert str(client.requests[0].timeframe) == "30Min"
    assert client.requests[0].feed.value == "iex"


def test_get_historical_intraday_bars_uses_trading_session_window() -> None:
    client = FakeDataClient()
    end = datetime(2026, 6, 1, 15, 0, tzinfo=timezone.utc)

    get_historical_intraday_bars(
        ["SPY"],
        lookback_bars=27,
        data_client=client,
        end_date=end,
    )

    assert end.replace(tzinfo=None) - client.requests[0].start >= timedelta(days=7)


def test_create_trading_client_uses_configured_endpoint_without_paper_mode(monkeypatch) -> None:
    captured = {}

    class FakeTradingClient:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("src.brokerages.alpaca_client.TradingClient", FakeTradingClient)

    create_trading_client(
        Config(
            alpaca_api_key="key",
            alpaca_api_secret="secret",
            alpaca_base_url="https://paper-api.alpaca.markets",
        )
    )

    assert captured["paper"] is False
    assert captured["url_override"] == "https://paper-api.alpaca.markets"


def test_is_market_open_reads_alpaca_clock() -> None:
    class FakeTradingClient:
        def get_clock(self):
            return type("Clock", (), {"is_open": True})()

    assert is_market_open(FakeTradingClient())


def test_is_market_open_fails_closed() -> None:
    class FakeTradingClient:
        def get_clock(self):
            raise RuntimeError("clock unavailable")

    assert not is_market_open(FakeTradingClient())


def test_validate_short_sale_feasibility_accepts_shortable_asset() -> None:
    class FakeTradingClient:
        def get_account(self):
            return type(
                "Account",
                (),
                {
                    "shorting_enabled": True,
                    "trading_blocked": False,
                    "account_blocked": False,
                    "buying_power": "10000",
                },
            )()

        def get_asset(self, symbol):
            assert symbol == "SPY"
            return type(
                "Asset",
                (),
                {"tradable": True, "shortable": True, "easy_to_borrow": True, "marginable": True},
            )()

    result = validate_short_sale_feasibility(FakeTradingClient(), "SPY", quantity=5, target_shares=-5, latest_price=100.0)

    assert result["shortable"] is True
    assert result["short_notional_after"] == 500.0


def test_validate_short_sale_feasibility_rejects_unshortable_asset() -> None:
    class FakeTradingClient:
        def get_account(self):
            return type(
                "Account",
                (),
                {
                    "shorting_enabled": True,
                    "trading_blocked": False,
                    "account_blocked": False,
                    "buying_power": "10000",
                },
            )()

        def get_asset(self, _symbol):
            return type(
                "Asset",
                (),
                {"tradable": True, "shortable": False, "easy_to_borrow": True, "marginable": True},
            )()

    result = validate_short_sale_feasibility(FakeTradingClient(), "XYZ", quantity=5, target_shares=-5, latest_price=100.0)

    assert result["shortable"] is False
    assert "shortable" in str(result["reason"])


def test_validate_short_sale_feasibility_rejects_account_without_shorting() -> None:
    class FakeTradingClient:
        def get_account(self):
            return type(
                "Account",
                (),
                {
                    "shorting_enabled": False,
                    "trading_blocked": False,
                    "account_blocked": False,
                    "buying_power": "10000",
                },
            )()

    result = validate_short_sale_feasibility(FakeTradingClient(), "SPY", quantity=5, target_shares=-5, latest_price=100.0)

    assert result["shortable"] is False
    assert result["reason"] == "account shorting is not enabled"


def test_submit_option_limit_order_uses_buy_to_open_intent() -> None:
    captured = {}

    class FakeTradingClient:
        def submit_order(self, order_data):
            captured["order"] = order_data
            return type("Order", (), {"id": "order-1"})()

    submit_option_limit_order(FakeTradingClient(), "SPY260116C00100000", "buy", 1, 2.15)

    order = captured["order"]
    assert order.symbol == "SPY260116C00100000"
    assert order.limit_price == 2.15
    assert order.position_intent.value == "buy_to_open"
