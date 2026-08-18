from __future__ import annotations

from datetime import date, timedelta

from src.core.config import Config
from src.algorithms.options import swing as options_trader


class Contract:
    def __init__(self, symbol: str, strike: float, delta: float, open_interest: int = 500) -> None:
        self.symbol = symbol
        self.type = "call"
        self.expiration_date = date.today() + timedelta(days=45)
        self.strike_price = strike
        self.delta = delta
        self.open_interest = open_interest


class Quote:
    def __init__(self, bid: float, ask: float) -> None:
        self.bid_price = bid
        self.ask_price = ask


def test_select_option_candidate_filters_and_ranks(monkeypatch) -> None:
    config = Config(
        options_swing_max_premium=500.0,
        options_swing_min_open_interest=100,
        options_swing_max_contracts=2,
    )
    contracts = [
        Contract("SPY_BAD", 120.0, 0.8),
        Contract("SPY_GOOD", 101.0, 0.48),
    ]

    monkeypatch.setattr(options_trader, "get_option_contracts", lambda *args, **kwargs: contracts)
    monkeypatch.setattr(
        options_trader,
        "get_option_latest_quotes",
        lambda _client, symbols: {
            "SPY_BAD": Quote(1.0, 1.2),
            "SPY_GOOD": Quote(2.0, 2.2),
        },
    )

    candidate = options_trader.select_option_candidate(
        config,
        trading_client=object(),
        option_data_client=object(),
        underlying="SPY",
        side="LONG",
        underlying_price=100.0,
    )

    assert candidate is not None
    assert candidate.contract_symbol == "SPY_GOOD"
    assert candidate.quantity == 2
    assert candidate.limit_price == 2.1


def test_run_options_once_submits_buy_to_open(monkeypatch) -> None:
    submitted = []

    monkeypatch.setattr(
        options_trader,
        "load_controls",
        lambda: {
            "options_trading_enabled": True,
            "options_strategy": "options_swing_rally_rotation",
            "options_trading_account_id": "paper-options",
            "trading_account_id": "paper",
        },
    )
    monkeypatch.setattr(options_trader, "get_config", lambda account_id=None, strategy_id=None: Config(account_id=account_id or "paper"))
    monkeypatch.setattr(options_trader, "create_trading_client", lambda config: object())
    monkeypatch.setattr(options_trader, "is_market_open", lambda client: True)
    monkeypatch.setattr(options_trader, "create_data_client", lambda config: object())
    monkeypatch.setattr(options_trader, "create_option_data_client", lambda config: object())
    monkeypatch.setattr(options_trader, "get_account_equity", lambda client: 10_000.0)
    monkeypatch.setattr(options_trader, "get_positions", lambda client: {})
    monkeypatch.setattr(options_trader, "rally_rotation_option_signals", lambda config, data_client=None: [{"symbol": "SPY", "side": "LONG", "score": 0.4}])
    monkeypatch.setattr(options_trader, "get_latest_price", lambda symbol, data_client, data_feed=None: 100.0)
    monkeypatch.setattr(
        options_trader,
        "select_option_candidate",
        lambda *args, **kwargs: options_trader.OptionCandidate(
            underlying="SPY",
            side="LONG",
            contract_symbol="SPY260116C00100000",
            contract_type="call",
            expiration_date="2026-01-16",
            strike_price=100.0,
            delta=0.5,
            bid=2.0,
            ask=2.2,
            limit_price=2.1,
            quantity=1,
            estimated_premium=210.0,
            score=0.4,
        ),
    )

    def fake_submit(_client, symbol, side, qty, limit_price, position_intent):
        submitted.append((symbol, side, qty, limit_price, position_intent))
        return type("Order", (), {"id": "option-order-1"})()

    monkeypatch.setattr(options_trader, "submit_option_limit_order", fake_submit)

    results = options_trader.run_options_once()

    assert submitted == [("SPY260116C00100000", "buy", 1, 2.1, "buy_to_open")]
    assert results[0]["order_id"] == "option-order-1"


def test_run_options_once_exits_when_market_clock_is_closed(monkeypatch) -> None:
    called = {"data_client": False}

    monkeypatch.setattr(
        options_trader,
        "load_controls",
        lambda: {
            "options_trading_enabled": True,
            "options_strategy": "options_swing_rally_rotation",
            "options_trading_account_id": "paper-options",
            "trading_account_id": "paper",
        },
    )
    monkeypatch.setattr(options_trader, "get_config", lambda account_id=None, strategy_id=None: Config(account_id=account_id or "paper"))
    monkeypatch.setattr(options_trader, "create_trading_client", lambda config: object())
    monkeypatch.setattr(options_trader, "is_market_open", lambda client: False)
    monkeypatch.setattr(options_trader, "create_data_client", lambda config: called.__setitem__("data_client", True))

    assert options_trader.run_options_once() == []
    assert not called["data_client"]
