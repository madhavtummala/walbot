from __future__ import annotations

from src import orders


class FakeOrder:
    id = "order-1"


def test_sync_positions_sells_positions_missing_from_targets(monkeypatch) -> None:
    submitted = []

    def fake_submit(_client, symbol, side, qty):
        submitted.append((symbol, side, qty))
        return FakeOrder()

    monkeypatch.setattr(orders, "submit_market_order", fake_submit)

    result = orders.sync_positions_to_targets(
        trading_client=object(),
        latest_prices={"SPY": 100.0},
        current_positions={"SPY": 3},
        target_weights={},
        equity=1_000.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    assert submitted == [("SPY", "sell", 3)]
    assert result[0]["action"] == "sell"
    assert result[0]["target_weight"] == 0.0


def test_sync_positions_submits_sells_before_buys(monkeypatch) -> None:
    submitted = []

    def fake_submit(_client, symbol, side, qty):
        submitted.append((symbol, side, qty))
        return FakeOrder()

    monkeypatch.setattr(orders, "submit_market_order", fake_submit)

    result = orders.sync_positions_to_targets(
        trading_client=object(),
        latest_prices={"AAA": 100.0, "BBB": 100.0},
        current_positions={"AAA": 2},
        target_weights={"AAA": 0.0, "BBB": 0.5},
        equity=1_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    assert submitted == [("AAA", "sell", 2), ("BBB", "buy", 5)]
    assert [order["target_weight"] for order in result] == [0.0, 0.5]
