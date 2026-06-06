from __future__ import annotations

from src.core import orders as orders


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


def test_sync_positions_executes_negative_target_weight_as_short(monkeypatch) -> None:
    submitted = []

    def fake_submit(_client, symbol, side, qty):
        submitted.append((symbol, side, qty))
        return FakeOrder()

    monkeypatch.setattr(orders, "submit_market_order", fake_submit)
    monkeypatch.setattr(
        orders,
        "validate_short_sale_feasibility",
        lambda *_args, **_kwargs: {"shortable": True, "reason": "ok"},
    )

    result = orders.sync_positions_to_targets(
        trading_client=object(),
        latest_prices={"AAA": 100.0},
        current_positions={},
        target_weights={"AAA": -0.5},
        equity=1_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    assert submitted == [("AAA", "sell", 5)]
    assert result[0]["target_weight"] == -0.5
    assert result[0]["target_shares"] == -5
    assert result[0]["position_intent"] == "sell_short"


def test_sync_positions_skips_infeasible_short_sale(monkeypatch) -> None:
    submitted = []

    monkeypatch.setattr(orders, "submit_market_order", lambda *_args, **_kwargs: submitted.append(_args))
    monkeypatch.setattr(
        orders,
        "validate_short_sale_feasibility",
        lambda *_args, **_kwargs: {"shortable": False, "reason": "asset is not shortable"},
    )

    result = orders.sync_positions_to_targets(
        trading_client=object(),
        latest_prices={"AAA": 100.0},
        current_positions={},
        target_weights={"AAA": -0.5},
        equity=1_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    assert submitted == []
    assert result[0]["action"] == "skip"
    assert result[0]["approval_status"] == "short_sale_not_feasible"
    assert result[0]["reason"] == "asset is not shortable"


def test_sync_positions_skips_when_approval_is_denied(monkeypatch) -> None:
    submitted = []

    monkeypatch.setattr(orders, "submit_market_order", lambda *_args, **_kwargs: submitted.append(_args))
    monkeypatch.setattr(orders, "request_trade_approval", lambda *_args, **_kwargs: False)

    result = orders.sync_positions_to_targets(
        trading_client=object(),
        latest_prices={"AAA": 100.0},
        current_positions={},
        target_weights={"AAA": 0.5},
        equity=1_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
        require_approval=True,
    )

    assert submitted == []
    assert result[0]["action"] == "skip"
    assert result[0]["approval_status"] == "not_approved"


def test_sync_positions_submits_after_approval(monkeypatch) -> None:
    submitted = []

    def fake_submit(_client, symbol, side, qty):
        submitted.append((symbol, side, qty))
        return FakeOrder()

    monkeypatch.setattr(orders, "submit_market_order", fake_submit)
    monkeypatch.setattr(orders, "request_trade_approval", lambda *_args, **_kwargs: True)

    result = orders.sync_positions_to_targets(
        trading_client=object(),
        latest_prices={"AAA": 100.0},
        current_positions={},
        target_weights={"AAA": 0.5},
        equity=1_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
        require_approval=True,
    )

    assert submitted == [("AAA", "buy", 5)]
    assert result[0]["order_id"] == "order-1"
