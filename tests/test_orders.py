from __future__ import annotations
from typing import Any

from src.core import orders as orders
from src.core.interfaces import OrderRequest


class FakeBrokerage:
    def __init__(self):
        self.submitted = []

    def submit_order(self, request: OrderRequest) -> dict[str, Any]:
        self.submitted.append((request.symbol, request.action, request.quantity))
        return {"order_id": "order-1"}

    def validate_short_sale_feasibility(
        self, symbol: str, quantity: int, target_shares: int, latest_price: float
    ) -> dict[str, Any]:
        return {"shortable": True, "reason": "ok"}


def test_sync_positions_sells_positions_missing_from_targets() -> None:
    brokerage = FakeBrokerage()
    result = orders.sync_positions_to_targets(
        brokerage=brokerage,
        latest_prices={"SPY": 100.0},
        current_positions={"SPY": 3},
        target_weights={},
        equity=1_000.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    assert brokerage.submitted == [("SPY", "sell", 3)]
    assert result[0]["action"] == "sell"
    assert result[0]["target_weight"] == 0.0


def test_sync_positions_submits_sells_before_buys() -> None:
    brokerage = FakeBrokerage()
    result = orders.sync_positions_to_targets(
        brokerage=brokerage,
        latest_prices={"AAA": 100.0, "BBB": 100.0},
        current_positions={"AAA": 2},
        target_weights={"AAA": 0.0, "BBB": 0.5},
        equity=1_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    assert brokerage.submitted == [("AAA", "sell", 2), ("BBB", "buy", 5)]
    assert [order["target_weight"] for order in result] == [0.0, 0.5]


def test_sync_positions_executes_negative_target_weight_as_short() -> None:
    brokerage = FakeBrokerage()
    result = orders.sync_positions_to_targets(
        brokerage=brokerage,
        latest_prices={"AAA": 100.0},
        current_positions={},
        target_weights={"AAA": -0.5},
        equity=1_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    assert brokerage.submitted == [("AAA", "sell", 5)]
    assert result[0]["target_weight"] == -0.5
    assert result[0]["target_shares"] == -5
    assert result[0]["position_intent"] == "sell_short"


def test_sync_positions_skips_infeasible_short_sale() -> None:
    brokerage = FakeBrokerage()
    brokerage.validate_short_sale_feasibility = lambda *a, **kw: {"shortable": False, "reason": "asset is not shortable"}

    result = orders.sync_positions_to_targets(
        brokerage=brokerage,
        latest_prices={"AAA": 100.0},
        current_positions={},
        target_weights={"AAA": -0.5},
        equity=1_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    assert brokerage.submitted == []
    assert result[0]["action"] == "skip"
    assert result[0]["approval_status"] == "short_sale_not_feasible"
    assert result[0]["reason"] == "asset is not shortable"


def test_sync_positions_skips_when_approval_is_denied(monkeypatch) -> None:
    monkeypatch.setattr(orders, "request_trade_approval", lambda *a, **kw: False)

    brokerage = FakeBrokerage()
    result = orders.sync_positions_to_targets(
        brokerage=brokerage,
        latest_prices={"AAA": 100.0},
        current_positions={},
        target_weights={"AAA": 0.5},
        equity=1_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
        require_approval=True,
    )

    assert brokerage.submitted == []
    assert result[0]["action"] == "skip"
    assert result[0]["approval_status"] == "not_approved"


def test_sync_positions_submits_after_approval(monkeypatch) -> None:
    monkeypatch.setattr(orders, "request_trade_approval", lambda *a, **kw: True)

    brokerage = FakeBrokerage()
    result = orders.sync_positions_to_targets(
        brokerage=brokerage,
        latest_prices={"AAA": 100.0},
        current_positions={},
        target_weights={"AAA": 0.5},
        equity=1_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
        require_approval=True,
    )

    assert brokerage.submitted == [("AAA", "buy", 5)]
    assert result[0]["order_id"] == "order-1"
