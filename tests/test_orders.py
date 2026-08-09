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


def _sync(brokerage, latest_prices, current_positions, target_weights, equity, **kwargs):
    """Plan then submit, the pairing ``pipeline.place_orders`` performs in production."""
    submit_keys = {"require_approval", "approval_timeout_seconds", "approval_poll_seconds"}
    planned = orders.plan_position_orders(
        latest_prices,
        current_positions,
        target_weights,
        equity,
        supports_fractional_shares=getattr(brokerage, "supports_fractional_shares", False),
        **{k: v for k, v in kwargs.items() if k not in submit_keys},
    )
    return orders.submit_planned_orders(
        brokerage, planned, **{k: v for k, v in kwargs.items() if k in submit_keys}
    )


def test_sync_positions_sells_positions_missing_from_targets() -> None:
    brokerage = FakeBrokerage()
    result = _sync(
        brokerage,
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
    result = _sync(
        brokerage,
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
    result = _sync(
        brokerage,
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

    result = _sync(
        brokerage,
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
    result = _sync(
        brokerage,
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
    result = _sync(
        brokerage,
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


class FractionalBrokerage(FakeBrokerage):
    supports_fractional_shares = True


def test_plan_orders_size_whole_shares_by_default() -> None:
    planned = orders.plan_position_orders(
        latest_prices={"SPY": 140.0},
        current_positions={},
        target_weights={"SPY": 0.3},
        equity=10_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    # 3000 / 140 = 21.428..., truncated to whole shares.
    assert planned[0]["quantity"] == 21
    assert isinstance(planned[0]["quantity"], int)


def test_plan_orders_size_fractional_shares_when_supported() -> None:
    planned = orders.plan_position_orders(
        latest_prices={"SPY": 140.0},
        current_positions={},
        target_weights={"SPY": 0.3},
        equity=10_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
        supports_fractional_shares=True,
    )

    # 3000 / 140 = 21.428..., kept to 2dp and truncated so it never overshoots 3000 dollars.
    assert planned[0]["quantity"] == 21.42
    assert planned[0]["trade_dollars"] <= 3_000.0


def test_short_targets_stay_whole_shares_even_when_fractional_supported() -> None:
    planned = orders.plan_position_orders(
        latest_prices={"SPY": 140.0},
        current_positions={},
        target_weights={"SPY": -0.3},
        equity=10_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
        supports_fractional_shares=True,
    )

    assert planned[0]["target_shares"] == -21
    assert float(planned[0]["quantity"]).is_integer()


def test_sync_positions_uses_brokerage_fractional_capability() -> None:
    brokerage = FractionalBrokerage()
    _sync(
        brokerage,
        latest_prices={"SPY": 140.0},
        current_positions={},
        target_weights={"SPY": 0.3},
        equity=10_000.0,
        cash_buffer=0.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    symbol, side, quantity = brokerage.submitted[0]
    assert (symbol, side) == ("SPY", "buy")
    assert not float(quantity).is_integer()


def test_sync_positions_closes_fractional_holding_exactly() -> None:
    brokerage = FractionalBrokerage()
    _sync(
        brokerage,
        latest_prices={"SPY": 100.0},
        current_positions={"SPY": 2.5},
        target_weights={},
        equity=1_000.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    assert brokerage.submitted == [("SPY", "sell", 2.5)]


class RejectingBrokerage(FakeBrokerage):
    """Rejects orders for ``reject_symbol``, the way Alpaca rejects shares held for an open order."""

    def __init__(self, reject_symbol: str):
        super().__init__()
        self.reject_symbol = reject_symbol

    def submit_order(self, request: OrderRequest) -> dict[str, Any]:
        if request.symbol == self.reject_symbol:
            raise RuntimeError("insufficient qty available for order")
        return super().submit_order(request)


def test_rejected_order_does_not_abort_the_rest_of_the_batch() -> None:
    brokerage = RejectingBrokerage("AAA")
    results = _sync(
        brokerage,
        latest_prices={"AAA": 100.0, "BBB": 100.0},
        current_positions={"AAA": 5, "BBB": 5},
        target_weights={},
        equity=1_000.0,
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    by_symbol = {row["symbol"]: row for row in results}
    assert by_symbol["AAA"]["status"] == "rejected"
    assert "insufficient qty" in by_symbol["AAA"]["reason"]
    assert by_symbol["BBB"]["status"] == "submitted"
    assert brokerage.submitted == [("BBB", "sell", 5)]
