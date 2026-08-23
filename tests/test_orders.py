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
    planned = orders.plan_position_orders(
        latest_prices,
        current_positions,
        target_weights,
        equity,
        supports_fractional_shares=getattr(brokerage, "supports_fractional_shares", False),
        **kwargs,
    )
    return orders.submit_planned_orders(brokerage, planned)


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
        min_trade_dollars=1.0,
        rebalance_threshold=0.0,
    )

    assert brokerage.submitted == []
    assert result[0]["action"] == "skip"
    assert result[0]["approval_status"] == "short_sale_not_feasible"
    assert result[0]["reason"] == "asset is not shortable"


class FractionalBrokerage(FakeBrokerage):
    supports_fractional_shares = True


def test_plan_orders_size_whole_shares_by_default() -> None:
    planned = orders.plan_position_orders(
        latest_prices={"SPY": 140.0},
        current_positions={},
        target_weights={"SPY": 0.3},
        equity=10_000.0,
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


# --------------------------------------------------------------------------------------
# Order funding: fitting a planned batch to the money that will be there to pay for it.
# --------------------------------------------------------------------------------------


def _buy(symbol: str, quantity: float, price: float) -> dict:
    return {
        "symbol": symbol,
        "action": "buy",
        "quantity": quantity,
        "target_weight": 0.1,
        "target_shares": quantity,
        "current_shares": 0,
        "trade_dollars": quantity * price,
        "latest_price": price,
        "position_intent": "buy_to_open",
        "opens_short": False,
        "short_shares_after": 0,
    }


def _sell(symbol: str, quantity: float, price: float, opens_short: bool = False) -> dict:
    return {
        **_buy(symbol, quantity, price),
        "action": "sell",
        "target_shares": -quantity if opens_short else 0,
        "current_shares": 0 if opens_short else quantity,
        "position_intent": "sell_short" if opens_short else "sell_to_close",
        "opens_short": opens_short,
    }


def test_funding_leaves_an_affordable_batch_alone() -> None:
    planned = [_buy("SPY", 10, 100.0), _buy("QQQ", 5, 200.0)]

    fundable, unfunded, report = orders.fund_planned_orders(planned, buying_power=5_000.0)

    assert unfunded == []
    assert [order["quantity"] for order in fundable] == [10, 5]
    assert {order["funding_status"] for order in fundable} == {"full"}
    assert report["shortfall"] == 0.0


def test_funding_counts_sale_proceeds_toward_the_buys() -> None:
    """Sells are planned first precisely so their proceeds pay for the buys."""
    planned = [_sell("GLD", 10, 100.0), _buy("SPY", 10, 100.0)]

    fundable, unfunded, report = orders.fund_planned_orders(planned, buying_power=200.0)

    assert unfunded == []
    assert report["sale_proceeds"] == 1_000.0
    assert [order["quantity"] for order in fundable] == [10, 10]


def test_funding_ignores_short_sale_proceeds() -> None:
    """A short sale brings a margin requirement, not spendable cash."""
    planned = [_sell("GLD", 10, 100.0, opens_short=True), _buy("SPY", 10, 100.0)]

    fundable, _, report = orders.fund_planned_orders(planned, buying_power=200.0)

    assert report["sale_proceeds"] == 0.0
    # Only the $200 of buying power funds the buy, so it lands at 2 shares rather than 10.
    buy = [order for order in fundable if order["action"] == "buy"][0]
    assert buy["quantity"] == 2 and buy["funding_status"] == "reduced"


def test_funding_reserve_is_held_back_from_buying_power() -> None:
    planned = [_buy("SPY", 10, 100.0)]

    _, unfunded, report = orders.fund_planned_orders(
        planned, buying_power=1_000.0, reserve=200.0, min_trade_dollars=1.0
    )

    assert unfunded == []
    assert report["budget"] == 800.0


def test_funding_shrinks_the_logged_rejections_to_fit() -> None:
    """The four legs a live run had refused, at the sizes and balances it reported.

    Each was submitted at its full target against a balance that could not cover it, and the
    broker refused them one at a time. Fitted to the batch, they go out smaller instead.
    """
    planned = [
        _buy("XSD", 3.68, 1082.47 / 3.68),
        _buy("SLV", 86.63, 3626.33 / 86.63),
        _buy("VGK", 18.95, 1533.24 / 18.95),
    ]

    fundable, unfunded, report = orders.fund_planned_orders(
        planned,
        buying_power=802.74 + 2_734.40 + 1_429.75,
        supports_fractional_shares=True,
        min_trade_dollars=1.0,
    )

    assert unfunded == []
    assert {order["funding_status"] for order in fundable} == {"reduced"}
    # Every leg keeps its share of the portfolio rather than the first ones taking it all.
    assert report["funded_notional"] <= report["budget"] + orders.FUNDING_EPSILON
    assert sorted(report["reduced"]) == ["SLV", "VGK", "XSD"]
    for order in fundable:
        assert order["quantity"] < order["requested_quantity"]
        assert "Reduced" in order["reason"]


def test_pro_rata_keeps_the_shape_of_the_intended_portfolio() -> None:
    planned = [_buy("AAA", 100, 10.0), _buy("BBB", 50, 10.0)]

    fundable, _, _ = orders.fund_planned_orders(
        planned, buying_power=750.0, supports_fractional_shares=True, min_trade_dollars=1.0
    )

    quantities = {order["symbol"]: order["quantity"] for order in fundable}
    # Half the budget short, so both legs land at half size and the 2:1 ratio survives.
    assert quantities == {"AAA": 50.0, "BBB": 25.0}


def test_greedy_fills_whole_legs_rather_than_shaving_all_of_them() -> None:
    """On a whole-share brokerage, shaving every DCA leg can deploy nothing at all."""
    planned = [_buy("AAA", 1, 300.0), _buy("BBB", 1, 300.0), _buy("CCC", 1, 300.0)]

    fundable, unfunded, _ = orders.fund_planned_orders(
        planned, buying_power=620.0, policy=orders.FUNDING_GREEDY, min_trade_dollars=1.0
    )

    assert [order["quantity"] for order in fundable] == [1, 1]
    assert [order["symbol"] for order in unfunded] == ["CCC"]
    assert "Insufficient funds" in unfunded[0]["reason"]


def test_funding_sells_cash_equivalents_to_cover_a_shortfall() -> None:
    """The DCA case: only buys, and the idle cash is parked in T-bills."""
    planned = [_buy("VTI", 10, 100.0)]

    fundable, unfunded, report = orders.fund_planned_orders(
        planned,
        buying_power=200.0,
        cash_equivalents={"SGOV": {"shares": 100.0, "price": 100.0, "value": 10_000.0}},
        min_trade_dollars=1.0,
    )

    assert unfunded == []
    liquidation, buy = fundable
    assert (liquidation["symbol"], liquidation["action"]) == ("SGOV", "sell")
    assert liquidation["funding_source"] == "cash_equivalent"
    # Rounded up, so a fill worse than the last print still leaves the batch funded.
    assert liquidation["quantity"] == 9
    assert buy["quantity"] == 10 and buy["funding_status"] == "full"
    assert report["cash_equivalents_liquidated"] == 900.0


def test_funding_never_liquidates_what_the_batch_already_trades() -> None:
    """Selling SGOV to fund a buy of SGOV would only undo itself."""
    planned = [_buy("SGOV", 10, 100.0)]

    fundable, unfunded, report = orders.fund_planned_orders(
        planned,
        buying_power=200.0,
        cash_equivalents={"SGOV": {"shares": 100.0, "price": 100.0, "value": 10_000.0}},
        min_trade_dollars=1.0,
    )

    assert report["cash_equivalents_liquidated"] == 0.0
    assert len(fundable) == 1 and fundable[0]["action"] == "buy"
    assert fundable[0]["quantity"] == 2


def test_funding_liquidates_only_what_it_is_short() -> None:
    planned = [_buy("VTI", 10, 100.0)]

    _, _, report = orders.fund_planned_orders(
        planned,
        buying_power=900.0,
        cash_equivalents={"SGOV": {"shares": 100.0, "price": 10.0, "value": 1_000.0}},
        min_trade_dollars=1.0,
    )

    # $100 short of $1000, so ~$100 is freed rather than the whole sleeve.
    assert 100.0 <= report["cash_equivalents_liquidated"] <= 110.0


def test_funding_drops_a_leg_it_cannot_clear_the_minimum_for() -> None:
    planned = [_buy("SPY", 10, 100.0)]

    # $40 of budget would buy 0.4 shares -- a real trade, but not one worth its costs.
    fundable, unfunded, report = orders.fund_planned_orders(
        planned, buying_power=40.0, min_trade_dollars=50.0, supports_fractional_shares=True
    )

    assert fundable == []
    assert [order["symbol"] for order in unfunded] == ["SPY"]
    assert unfunded[0]["funding_status"] == "unfunded"
    assert report["unfunded"] == ["SPY"]


def test_funding_an_empty_batch_is_a_no_op() -> None:
    assert orders.fund_planned_orders([], buying_power=1_000.0) == ([], [], {})
