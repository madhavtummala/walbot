"""Reconciliation: making the broker's working orders match what a run decided.

The property under test throughout is idempotence. A reconciler that is not idempotent cannot be
driven by a cron -- running twice in a minute would double the book, and skipping an hour would
leave it wrong until someone noticed. Every test here is really asking "does asserting the same
desired state twice do nothing the second time".
"""

from __future__ import annotations

from typing import Any

import pytest

from src.algorithms.options_flip.algorithm import OptionsFlipAlgorithm
from src.algorithms.reconcile import ORDER_IDS_KEY
from src.core.interfaces import AlgorithmPlan, DesiredOrder, OrderRequest
from src.data.state_store import ephemeral_state


class FakeBrokerage:
    """A broker that remembers what is working, so reconciliation can be observed."""

    supports_options = True

    def __init__(self, working: list[dict[str, Any]] | None = None):
        self.working = list(working or [])
        self.submitted: list[OrderRequest] = []
        self.cancelled: list[str] = []
        self.replaced: list[tuple[str, OrderRequest]] = []
        self._next_id = 100

    def get_orders(self, status: str = "WORKING") -> list[dict[str, Any]]:
        return list(self.working)

    def submit_order(self, request: OrderRequest) -> dict[str, Any]:
        self.submitted.append(request)
        self._next_id += 1
        return {"order_id": str(self._next_id), "status": "accepted", "symbol": request.symbol}

    def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(str(order_id))
        self.working = [o for o in self.working if str(o.get("order_id")) != str(order_id)]

    def replace_order(self, order_id: str, request: OrderRequest) -> dict[str, Any]:
        self.replaced.append((str(order_id), request))
        self._next_id += 1
        return {"order_id": str(self._next_id), "status": "accepted", "symbol": request.symbol}


class Config:
    account_id = "test"


def bid(price: float = 1.15, quantity: int = 1) -> OrderRequest:
    return OrderRequest(
        symbol="QQQM  260220C00100000", action="buy", quantity=quantity,
        order_type="limit", limit_price=price, asset_type="option",
    )


def working_bid(order_id: str = "100", price: float = 1.15, quantity: int = 1) -> dict[str, Any]:
    return {
        "order_id": order_id, "symbol": "QQQM  260220C00100000", "action": "buy",
        "quantity": float(quantity), "order_type": "limit", "limit_price": price,
        "stop_price": 0.0, "status": "WORKING", "asset_type": "option",
    }


def run(brokerage: FakeBrokerage, orders, state=None) -> dict[str, Any]:
    """Drive reconciliation the way the algorithm does, so the tests exercise the real path."""
    plan = AlgorithmPlan(strategy="options_flip", desired_orders=list(orders), state=dict(state or {}))
    with ephemeral_state():
        return OptionsFlipAlgorithm({}).execute(plan, Config(), brokerage)


def test_a_wanted_order_that_does_not_exist_is_submitted() -> None:
    brokerage = FakeBrokerage()

    outcome = run(brokerage, [DesiredOrder(key="QQQM:entry", request=bid())])

    assert len(brokerage.submitted) == 1
    assert outcome["state"][ORDER_IDS_KEY] == {"QQQM:entry": "101"}


def test_an_unchanged_order_is_left_alone() -> None:
    brokerage = FakeBrokerage([working_bid("100")])

    outcome = run(brokerage, [DesiredOrder(key="QQQM:entry", request=bid(1.15))],
        state={ORDER_IDS_KEY: {"QQQM:entry": "100"}},
    )

    assert brokerage.submitted == [] and brokerage.replaced == [] and brokerage.cancelled == []
    assert outcome["order_results"][0]["reconciled"] == "unchanged"


def test_reconciling_twice_changes_nothing_the_second_time() -> None:
    """The property the whole 5-minute cadence rests on."""
    brokerage = FakeBrokerage()
    orders = [DesiredOrder(key="QQQM:entry", request=bid())]

    first = run(brokerage, orders)
    brokerage.working = [working_bid(first["state"][ORDER_IDS_KEY]["QQQM:entry"])]
    second = run(brokerage, orders, state=first["state"])

    assert len(brokerage.submitted) == 1
    assert second["order_results"][0]["reconciled"] == "unchanged"


def test_a_re_priced_order_is_replaced_not_duplicated() -> None:
    brokerage = FakeBrokerage([working_bid("100", price=1.15)])

    run(brokerage, [DesiredOrder(key="QQQM:entry", request=bid(1.40))],
        state={ORDER_IDS_KEY: {"QQQM:entry": "100"}},
    )

    assert brokerage.replaced and not brokerage.submitted
    assert brokerage.replaced[0][1].limit_price == 1.40


def test_a_move_inside_the_tolerance_does_not_churn_the_book() -> None:
    brokerage = FakeBrokerage([working_bid("100", price=1.15)])

    run(brokerage,
        [DesiredOrder(key="QQQM:entry", request=bid(1.16), replace_tolerance=0.05)],
        state={ORDER_IDS_KEY: {"QQQM:entry": "100"}},
    )

    assert not brokerage.replaced


def test_an_order_no_longer_wanted_is_cancelled() -> None:
    brokerage = FakeBrokerage([working_bid("100")])

    run(brokerage, [], state={ORDER_IDS_KEY: {"QQQM:entry": "100"}})

    assert brokerage.cancelled == ["100"]


def test_a_quantity_change_always_replaces() -> None:
    brokerage = FakeBrokerage([working_bid("100", quantity=1)])

    run(brokerage,
        [DesiredOrder(key="QQQM:entry", request=bid(1.15, quantity=2), replace_tolerance=0.99)],
        state={ORDER_IDS_KEY: {"QQQM:entry": "100"}},
    )

    # Tolerance governs prices only: a different size is a different order, not a re-price.
    assert brokerage.replaced


def test_an_order_that_filled_between_runs_is_not_cancelled() -> None:
    # Recorded by us, absent from the broker's working set: it filled or was cancelled by hand.
    brokerage = FakeBrokerage([])

    outcome = run(brokerage, [], state={ORDER_IDS_KEY: {"QQQM:entry": "100"}})

    assert brokerage.cancelled == []
    assert outcome["state"][ORDER_IDS_KEY] == {}


def test_a_cold_start_with_untracked_working_orders_leaves_them_alone() -> None:
    # We have no memory of order 999, so it is not ours to cancel.
    brokerage = FakeBrokerage([working_bid("999")])

    run(brokerage, [DesiredOrder(key="QQQM:entry", request=bid())], state={})

    assert brokerage.cancelled == []
    assert len(brokerage.submitted) == 1


def test_one_rejected_order_does_not_abandon_the_rest() -> None:
    class PartlyRefusing(FakeBrokerage):
        def submit_order(self, request):
            if request.order_type == "limit":
                raise RuntimeError("rejected by venue")
            return super().submit_order(request)

    brokerage = PartlyRefusing()
    stop = OrderRequest(
        symbol="QQQM  260220C00100000", action="sell", quantity=1,
        order_type="stop", stop_price=1.50, asset_type="option",
    )

    outcome = run(brokerage, [
        DesiredOrder(key="QQQM:entry", request=bid()),
        DesiredOrder(key="QQQM:stop", request=stop),
    ])

    actions = {result["key"]: result["reconciled"] for result in outcome["order_results"]}
    assert actions == {"QQQM:entry": "rejected", "QQQM:stop": "submitted"}


def test_a_venue_that_refuses_replace_still_re_prices() -> None:
    """Alpaca refuses to replace an order in ``accepted`` status.

    Without a fallback the bid would freeze at whatever price the first fire of the day set,
    which for a strategy whose entire mechanism is walking a resting bid in is total failure --
    and a silent one, since nothing errors.
    """
    class RefusingReplace(FakeBrokerage):
        def replace_order(self, order_id, request):
            raise RuntimeError("cannot replace order in accepted status")

    brokerage = RefusingReplace([working_bid("100", price=1.15)])

    outcome = run(brokerage, [DesiredOrder(key="QQQM:entry", request=bid(1.40))],
        state={ORDER_IDS_KEY: {"QQQM:entry": "100"}},
    )

    assert brokerage.cancelled == ["100"]
    assert brokerage.submitted[0].limit_price == 1.40
    result = outcome["order_results"][0]
    # Says which route it took, because the two differ in whether a gap existed.
    assert result["reconciled"] == "resubmitted"
    assert result["previous_order_id"] == "100"
    assert outcome["state"][ORDER_IDS_KEY] == {"QQQM:entry": "101"}


def test_when_both_replace_and_resubmit_fail_no_dead_id_is_recorded() -> None:
    class RefusingBoth(FakeBrokerage):
        def replace_order(self, order_id, request):
            raise RuntimeError("cannot replace")

        def submit_order(self, request):
            raise RuntimeError("rejected by venue")

    brokerage = RefusingBoth([working_bid("100", price=1.15)])

    outcome = run(brokerage, [DesiredOrder(key="QQQM:entry", request=bid(1.40))],
        state={ORDER_IDS_KEY: {"QQQM:entry": "100"}},
    )

    # The old order was cancelled on the way through, so keeping its id would have the next run
    # believe a dead order is still working -- and leave the desired order never placed.
    assert outcome["state"][ORDER_IDS_KEY] == {}
    assert outcome["order_results"][0]["reconciled"] == "rejected"


def test_state_is_persisted_even_when_nothing_was_placed() -> None:
    """The trap this class exists to avoid: memory that only survives a run that traded."""
    brokerage = FakeBrokerage()
    plan = AlgorithmPlan(strategy="options_flip", desired_orders=[], state={"symbols": {"QQQM": {"x": 1}}})

    with ephemeral_state() as store:
        OptionsFlipAlgorithm({}).execute(plan, Config(), brokerage)

    assert store["algorithm_state:options_flip:test"]["symbols"] == {"QQQM": {"x": 1}}


def test_a_brokerage_that_cannot_list_orders_is_refused_loudly() -> None:
    class Blind:
        def get_orders(self, status="WORKING"):
            raise NotImplementedError("no order listing")

    # Proceeding would re-submit the whole book every run, since nothing would look present.
    with pytest.raises(NotImplementedError, match="reconciliation"):
        run(Blind(), [DesiredOrder(key="QQQM:entry", request=bid())])


def test_results_keep_the_trade_side_separate_from_the_reconciler_verb() -> None:
    """The order journal files ``action`` under ``side``, so it must stay buy/sell."""
    brokerage = FakeBrokerage([working_bid("100")])

    outcome = run(brokerage, [DesiredOrder(key="QQQM:entry", request=bid(1.15))],
        state={ORDER_IDS_KEY: {"QQQM:entry": "100"}},
    )

    result = outcome["order_results"][0]
    assert result["action"] == "buy"
    assert result["reconciled"] == "unchanged"
    assert result["symbol"] == "QQQM  260220C00100000"


def test_a_cancelled_order_still_reports_what_it_was() -> None:
    brokerage = FakeBrokerage([working_bid("100")])

    outcome = run(brokerage, [], state={ORDER_IDS_KEY: {"QQQM:entry": "100"}})

    result = outcome["order_results"][0]
    assert (result["reconciled"], result["action"], result["symbol"]) == (
        "cancelled", "buy", "QQQM  260220C00100000"
    )
