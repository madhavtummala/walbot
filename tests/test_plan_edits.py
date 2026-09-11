from __future__ import annotations

import pytest

from src.core.interfaces import MODE_INCREMENTAL, MODE_TARGET, AlgorithmPlan, DesiredOrder, Intent, OrderRequest
from src.core.orders import plan_share_orders, resolve_target_shares
from src.core.plan_edits import PlanEditRefused, apply_edits

PRICES = {"AAA": 100.0, "BBB": 50.0}


def _plan(mode: str = MODE_TARGET, **overrides) -> AlgorithmPlan:
    fields = {
        "strategy": "rally_rotation",
        "mode": mode,
        "intents": [Intent(symbol="AAA", kind="weight", value=0.5), Intent(symbol="BBB", kind="weight", value=0.5)],
        "latest_prices": dict(PRICES),
    }
    fields.update(overrides)
    return AlgorithmPlan(**fields)


def _orders(plan: AlgorithmPlan, positions: dict[str, float], equity: float = 10_000.0) -> dict[str, dict]:
    """Run the edited plan through the real sizer, so an edit is judged by the orders it causes."""
    target_shares = resolve_target_shares(plan.intents, plan.mode, positions, PRICES, equity)
    planned = plan_share_orders(PRICES, positions, target_shares, equity, min_trade_dollars=0.0)
    return {row["symbol"]: row for row in planned}


def test_no_edits_leaves_the_plan_alone() -> None:
    plan = _plan()
    edited, applied = apply_edits(plan, [], positions={"AAA": 10.0})

    assert edited is plan
    assert applied == []


def test_skip_under_target_mode_places_no_order_for_that_symbol() -> None:
    """The trap: under ``target`` the intent list *is* the portfolio, so a row deleted as a
    "veto" targets zero and sells the position. Pinning to the holding is what makes it a
    genuine no-op -- asserted through the sizer rather than on the intent, because the intent
    is only the means."""
    edited, applied = apply_edits(_plan(), [{"op": "skip", "symbol": "AAA"}], positions={"AAA": 10.0})

    orders = _orders(edited, {"AAA": 10.0})
    assert "AAA" not in orders  # not sold, not topped up
    assert orders["BBB"]["action"] == "buy"  # the rest of the plan still runs
    assert applied == [{"op": "skip", "symbol": "AAA", "effect": "held at 10 shares"}]


def test_deleting_the_row_instead_would_have_sold_it() -> None:
    """Nails down *why* skip is spelled as a pin. This is the behaviour being avoided."""
    naive = _plan(intents=[Intent(symbol="BBB", kind="weight", value=0.5)])

    orders = _orders(naive, {"AAA": 10.0})

    assert orders["AAA"]["action"] == "sell"
    assert orders["AAA"]["target_shares"] == 0


def test_skip_under_incremental_mode_drops_the_delta() -> None:
    """Incremental intents are deltas applied on top of the book, so an unlisted symbol is
    genuinely untouched -- here removal is the correct spelling."""
    plan = _plan(MODE_INCREMENTAL, intents=[Intent(symbol="AAA", kind="shares", value=5.0)])

    edited, applied = apply_edits(plan, [{"op": "skip", "symbol": "AAA"}], positions={"AAA": 10.0})

    assert edited.intents == []
    assert _orders(edited, {"AAA": 10.0}) == {}
    assert applied == [{"op": "skip", "symbol": "AAA", "effect": "increment dropped"}]


def test_exit_closes_the_position_under_either_mode() -> None:
    for mode in (MODE_TARGET, MODE_INCREMENTAL):
        plan = _plan(mode, intents=[Intent(symbol="AAA", kind="shares", value=5.0)])

        edited, _ = apply_edits(plan, [{"op": "exit", "symbol": "AAA"}], positions={"AAA": 10.0})

        orders = _orders(edited, {"AAA": 10.0})
        assert orders["AAA"]["action"] == "sell", mode
        assert orders["AAA"]["target_shares"] == 0, mode


def test_a_held_symbol_the_plan_never_mentioned_can_still_be_skipped() -> None:
    """Under target mode an unmentioned holding is already on its way to zero, so "leave it"
    needs a row of its own to say so."""
    edited, applied = apply_edits(_plan(), [{"op": "skip", "symbol": "CCC"}], positions={"CCC": 4.0})

    assert [intent.symbol for intent in edited.intents] == ["AAA", "BBB", "CCC"]
    assert applied == [{"op": "skip", "symbol": "CCC", "effect": "held at 4 shares"}]


def test_an_edit_keeps_whatever_the_algorithm_attached_to_the_intent() -> None:
    plan = _plan(intents=[Intent(symbol="AAA", kind="weight", value=0.5, extra={"band": "upper"})])

    edited, _ = apply_edits(plan, [{"op": "skip", "symbol": "AAA"}], positions={"AAA": 10.0})

    assert edited.intents[0].extra == {"band": "upper"}


def test_editing_an_order_book_plan_is_refused() -> None:
    plan = AlgorithmPlan(
        strategy="options_flip",
        desired_orders=[DesiredOrder(key="GLD:stop", request=OrderRequest(symbol="GLD", action="sell", quantity=1))],
    )

    with pytest.raises(PlanEditRefused, match="cancels a resting order"):
        apply_edits(plan, [{"op": "skip", "symbol": "GLD"}], positions={})


def test_an_op_that_carries_an_amount_is_refused() -> None:
    """There is no edit that takes a size, so the fence holds on the op name alone."""
    with pytest.raises(PlanEditRefused, match="Unknown edit op"):
        apply_edits(_plan(), [{"op": "set_weight", "symbol": "AAA", "value": 0.9}], positions={"AAA": 10.0})


def test_an_unrelated_symbol_is_refused_rather_than_ignored() -> None:
    with pytest.raises(PlanEditRefused, match="neither proposed by this plan nor held"):
        apply_edits(_plan(), [{"op": "skip", "symbol": "ZZZ"}], positions={"AAA": 10.0})


def test_conflicting_ops_for_one_symbol_are_refused() -> None:
    edits = [{"op": "skip", "symbol": "AAA"}, {"op": "exit", "symbol": "AAA"}]

    with pytest.raises(PlanEditRefused, match="Conflicting edits"):
        apply_edits(_plan(), edits, positions={"AAA": 10.0})


def test_a_malformed_edit_is_refused() -> None:
    with pytest.raises(PlanEditRefused, match="must be an object"):
        apply_edits(_plan(), ["skip AAA"], positions={"AAA": 10.0})

    with pytest.raises(PlanEditRefused, match="names no symbol"):
        apply_edits(_plan(), [{"op": "skip"}], positions={"AAA": 10.0})
