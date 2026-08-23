from __future__ import annotations

import pytest

from src.core.interfaces import (
    MODE_INCREMENTAL,
    MODE_TARGET,
    AlgorithmPlan,
    Intent,
    intents_from_weights,
    weights_from_intents,
)
from src.core.orders import resolve_target_shares


def test_intent_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown intent kind"):
        Intent(symbol="AAA", kind="dollars", value=1.0)


def test_plan_derives_weights_from_its_intents() -> None:
    """Weights are a view of the intents, so the two cannot disagree."""
    plan = AlgorithmPlan(strategy="x", intents=intents_from_weights({"AAA": 0.5}))
    assert plan.intents == [Intent("AAA", "weight", 0.5)]
    assert plan.target_weights == {"AAA": 0.5}


def test_weight_helpers_round_trip() -> None:
    weights = {"AAA": 0.5, "BBB": 0.25}
    assert weights_from_intents(intents_from_weights(weights)) == weights


def test_weights_from_intents_ignores_non_weight_kinds() -> None:
    assert weights_from_intents([Intent("AAA", "notional", 200.0)]) == {}


# --------------------------------------------------------------------------------------
# Resolution: the step where the two modes genuinely differ.
# --------------------------------------------------------------------------------------


def test_target_mode_exits_a_holding_absent_from_the_intents() -> None:
    shares = resolve_target_shares(
        [Intent("AAA", "weight", 0.5)],
        MODE_TARGET,
        current_positions={"AAA": 10, "OLD": 7},
        latest_prices={"AAA": 100.0, "OLD": 50.0},
        investable_equity=10_000.0,
    )

    assert shares == {"AAA": 50.0, "OLD": 0.0}


def test_incremental_mode_leaves_unlisted_holdings_alone() -> None:
    shares = resolve_target_shares(
        [Intent("AAA", "notional", 250.0)],
        MODE_INCREMENTAL,
        current_positions={"AAA": 10, "OTHER": 7},
        latest_prices={"AAA": 100.0, "OTHER": 50.0},
        investable_equity=10_000.0,
    )

    # 250/100 floors to 2 whole shares, added on top of the 10 already held. OTHER is untouched,
    # which is the whole point: a DCA buy must not disturb positions it does not own.
    assert shares == {"AAA": 12.0}


def test_incremental_notional_below_one_share_asks_for_no_trade() -> None:
    shares = resolve_target_shares(
        [Intent("VOO", "notional", 100.0)],
        MODE_INCREMENTAL,
        current_positions={"VOO": 3},
        latest_prices={"VOO": 500.0},
        investable_equity=10_000.0,
    )

    assert shares == {"VOO": 3.0}


def test_incremental_notional_buys_fractionally_when_the_broker_allows_it() -> None:
    shares = resolve_target_shares(
        [Intent("VOO", "notional", 100.0)],
        MODE_INCREMENTAL,
        current_positions={},
        latest_prices={"VOO": 500.0},
        investable_equity=10_000.0,
        supports_fractional_shares=True,
    )

    assert shares == {"VOO": 0.2}


def test_negative_notional_sells() -> None:
    shares = resolve_target_shares(
        [Intent("AAA", "notional", -250.0)],
        MODE_INCREMENTAL,
        current_positions={"AAA": 10},
        latest_prices={"AAA": 100.0},
        investable_equity=10_000.0,
    )

    assert shares == {"AAA": 8.0}


def test_share_intents_need_no_price() -> None:
    shares = resolve_target_shares(
        [Intent("AAA", "shares", 4.0)],
        MODE_INCREMENTAL,
        current_positions={"AAA": 1},
        latest_prices={},
        investable_equity=0.0,
    )

    assert shares == {"AAA": 5.0}


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown intent mode"):
        resolve_target_shares([], "sideways", {}, {}, 0.0)
