from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.algorithms.base import BaseAlgorithm
from src.algorithms.fast_momentum import DefensiveMomentumConfig, apply_stickiness
from src.algorithms.registry import register_algorithm
from src.core import pipeline
from src.core.config import Config
from src.core.interfaces import AlgorithmResult, Intent, PortfolioSnapshot


class PassthroughAlgorithm(BaseAlgorithm):
    """The thinnest possible plugin, so these tests exercise the pipeline and not a strategy."""

    algorithm_id = "passthrough"


register_algorithm("passthrough", PassthroughAlgorithm)


class RecordingBrokerage:
    supports_fractional_shares = False

    def __init__(self, positions: dict[str, float] | None = None, equity: float = 10_000.0):
        self.positions = dict(positions or {})
        self.equity = equity
        self.submitted: list[tuple[str, str, float]] = []

    def get_account_state(self) -> dict[str, Any]:
        return {"equity": self.equity, "cash": 0.0, "buying_power": self.equity}

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def submit_order(self, request) -> dict[str, Any]:
        self.submitted.append((request.symbol, request.action, request.quantity))
        return {"order_id": f"order-{len(self.submitted)}"}

    def validate_short_sale_feasibility(self, *a, **kw) -> dict[str, Any]:
        return {"shortable": True, "reason": "ok"}


def _result(**overrides) -> AlgorithmResult:
    defaults = {
        "strategy": "passthrough",
        "target_weights": {"AAA": 0.5},
        "signals": {"AAA": {"score": 1.0}},
        "latest_prices": {"AAA": 100.0},
    }
    defaults.update(overrides)
    return AlgorithmResult(**defaults)


def _config(**overrides) -> Config:
    return Config(cash_buffer=0.0, min_trade_dollars=1.0, rebalance_threshold=0.0, **overrides)


def test_snapshot_weights_ignore_unpriced_symbols() -> None:
    snapshot = PortfolioSnapshot(positions={"AAA": 10, "BBB": 5}, equity=1_000.0)

    assert snapshot.weights({"AAA": 50.0}) == {"AAA": 0.5}


def test_snapshot_weights_are_empty_without_equity() -> None:
    assert PortfolioSnapshot(positions={"AAA": 10}, equity=0.0).weights({"AAA": 50.0}) == {}


def test_place_orders_sizes_from_the_result_prices() -> None:
    brokerage = RecordingBrokerage()

    outcome = pipeline.place_orders(_result(), _config(), brokerage)

    assert outcome["status"] == "submitted"
    assert brokerage.submitted == [("AAA", "buy", 50)]  # 50% of 10k at 100


def test_place_orders_exits_a_holding_left_out_of_the_targets() -> None:
    brokerage = RecordingBrokerage(positions={"BBB": 4})

    outcome = pipeline.place_orders(
        _result(latest_prices={"AAA": 100.0, "BBB": 100.0}), _config(), brokerage
    )

    assert ("BBB", "sell", 4) in brokerage.submitted
    assert outcome["diff"][0]["symbol"] in {"AAA", "BBB"}


def test_place_orders_refuses_a_stale_result() -> None:
    stale = _result(as_of=datetime.now(timezone.utc) - timedelta(hours=1))

    with pytest.raises(pipeline.StaleResultError, match="too stale"):
        pipeline.place_orders(stale, _config(), RecordingBrokerage())


def test_place_orders_rejects_a_symbol_the_result_never_priced() -> None:
    result = _result(target_weights={"AAA": 0.3, "ZZZ": 0.2})

    with pytest.raises(ValueError, match="ZZZ"):
        pipeline.place_orders(result, _config(), RecordingBrokerage())


def test_place_orders_honours_an_agents_edited_weights() -> None:
    brokerage = RecordingBrokerage()

    pipeline.place_orders(_result(), _config(), brokerage, target_weights={"AAA": 0.25})

    assert brokerage.submitted == [("AAA", "buy", 25)]


def test_weight_diff_reports_direction_and_magnitude() -> None:
    rows = pipeline.weight_diff({"AAA": 0.4, "BBB": 0.2}, {"AAA": 0.1, "BBB": 0.2})

    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["AAA"]["action"] == "trim"
    assert by_symbol["AAA"]["change"] == pytest.approx(-0.3)
    assert by_symbol["BBB"]["action"] == "hold"
    assert rows[0]["symbol"] == "AAA"  # largest change first


# --------------------------------------------------------------------------------------
# Stickiness, now applied in step 2 against an already-chosen set.
# --------------------------------------------------------------------------------------


def _momentum_config(**overrides) -> DefensiveMomentumConfig:
    settings = {"max_positions": 2, "min_score_delta_to_replace": 0.5, **overrides}
    return DefensiveMomentumConfig(**settings)


def test_stickiness_retains_a_held_symbol_a_marginal_challenger_would_replace() -> None:
    kept = apply_stickiness(
        target_weights={"NEW": 0.5, "KEEP": 0.5},
        scores_by_symbol={"HELD": {"score": 1.0}, "NEW": {"score": 1.2}, "KEEP": {"score": 2.0}},
        current_weights={"HELD": 0.5, "KEEP": 0.5},
        config=_momentum_config(),
    )

    # NEW beats HELD by only 0.2, under the 0.5 delta, so the incumbent stays.
    assert kept["HELD"] == 0.5
    assert kept["NEW"] == 0.0


def test_stickiness_yields_to_a_clearly_better_challenger() -> None:
    kept = apply_stickiness(
        target_weights={"NEW": 0.5, "KEEP": 0.5},
        scores_by_symbol={"HELD": {"score": 1.0}, "NEW": {"score": 2.5}, "KEEP": {"score": 3.0}},
        current_weights={"HELD": 0.5, "KEEP": 0.5},
        config=_momentum_config(),
    )

    assert kept == {"NEW": 0.5, "KEEP": 0.5}


def test_stickiness_is_off_when_the_delta_is_zero() -> None:
    kept = apply_stickiness(
        target_weights={"NEW": 1.0},
        scores_by_symbol={"HELD": {"score": 1.0}, "NEW": {"score": 1.1}},
        current_weights={"HELD": 1.0},
        config=_momentum_config(min_score_delta_to_replace=0.0),
    )

    assert kept == {"NEW": 1.0}


def test_exposure_capped_algorithms_do_not_apply_the_cash_buffer_twice() -> None:
    """fast_momentum/invest_spy bake cash into their weights via max_gross_exposure.

    Applying the account cash_buffer on top would silently under-invest by that buffer.
    """
    from src.algorithms.registry import get_algorithm_class

    config = Config(cash_buffer=0.02)
    for strategy in ("fast_momentum", "invest_spy"):
        sizing = get_algorithm_class(strategy).from_config(config).sizing(config)
        assert sizing["cash_buffer"] == 0.0, strategy

    # A strategy that does not cap exposure still honours the account buffer.
    assert PassthroughAlgorithm.from_config(config).sizing(config)["cash_buffer"] == 0.02
