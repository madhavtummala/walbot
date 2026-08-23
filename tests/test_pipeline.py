from __future__ import annotations

from typing import Any

import pytest

from src.algorithms.base import BaseAlgorithm
from src.algorithms.registry import register_algorithm
from src.core import pipeline
from src.core.config import Config
from src.core.interfaces import PortfolioSnapshot, intents_from_weights


class PassthroughAlgorithm(BaseAlgorithm):
    """The thinnest possible algorithm, so these tests exercise the pipeline, not a strategy."""

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


def _orders(config, brokerage, weights=None, **kwargs) -> dict[str, Any]:
    """Place a one-symbol weight plan, spelling out the arguments the algorithm supplies."""
    return pipeline.place_orders(
        intents_from_weights(weights or {"AAA": 0.5}),
        config,
        brokerage,
        latest_prices=kwargs.pop("latest_prices", {"AAA": 100.0}),
        signals={"AAA": {"score": 1.0}},
        min_trade_dollars=float(getattr(config, "min_trade_dollars", 0.0) or 0.0),
        **kwargs,
    )


def _config(**overrides) -> Config:
    return Config(**{"cash_buffer": 0.0, "min_trade_dollars": 1.0, "rebalance_threshold": 0.0, **overrides})


def test_snapshot_weights_ignore_unpriced_symbols() -> None:
    snapshot = PortfolioSnapshot(positions={"AAA": 10, "BBB": 5}, equity=1_000.0)

    assert snapshot.weights({"AAA": 50.0}) == {"AAA": 0.5}


def test_snapshot_weights_are_empty_without_equity() -> None:
    assert PortfolioSnapshot(positions={"AAA": 10}, equity=0.0).weights({"AAA": 50.0}) == {}


def test_place_orders_sizes_from_the_prices_on_the_plan() -> None:
    brokerage = RecordingBrokerage()

    outcome = _orders(_config(), brokerage)

    assert outcome["status"] == "submitted"
    assert brokerage.submitted == [("AAA", "buy", 50)]  # 50% of 10k at 100


def test_place_orders_exits_a_holding_left_out_of_the_targets() -> None:
    brokerage = RecordingBrokerage(positions={"BBB": 4})

    outcome = _orders(_config(), brokerage, latest_prices={"AAA": 100.0, "BBB": 100.0})

    assert ("BBB", "sell", 4) in brokerage.submitted
    assert outcome["diff"][0]["symbol"] in {"AAA", "BBB"}


def test_place_orders_rejects_a_symbol_the_plan_never_priced() -> None:
    with pytest.raises(ValueError, match="ZZZ"):
        _orders(_config(), RecordingBrokerage(), weights={"AAA": 0.3, "ZZZ": 0.2})


def test_place_orders_acts_on_whatever_intents_it_is_handed() -> None:
    """An agent edits the plan's intents; this step has no opinion about where they came from."""
    brokerage = RecordingBrokerage()

    _orders(_config(), brokerage, weights={"AAA": 0.25})

    assert brokerage.submitted == [("AAA", "buy", 25)]


def test_weight_diff_reports_direction_and_magnitude() -> None:
    rows = pipeline.weight_diff({"AAA": 0.4, "BBB": 0.2}, {"AAA": 0.1, "BBB": 0.2})

    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["AAA"]["action"] == "trim"
    assert by_symbol["AAA"]["change"] == pytest.approx(-0.3)
    assert by_symbol["BBB"]["action"] == "hold"
    assert rows[0]["symbol"] == "AAA"  # largest change first


def test_no_algorithm_sizes_against_a_cash_buffer() -> None:
    """Sizing states a portfolio; funding decides what the account can pay for.

    While the buffer was a haircut on targets, every exposure-capped algorithm had to override
    it to zero or under-invest twice over. It is an account-level floor applied once against
    buying power now, so no algorithm should be carrying one at all.
    """
    from src.algorithms.registry import get_algorithm_class

    config = Config(cash_buffer=0.02)
    for strategy in ("rally_rotation", "bursty_dca"):
        sizing = get_algorithm_class(strategy).from_config(config).sizing(config)
        assert "cash_buffer" not in sizing, strategy

    assert "cash_buffer" not in PassthroughAlgorithm.from_config(config).sizing(config)


# --------------------------------------------------------------------------------------
# Order funding, as a caller of place_orders sees it.
# --------------------------------------------------------------------------------------


class FundingBrokerage(RecordingBrokerage):
    """A brokerage whose buying power is separate from its equity, and which parks cash."""

    supports_fractional_shares = True

    def __init__(self, buying_power: float, cash_equivalents=None, **kwargs):
        super().__init__(**kwargs)
        self.buying_power = buying_power
        self._cash_equivalents = cash_equivalents or {}

    def get_account_state(self) -> dict[str, Any]:
        return {"equity": self.equity, "cash": self.buying_power, "buying_power": self.buying_power}

    def get_cash_equivalents(self) -> dict[str, dict[str, float]]:
        return dict(self._cash_equivalents)


def test_place_orders_trims_a_batch_to_available_buying_power() -> None:
    """Sizing targets the book; only funding knows what the account can pay for."""
    brokerage = FundingBrokerage(buying_power=2_500.0, equity=10_000.0)

    outcome = _orders(_config(), brokerage)

    # A 50% target on $10k equity is $5,000 of AAA, against $2,500 that can actually be spent.
    assert brokerage.submitted == [("AAA", "buy", 25.0)]
    assert outcome["status"] == pipeline.STATUS_SUBMITTED_REDUCED
    assert outcome["funding"]["reduced"] == ["AAA"]
    assert outcome["funding"]["buying_power"] == 2_500.0
    assert outcome["rejected"] == []


def test_place_orders_holds_back_the_account_cash_buffer() -> None:
    brokerage = FundingBrokerage(buying_power=10_000.0, equity=10_000.0)

    outcome = _orders(_config(cash_buffer=0.02), brokerage)

    assert outcome["funding"]["reserve"] == 200.0
    assert outcome["funding"]["budget"] == 9_800.0
    # Still affordable, so the buffer costs the plan nothing.
    assert outcome["status"] == pipeline.STATUS_SUBMITTED


def test_place_orders_liquidates_cash_equivalents_to_fund_a_batch() -> None:
    brokerage = FundingBrokerage(
        buying_power=1_000.0,
        equity=10_000.0,
        cash_equivalents={"SGOV": {"shares": 500.0, "price": 100.0, "value": 50_000.0}},
    )

    outcome = _orders(_config(), brokerage)

    assert ("SGOV", "sell") == brokerage.submitted[0][:2]
    assert ("AAA", "buy", 50.0) == brokerage.submitted[1]
    assert outcome["funding"]["cash_equivalents_liquidated"] > 0
    assert outcome["status"] == pipeline.STATUS_SUBMITTED


def test_place_orders_reports_an_unfundable_leg_with_its_reason() -> None:
    """An agent has to be able to tell a deliberate trim from a broker refusal."""
    brokerage = FundingBrokerage(buying_power=0.0, equity=10_000.0)

    outcome = _orders(_config(min_trade_dollars=50.0), brokerage)

    assert brokerage.submitted == []
    assert outcome["status"] == "unfunded"
    assert [row["symbol"] for row in outcome["unfunded"]] == ["AAA"]
    assert "Insufficient funds" in outcome["unfunded"][0]["reason"]
    # Nothing was refused by the broker; the batch never asked it to.
    assert outcome["rejected"] == []


def test_final_weights_reflect_what_was_submitted_not_what_was_planned() -> None:
    """Reporting the pre-funding target would claim a fill that never happened."""
    brokerage = FundingBrokerage(buying_power=2_500.0, equity=10_000.0)

    outcome = _orders(_config(), brokerage)

    # 25 shares at $100 against $10k equity is a quarter of the book, not the half targeted.
    assert outcome["final_weights"] == {"AAA": 0.25}
