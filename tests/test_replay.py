"""The generic backtest harness: one path for every algorithm."""

from __future__ import annotations

import pandas as pd
import pytest

from src.algorithms.registry import get_algorithm_class
from src.core.config import ALGORITHM_IDS, Config
from src.core.interfaces import (
    MODE_INCREMENTAL,
    MODE_TARGET,
    AlgorithmDecision,
    AlgorithmRequirements,
    Intent,
)
from src.data.state_store import ephemeral_state, load_state, save_state
from src.execution.replay import Coverage, _target_shares, replay


DATES = pd.bdate_range("2026-01-05", periods=20, tz="UTC")


def _history(price: float = 100.0) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(
        {"timestamp": DATES, "open": price, "high": price, "low": price,
         "close": price, "adjusted_close": price, "volume": 1_000}
    ).set_index("timestamp")
    return {"AAA": frame}


class _Recorder:
    """Minimal algorithm: records the contexts it was handed and buys a fixed weight."""

    algorithm_id = "recorder"
    schedule = None

    def __init__(self, weight: float = 0.5, mode: str = MODE_TARGET) -> None:
        self.weight = weight
        self.mode = mode
        self.contexts: list = []
        self.settled: list = []

    def requirements(self, config, positions) -> AlgorithmRequirements:
        return AlgorithmRequirements(price_symbols=["AAA"])

    def analyze(self, context) -> AlgorithmDecision:
        self.contexts.append(context)
        return AlgorithmDecision(
            intents=[Intent("AAA", "weight", self.weight)], mode=self.mode
        )

    def refine(self, intents, signals, snapshot, latest_prices, config):
        return list(intents)

    def settle(self, config, order_results, intents) -> None:
        self.settled.append(order_results)


def _replay(algorithm, **kwargs):
    return replay(
        algorithm,
        Config(),
        daily_history=_history(),
        trade_dates=list(DATES),
        should_run=lambda date: True,
        starting_equity=10_000.0,
        **kwargs,
    )


def test_replay_never_lets_a_decision_see_the_price_it_trades_at() -> None:
    """Signals come from the previous bar and execute at this bar's close. Without this the
    backtest trades on information it could not have had, and every strategy looks prescient.
    """
    algorithm = _Recorder()
    _replay(algorithm)

    assert algorithm.contexts, "analyze was never called"
    for index, context in enumerate(algorithm.contexts, start=1):
        assert pd.Timestamp(context.timestamp) == DATES[index - 1]
        # The context's bars stop at the signal date, not the trade date.
        assert context.bars_by_symbol["AAA"]["timestamp"].max() == DATES[index - 1]


def test_replay_calls_settle_with_what_actually_filled() -> None:
    """Stateful algorithms draw down on fills, so the replay has to report them like live."""
    algorithm = _Recorder()
    _replay(algorithm)

    fills = [order for batch in algorithm.settled for order in batch]
    assert fills, "settle never saw a fill"
    assert {order["symbol"] for order in fills} == {"AAA"}
    assert all(order["status"] == "submitted" for order in fills)
    assert all(order["quantity"] > 0 and order["latest_price"] > 0 for order in fills)


def test_replay_never_spends_cash_it_does_not_have() -> None:
    history, _ = _replay(_Recorder(weight=5.0))  # asks for 5x equity

    assert (history["cash"] >= -0.01).all()


def test_replay_state_is_isolated_from_the_live_store() -> None:
    """A backtest replays algorithms that carry state between runs -- DCA's accrued budget,
    Fast Momentum's intraday risk flag. It starts from a clean slate rather than inheriting
    the live account's state, so a backtest is reproducible instead of depending on whatever
    the account happens to have accrued, and it cannot write back over it either.
    """
    from src.data.state_store import delete_state

    save_state("replay_probe", {"live": True})
    try:
        with ephemeral_state():
            assert load_state("replay_probe", None) is None  # clean slate, not the live value
            save_state("replay_probe", {"live": False})
            assert load_state("replay_probe", None) == {"live": False}
        assert load_state("replay_probe", None) == {"live": True}  # write never leaked
    finally:
        delete_state("replay_probe")


def test_replay_state_carries_across_dates_within_one_run() -> None:
    """One store for the whole replay: accrual has to carry from one date to the next, or a
    budget would reset every day and never reach a tradable amount."""
    algorithm = _Recorder()
    with ephemeral_state() as store:
        save_state("carried", 1)
        _replay(algorithm)
        assert load_state("carried", None) == 1
        assert "carried" in store


def test_target_mode_exits_a_holding_the_algorithm_stopped_asking_for() -> None:
    positions = {"AAA": 10.0, "BBB": 5.0}
    prices = {"AAA": 100.0, "BBB": 50.0}

    desired = _target_shares(
        [Intent("AAA", "weight", 0.5)], MODE_TARGET, positions, prices, 10_000.0
    )

    assert desired["AAA"] == pytest.approx(50.0)
    assert desired["BBB"] == 0.0


def test_incremental_mode_leaves_untouched_holdings_alone() -> None:
    positions = {"AAA": 10.0, "BBB": 5.0}
    prices = {"AAA": 100.0, "BBB": 50.0}

    desired = _target_shares(
        [Intent("AAA", "notional", 500.0)], MODE_INCREMENTAL, positions, prices, 10_000.0
    )

    assert desired["AAA"] == pytest.approx(15.0)
    assert desired["BBB"] == 5.0


def test_coverage_reports_the_history_the_cache_could_not_supply() -> None:
    """A window the cache cannot reach scores every symbol near zero, which reads as a poor
    strategy rather than an unsupported window unless it is reported.

    Measured in minutes, not symbols: a symbol the cache holds two sessions of, against a
    twelve-session horizon, is not covered in any sense the score would recognise."""
    coverage = Coverage(history_requested=10_000, history_supplied=4_000, missing_symbols={"AAA"})
    reported = coverage.as_dict()

    assert reported["history_ratio"] == 0.4
    assert reported["missing_symbols"] == ["AAA"]
    assert reported["complete"] is False
    assert Coverage().as_dict()["complete"] is True


def test_every_registered_algorithm_declares_what_the_replay_needs() -> None:
    """The replay satisfies ``requirements()``; an algorithm that fetches its own data inside
    ``analyze`` instead cannot be replayed and would need a hand-written backtest branch.
    """
    for algorithm_id in sorted(ALGORITHM_IDS):
        algorithm = get_algorithm_class(algorithm_id)
        requirements = algorithm.from_config(Config()).requirements(Config(), {})
        assert isinstance(requirements, AlgorithmRequirements)
        assert requirements.price_symbols, f"{algorithm_id} declares no symbols"
        assert algorithm.schedule.weekdays, f"{algorithm_id} has no schedule"
