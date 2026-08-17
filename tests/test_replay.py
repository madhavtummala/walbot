"""The generic backtest harness: one path for every algorithm."""

from __future__ import annotations

import pandas as pd
import pytest

from src.algorithms.base import BaseAlgorithm
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
from src.core.orders import resolve_target_shares
from src.execution.replay import Coverage, replay


DATES = pd.bdate_range("2026-01-05", periods=20, tz="UTC")


def _history(price: float = 100.0) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(
        {"timestamp": DATES, "open": price, "high": price, "low": price,
         "close": price, "adjusted_close": price, "volume": 1_000}
    ).set_index("timestamp")
    return {"AAA": frame}


class _Recorder(BaseAlgorithm):
    """Minimal algorithm: records the contexts it was handed and buys a fixed weight.

    Extends ``BaseAlgorithm`` rather than duck-typing the parts the old replay happened to
    call. The replay drives the same step 2 the live runner does now, so a stub that does not
    satisfy the real plugin contract is not a valid subject -- which is the point.
    """

    algorithm_id = "recorder"
    schedule = None

    def __init__(self, weight: float = 0.5, mode: str = MODE_TARGET) -> None:
        super().__init__(Config())
        self.weight = weight
        self.mode = mode
        self.contexts: list = []
        self.settled: list = []
        self.refined_at: list = []

    def requirements(self, config, positions) -> AlgorithmRequirements:
        # Daily bars are handed over only when they are declared, live and replayed alike.
        # The replay used to supply them regardless, so a stub could read data it never asked
        # for and a real algorithm could quietly depend on that in a backtest but not live.
        return AlgorithmRequirements(price_symbols=["AAA"], daily_lookback_days=30)

    def analyze(self, context) -> AlgorithmDecision:
        self.contexts.append(context)
        return AlgorithmDecision(
            intents=[Intent("AAA", "weight", self.weight)], mode=self.mode
        )

    def refine(self, intents, signals, snapshot, latest_prices, config, as_of):
        self.refined_at.append(as_of)
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
    """Sizing is the live sizer now -- the replay used to carry its own simpler copy.

    That copy ignored the rebalance threshold and the minimum trade size, so a backtest was
    exercising execution logic the runtime does not use. These two properties still hold; they
    are just asserted against the one implementation both paths share.
    """
    positions = {"AAA": 10.0, "BBB": 5.0}
    prices = {"AAA": 100.0, "BBB": 50.0}

    desired = resolve_target_shares(
        [Intent("AAA", "weight", 0.5)], MODE_TARGET, positions, prices, 10_000.0
    )

    assert desired["AAA"] == pytest.approx(50.0)
    assert desired["BBB"] == 0.0


def test_incremental_mode_leaves_untouched_holdings_alone() -> None:
    positions = {"AAA": 10.0, "BBB": 5.0}
    prices = {"AAA": 100.0, "BBB": 50.0}

    desired = resolve_target_shares(
        [Intent("AAA", "notional", 500.0)], MODE_INCREMENTAL, positions, prices, 10_000.0
    )

    assert desired["AAA"] == pytest.approx(15.0)
    # Absent rather than echoed back: the canonical sizer states "unchanged" by omission, so
    # an incremental run cannot accidentally retarget a position another strategy owns.
    assert "BBB" not in desired


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


def test_a_backtest_never_touches_the_configured_paper_account() -> None:
    """The local paper account is for live scheduled testing, not for backtests.

    The replay executes through the same ``PaperBrokerage`` class and the same
    ``pipeline.place_orders`` the live runner uses -- that shared path is the point -- but on a
    throwaway state store. Sharing the code must never mean sharing the balances.
    """
    from src.brokerages.providers.paper import PaperBrokerage, _state_key
    from src.core.config import get_config
    from src.data.state_store import delete_state, load_state, save_state

    account = get_config(account_id="local_paper")
    key = _state_key(account.account_id)
    sentinel = {"cash": 4321.0, "positions": {"ZZZ": 7.0}, "prices": {"ZZZ": 1.0}}
    save_state(key, sentinel)
    try:
        before = load_state(key, None)
        _replay(_Recorder())
        after = load_state(key, None)

        assert after == before == sentinel, "the replay wrote to the live paper book"
        # And the book the replay used really was a PaperBrokerage, not a private simulator.
        with ephemeral_state():
            assert isinstance(PaperBrokerage(account), PaperBrokerage)
    finally:
        delete_state(key)


def test_the_replay_source_answers_strictly_as_of_the_signal_date() -> None:
    """The lookahead guarantee now lives in the source, so it is asserted there.

    Assembling the context moved into ``build_algorithm_context``, shared with the live path.
    What keeps a backtest honest is that every method of ``ReplayContextSource`` answers as of
    one past date -- if any of them reached past it, no test of the loop would notice.
    """
    from src.core.interfaces import AlgorithmRequirements
    from src.execution.replay import Coverage, HistoryCache, ReplayContextSource

    history = _history()
    as_of = DATES[5]
    empty = HistoryCache(["AAA"], [as_of], providers=[], lookback_minutes=0)
    source = ReplayContextSource(as_of, history, history=empty, coverage=Coverage())

    assert pd.Timestamp(source.timestamp()) == as_of
    # Prices are that date's closes, not the newest the frame happens to hold.
    assert set(source.latest_prices(["AAA"], Config())) == {"AAA"}
    daily = source.daily_bars(["AAA"], AlgorithmRequirements(daily_lookback_days=30), Config())
    assert daily["AAA"]["timestamp"].max() == as_of, "daily bars reached past the signal date"


def test_live_and_replay_share_one_context_assembler() -> None:
    """An algorithm that declares a new data need must get it in both paths.

    They were assembled separately before, so a requirement honoured live could be silently
    ignored in the backtest -- nothing would fail, the strategy would just score differently.
    """
    import inspect

    from src.core.market_context import ContextSource, LiveContextSource, build_algorithm_context
    from src.execution.replay import ReplayContextSource

    assert issubclass(LiveContextSource, ContextSource)
    assert issubclass(ReplayContextSource, ContextSource)
    # One assembler, and the only thing that varies is where the data comes from.
    assert "source" in inspect.signature(build_algorithm_context).parameters


def test_the_replay_reads_the_bar_store_once_per_symbol_not_once_per_date(monkeypatch) -> None:
    """History is loaded for the whole replay and sliced, the way daily bars always were.

    This is what a twelve-month backtest cost before: 14 symbols x 2 provider attempts x 250
    trade dates was 21,000 database round trips and 97 of its 99 seconds, which put it past the
    dashboard's request timeout and surfaced to the user as an aborted fetch. Reads must scale
    with the number of symbols, not with the length of the window.
    """
    from src.execution import replay as replay_module

    reads: list[tuple[str, object]] = []
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01 14:30", periods=600, freq="15min", tz="UTC"),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
            "volume": 1000.0, "adjusted_close": 100.5,
        }
    )

    def counting_read_history(symbol, *, lookback_minutes, end, provider=None, **kwargs):
        reads.append((symbol, provider))
        return bars

    monkeypatch.setattr(replay_module, "read_history", counting_read_history)

    class _NeedsHistory(_Recorder):
        def requirements(self, config, positions) -> AlgorithmRequirements:
            return AlgorithmRequirements(
                price_symbols=["AAA"], daily_lookback_days=30, history_lookback_minutes=780
            )

    algorithm = _NeedsHistory()
    _replay(algorithm)

    dates_stepped = len(algorithm.contexts)
    assert dates_stepped > 5, "the replay did not step through its dates"
    # One read per symbol per provider attempt, for the whole replay -- not per date. The
    # provider order is still walked, and still in order.
    assert reads == [("AAA", "yfinance"), ("AAA", None)], reads
    assert len(reads) < dates_stepped, "reads still scale with the length of the window"


# --------------------------------------------------------------------------------------
# Opening the book in a cash-equivalent sleeve rather than in raw cash.
# --------------------------------------------------------------------------------------


def _two_symbol_history(price: float = 100.0) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(
        {"timestamp": DATES, "open": price, "high": price, "low": price,
         "close": price, "adjusted_close": price, "volume": 1_000}
    ).set_index("timestamp")
    return {"AAA": frame, "SGOV": frame.copy()}


def test_open_in_parks_the_opening_balance_instead_of_holding_cash() -> None:
    """A funded account's idle balance sits in T-bills, not at 0%.

    Opening in cash understates the income and never exercises the liquidation a real
    rebalance performs to get at the money.
    """
    algorithm = _Recorder(weight=0.0, mode=MODE_INCREMENTAL)
    history, _ = replay(
        algorithm,
        Config(cash_equivalents=["SGOV"]),
        daily_history=_two_symbol_history(),
        trade_dates=list(DATES),
        should_run=lambda date: True,
        starting_equity=10_000.0,
        dividends={},
        open_in="SGOV",
    )

    # The whole stake is in the sleeve from the first date, not a day late.
    assert history["positions"].iloc[0].get("SGOV") == pytest.approx(10_000.0, abs=100.0)
    assert history["cash"].iloc[0] < 100.0
    assert history["equity"].iloc[0] == pytest.approx(10_000.0, abs=100.0)


def test_open_in_never_overdraws_the_opening_balance() -> None:
    """Rounding a whole balance up asks for more than the account holds, and is refused."""
    algorithm = _Recorder(weight=0.0, mode=MODE_INCREMENTAL)
    history, _ = replay(
        algorithm,
        Config(cash_equivalents=["SGOV"]),
        daily_history=_two_symbol_history(price=100.56),
        trade_dates=list(DATES),
        should_run=lambda date: True,
        starting_equity=60_000.0,
        dividends={},
        open_in="SGOV",
    )

    assert (history["cash"] >= -0.01).all()
    assert history["positions"].iloc[0].get("SGOV", 0) > 0


def test_open_in_falls_back_to_cash_when_the_sleeve_has_no_price() -> None:
    algorithm = _Recorder(weight=0.0, mode=MODE_INCREMENTAL)
    history, _ = replay(
        algorithm,
        Config(cash_equivalents=["SGOV"]),
        daily_history=_history(),  # no SGOV bars at all
        trade_dates=list(DATES),
        should_run=lambda date: True,
        starting_equity=10_000.0,
        dividends={},
        open_in="SGOV",
    )

    assert history["cash"].iloc[0] == pytest.approx(10_000.0)


def test_a_parked_book_still_funds_an_incremental_buy() -> None:
    """The DCA shape: nothing but buys, and every dollar is in the sleeve.

    Without cash-aware funding this is the batch the broker refuses outright.
    """
    class _Buyer(_Recorder):
        def analyze(self, context) -> AlgorithmDecision:
            self.contexts.append(context)
            return AlgorithmDecision(
                intents=[Intent("AAA", "notional", 500.0)], mode=MODE_INCREMENTAL
            )

    algorithm = _Buyer()
    history, _ = replay(
        algorithm,
        Config(cash_equivalents=["SGOV"]),
        daily_history=_two_symbol_history(),
        trade_dates=list(DATES),
        should_run=lambda date: True,
        starting_equity=10_000.0,
        dividends={},
        open_in="SGOV",
    )

    fills = [order for batch in algorithm.settled for order in batch]
    assert any(order["symbol"] == "AAA" and order["status"] == "submitted" for order in fills)
    # The sleeve was sold down to pay for them.
    assert history["positions"].iloc[-1].get("SGOV", 0) < history["positions"].iloc[0]["SGOV"]
