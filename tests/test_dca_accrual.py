"""Budget behaviour for DCA and Bursty DCA.

Each of these fails silently otherwise: the bot keeps running, keeps placing orders, and
simply spends the wrong amount.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.algorithms.dca import accrual
from src.algorithms.dca.accrual import HOURS_IN_MONTH, SymbolState, accrue, min_executable
from src.algorithms.dca.bot import DCAAlgorithm
from src.algorithms.dca.bursty import BurstyConfig, BurstyDCAAlgorithm, evaluate_trigger
from src.core.interfaces import MODE_INCREMENTAL, AlgorithmContext, PortfolioSnapshot

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FakeConfig:
    """Only the attributes the DCA algorithms actually read."""

    def __init__(self, min_trade_dollars: float = 0.0, algorithm_configs: dict[str, Any] | None = None) -> None:
        self.account_id = "test"
        self.min_trade_dollars = min_trade_dollars
        self.algorithm_configs = dict(algorithm_configs or {})


@pytest.fixture
def state_store(monkeypatch) -> dict[str, Any]:
    """Back the accrual state store with a plain dict for the duration of a test."""
    store: dict[str, Any] = {}
    monkeypatch.setattr(accrual, "load_state", lambda key, default: store.get(key, default))
    monkeypatch.setattr(accrual, "save_state", lambda key, value: store.__setitem__(key, value))
    return store


def _plan(symbol: str = "AAA", monthly_budget: float = 100.0) -> dict[str, Any]:
    return {
        "buy": {"items": [{"symbol": symbol, "amount": monthly_budget}]},
        "sell": {"items": []},
    }


class Simulation:
    """Drive an algorithm over a series of runs, filling its orders the way step 2 would."""

    def __init__(self, algorithm, config, prices, *, fractional: bool = False):
        self.algorithm = algorithm
        self.config = config
        self.prices = prices
        self.fractional = fractional
        self.positions: dict[str, float] = {}
        self.deployed = 0.0
        self.trades: list[tuple[datetime, str, float]] = []

    def run(self, now: datetime) -> None:
        result = self.algorithm.analyze(
            AlgorithmContext(
                config=self.config,
                latest_prices=self.prices,
                positions={},
                equity=0.0,
                account_id="test",
                timestamp=now,
            )
        )
        assert result.mode == MODE_INCREMENTAL
        snapshot = PortfolioSnapshot(positions=dict(self.positions), equity=100_000.0)
        intents = self.algorithm.refine(
            result.resolved_intents(), result.signals, snapshot, self.prices, self.config, now
        )

        order_results = []
        for intent in intents:
            price = self.prices[intent.symbol]
            raw_shares = intent.value / price
            shares = round(raw_shares, 2) if self.fractional else float(int(raw_shares))
            if shares == 0:
                continue
            self.positions[intent.symbol] = self.positions.get(intent.symbol, 0.0) + shares
            self.deployed += shares * price
            self.trades.append((now, intent.symbol, shares * price))
            order_results.append(
                {
                    "symbol": intent.symbol,
                    "status": "submitted",
                    "quantity": abs(shares),
                    "latest_price": price,
                }
            )
        self.algorithm.settle(self.config, order_results, intents)


def _dca_simulation(monkeypatch, *, prices, plan, min_trade_dollars=0.0, fractional=False) -> Simulation:
    config = FakeConfig(min_trade_dollars=min_trade_dollars)
    monkeypatch.setattr(DCAAlgorithm, "plan", lambda self, _config: plan)
    monkeypatch.setattr("src.algorithms.dca.bot.broker_supports_fractional_shares", lambda account_id: fractional)
    return Simulation(DCAAlgorithm(config), config, prices, fractional=fractional)


# --------------------------------------------------------------------------------------
# Accrual arithmetic
# --------------------------------------------------------------------------------------


def test_first_run_only_seeds_the_clock() -> None:
    """Budget is never accrued retroactively for time before the plan was known."""
    state = accrue(SymbolState(), monthly_budget=100.0, now=START)

    assert state.accrued == 0.0
    assert state.last_run_at == START.isoformat()


def test_accrual_is_proportional_to_elapsed_wall_clock_time() -> None:
    seeded = accrue(SymbolState(), 100.0, START)
    half_a_month = accrue(seeded, 100.0, START + timedelta(hours=HOURS_IN_MONTH / 2))

    assert half_a_month.accrued == pytest.approx(50.0)


def test_a_long_outage_cannot_deploy_unbounded_catch_up() -> None:
    seeded = accrue(SymbolState(), 100.0, START)
    after_a_year = accrue(seeded, 100.0, START + timedelta(days=365))

    assert after_a_year.accrued == pytest.approx(100.0)


def test_month_rollover_resets_the_cumulative_deployment_cap() -> None:
    january = accrue(SymbolState(deployed_this_month=250.0, month="2026-01"), 100.0, START)
    february = accrue(january, 100.0, datetime(2026, 2, 2, tzinfo=timezone.utc))

    assert january.deployed_this_month == 250.0
    assert february.deployed_this_month == 0.0


def test_min_executable_is_one_share_on_a_whole_share_brokerage() -> None:
    assert min_executable(500.0, min_trade_dollars=1.0, supports_fractional_shares=False) == 500.0
    assert min_executable(500.0, min_trade_dollars=1.0, supports_fractional_shares=True) == 1.0


# --------------------------------------------------------------------------------------
# 1. Budget conservation
# --------------------------------------------------------------------------------------


def test_a_month_of_hourly_runs_deploys_about_one_month_of_budget(monkeypatch, state_store) -> None:
    simulation = _dca_simulation(monkeypatch, prices={"AAA": 25.0}, plan=_plan(monthly_budget=100.0))

    for hour in range(int(HOURS_IN_MONTH) + 1):
        simulation.run(START + timedelta(hours=hour))

    floor_dollars = min_executable(25.0, 0.0, False)
    assert simulation.deployed == pytest.approx(100.0, abs=floor_dollars)


# --------------------------------------------------------------------------------------
# 2. Cadence independence -- the reason the budget is not divided by run count
# --------------------------------------------------------------------------------------


def test_hourly_and_daily_cadences_deploy_the_same_monthly_total(monkeypatch, state_store) -> None:
    hourly = _dca_simulation(monkeypatch, prices={"AAA": 25.0}, plan=_plan(monthly_budget=100.0))
    for hour in range(int(HOURS_IN_MONTH) + 1):
        hourly.run(START + timedelta(hours=hour))

    state_store.clear()
    daily = _dca_simulation(monkeypatch, prices={"AAA": 25.0}, plan=_plan(monthly_budget=100.0))
    for day in range(int(HOURS_IN_MONTH / 24) + 1):
        daily.run(START + timedelta(days=day))

    assert hourly.deployed == pytest.approx(daily.deployed, abs=25.0)
    # ~141 hourly runs against ~31 daily ones: dividing the budget per run would have made
    # these differ by more than 4x.
    assert hourly.deployed == pytest.approx(100.0, abs=25.0)


def test_an_empty_plan_trades_nothing(monkeypatch, state_store) -> None:
    """Turning DCA off is the algorithm bot's switch, not the plan's -- a disabled bot never
    reaches ``analyze`` at all. Emptying the plan is the only "off" the algorithm itself has.
    """
    simulation = _dca_simulation(
        monkeypatch, prices={"AAA": 25.0}, plan={"buy": {"items": []}, "sell": {"items": []}}
    )

    for day in range(31):
        simulation.run(START + timedelta(days=day))

    assert simulation.deployed == 0.0


# --------------------------------------------------------------------------------------
# 4. Whole-share brokerages
# --------------------------------------------------------------------------------------


def test_a_small_budget_against_an_expensive_share_eventually_trades(monkeypatch, state_store) -> None:
    """$100/month against a $500 ETF on Schwab: it must accrue ~5 months, then actually trade."""
    simulation = _dca_simulation(monkeypatch, prices={"VOO": 500.0}, plan=_plan("VOO", 100.0))

    for day in range(int(6 * HOURS_IN_MONTH / 24)):
        simulation.run(START + timedelta(days=day))

    assert simulation.trades, "the position must eventually trade rather than never"
    first_trade_months = (simulation.trades[0][0] - START).total_seconds() / 3600 / HOURS_IN_MONTH
    assert 4.5 <= first_trade_months <= 5.5
    assert simulation.deployed == pytest.approx(500.0)


def test_a_fractional_brokerage_trades_the_same_budget_immediately(monkeypatch, state_store) -> None:
    simulation = _dca_simulation(
        monkeypatch, prices={"VOO": 500.0}, plan=_plan("VOO", 100.0), fractional=True, min_trade_dollars=1.0
    )

    for day in range(31):
        simulation.run(START + timedelta(days=day))

    assert simulation.deployed == pytest.approx(100.0, abs=5.0)


def test_the_view_warns_when_a_budget_takes_months_to_clear_one_share() -> None:
    algorithm = DCAAlgorithm(FakeConfig())

    warning = algorithm.whole_share_warning("VOO", monthly_budget=100.0, price=500.0, fractional=False)

    assert "~5 months" in warning
    assert algorithm.whole_share_warning("VOO", 100.0, 500.0, fractional=True) == ""


# --------------------------------------------------------------------------------------
# 3. Clamp vs accrual -- the guard that is stated in months, not per-run increments
# --------------------------------------------------------------------------------------


def _bursty_simulation(monkeypatch, *, prices, plan, fires, settings=None) -> Simulation:
    config = FakeConfig(algorithm_configs={} if settings is None else settings)
    monkeypatch.setattr(BurstyDCAAlgorithm, "plan", lambda self, _config: plan)
    monkeypatch.setattr("src.algorithms.dca.bot.broker_supports_fractional_shares", lambda account_id: True)
    monkeypatch.setattr(
        BurstyDCAAlgorithm,
        "trigger",
        lambda self, symbol, context, plan, months_behind=0.0: {
            "fires": context.timestamp in fires,
            "reason": "Valley" if context.timestamp in fires else "Waiting for valley",
            "detail": {},
        },
    )
    return Simulation(BurstyDCAAlgorithm(config), config, prices, fractional=True)


def test_bursty_deployment_tracks_the_value_path_within_the_clamp(monkeypatch, state_store) -> None:
    """60 sessions, 3 triggers.

    The clamp is expressed against the monthly budget, so a burst can deploy several months of
    path in one trade. Clamping against the per-run increment instead would leave the position
    permanently behind the path while erroring nowhere -- which is what this pins down.
    """
    sessions = [START + timedelta(days=day) for day in range(60)]
    fires = {sessions[20], sessions[40], sessions[59]}
    simulation = _bursty_simulation(
        monkeypatch, prices={"AAA": 50.0}, plan=_plan(monthly_budget=300.0), fires=fires
    )

    for session in sessions:
        simulation.run(session)

    assert len(simulation.trades) == 3
    elapsed_months = (sessions[-1] - START).total_seconds() / 3600 / HOURS_IN_MONTH
    path_value = 300.0 * elapsed_months
    # Deployment tracks the path, not the number of sessions that happened to fire.
    assert simulation.deployed == pytest.approx(path_value, rel=0.05)


def test_a_single_burst_is_clamped_to_a_multiple_of_the_monthly_budget(monkeypatch, state_store) -> None:
    sessions = [START + timedelta(days=day) for day in range(180)]
    simulation = _bursty_simulation(
        monkeypatch, prices={"AAA": 50.0}, plan=_plan(monthly_budget=100.0), fires={sessions[-1]}
    )

    for session in sessions:
        simulation.run(session)

    # Six months of path is available, but one trade may deploy at most 3x the monthly budget.
    assert len(simulation.trades) == 1
    assert simulation.trades[0][2] == pytest.approx(300.0, abs=50.0)


def test_cumulative_monthly_deployment_is_capped(monkeypatch, state_store) -> None:
    sessions = [START + timedelta(days=day) for day in range(28)]
    settings = {"bursty_dca": {"max_monthly_multiple": 1.0, "max_trade_multiple": 3.0}}
    simulation = _bursty_simulation(
        monkeypatch,
        prices={"AAA": 50.0},
        plan=_plan(monthly_budget=100.0),
        fires=set(sessions),
        settings=settings,
    )

    for session in sessions:
        simulation.run(session)

    assert simulation.deployed <= 100.0 + 1e-6


# --------------------------------------------------------------------------------------
# Bursty triggers
# --------------------------------------------------------------------------------------


def _bars(closes: list[float]):
    import pandas as pd

    return pd.DataFrame({"close": closes})


def test_the_regime_gate_blocks_buying_into_a_decline() -> None:
    falling = _bars([200.0 - index for index in range(220)])

    outcome = evaluate_trigger(falling, buying=True, settings=BurstyConfig())

    assert not outcome["fires"]
    assert outcome["reason"] == "Below 200-day MA"


def test_an_uptrend_without_a_dip_waits_for_a_valley() -> None:
    rising = _bars([100.0 + index for index in range(220)])

    outcome = evaluate_trigger(rising, buying=True, settings=BurstyConfig())

    assert not outcome["fires"]
    assert outcome["reason"] == "Waiting for valley"


def test_a_dip_inside_an_uptrend_fires() -> None:
    closes = [100.0 + index for index in range(220)]
    closes[-1] = closes[-1] * 0.90  # a sharp one-day drop, still far above the 200-day MA
    outcome = evaluate_trigger(_bars(closes), buying=True, settings=BurstyConfig())

    assert outcome["fires"]
    assert outcome["detail"]["close"] < outcome["detail"]["ma_200"] * 2


def test_no_history_never_fires() -> None:
    assert not evaluate_trigger(None, buying=True, settings=BurstyConfig())["fires"]


def test_the_budget_line_restates_a_monthly_budget_at_a_human_scale() -> None:
    from src.algorithms.dca.bot import budget_line

    assert budget_line(100.0) == "$100/month ~ $4.60/trading day"
    assert budget_line(100.0, equity=9_000.0).endswith("~ 1.1% of equity")
    assert budget_line(0.0) == ""


# =========================================================================================
# Backlog-aware pickiness
# =========================================================================================


def test_the_oversold_test_relaxes_as_the_budget_falls_behind() -> None:
    """Waiting for a dip is the strategy; waiting forever is not.

    In a market that only rises, %B never breaks the lower band and RSI(2) never reaches 10, so
    the budget accrues and the eventual entry lands above every price that was declined. This
    lets the two oversold thresholds ease in proportion to how far behind the budget has fallen.
    """
    from src.algorithms.dca.bursty import relaxed_thresholds

    strict = BurstyConfig(backlog_relax_months=0.0)
    assert relaxed_thresholds(strict, months_behind=6.0) == (strict.percent_b_threshold, strict.rsi_threshold)

    adaptive = BurstyConfig(backlog_relax_months=3.0, rsi_threshold=10.0, percent_b_threshold=0.0)
    assert relaxed_thresholds(adaptive, 0.0) == (0.0, 10.0)
    # Halfway behind: halfway between picky and "buy on anything" (%B 1.0, RSI 100).
    percent_b, rsi = relaxed_thresholds(adaptive, 1.5)
    assert percent_b == pytest.approx(0.5)
    assert rsi == pytest.approx(55.0)
    # Past the horizon it stops easing rather than running away.
    assert relaxed_thresholds(adaptive, 99.0) == (1.0, 100.0)


def test_a_backlogged_symbol_buys_where_an_on_schedule_one_waits() -> None:
    """The whole point: same bars, same day, different answer because of the backlog."""
    rising = _bars([100 + index * 0.5 for index in range(260)])
    adaptive = BurstyConfig(backlog_relax_months=3.0)

    assert not evaluate_trigger(rising, buying=True, settings=adaptive, months_behind=0.0)["fires"]
    assert evaluate_trigger(rising, buying=True, settings=adaptive, months_behind=3.0)["fires"]


def test_the_regime_gate_never_relaxes() -> None:
    """Relaxing the trend filter would trade a small problem for this design's worst one."""
    falling = _bars([200 - index * 0.5 for index in range(260)])

    outcome = evaluate_trigger(falling, buying=True, settings=BurstyConfig(backlog_relax_months=3.0),
                               months_behind=99.0)

    assert not outcome["fires"]
    assert "day MA" in outcome["reason"]


def test_a_repaying_symbol_does_not_read_as_accruing_negative_dollars() -> None:
    """``settle`` subtracts the filled notional with no floor, so accrual can go negative.

    That is deliberate -- it is what keeps the long-run spend rate honest after Bursty DCA
    sizes a trade above what had accrued in order to catch up to its value path. Rendered
    raw, though, it reached the dashboard as "Accruing ($-450 of $1)".
    """
    from src.algorithms.dca.bot import DCAAlgorithm

    reason = DCAAlgorithm({}).reason(
        SymbolState(accrued=-450.0),
        floor_dollars=1.0,
        trigger={"fires": True, "detail": {}},
        ready=False,
    )

    assert reason == "Ahead of plan (repaying $450)"


def test_bursty_never_promises_a_trade_value_averaging_can_still_veto() -> None:
    """The signal view is step 1; value averaging sizes the trade in step 2.

    A symbol already at or above its value path is dropped in ``refine``, which the view
    cannot see -- it has no portfolio. A bare "Ready" told the reader an order was coming that
    would never be placed, so the row has to stop short of promising one.
    """
    from src.algorithms.dca.bursty import BurstyDCAAlgorithm

    algorithm = BurstyDCAAlgorithm({})
    ready_state = SymbolState(accrued=500.0)
    fires = {"fires": True, "detail": {"value_averaged": True}}

    assert algorithm.reason(ready_state, 1.0, fires, ready=True) == "Ready (sized to value path)"
    # With value averaging off, step 2 takes the intent as sized, so "Ready" is the truth.
    plain = {"fires": True, "detail": {"value_averaged": False}}
    assert algorithm.reason(ready_state, 1.0, plain, ready=True) == "Ready"
