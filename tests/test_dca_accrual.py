"""Budget behaviour for Bursty DCA.

Each of these fails silently otherwise: the bot keeps running, keeps placing orders, and
simply spends the wrong amount.

The accrual half of the algorithm is exercised through a deliberately neutral tuning --
``scaling_factor: 0`` and ``relax_depth: 0``, so both sizing factors collapse to 1.0 and a run
deploys the plain plan rate. That is what the retired steady ``DCAAlgorithm`` used to provide
as a separate class, and it is what isolates accrual arithmetic from the sizing model.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

import src.algorithms.bursty_dca.algorithm as mod
from src.algorithms.bursty_dca.algorithm import (
    HOURS_IN_MONTH,
    MAX_SIGMA,
    BurstyDCAAlgorithm,
    SymbolState,
    accrue,
    conviction,
    evaluate_valuation,
    min_executable,
    planned_order_size,
    willingness,
)
from src.algorithms.bursty_dca.config import BurstyConfig
from src.core.interfaces import ACTION_BLOCKED, MODE_INCREMENTAL, AlgorithmContext

START = datetime(2026, 1, 1, tzinfo=timezone.utc)

#: Tuning that removes both sizing factors, leaving the plan rate alone: no valuation response,
#: no backlog response, and a monthly cap of exactly one budget.
STEADY_SETTINGS = {
    "bursty_dca": {"scaling_factor": 0.0, "relax_depth": 0.0, "max_monthly_multiple": 1.0}
}


class FakeConfig:
    """Only the attributes the algorithm actually reads."""

    def __init__(self, algorithm_configs: dict[str, Any] | None = None) -> None:
        self.account_id = "test"
        self.algorithm_configs = dict(algorithm_configs or {})


def _plan(symbol: str = "AAA", monthly_budget: float = 100.0) -> dict[str, Any]:
    return {
        "buy": {"items": [{"symbol": symbol, "amount": monthly_budget}]},
        "sell": {"items": []},
    }


def _bars(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": closes})


def _flat_bars(price: float, count: int = 260) -> pd.DataFrame:
    """A series that oscillates around ``price`` and *ends on it*, so valuation reads neutral.

    Two things have to be true at once and it is easy to get only one. A dead-flat series has
    no deviation to divide by, so ``evaluate_valuation`` refuses it outright -- there must be
    variation. But the last close is what gets scored, so a series that merely oscillates
    leaves it a full sigma off the mean and a "neutral" fixture silently tests a 0.5x
    conviction instead of a 1.0x one.
    """
    closes = [price + (0.5 if index % 2 else -0.5) for index in range(count)]
    closes[-1] = price
    return _bars(closes)


class Simulation:
    """Drive an algorithm over a series of runs, filling its orders the way ``execute`` would.

    Carries the state by hand -- context in, plan out, ``state_after`` deciding what the next
    run sees -- because that is exactly what the live path does through DuckDB and the replay
    does through its ephemeral store. Nothing here has to be patched to keep the two apart:
    the algorithm never reaches for a store, so the store this harness does not have cannot
    leak into a test.
    """

    def __init__(self, algorithm, config, prices, *, bars=None, fractional: bool = False):
        self.algorithm = algorithm
        self.config = config
        self.prices = prices
        self.bars = bars or {}
        self.fractional = fractional
        self.state: dict[str, Any] = {}
        self.positions: dict[str, float] = {}
        self.deployed = 0.0
        self.trades: list[tuple[datetime, str, float]] = []

    def run(self, now: datetime) -> None:
        plan = self.algorithm.plan(
            AlgorithmContext(
                config=self.config,
                latest_prices=self.prices,
                daily_bars_by_symbol=dict(self.bars),
                positions=dict(self.positions),
                equity=100_000.0,
                account_id="test",
                state=self.state,
                timestamp=now,
            )
        )
        assert plan.mode == MODE_INCREMENTAL

        order_results = []
        for intent in plan.intents:
            price = self.prices[intent.symbol]
            raw_shares = intent.value / price
            shares = round(raw_shares, 2) if self.fractional else float(int(raw_shares))
            if shares == 0:
                continue
            self.positions[intent.symbol] = self.positions.get(intent.symbol, 0.0) + shares
            self.deployed += shares * price
            self.trades.append((now, intent.symbol, shares * price))
            order_results.append({
                "symbol": intent.symbol,
                "status": "submitted",
                "quantity": abs(shares),
                "latest_price": price,
            })
        self.state = self.algorithm.state_after(plan, {"order_results": order_results})


def _simulation(monkeypatch, *, prices, plan, settings=None, fractional=True) -> Simulation:
    config = FakeConfig(algorithm_configs=settings or {})
    monkeypatch.setattr(BurstyDCAAlgorithm, "budget_plan", lambda self, _config: plan)
    monkeypatch.setattr(
        "src.algorithms.bursty_dca.algorithm.broker_supports_fractional_shares",
        lambda account_id: fractional,
    )
    bars = {symbol: _flat_bars(price) for symbol, price in prices.items()}
    return Simulation(BurstyDCAAlgorithm(config), config, prices, bars=bars, fractional=fractional)


def _steady_simulation(monkeypatch, *, prices, plan, fractional=False) -> Simulation:
    """The algorithm with both sizing factors tuned out, so only accrual and the cap act."""
    return _simulation(
        monkeypatch, prices=prices, plan=plan, settings=STEADY_SETTINGS, fractional=fractional
    )


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


def test_min_executable_is_derived_from_share_rounding_not_from_config() -> None:
    """There is no configured trade minimum any more: the floor is what rounding will keep."""
    assert min_executable(500.0, supports_fractional_shares=False) == 500.0
    # Two decimal places of fractional precision, so a hundredth of a share.
    assert min_executable(500.0, supports_fractional_shares=True) == 5.0


# --------------------------------------------------------------------------------------
# 1. Budget conservation
# --------------------------------------------------------------------------------------


def test_a_month_of_hourly_runs_deploys_about_one_month_of_budget(monkeypatch) -> None:
    simulation = _steady_simulation(monkeypatch, prices={"AAA": 25.0}, plan=_plan(monthly_budget=100.0))

    for hour in range(int(HOURS_IN_MONTH) + 1):
        simulation.run(START + timedelta(hours=hour))

    assert simulation.deployed == pytest.approx(100.0, abs=min_executable(25.0, False))


# --------------------------------------------------------------------------------------
# 2. Cadence independence -- the reason the budget is not divided by run count
# --------------------------------------------------------------------------------------


def test_hourly_and_daily_cadences_deploy_the_same_monthly_total(monkeypatch) -> None:
    hourly = _steady_simulation(monkeypatch, prices={"AAA": 25.0}, plan=_plan(monthly_budget=100.0))
    for hour in range(int(HOURS_IN_MONTH) + 1):
        hourly.run(START + timedelta(hours=hour))

    # A fresh simulation starts from an empty state, which is the whole isolation story now:
    # the two cadences share nothing because neither reaches for a store.
    daily = _steady_simulation(monkeypatch, prices={"AAA": 25.0}, plan=_plan(monthly_budget=100.0))
    for day in range(int(HOURS_IN_MONTH / 24) + 1):
        daily.run(START + timedelta(days=day))

    assert hourly.deployed == pytest.approx(daily.deployed, abs=25.0)
    # ~141 hourly runs against ~31 daily ones: dividing the budget per run would have made
    # these differ by more than 4x.
    assert hourly.deployed == pytest.approx(100.0, abs=25.0)


def test_an_empty_plan_trades_nothing(monkeypatch) -> None:
    """Turning the algorithm off is the binding's switch, not the plan's -- a disabled binding
    never reaches ``plan`` at all. Emptying the plan is the only "off" the algorithm has.
    """
    simulation = _steady_simulation(
        monkeypatch, prices={"AAA": 25.0}, plan={"buy": {"items": []}, "sell": {"items": []}}
    )

    for day in range(31):
        simulation.run(START + timedelta(days=day))

    assert simulation.deployed == 0.0


# --------------------------------------------------------------------------------------
# 3. Whole-share brokerages
# --------------------------------------------------------------------------------------


def test_a_small_budget_against_an_expensive_share_eventually_trades(monkeypatch) -> None:
    """$100/month against a $500 ETF on Schwab: it must accrue ~5 months, then actually trade.

    Sizing alone can never get there -- the factors would have to reach 5x and ``willingness``
    saturates below 2x -- so this is specifically the accrued-balance lift in
    ``planned_order_size``, and without it the money accrues forever.
    """
    simulation = _steady_simulation(monkeypatch, prices={"VOO": 500.0}, plan=_plan("VOO", 100.0))

    for day in range(int(6 * HOURS_IN_MONTH / 24)):
        simulation.run(START + timedelta(days=day))

    assert simulation.trades, "the position must eventually trade rather than never"
    first_trade_months = (simulation.trades[0][0] - START).total_seconds() / 3600 / HOURS_IN_MONTH
    assert 4.5 <= first_trade_months <= 5.5
    assert simulation.deployed == pytest.approx(500.0)


def test_a_fractional_brokerage_trades_the_same_budget_immediately(monkeypatch) -> None:
    simulation = _steady_simulation(
        monkeypatch, prices={"VOO": 500.0}, plan=_plan("VOO", 100.0), fractional=True
    )

    for day in range(31):
        simulation.run(START + timedelta(days=day))

    assert simulation.deployed == pytest.approx(100.0, abs=5.0)


# --------------------------------------------------------------------------------------
# 4. The monthly cap
# --------------------------------------------------------------------------------------


def test_cumulative_monthly_deployment_is_capped(monkeypatch) -> None:
    settings = {"bursty_dca": {"max_monthly_multiple": 1.0}}
    simulation = _simulation(
        monkeypatch, prices={"AAA": 50.0}, plan=_plan(monthly_budget=100.0), settings=settings
    )

    for day in range(28):
        simulation.run(START + timedelta(days=day))

    assert simulation.deployed <= 100.0 + 1e-6


# --------------------------------------------------------------------------------------
# 5. Valuation: distance from the moving average, in sigma
# --------------------------------------------------------------------------------------


def test_valuation_is_signed_so_a_rich_price_can_be_reported_as_rich() -> None:
    """The predecessor measured drawdown from a running peak and could only say "cheap" or
    "neutral". Two-sidedness is the whole reason this replaced it."""
    rising = [100.0 + index * 0.5 for index in range(260)]

    cheap = evaluate_valuation(_bars(rising[:-1] + [rising[-1] * 0.5]), BurstyConfig())
    dear = evaluate_valuation(_bars(rising), BurstyConfig())

    assert cheap["detail"]["z"] > 0
    assert dear["detail"]["z"] < 0


def test_dislocation_is_clamped_so_a_bad_bar_cannot_size_an_order() -> None:
    closes = [100.0 + (index % 3) for index in range(260)]
    closes[-1] = 0.01  # a split adjustment that never happened

    outcome = evaluate_valuation(_bars(closes), BurstyConfig())

    assert outcome["detail"]["z"] == MAX_SIGMA


def test_a_series_with_no_deviation_sizes_at_plan_rather_than_refusing() -> None:
    """A price that has not moved is not dislocated, so the honest reading is neutral -- and
    neutral means straight DCA, not a symbol that accrues forever over a data quirk."""
    outcome = evaluate_valuation(_bars([200.0] * 260), BurstyConfig())

    assert outcome["ok"] is True
    assert outcome["detail"]["z"] == 0.0
    assert conviction(outcome["detail"]["z"], buying=True, settings=BurstyConfig()) == 1.0


def test_valuation_refuses_a_series_shorter_than_its_window() -> None:
    assert evaluate_valuation(None, BurstyConfig())["ok"] is False
    assert evaluate_valuation(_bars([100.0] * 10), BurstyConfig())["ok"] is False


def test_conviction_reads_the_same_dislocation_backwards_for_a_sell() -> None:
    """A buy wants price below its average, a sell wants it above. One number, two readings."""
    settings = BurstyConfig(scaling_factor=0.5)

    assert conviction(1.0, buying=True, settings=settings) == pytest.approx(1.5)
    assert conviction(1.0, buying=False, settings=settings) == pytest.approx(0.5)
    assert conviction(-1.0, buying=False, settings=settings) == pytest.approx(1.5)


def test_conviction_never_goes_negative() -> None:
    """A buy bucket that looks badly priced buys nothing. It must never invert into a sell."""
    assert conviction(-3.0, buying=True, settings=BurstyConfig(scaling_factor=2.0)) == 0.0


# --------------------------------------------------------------------------------------
# 6. Backlog resistance -- the successor to the old relax threshold
# --------------------------------------------------------------------------------------


def test_willingness_is_one_at_plan_and_moves_smoothly_either_side() -> None:
    settings = BurstyConfig(relax_months=2.0, relax_depth=0.7)

    assert willingness(0.0, settings) == pytest.approx(1.0)
    assert willingness(2.0, settings) > 1.0
    assert willingness(-2.0, settings) < 1.0
    # Symmetric about the plan rate: falling a month behind and running a month ahead pull the
    # size by the same amount in opposite directions.
    assert willingness(1.0, settings) - 1.0 == pytest.approx(1.0 - willingness(-1.0, settings))


def test_willingness_saturates_rather_than_growing_without_bound() -> None:
    """A threshold changed character in one step and a linear term would size a decade of
    absence into a single order. The curve is bounded by construction."""
    settings = BurstyConfig(relax_months=2.0, relax_depth=0.7)

    assert willingness(100.0, settings) <= 1.7 + 1e-9
    assert willingness(-100.0, settings) >= 0.3 - 1e-9


def test_resistance_is_continuous_rather_than_a_threshold() -> None:
    """The property the old ``backlog_relax_months`` threshold could not have: no two nearby
    backlogs behave dramatically differently."""
    settings = BurstyConfig()
    steps = [willingness(months / 10.0, settings) for months in range(-40, 41)]

    assert all(later > earlier for earlier, later in zip(steps, steps[1:]))
    assert max(abs(later - earlier) for earlier, later in zip(steps, steps[1:])) < 0.05


def test_a_strong_signal_still_deploys_against_heavy_resistance() -> None:
    """The reason conviction scales the overdraft rather than a fixed number sitting there:
    an exceptional price can borrow months forward, an ordinary one at the same backlog cannot
    borrow at all. A fixed overdraft clamps both to the same figure and erases the distinction
    exactly where it matters."""
    settings = BurstyConfig(scaling_factor=0.5)
    overspent = -4.0

    exceptional = planned_order_size(500.0, MAX_SIGMA, True, overspent, 0.0, settings)
    ordinary = planned_order_size(500.0, 0.0, True, overspent, 0.0, settings)

    assert exceptional > 400.0
    assert ordinary == 0.0


def test_the_long_run_spend_rate_is_the_plan_rate_at_default_tuning(monkeypatch) -> None:
    """Measured over months, at defaults, with nothing tuned out.

    The steady tuning cannot catch this because it zeroes the very factor that causes it, and a
    single month cannot either: ``max_monthly_multiple`` alone holds any cadence to 3x budget
    within one month, so a one-month test passes whether or not ``spending_allowance`` exists.
    It just passes at three times the plan rate, every month, forever.

    The allowance is what makes the *rate* the plan rate rather than the cap rate, because it
    bounds each run against the balance actually accrued instead of against a monthly ceiling
    that resets.
    """
    budget = 600.0
    months = 3
    settings = BurstyConfig()

    hourly = _simulation(monkeypatch, prices={"AAA": 20.0}, plan=_plan(monthly_budget=budget))
    for hour in range(int(months * HOURS_IN_MONTH) + 1):
        hourly.run(START + timedelta(hours=hour))

    daily = _simulation(monkeypatch, prices={"AAA": 20.0}, plan=_plan(monthly_budget=budget))
    for day in range(int(months * HOURS_IN_MONTH / 24) + 1):
        daily.run(START + timedelta(days=day))

    # ~4.5x the runs, the same money: cadence controls only the opportunity to act.
    assert hourly.deployed == pytest.approx(daily.deployed, rel=0.15)

    # The plan rate plus the bounded overdraft, and nowhere near the 3x/month the monthly cap
    # would permit on its own.
    overdraft = budget * settings.relax_months
    for deployed in (hourly.deployed, daily.deployed):
        assert 0.75 * months * budget <= deployed <= months * budget + overdraft


# --------------------------------------------------------------------------------------
# 7. The preview and the order must agree
# --------------------------------------------------------------------------------------


def test_planned_order_size_matches_what_a_run_would_place() -> None:
    """size = |budget| x conviction x willingness, clamped to the month's remaining cap room --
    the same formula ``plan`` applies, so the Next-order column cannot drift from what
    actually reaches the market."""
    settings = BurstyConfig(scaling_factor=0.5, relax_depth=0.0, max_monthly_multiple=3.0)

    assert planned_order_size(100.0, 0.0, True, 0.0, 0.0, settings) == pytest.approx(100.0)
    # 1 sigma below the average scales the buy to 1.5x budget.
    assert planned_order_size(100.0, 1.0, True, 0.0, 0.0, settings) == pytest.approx(150.0)
    # A big dislocation clamps at the monthly cap (3x).
    assert planned_order_size(100.0, 20.0, True, 0.0, 0.0, settings) == pytest.approx(300.0)
    # Cap room is what is left after this month's deployments.
    assert planned_order_size(100.0, 20.0, True, 0.0, 250.0, settings) == pytest.approx(50.0)
    # Sells size off the absolute budget too.
    assert planned_order_size(-100.0, -1.0, False, 0.0, 0.0, settings) == pytest.approx(150.0)


def test_the_accrued_lift_never_spends_money_the_symbol_has_not_banked() -> None:
    settings = BurstyConfig(scaling_factor=0.0, relax_depth=0.0)

    # Banked enough for a share: send one rather than accruing forever.
    lifted = planned_order_size(100.0, 0.0, True, 0.0, 0.0, settings, accrued=520.0, floor_dollars=500.0)
    assert lifted == pytest.approx(500.0)

    # Not banked enough: stay small and keep waiting.
    waiting = planned_order_size(100.0, 0.0, True, 0.0, 0.0, settings, accrued=310.0, floor_dollars=500.0)
    assert waiting == pytest.approx(100.0)


def test_the_accrued_lift_cannot_resurrect_an_order_conviction_zeroed() -> None:
    """A bucket that wants none of its budget wants none of it at any size, however much has
    accrued -- otherwise the lift would sell a name the model just called cheap."""
    settings = BurstyConfig(scaling_factor=2.0)

    size = planned_order_size(
        100.0, 3.0, buying=False, backlog_months=0.0, deployed_this_month=0.0,
        settings=settings, accrued=10_000.0, floor_dollars=500.0,
    )

    assert size == 0.0


def test_plan_signals_expose_both_factors_verbatim(monkeypatch) -> None:
    """The dashboard shows why an order is the size it is without parsing prose."""
    config = FakeConfig()
    monkeypatch.setattr(BurstyDCAAlgorithm, "budget_plan", lambda self, _config: _plan(monthly_budget=25.0))
    monkeypatch.setattr(
        "src.algorithms.bursty_dca.algorithm.broker_supports_fractional_shares", lambda account_id: True
    )

    result = BurstyDCAAlgorithm(config).plan(
        AlgorithmContext(
            config=config,
            latest_prices={"AAA": 100.0},
            daily_bars_by_symbol={"AAA": _flat_bars(100.0)},
            positions={},
            equity=0.0,
            account_id="test",
            timestamp=START,
        )
    )

    row = result.signals["AAA"]
    assert row["conviction"] == pytest.approx(1.0, abs=0.05)
    # First run only seeds the clock, so nothing has accrued and the backlog is exactly zero.
    assert row["backlog_months"] == 0.0
    assert row["willingness"] == pytest.approx(1.0)
    assert row["plan_multiple"] == pytest.approx(1.0, abs=0.05)


# --------------------------------------------------------------------------------------
# 8. Headlines
# --------------------------------------------------------------------------------------


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "monthly_budget": 100.0,
        "fractional": True,
        "deployed": False,
        "state": SymbolState(),
        "size": 0.0,
        "floor_dollars": 1.0,
        "valuation": {"ok": True, "reason": "0.0σ below its average", "detail": {"z": 0.0}},
    }
    return {**row, **overrides}


def test_the_headline_reports_the_deployment_and_its_departure_from_budget() -> None:
    from src.algorithms.bursty_dca.signals import _headline

    headline = _headline(_row(monthly_budget=500.0, deployed=True, size=750.0))

    assert headline == "Deploying $750 (1.5x budget)"


def test_the_headline_says_which_way_the_price_went_when_conviction_zeroes() -> None:
    """"Holding off" alone gives a reader nothing to act on; the direction is the reason."""
    from src.algorithms.bursty_dca.signals import _headline

    valuation = {"ok": True, "reason": "2.5σ above its average", "detail": {"z": -2.5}}

    assert "2.5σ above its average" in _headline(_row(size=0.0, valuation=valuation))


def test_the_view_says_how_far_off_a_budget_is_from_clearing_one_share() -> None:
    """A $100/month budget against a $500 share waits months. Say how many and how much is
    banked, rather than reporting "too small" every run and leaving the reader to divide."""
    from src.algorithms.bursty_dca.signals import _headline

    headline = _headline(
        _row(fractional=False, size=100.0, floor_dollars=500.0, state=SymbolState(accrued=131.0))
    )

    assert "$131 banked" in headline
    assert "~4 months away" in headline


def test_a_priceless_symbol_says_so_rather_than_reporting_a_size() -> None:
    from src.algorithms.bursty_dca.signals import _headline

    valuation = {"ok": False, "reason": "No price history", "detail": {}}

    assert _headline(_row(valuation=valuation)) == "No price history"


# --------------------------------------------------------------------------------------
# 9. Selling trims, it never shorts
# --------------------------------------------------------------------------------------


def _sell_plan(symbol: str = "AAA", amount: float = 25.0) -> dict[str, Any]:
    return {
        "buy": {"items": []},
        "sell": {"items": [{"symbol": symbol, "amount": amount}]},
    }


def test_a_sell_with_nothing_held_says_so_instead_of_passing_quietly(monkeypatch) -> None:
    """A sell budget clamps to what is actually held, so an empty book zeroes the order before
    the price has any say. That is the whole reason the row cannot act -- and without a gate
    saying it, the row passed every visible test and still went nowhere."""
    config = FakeConfig()
    monkeypatch.setattr(BurstyDCAAlgorithm, "budget_plan", lambda self, _config: _sell_plan())
    monkeypatch.setattr(
        "src.algorithms.bursty_dca.algorithm.broker_supports_fractional_shares", lambda account_id: True
    )

    def context(positions: dict[str, float]) -> AlgorithmContext:
        return AlgorithmContext(
            config=config,
            latest_prices={"AAA": 100.0},
            daily_bars_by_symbol={"AAA": _flat_bars(100.0)},
            positions=positions,
            equity=0.0,
            account_id="test",
            timestamp=START,
        )

    algorithm = BurstyDCAAlgorithm(config)

    empty = algorithm.plan(context({}))
    row = empty.signals["AAA"]
    assert empty.intents == []
    assert row["action"] == ACTION_BLOCKED
    # The headline names the book, not the price.
    assert row["reason"].startswith("Nothing held to sell")
    trim = next(check for check in row["checks"] if check["label"] == "Position to trim")
    assert trim["ok"] is False and trim["blocking"] is True

    # The same plan against a held position orders the sell, and the gate passes.
    holding = algorithm.plan(context({"AAA": 3.0}))
    sell = next(intent for intent in holding.intents if intent.symbol == "AAA")
    assert sell.value < 0
    trim = next(check for check in holding.signals["AAA"]["checks"] if check["label"] == "Position to trim")
    assert trim["ok"] is True and trim["blocking"] is False
