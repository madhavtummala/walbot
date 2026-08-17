from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.algorithms.dual_momentum import (
    DualMomentumAlgorithm,
    base_scores,
    DualMomentumConfig,
    _in_cooldown,
    _record_exits,
    resolve_themes,
    analyze_universe,
    apply_turnover_filters,
    partial_adjustment,
    action_due,
    advance_run,
    crash_stop,
    theme_ranks,
    record_action,
    covariance_matrix,
    defensive_weights,
    eligibility,
    hold_eligibility,
    universe_data_ok,
    park_residual,
    score_to_weights,
    theme_allocation,
    track_eligibility,
    volatility_scale,
    zscores,
)
from src.core.interfaces import AlgorithmContext, PortfolioSnapshot
from src.data.state_store import ephemeral_state


# =========================================================================================
# Fixtures: synthetic bars with a shape we control exactly
# =========================================================================================


def daily_bars(start: float, end: float, days: int = 140) -> pd.DataFrame:
    """A straight line from ``start`` to ``end``, long enough to clear the 100-day window."""
    dates = pd.date_range("2026-01-01", periods=days, freq="D", tz="UTC")
    closes = [start + (end - start) * (index / max(days - 1, 1)) for index in range(days)]
    return pd.DataFrame({"timestamp": dates, "close": closes, "open": closes,
                         "high": closes, "low": closes, "volume": [1_000_000] * days})


def intraday_bars(start: float, end: float, bars: int = 400, freq: str = "15min") -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=bars, freq=freq, tz="UTC")
    closes = [start + (end - start) * (index / max(bars - 1, 1)) for index in range(bars)]
    return pd.DataFrame({"timestamp": dates, "close": closes, "open": closes,
                         "high": closes, "low": closes, "volume": [10_000] * bars})


class Runtime:
    """Minimal stand-in for the runtime config object algorithms read."""

    def __init__(self, **tuning):
        self.algorithm_configs = {"dual_momentum": tuning}
        self.account_id = "test"
        self.symbols = []
        self.cash_buffer = 0.0
        self.min_trade_dollars = 0.0
        self.rebalance_threshold = 0.0


def context_for(config: Runtime, daily: dict, intraday: dict, timestamp: datetime | None = None) -> AlgorithmContext:
    return AlgorithmContext(
        config=config,
        bars_by_symbol=daily,
        history_bars_by_symbol=intraday,
        latest_prices={symbol: float(frame["close"].iloc[-1]) for symbol, frame in daily.items()},
        timestamp=timestamp or datetime(2026, 6, 5, 15, 0, tzinfo=timezone.utc),
    )


# =========================================================================================
# Absolute momentum, and the data check that is all that remains of the regime layer
# =========================================================================================


def test_a_falling_universe_holds_the_defensive_sleeve_rather_than_the_least_bad_etf() -> None:
    """Cross-sectional ranking alone always owns something. Absolute momentum is what fixes it.

    This used to be the regime gate's job. With that gate removed, per-name ``eligibility`` is
    the only thing standing between a falling market and a fully invested book -- so this test
    matters more than it did, not less.
    """
    config = Runtime(risk_on_universe=["AAA", "BBB"], defensive_universe=["BIL"], benchmark="AAA")
    strategy = DualMomentumConfig.from_runtime_config(config)
    falling = {
        "AAA": daily_bars(120, 80),
        "BBB": daily_bars(120, 60),   # least bad is still falling
        "BIL": daily_bars(100, 101),
    }
    intraday = {"AAA": intraday_bars(120, 80), "BBB": intraday_bars(120, 60), "BIL": intraday_bars(100, 101)}

    outcome = analyze_universe(context_for(config, falling, intraday), strategy)

    assert outcome["weights"]["BIL"] > 0
    assert outcome["weights"]["AAA"] == 0
    assert outcome["weights"]["BBB"] == 0
    assert not outcome["entries"], "nothing should have qualified"


def test_a_thin_universe_is_reported_as_a_data_gap_not_a_bear_market() -> None:
    """A short cache would otherwise read as "below its 100-day average", which is a lie.

    The one piece of the regime layer that survives, and the reason it survives: this is a
    statement about the data, not about the market.
    """
    config = Runtime(risk_on_universe=["AAA", "BBB", "CCC", "DDD"], defensive_universe=["BIL"])
    strategy = DualMomentumConfig.from_runtime_config(config)
    scored = {
        "AAA": {"symbol": "AAA", "above_moving_average": True, "enough_history": True, "daily_bars": 140},
        "BBB": {"symbol": "BBB", "above_moving_average": False, "enough_history": False, "daily_bars": 0},
        "CCC": {"symbol": "CCC", "above_moving_average": False, "enough_history": False, "daily_bars": 0},
        "DDD": {"symbol": "DDD", "above_moving_average": False, "enough_history": False, "daily_bars": 0},
    }

    data = universe_data_ok(scored, strategy)

    assert data["data_ok"] is False
    assert data["coverage"] == 0.25
    assert "1 of 4 risk-on names" in data["detail"]
    assert "average" not in data["detail"], "a data gap must not be reported as a trend reading"


def test_the_data_check_has_no_opinion_about_trends() -> None:
    """What removing the breadth gate means concretely.

    Three of four names below their own average used to force the whole book defensive. Now it
    is not the portfolio layer's business at all -- ``eligibility`` will decline those three
    individually, and the fourth is still allowed to be held.
    """
    config = Runtime(risk_on_universe=["AAA", "BBB", "CCC", "DDD"], benchmark="AAA")
    strategy = DualMomentumConfig.from_runtime_config(config)
    scored = {
        "AAA": {"symbol": "AAA", "above_moving_average": True, "enough_history": True},
        "BBB": {"symbol": "BBB", "above_moving_average": False, "enough_history": True},
        "CCC": {"symbol": "CCC", "above_moving_average": False, "enough_history": True},
        "DDD": {"symbol": "DDD", "above_moving_average": False, "enough_history": True},
    }

    data = universe_data_ok(scored, strategy)

    assert data["data_ok"] is True, "every name is judgeable, so there is nothing to refuse"
    assert data["coverage"] == 1.0


# =========================================================================================
# Absolute eligibility
# =========================================================================================


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"has_daily": True, "enough_history": True, "above_moving_average": False,
          "abs_return": 0.2, "fast_return": 0.1}, "Below its"),
        ({"has_daily": True, "enough_history": True, "above_moving_average": True,
          "abs_return": -0.01, "fast_return": 0.1}, "60-day return"),
        ({"has_daily": True, "enough_history": True, "above_moving_average": True,
          "abs_return": 0.2, "fast_return": -0.10}, "20-day return"),
        ({"has_daily": True, "enough_history": False, "daily_bars": 12,
          "above_moving_average": True, "abs_return": 0.2, "fast_return": 0.1}, "12 of 100"),
    ],
)
def test_each_absolute_gate_rejects_and_says_why(row: dict, expected: str) -> None:
    ok, reason = eligibility(row, DualMomentumConfig())

    assert ok is False
    assert expected in reason


def test_an_ineligible_name_is_never_ranked_so_a_thin_field_means_holding_less() -> None:
    config = Runtime(
        risk_on_universe=["AAA", "BBB", "CCC"], defensive_universe=["BIL"], benchmark="AAA",
        max_positions=3, min_base_score=-99,
    )
    strategy = DualMomentumConfig.from_runtime_config(config)
    daily = {
        "AAA": daily_bars(80, 130),    # eligible
        "BBB": daily_bars(130, 90),    # below its own average
        "CCC": daily_bars(130, 95),    # below its own average
        "BIL": daily_bars(100, 101),
    }
    intraday = {symbol: intraday_bars(frame["close"].iloc[0], frame["close"].iloc[-1])
                for symbol, frame in daily.items()}

    outcome = analyze_universe(context_for(config, daily, intraday), strategy)

    assert [row["symbol"] for row in outcome["ranked"]] == ["AAA"]
    assert outcome["weights"]["BBB"] == 0
    assert outcome["weights"]["CCC"] == 0


# =========================================================================================
# Timing as a flag, not a score bonus
# =========================================================================================


def test_an_entry_must_clear_the_score_floor_but_a_holding_need_not() -> None:
    """Entry and exit are deliberately asymmetric, and always have been.

    The asymmetry used to be demonstrated through the entry-timing gate. That gate is gone, so
    the floor stands in for it here -- the property under test is the same one: ``analyze``
    proposing nothing for a name does not mean step 2 should sell it.
    """
    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"], benchmark="AAA",
                     min_base_score=99)
    daily = {"AAA": daily_bars(80, 130), "BIL": daily_bars(100, 101)}
    intraday = {"AAA": intraday_bars(130, 130), "BIL": intraday_bars(101, 101)}
    context = context_for(config, daily, intraday)
    algorithm = DualMomentumAlgorithm(config)

    decision = algorithm.analyze(context)
    assert decision.target_weights["AAA"] == 0, "below the floor, no new entry"

    with ephemeral_state():
        holder = PortfolioSnapshot(positions={"AAA": 10}, equity=10_000.0)
        kept = algorithm.refine_weights(dict(decision.target_weights), decision.signals, holder,
                                        context.latest_prices, config, context.timestamp)

    assert kept["AAA"] > 0, "an eligible holding is kept even when it would not be bought today"


def test_a_holding_survives_the_intent_round_trip_the_pipeline_actually_performs() -> None:
    """The same guarantee, through ``refine`` rather than ``refine_weights``.

    ``pipeline.place_orders`` drops zero-weight intents before step 2 runs, so the weight dict
    that reaches ``refine_weights`` in production contains only what ``analyze`` proposed this
    bar. ``settle`` used to iterate that dict alone, which discarded every keep decision about
    a name absent from it -- and ``MODE_TARGET`` reads an absent symbol as "sell it all".

    The test above cannot catch that: it hands over the complete vector, zeros included.
    """
    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"], benchmark="AAA",
                     min_base_score=99)
    daily = {"AAA": daily_bars(80, 130), "BIL": daily_bars(100, 101)}
    intraday = {"AAA": intraday_bars(130, 130), "BIL": intraday_bars(101, 101)}
    context = context_for(config, daily, intraday)
    algorithm = DualMomentumAlgorithm(config)
    decision = algorithm.analyze(context)
    assert decision.target_weights["AAA"] == 0, "below the floor, no new entry"

    # Exactly what the pipeline passes on: weight intents, zero-valued ones removed.
    proposed = [i for i in decision.resolved_intents() if i.kind != "weight" or i.value]
    assert "AAA" not in {i.symbol for i in proposed}, "the fixture only bites while this holds"

    with ephemeral_state():
        holder = PortfolioSnapshot(positions={"AAA": 10}, equity=10_000.0)
        final = algorithm.refine(proposed, decision.signals, holder,
                                 context.latest_prices, config, context.timestamp)

    weights = {intent.symbol: intent.value for intent in final if intent.kind == "weight"}
    assert weights.get("AAA", 0) > 0, "a kept holding must survive a proposal that omits it"


# =========================================================================================
# Ranking hysteresis
# =========================================================================================


def test_a_challenger_must_win_by_the_replacement_margin() -> None:
    rows = {
        "HELD": {"base_score": 1.00},
        "CLOSE": {"base_score": 1.20},
        "CLEAR": {"base_score": 1.80},
    }
    config = DualMomentumConfig(max_positions=1, min_score_delta_to_replace=0.35)

    scores = {symbol: float(row["base_score"]) for symbol, row in rows.items()}
    assert resolve_themes({"HELD"}, {"CLOSE"}, scores, config) == {"HELD"}
    assert resolve_themes({"HELD"}, {"CLEAR"}, scores, config) == {"CLEAR"}


def test_free_slots_are_filled_before_anything_is_displaced() -> None:
    rows = {"HELD": {"base_score": 1.0}, "NEW": {"base_score": 0.2}}
    config = DualMomentumConfig(max_positions=2, min_score_delta_to_replace=0.35)

    scores = {symbol: float(row["base_score"]) for symbol, row in rows.items()}
    assert resolve_themes({"HELD"}, {"NEW"}, scores, config) == {"HELD", "NEW"}


def test_a_name_that_just_exited_waits_out_its_cooldown() -> None:
    config = DualMomentumConfig(cooldown_after_exit=4, risk_refresh_minutes=15)
    exit_time = datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc)
    state: dict = {}
    _record_exits(state, {"AAA"}, set(), exit_time.isoformat())

    assert _in_cooldown(state, "AAA", exit_time + timedelta(minutes=30), config) is True
    assert _in_cooldown(state, "AAA", exit_time + timedelta(minutes=61), config) is False


def test_re_entering_clears_the_exit_record() -> None:
    state: dict = {}
    _record_exits(state, {"AAA"}, set(), "2026-06-05T14:00:00+00:00")
    _record_exits(state, set(), {"AAA"}, "2026-06-05T15:00:00+00:00")

    assert state["exited_at"] == {}


# =========================================================================================
# Sizing, volatility target, turnover
# =========================================================================================


def test_a_zero_tilt_sizes_on_score_alone() -> None:
    """Volatility leaves sizing entirely, rather than being neutralised twice.

    The score is already a cross-sectional comparison; dividing by volatility on top of it
    underweighted the leaders, which cost return *and* drawdown on full-coverage replay.
    """
    config = DualMomentumConfig(min_base_score=0.0, name_weight_max=1.0, risk_on_gross_max=1.0,
                                volatility_tilt=0.0)
    rows = [
        {"symbol": "CALM", "base_score": 1.0, "annual_volatility": 0.10},
        {"symbol": "WILD", "base_score": 1.0, "annual_volatility": 0.40},
    ]

    weights = score_to_weights(rows, config)

    assert weights["CALM"] == pytest.approx(weights["WILD"])
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-6)


def test_a_positive_tilt_leans_into_the_volatile_name() -> None:
    """The shipped default: same score, bigger position in the wilder name."""
    config = DualMomentumConfig(min_base_score=0.0, name_weight_max=1.0, risk_on_gross_max=1.0,
                                volatility_tilt=1.0)
    rows = [
        {"symbol": "CALM", "base_score": 1.0, "annual_volatility": 0.10},
        {"symbol": "WILD", "base_score": 1.0, "annual_volatility": 0.40},
    ]

    weights = score_to_weights(rows, config)

    assert weights["WILD"] == pytest.approx(4 * weights["CALM"], rel=1e-6)


def test_a_negative_tilt_restores_risk_parity() -> None:
    config = DualMomentumConfig(min_base_score=0.0, name_weight_max=1.0, risk_on_gross_max=1.0,
                                volatility_tilt=-1.0)
    rows = [
        {"symbol": "CALM", "base_score": 1.0, "annual_volatility": 0.10},
        {"symbol": "WILD", "base_score": 1.0, "annual_volatility": 0.40},
    ]

    weights = score_to_weights(rows, config)

    assert weights["CALM"] == pytest.approx(4 * weights["WILD"], rel=1e-6)
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-6)


def test_the_name_cap_binds_before_the_gross_cap() -> None:
    config = DualMomentumConfig(min_base_score=0.0, name_weight_max=0.35, risk_on_gross_max=1.0)
    rows = [{"symbol": "ONE", "base_score": 5.0, "annual_volatility": 0.10},
            {"symbol": "TWO", "base_score": 0.1, "annual_volatility": 0.10}]

    weights = score_to_weights(rows, config)

    assert weights["ONE"] == pytest.approx(0.35)
    assert sum(weights.values()) < 1.0, "a thin field holds less rather than concentrating"


def test_volatility_scaling_only_engages_above_the_trigger() -> None:
    config = DualMomentumConfig(target_portfolio_vol=0.12, high_vol_trigger=1.5, vol_scale_floor=0.25)
    covariance = {"AAA": {"AAA": 0.02}}   # ~14% annualised, inside 1.5x target

    quiet = volatility_scale({"AAA": 1.0}, covariance, config)

    assert quiet["engaged"] is False
    assert quiet["scale"] == 1.0


def test_volatility_scaling_pulls_a_hot_portfolio_back_toward_target() -> None:
    config = DualMomentumConfig(target_portfolio_vol=0.12, high_vol_trigger=1.5, vol_scale_floor=0.25)
    covariance = {"AAA": {"AAA": 0.09}}   # 30% annualised: above the trigger, above the floor

    hot = volatility_scale({"AAA": 1.0}, covariance, config)

    assert hot["engaged"] is True
    assert hot["scale"] == pytest.approx(0.12 / 0.30, rel=1e-3)
    assert hot["below_floor"] is False


def test_the_shipped_target_does_not_tax_a_normal_book_of_these_etfs() -> None:
    """The default is a crash brake, not a governor.

    The spec's 12% target against a 23-58% volatility universe would have held the book at
    28-55% invested permanently. A realistic three-name book has to come through unscaled,
    or the strategy spends its life in cash while momentum is working.
    """
    config = DualMomentumConfig()   # shipped defaults
    # QQQM + AIQ + GTEK at equal weight, correlation 0.8: about 32% ex-ante volatility.
    vols = {"QQQM": 0.235, "AIQ": 0.334, "GTEK": 0.468}
    covariance = {
        left: {right: vols[left] * vols[right] * (1.0 if left == right else 0.8) for right in vols}
        for left in vols
    }
    weights = {symbol: 1 / 3 for symbol in vols}

    outcome = volatility_scale(weights, covariance, config)

    assert 0.30 < outcome["portfolio_volatility"] < 0.35
    assert outcome["engaged"] is False
    assert outcome["scale"] == 1.0


def test_the_target_still_engages_when_volatility_actually_explodes() -> None:
    config = DualMomentumConfig()
    covariance = {"AAA": {"AAA": 0.36}}   # 60% annualised

    outcome = volatility_scale({"AAA": 1.0}, covariance, config)

    assert outcome["engaged"] is True
    assert outcome["scale"] == pytest.approx(0.30 / 0.60, rel=1e-3)


def test_a_portfolio_too_volatile_even_at_the_floor_goes_defensive() -> None:
    config = DualMomentumConfig(target_portfolio_vol=0.12, high_vol_trigger=1.5, vol_scale_floor=0.25)
    covariance = {"AAA": {"AAA": 4.0}}    # 200% annualised

    extreme = volatility_scale({"AAA": 1.0}, covariance, config)

    assert extreme["below_floor"] is True


def test_portfolio_volatility_uses_covariance_not_just_variances() -> None:
    """Two correlated names are riskier together than the diagonal alone would say."""
    config = DualMomentumConfig(target_portfolio_vol=0.10, high_vol_trigger=1.0)
    weights = {"AAA": 0.5, "BBB": 0.5}
    independent = {"AAA": {"AAA": 0.04, "BBB": 0.0}, "BBB": {"AAA": 0.0, "BBB": 0.04}}
    correlated = {"AAA": {"AAA": 0.04, "BBB": 0.04}, "BBB": {"AAA": 0.04, "BBB": 0.04}}

    assert (
        volatility_scale(weights, correlated, config)["portfolio_volatility"]
        > volatility_scale(weights, independent, config)["portfolio_volatility"]
    )


def test_covariance_is_annualised_from_daily_returns() -> None:
    config = DualMomentumConfig(vol_estimation_days=30)
    bars = {"AAA": daily_bars(100, 130), "BBB": daily_bars(100, 90)}

    covariance = covariance_matrix(bars, ["AAA", "BBB"], config)

    assert covariance["AAA"]["AAA"] >= 0
    assert covariance["AAA"]["BBB"] == pytest.approx(covariance["BBB"]["AAA"])


def test_trivial_trades_are_left_alone() -> None:
    config = DualMomentumConfig(rebalance_weight_threshold=0.03, minimum_trade_notional=100.0,
                                minimum_trade_nav_fraction=0.005)
    current = {"AAA": 0.30, "BBB": 0.30}

    filtered = apply_turnover_filters({"AAA": 0.31, "BBB": 0.40}, current, 10_000.0, config)

    assert filtered["AAA"] == 0.30, "inside the rebalance threshold"
    assert filtered["BBB"] == 0.40


def test_a_trade_below_the_notional_floor_is_skipped() -> None:
    config = DualMomentumConfig(rebalance_weight_threshold=0.0, minimum_trade_notional=100.0,
                                minimum_trade_nav_fraction=0.005)

    filtered = apply_turnover_filters({"AAA": 0.05}, {"AAA": 0.0}, 1_000.0, config)

    assert filtered["AAA"] == 0.0, "$50 of a $1,000 account is below the $100 floor"


# =========================================================================================
# Scoring maths
# =========================================================================================


def test_robust_zscores_survive_one_event_driven_spike() -> None:
    values = {"A": 0.01, "B": 0.02, "C": 0.03, "SPIKE": 5.0}

    robust = zscores(values, robust=True)
    classic = zscores(values, robust=False)

    # Under a classic z-score the spike compresses everyone else toward one value.
    assert abs(classic["C"] - classic["A"]) < 0.02
    assert abs(robust["C"] - robust["A"]) > 1.0


def test_raw_ranking_prefers_the_louder_name_and_risk_adjusted_does_not() -> None:
    """Sizing divides by volatility, but sizing only reaches names that were selected.

    Both ETFs here have the identical trend per unit of their own risk. Under raw returns the
    volatile one scores four times higher and takes the slot; the calm one is never sized at
    all, so the volatility divisor in the weighting cannot correct for it.
    """
    features = {
        "WILD": {"symbol": "WILD", "annual_volatility": 0.60,
                 "return_series": {h: [0.12] * 3 for h in ("nano", "micro", "meso", "macro")}},
        "CALM": {"symbol": "CALM", "annual_volatility": 0.15,
                 "return_series": {h: [0.03] * 3 for h in ("nano", "micro", "meso", "macro")}},
    }

    raw = base_scores(features, DualMomentumConfig(risk_adjusted_score=False))
    adjusted = base_scores(features, DualMomentumConfig(risk_adjusted_score=True))

    assert raw["WILD"]["base_score"] > raw["CALM"]["base_score"]
    # Same return per unit of risk, so neither leads.
    assert adjusted["WILD"]["base_score"] == pytest.approx(adjusted["CALM"]["base_score"], abs=1e-9)


def test_an_unmeasurable_volatility_borrows_the_median_rather_than_dividing_by_zero() -> None:
    features = {
        "GOOD": {"symbol": "GOOD", "annual_volatility": 0.20,
                 "return_series": {h: [0.05] * 3 for h in ("nano", "micro", "meso", "macro")}},
        "THIN": {"symbol": "THIN", "annual_volatility": 0.0,
                 "return_series": {h: [0.05] * 3 for h in ("nano", "micro", "meso", "macro")}},
    }

    scored = base_scores(features, DualMomentumConfig(risk_adjusted_score=True))

    assert math.isfinite(scored["THIN"]["base_score"])
    assert scored["THIN"]["base_score"] == pytest.approx(scored["GOOD"]["base_score"], abs=1e-9)


def test_zscores_of_an_identical_cross_section_are_zero() -> None:
    assert zscores({"A": 0.5, "B": 0.5}, robust=True) == {"A": 0.0, "B": 0.0}


def test_the_defensive_sleeve_is_itself_ranked() -> None:
    config = DualMomentumConfig(defensive_universe=["BIL", "IEF", "AGG"], defensive_max_positions=2)
    scored = {
        "BIL": {"symbol": "BIL", "abs_return": 0.01},
        "IEF": {"symbol": "IEF", "abs_return": 0.05},
        "AGG": {"symbol": "AGG", "abs_return": -0.02},
    }

    weights = defensive_weights(scored, config)

    assert set(weights) == {"IEF", "BIL"}
    assert sum(weights.values()) == pytest.approx(1.0)


# =========================================================================================
# Risk stops
# =========================================================================================


def test_a_session_breaker_cannot_fire_at_this_algorithms_cadence() -> None:
    """Why ``intraday_drawdown_limit`` was removed rather than merely switched off.

    The shared breaker measures the drop from the equity it first saw *this session*. Dual
    Momentum runs ``DAILY_AT_OPEN``, so every run opens a new session and rebases that
    reference to the current equity -- the drawdown it computes is identically zero however
    far the book has fallen since yesterday. A knob that reads as crash protection and
    provides none is worse than no knob.

    Kept as a test because the trap is in the interaction, not in either piece: the function
    is correct, and Fast Momentum, which runs intraday, still relies on it.
    """
    from src.algorithms.risk import session_drawdown_breached

    state: dict = {}
    monday = datetime(2026, 6, 5, 15, tzinfo=timezone.utc)
    tuesday = datetime(2026, 6, 8, 15, tzinfo=timezone.utc)

    assert session_drawdown_breached(state, 10_000.0, -0.015, monday) is False
    # Down 20% overnight, and the next daily run still reads a drawdown of exactly zero.
    assert session_drawdown_breached(state, 8_000.0, -0.015, tuesday) is False
    assert state["session_drawdown"] == 0.0

    # Called twice inside one session -- Fast Momentum's cadence -- it works as intended.
    assert session_drawdown_breached(state, 7_800.0, -0.015, tuesday) is True


def test_a_single_session_collapse_still_sells_the_holding() -> None:
    """``max_daily_drop`` is the stop that survives, and it reads close-to-close returns."""
    config = DualMomentumConfig(max_daily_drop=0.10)
    held = {"symbol": "AAA", "eligible": 1, "above_moving_average": True,
            "ma_distance": 0.20, "abs_return": 0.30, "fast_return": -0.12,
            "enough_history": True, "nano_return": -0.15}

    stays, why = hold_eligibility(held, config)

    assert stays is False
    assert "15" in why or "drop" in why.lower()


# =========================================================================================
# Wiring
# =========================================================================================


def test_the_algorithm_is_registered_and_declares_what_it_needs() -> None:
    from src.algorithms.registry import get_algorithm_class
    from src.core.config import ALGORITHM_IDS
    from src.core.strategy_models import STRATEGY_LABELS

    assert "dual_momentum" in ALGORITHM_IDS
    assert STRATEGY_LABELS["dual_momentum"] == "Dual Momentum"

    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"], benchmark="QQQM")
    algorithm = get_algorithm_class("dual_momentum")(config)
    requirements = algorithm.requirements(config, {"ZZZ": 3})

    # The benchmark has to be fetched even though it is never traded.
    assert "QQQM" in requirements.price_symbols
    assert "ZZZ" in requirements.price_symbols, "held positions are priced too"
    # No intraday window at all: every feature is computed from the daily bars below.
    assert requirements.history_lookback_minutes == 0
    assert requirements.daily_lookback_days >= 100
    assert requirements.paper_only is True
    assert requirements.needs_sentiment is False, "sentiment is off until it is phased in"


def test_sentiment_is_requested_only_once_it_is_switched_on() -> None:
    from src.algorithms.registry import get_algorithm_class

    config = Runtime(sentiment_size_scale=0.05)
    algorithm = get_algorithm_class("dual_momentum")(config)

    assert algorithm.requirements(config, {}).needs_sentiment is True


def test_sizing_carries_no_cash_buffer() -> None:
    """Exposure is stated by ``risk_on_gross_max``; funding is checked against buying power.

    The algorithm used to carry the account cash buffer here so a plan built over a retained
    sub-threshold position could still be paid for. Order funding measures that directly now,
    and a buffer applied at sizing time would just under-invest on top of it.
    """
    algorithm = DualMomentumAlgorithm(Runtime())

    assert "cash_buffer" not in algorithm.sizing(Runtime())


def test_every_signal_row_carries_the_audit_trail() -> None:
    """Config-driven decisions are only auditable if each gate's verdict is recorded."""
    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"], benchmark="AAA")
    algorithm = DualMomentumAlgorithm(config)
    daily = {"AAA": daily_bars(80, 130), "BIL": daily_bars(100, 101)}
    intraday = {"AAA": intraday_bars(100, 130), "BIL": intraday_bars(100, 101)}

    row = algorithm.analyze(context_for(config, daily, intraday)).signals["AAA"]

    for key in ("base_score", "rank", "eligible", "eligibility_reason",
                "annual_volatility", "target_weight", "defensive_weight",
                "data_ok", "data_detail", "universe_coverage", "vol_scale",
                "portfolio_volatility", "reason"):
        assert key in row, key


def test_a_close_is_not_blocked_by_the_rebalance_threshold() -> None:
    """A position under the threshold could never clear it, so it was held forever.

    The threshold suppresses small *adjustments* to a name the algorithm still wants. Applied
    to a close it traps the position instead: an 8-position book carried a mean of 12 names,
    5 of them frozen below the threshold and holding a fifth of the equity.
    """
    config = DualMomentumConfig(rebalance_weight_threshold=0.08, minimum_trade_notional=100.0,
                                minimum_trade_nav_fraction=0.0)

    # Held at 3% and targeted at zero: a 3% move, far under the 8% threshold.
    exited = apply_turnover_filters({"AAA": 0.0}, {"AAA": 0.03}, 10_000.0, config)
    assert exited["AAA"] == 0.0, "an exit must not be gated by the rebalance threshold"

    # A small *trim* of the same size is still suppressed -- that is what the threshold is for.
    trimmed = apply_turnover_filters({"AAA": 0.09}, {"AAA": 0.12}, 10_000.0, config)
    assert trimmed["AAA"] == 0.12, "a sub-threshold adjustment is still skipped"

    # The absolute notional floor still applies, so this cannot spray dust orders.
    tiny = apply_turnover_filters({"AAA": 0.0}, {"AAA": 0.005}, 10_000.0, config)
    assert tiny["AAA"] == 0.005, "below the notional floor it is still left alone"


def test_partial_adjustment_moves_a_fraction_of_the_way_but_exits_in_full() -> None:
    """``rebalance_step`` regularises turnover without stranding a rejected holding.

    The no-trade band and the partial step brake different things: the band filters small
    drift, the step damps a target that swings hard every session. An exit is exempt from
    both, or a name the strategy has dropped decays off the book over a week.
    """
    config = DualMomentumConfig(rebalance_step=0.5)

    stepped = partial_adjustment({"AAA": 0.40, "BBB": 0.10}, {"AAA": 0.20, "BBB": 0.30}, config)
    assert stepped["AAA"] == pytest.approx(0.30), "half way up from 0.20 toward 0.40"
    assert stepped["BBB"] == pytest.approx(0.20), "half way down from 0.30 toward 0.10"

    # An entry from nothing also arrives gradually -- that is the point of the knob.
    entry = partial_adjustment({"AAA": 0.40}, {}, config)
    assert entry["AAA"] == pytest.approx(0.20)

    # But a target of zero is honoured immediately.
    exit_now = partial_adjustment({"AAA": 0.0}, {"AAA": 0.30}, config)
    assert exit_now["AAA"] == 0.0, "an exit is never decayed"

    # The default is a full jump, so the knob is inert until it is turned on.
    assert partial_adjustment({"AAA": 0.40}, {"AAA": 0.20}, DualMomentumConfig())["AAA"] == 0.40


def test_the_per_name_cap_is_re_spread_rather_than_dropped() -> None:
    """Capping one name used to lose its overflow, leaving the book under-invested."""
    config = DualMomentumConfig(risk_on_gross_max=0.96, name_weight_max=0.5,
                                min_base_score=0.0, volatility_tilt=0.0)

    # One name: the cap binds and 46% cannot be deployed inside the per-name limit.
    one = score_to_weights([{"symbol": "AAA", "base_score": 1.0, "annual_volatility": 0.2}], config)
    assert one["AAA"] == pytest.approx(0.5)

    # Two names, wildly different scores: the stronger caps at 50% and its excess goes to the
    # other rather than evaporating, so the pair deploys the full 96%.
    two = score_to_weights([
        {"symbol": "AAA", "base_score": 10.0, "annual_volatility": 0.2},
        {"symbol": "BBB", "base_score": 1.0, "annual_volatility": 0.2},
    ], config)
    assert two["AAA"] == pytest.approx(0.5)
    assert sum(two.values()) == pytest.approx(0.96), "the overflow is re-spread, not dropped"


def test_undeployed_gross_is_parked_in_bills_not_left_as_cash() -> None:
    """A funded account holds bills, not idle cash, for whatever the risk sleeve cannot use."""
    config = DualMomentumConfig(risk_on_gross_max=0.96, name_weight_max=0.5)

    parked = park_residual({"AAA": 0.5}, {"BIL": 0.96}, config)

    assert parked["AAA"] == pytest.approx(0.5)
    assert parked["BIL"] == pytest.approx(0.46)
    assert sum(parked.values()) == pytest.approx(0.96)

    # Fully deployed: nothing to park.
    full = park_residual({"AAA": 0.48, "BBB": 0.48}, {"BIL": 0.96}, config)
    assert "BIL" not in full


def test_eligibility_becomes_a_state_rather_than_a_daily_coin_flip() -> None:
    """A name near a floor used to flip in and out on consecutive sessions."""
    config = DualMomentumConfig(eligibility_window=10, entry_min_eligible_days=8,
                                exit_max_eligible_days=3)
    state: dict = {}

    # Alternating in and out: never a settled enough signal to open.
    for flag in [1, 0] * 5:
        history = track_eligibility(state, {"AAA": {"eligible": flag}}, config)
    assert sum(history["AAA"]) == 5
    assert sum(history["AAA"]) < config.entry_min_eligible_days, "chop does not earn an entry"
    assert sum(history["AAA"]) > config.exit_max_eligible_days, "nor does it force an exit"

    # A settled signal does.
    for _ in range(10):
        history = track_eligibility(state, {"AAA": {"eligible": 1}}, config)
    assert sum(history["AAA"]) == 10

    # The window is bounded, so a sustained breakdown eventually clears the exit bar.
    for _ in range(8):
        history = track_eligibility(state, {"AAA": {"eligible": 0}}, config)
    assert sum(history["AAA"]) == 2 <= config.exit_max_eligible_days


def test_a_cold_state_store_does_not_liquidate_the_book() -> None:
    """With no history a count reads as zero, which must not be read as 'ineligible'."""
    config = DualMomentumConfig(eligibility_window=10, exit_max_eligible_days=3)
    state: dict = {}

    history = track_eligibility(state, {"AAA": {"eligible": 1}}, config)

    assert sum(history["AAA"]) == 1 <= config.exit_max_eligible_days
    assert len(history["AAA"]) < config.eligibility_window, (
        "the window is not full, so the exit rule must not fire yet"
    )


def test_a_holding_that_crashes_in_one_session_is_sold_at_once() -> None:
    """The portfolio-level breaker cannot fire at a daily cadence; this one can."""
    config = DualMomentumConfig(max_daily_drop=0.10, exit_threshold_slack=0.05,
                                etf_min_abs_return=0.0, etf_min_fast_return=-0.02)
    healthy = {"enough_history": 1, "ma_distance": 0.08, "abs_return": 0.20,
               "fast_return": 0.05, "nano_return": -0.01}

    stays, _ = hold_eligibility(healthy, config)
    assert stays

    crashed = {**healthy, "nano_return": -0.12}
    stays, why = hold_eligibility(crashed, config)
    assert stays is False
    assert "one session" in why

    # Still inside the limit: everything else about the name is fine, so it is kept.
    dipped = {**healthy, "nano_return": -0.08}
    assert hold_eligibility(dipped, config)[0] is True


def test_a_sibling_must_beat_the_sitting_name_before_it_takes_its_slot() -> None:
    """``intra_theme_delta_to_replace`` was documented everywhere and read by nothing.

    Which ETFs fill a theme's slots was decided on today's scores alone, so two near-tied
    siblings swapped places on any session that reordered them by a hair -- 13.9% of all
    turnover over 2023. Holding a slot must not also earn a bigger position, so the margin
    applies to selection only and the split stays proportional to raw score.
    """
    config = DualMomentumConfig(max_positions_per_theme=1, min_base_score=0.0,
                                volatility_tilt=0.0, risk_on_gross_max=1.0,
                                name_weight_max=1.0, intra_theme_delta_to_replace=0.5,
                                themes={"AAA": "growth", "BBB": "growth"})
    rows = {
        "AAA": {"symbol": "AAA", "eligible": 1, "base_score": 1.0, "annual_volatility": 0.2},
        "BBB": {"symbol": "BBB", "eligible": 1, "base_score": 1.3, "annual_volatility": 0.2},
    }

    # Nothing held: the higher score simply wins.
    fresh = theme_allocation({"growth"}, rows, config, {})
    assert fresh.get("BBB", 0.0) > 0 and fresh.get("AAA", 0.0) == 0

    # AAA sitting: BBB leads by 0.3, short of the 0.5 margin, so AAA keeps the slot.
    defended = theme_allocation({"growth"}, rows, config, {"AAA": 0.4})
    assert defended.get("AAA", 0.0) > 0, "a 0.3 lead must not displace the incumbent"
    assert defended.get("BBB", 0.0) == 0

    # Widen the lead past the margin and the swap goes through.
    rows["BBB"]["base_score"] = 1.8
    displaced = theme_allocation({"growth"}, rows, config, {"AAA": 0.4})
    assert displaced.get("BBB", 0.0) > 0 and displaced.get("AAA", 0.0) == 0

    # At zero the knob is off, which is the behaviour every earlier measurement was taken under.
    off = DualMomentumConfig(**{**config.__dict__, "intra_theme_delta_to_replace": 0.0})
    rows["BBB"]["base_score"] = 1.3
    assert theme_allocation({"growth"}, rows, off, {"AAA": 0.4}).get("BBB", 0.0) > 0


def test_the_no_trade_band_cannot_push_the_book_over_its_gross_budget() -> None:
    """The 2026-01-08 case: a 100% proposal reached planning as a 106.9% one.

    The band suppresses moves in both directions, but only the trims were funding anything.
    Holding two incumbents above target because their trims were too small to trade, while
    passing a new entry through at full size, produced a plan that could not be paid for --
    16 of 24 sessions that January, and on one of them the entry was dropped outright.
    """
    config = DualMomentumConfig(rebalance_weight_threshold=0.08, minimum_trade_notional=0.0,
                                minimum_trade_nav_fraction=0.0)
    target = {"IEMG": 0.066, "SLV": 0.475, "XSD": 0.291, "XBI": 0.121, "EWJ": 0.022, "GLD": 0.025}
    current = {"IEMG": 0.122, "SLV": 0.466, "XSD": 0.360}

    filtered = apply_turnover_filters(target, current, 10_000.0, config)

    # The suppressed trims stay suppressed -- that is what the band is for.
    assert filtered["IEMG"] == pytest.approx(0.122)
    assert filtered["XSD"] == pytest.approx(0.360)
    # The entry is what gives way, and it arrives smaller rather than at full size.
    assert 0 < filtered["XBI"] < 0.121
    assert sum(filtered.values()) == pytest.approx(sum(target.values()), abs=1e-6)


def test_a_shrunken_entry_survives_the_band_but_not_the_notional_floor() -> None:
    """The band is about drift, so it must not kill an opening that was merely resized.

    Re-applying ``rebalance_weight_threshold`` to the shrunken leg dropped the entry outright
    whenever the freed budget was smaller than the band -- the opposite of the intent. The
    dollar floor still applies, because a $12 order costs the same spread as a real one.
    """
    config = DualMomentumConfig(rebalance_weight_threshold=0.08, minimum_trade_notional=100.0,
                                minimum_trade_nav_fraction=0.0)
    target = {"HELD": 0.90, "NEW": 0.10}
    current = {"HELD": 0.95}

    # $10,000 book: the entry shrinks to 5% = $500, well over the floor, so it happens.
    big = apply_turnover_filters(target, current, 10_000.0, config)
    assert big["HELD"] == pytest.approx(0.95), "the 5% trim was under the band"
    assert big["NEW"] == pytest.approx(0.05), "shrunk to fit, not dropped"
    assert sum(big.values()) == pytest.approx(1.0)

    # $1,000 book: the same 5% is $50, under the floor, so it is not worth submitting.
    small = apply_turnover_filters(target, current, 1_000.0, config)
    assert small["NEW"] == 0.0


def test_each_kind_of_action_keeps_its_own_clock() -> None:
    """One clock per decision, counted in runs so "3 days" means three sessions.

    Wall-clock days do not survive a weekend: a three-calendar-day interval set on a Friday is
    satisfied by Monday, so it throttles Tuesday-to-Thursday decisions and waves every
    Friday-to-Monday one straight through.
    """
    state: dict = {}

    # A cold state store has never acted, so every clock is due.
    assert action_due(state, "theme_rotation", 3) is True

    advance_run(state)
    record_action(state, "theme_rotation")
    advance_run(state)
    assert action_due(state, "theme_rotation", 3) is False, "one session later"
    advance_run(state)
    assert action_due(state, "theme_rotation", 3) is False, "two sessions later"
    advance_run(state)
    assert action_due(state, "theme_rotation", 3) is True, "three sessions later"

    # The clocks are independent: rotating does not start the intra-theme one.
    assert action_due(state, "intra_theme", 3) is True

    # Zero means every run.
    assert action_due(state, "exit", 0) is True


def test_a_caller_that_does_not_act_does_not_restart_the_clock() -> None:
    """Otherwise a throttled decision that changed nothing would push the next one out."""
    state: dict = {}
    advance_run(state)

    assert action_due(state, "entry", 3) is True
    # ``action_due`` is a question, not a commitment -- only ``record_action`` starts the clock.
    assert action_due(state, "entry", 3) is True
    record_action(state, "entry")
    advance_run(state)
    assert action_due(state, "entry", 3) is False


def test_the_crash_stop_answers_to_no_clock() -> None:
    """``max_daily_drop`` must fire on the session it happens, whatever exit_interval_days says.

    Otherwise a name can gap 30% across a week of throttled sessions while the algorithm waits
    its turn to look at it.
    """
    config = DualMomentumConfig(max_daily_drop=0.10)

    assert crash_stop({"nano_return": -0.15}, config)[0] is False
    assert crash_stop({"nano_return": -0.05}, config)[0] is True
    # 0 turns it off entirely.
    assert crash_stop({"nano_return": -0.99}, DualMomentumConfig(max_daily_drop=0.0))[0] is True


def test_theme_rank_is_not_the_etf_rank_of_its_best_member() -> None:
    """Entry was gated on the ETF rank of a theme's best name, which is a different question.

    Metals holding SLV and GLD at ETF ranks 1 and 2 pushes every other theme's best name down
    the ladder, so with entry_rank_max=3 only the ETF ranked 3rd could bring its theme in --
    whatever that theme's own standing was.
    """
    ranks = theme_ranks({"metals": 2.9, "energy": 1.3, "us_growth": 0.8, "intl": 0.2})

    assert ranks == {"metals": 1, "energy": 2, "us_growth": 3, "intl": 4}
