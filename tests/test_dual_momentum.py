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
    _resolve_replacements,
    analyze_universe,
    apply_turnover_filters,
    confirm_regime,
    covariance_matrix,
    defensive_weights,
    eligibility,
    intraday_drawdown_breached,
    market_regime,
    score_to_weights,
    timing,
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
# The regime gate: the whole reason this exists as a fork
# =========================================================================================


def test_risk_off_holds_the_defensive_sleeve_rather_than_the_least_bad_etf() -> None:
    """Cross-sectional ranking alone always owns something. That is what the gate fixes."""
    config = Runtime(risk_on_universe=["AAA", "BBB"], defensive_universe=["BIL"], benchmark="AAA")
    strategy = DualMomentumConfig.from_runtime_config(config)
    falling = {
        "AAA": daily_bars(120, 80),
        "BBB": daily_bars(120, 60),   # least bad is still falling
        "BIL": daily_bars(100, 101),
    }
    intraday = {"AAA": intraday_bars(120, 80), "BBB": intraday_bars(120, 60), "BIL": intraday_bars(100, 101)}

    outcome = analyze_universe(context_for(config, falling, intraday), strategy)

    assert outcome["regime"]["risk_on"] is False
    assert outcome["weights"]["BIL"] > 0
    assert outcome["weights"]["AAA"] == 0
    assert outcome["weights"]["BBB"] == 0


def test_a_missing_benchmark_history_is_reported_as_a_data_gap_not_a_bear_market() -> None:
    """A short cache would otherwise read as "below its 100-day average", which is a lie."""
    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"], benchmark="QQQM")
    strategy = DualMomentumConfig.from_runtime_config(config)
    scored = {
        "AAA": {"symbol": "AAA", "above_moving_average": True, "enough_history": True, "daily_bars": 140},
        "QQQM": {"symbol": "QQQM", "above_moving_average": False, "enough_history": False, "daily_bars": 0},
    }

    regime = market_regime(scored, strategy)

    assert regime["risk_on"] is False
    assert regime["data_ok"] is False
    assert "no usable QQQM history" in regime["detail"]
    assert "average" not in regime["detail"]


def test_breadth_blocks_a_narrow_rally() -> None:
    config = Runtime(risk_on_universe=["AAA", "BBB", "CCC", "DDD"], benchmark="AAA", breadth_min=0.5)
    strategy = DualMomentumConfig.from_runtime_config(config)
    scored = {
        "AAA": {"symbol": "AAA", "above_moving_average": True, "enough_history": True, "abs_return": 0.2},
        "BBB": {"symbol": "BBB", "above_moving_average": False, "enough_history": True},
        "CCC": {"symbol": "CCC", "above_moving_average": False, "enough_history": True},
        "DDD": {"symbol": "DDD", "above_moving_average": False, "enough_history": True},
    }

    regime = market_regime(scored, strategy)

    assert regime["trend_ok"] and regime["momentum_ok"]
    assert regime["breadth"] == 0.25
    assert regime["risk_on"] is False
    assert "breadth" in regime["detail"]


def test_regime_hysteresis_needs_agreement_in_both_directions() -> None:
    config = DualMomentumConfig(regime_confirm_bars=2, regime_exit_confirm_bars=2)
    state: dict = {}

    state.update(confirm_regime(state, True, config))
    assert state["regime_risk_on"] is False, "one good reading is not a regime"
    state.update(confirm_regime(state, True, config))
    assert state["regime_risk_on"] is True

    state.update(confirm_regime(state, False, config))
    assert state["regime_risk_on"] is True, "one bad reading does not end it either"
    state.update(confirm_regime(state, False, config))
    assert state["regime_risk_on"] is False


def test_a_broken_streak_restarts_the_count() -> None:
    config = DualMomentumConfig(regime_confirm_bars=3)
    state: dict = {}
    state.update(confirm_regime(state, True, config))
    state.update(confirm_regime(state, True, config))
    state.update(confirm_regime(state, False, config))
    state.update(confirm_regime(state, True, config))
    state.update(confirm_regime(state, True, config))

    assert state["regime_risk_on"] is False


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


def test_the_pullback_setup_cannot_promote_a_name_that_fails_the_score_floor() -> None:
    """The reason it moved out of the score: as a bonus, a deep dip could outvote the trend."""
    weak = {
        "symbol": "WEAK",
        "momentum_change": -5.0,
        "micro_return": 0.01,
        "z": {"macro": 0.1, "meso": 0.9, "nano": -2.0},
        "base_score": -1.0,
    }
    config = DualMomentumConfig(min_base_score=0.25)

    ok, reason = timing(weak, config)

    assert ok is True and reason == "Pullback in uptrend"
    # Timing says "now", the score still says "not this one".
    assert weak["base_score"] < config.min_base_score


def test_timing_opens_on_acceleration_or_on_a_pullback_and_otherwise_waits() -> None:
    config = DualMomentumConfig()
    accelerating = {"momentum_change": 0.5, "z": {}, "micro_return": 0.0}
    quiet = {"momentum_change": -0.1, "z": {"macro": 0.0, "meso": 0.0, "nano": 0.0}, "micro_return": 0.0}

    assert timing(accelerating, config) == (True, "Momentum accelerating")
    assert timing(quiet, config)[0] is False


def test_an_entry_needs_timing_but_a_holding_does_not() -> None:
    """Timing decides when to enter, never whether to stay."""
    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"], benchmark="AAA",
                     regime_confirm_bars=1, min_base_score=-99)
    strategy = DualMomentumConfig.from_runtime_config(config)
    daily = {"AAA": daily_bars(80, 130), "BIL": daily_bars(100, 101)}
    # Flat intraday: no acceleration, no pullback, so the timing flag is false.
    intraday = {"AAA": intraday_bars(130, 130), "BIL": intraday_bars(101, 101)}
    context = context_for(config, daily, intraday)
    algorithm = DualMomentumAlgorithm(config)

    decision = algorithm.analyze(context)
    assert decision.signals["AAA"]["timing"] == 0
    assert decision.target_weights["AAA"] == 0, "no timing, no new entry"

    with ephemeral_state():
        holder = PortfolioSnapshot(positions={"AAA": 10}, equity=10_000.0)
        # Two runs: the first confirms the regime, the second acts on it.
        algorithm.refine_weights(dict(decision.target_weights), decision.signals, holder,
                                 context.latest_prices, config)
        kept = algorithm.refine_weights(dict(decision.target_weights), decision.signals, holder,
                                        context.latest_prices, config)

    assert kept["AAA"] > 0, "an eligible, ranked holding is kept even with the timing flag false"


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

    assert _resolve_replacements({"HELD"}, {"CLOSE"}, rows, config) == {"HELD"}
    assert _resolve_replacements({"HELD"}, {"CLEAR"}, rows, config) == {"CLEAR"}


def test_free_slots_are_filled_before_anything_is_displaced() -> None:
    rows = {"HELD": {"base_score": 1.0}, "NEW": {"base_score": 0.2}}
    config = DualMomentumConfig(max_positions=2, min_score_delta_to_replace=0.35)

    assert _resolve_replacements({"HELD"}, {"NEW"}, rows, config) == {"HELD", "NEW"}


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


def test_the_drawdown_breaker_latches_for_the_session_and_resets_the_next_one() -> None:
    config = DualMomentumConfig(intraday_drawdown_limit=-0.015)
    state: dict = {}

    assert intraday_drawdown_breached(state, 10_000.0, config, "2026-06-05") is False
    assert intraday_drawdown_breached(state, 9_800.0, config, "2026-06-05") is True
    # Recovering within the same session does not un-trip it.
    assert intraday_drawdown_breached(state, 10_050.0, config, "2026-06-05") is True
    # A new session starts clean, which is what makes a backtest meaningful.
    assert intraday_drawdown_breached(state, 10_050.0, config, "2026-06-06") is False


def test_the_breaker_parks_the_book_in_the_defensive_sleeve() -> None:
    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"], benchmark="AAA",
                     regime_confirm_bars=1, intraday_drawdown_limit=-0.01, min_base_score=-99)
    algorithm = DualMomentumAlgorithm(config)
    daily = {"AAA": daily_bars(80, 130), "BIL": daily_bars(100, 101)}
    intraday = {"AAA": intraday_bars(100, 130), "BIL": intraday_bars(100, 101)}
    context = context_for(config, daily, intraday)
    decision = algorithm.analyze(context)

    with ephemeral_state():
        healthy = PortfolioSnapshot(positions={}, equity=10_000.0)
        algorithm.refine_weights(dict(decision.target_weights), decision.signals, healthy,
                                 context.latest_prices, config)
        crashed = PortfolioSnapshot(positions={}, equity=9_000.0)
        final = algorithm.refine_weights(dict(decision.target_weights), decision.signals, crashed,
                                         context.latest_prices, config)

    assert final["BIL"] > 0
    assert final["AAA"] == 0


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
    # Twelve sessions of wall-clock time, not a bar count that means nothing without a grid.
    assert requirements.history_lookback_minutes > 320 * 15
    assert requirements.daily_lookback_days >= 100
    assert requirements.paper_only is True
    assert requirements.needs_sentiment is False, "sentiment is off until it is phased in"


def test_sentiment_is_requested_only_once_it_is_switched_on() -> None:
    from src.algorithms.registry import get_algorithm_class

    config = Runtime(sentiment_size_scale=0.05)
    algorithm = get_algorithm_class("dual_momentum")(config)

    assert algorithm.requirements(config, {}).needs_sentiment is True


def test_sizing_leaves_no_double_counted_cash_buffer() -> None:
    algorithm = DualMomentumAlgorithm(Runtime())

    assert algorithm.sizing(Runtime())["cash_buffer"] == 0.0


def test_every_signal_row_carries_the_audit_trail() -> None:
    """Config-driven decisions are only auditable if each gate's verdict is recorded."""
    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"], benchmark="AAA")
    algorithm = DualMomentumAlgorithm(config)
    daily = {"AAA": daily_bars(80, 130), "BIL": daily_bars(100, 101)}
    intraday = {"AAA": intraday_bars(100, 130), "BIL": intraday_bars(100, 101)}

    row = algorithm.analyze(context_for(config, daily, intraday)).signals["AAA"]

    for key in ("base_score", "rank", "eligible", "eligibility_reason", "timing", "timing_reason",
                "momentum_change", "annual_volatility", "target_weight", "defensive_weight",
                "regime_risk_on", "regime_detail", "regime_breadth", "vol_scale",
                "portfolio_volatility", "reason", "as_of"):
        assert key in row, key
