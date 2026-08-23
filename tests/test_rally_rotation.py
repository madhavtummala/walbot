from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.algorithms.rally_rotation.algorithm import RallyRotationAlgorithm
from src.algorithms.rally_rotation.config import RallyRotationConfig
from src.algorithms.rally_rotation.gates import (
    blocking,
    crash_stop,
    entry_checks,
    exit_checks,
    passes,
    universe_data_ok,
)
from src.algorithms.rally_rotation.memory import (
    action_due,
    observed_days,
    qualifying_days,
    record_action,
    resolve_positions,
    sessions_since,
    track_eligibility,
    track_ranking,
)
from src.algorithms.rally_rotation.scoring import base_scores, zscores
from src.algorithms.rally_rotation.sizing import (
    apply_turnover_filters,
    defensive_weights,
    park_residual,
    partial_adjustment,
    score_to_weights,
)
from src.core.interfaces import MARKET_TZ, AlgorithmContext


def tuning(config) -> RallyRotationConfig:
    """The algorithm's own knobs, through the contract rather than around it."""
    return RallyRotationAlgorithm(config).tuning(config)


def rank_universe(config, daily, intraday) -> dict:
    """Pass one on its own: what the market data alone proposes, layer by layer."""
    settings = tuning(config)
    return RallyRotationAlgorithm(config).rank_universe(context_for(config, daily, intraday), settings)


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
        self.algorithm_configs = {"rally_rotation": tuning}
        self.account_id = "test"
        self.symbols = []
        self.cash_buffer = 0.0
        self.min_trade_dollars = 0.0
        self.rebalance_threshold = 0.0


def context_for(config: Runtime, daily: dict, intraday: dict, timestamp: datetime | None = None) -> AlgorithmContext:
    return AlgorithmContext(
        config=config,
        daily_bars_by_symbol=daily,
        intraday_bars_by_symbol=intraday,
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
    config = Runtime(risk_on_universe=["AAA", "BBB"], defensive_universe=["BIL"])
    falling = {
        "AAA": daily_bars(120, 80),
        "BBB": daily_bars(120, 60),   # least bad is still falling
        "BIL": daily_bars(100, 101),
    }
    intraday = {"AAA": intraday_bars(120, 80), "BBB": intraday_bars(120, 60), "BIL": intraday_bars(100, 101)}

    outcome = rank_universe(config, falling, intraday)

    assert outcome["weights"]["BIL"] > 0
    assert outcome["weights"]["AAA"] == 0
    assert outcome["weights"]["BBB"] == 0
    assert not outcome["entries"], "nothing should have qualified"


def test_a_thin_universe_is_reported_as_a_data_gap_not_a_bear_market() -> None:
    """A short cache would otherwise read as "below its 100-day average", which is a lie.

    The one piece of the regime layer that survives, and the reason it survives: this is a
    statement about the data, not about the market.
    """
    strategy = tuning(Runtime(risk_on_universe=["AAA", "BBB", "CCC", "DDD"], defensive_universe=["BIL"]))
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
    is not the portfolio layer's business at all -- the entry gates decline those three
    individually, and the fourth is still allowed to be held.
    """
    strategy = tuning(Runtime(risk_on_universe=["AAA", "BBB", "CCC", "DDD"]))
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
          "abs_return": 0.2, "fast_return": 0.1}, "Above its 100-day average"),
        ({"has_daily": True, "enough_history": True, "above_moving_average": True,
          "abs_return": -0.01, "fast_return": 0.1}, "60-day return"),
        ({"has_daily": True, "enough_history": True, "above_moving_average": True,
          "abs_return": 0.2, "fast_return": -0.10}, "20-day return"),
        ({"has_daily": True, "enough_history": False, "daily_bars": 12,
          "above_moving_average": True, "abs_return": 0.2, "fast_return": 0.1}, "12 bars"),
    ],
)
def test_each_absolute_gate_rejects_and_says_why(row: dict, expected: str) -> None:
    """The gate that failed names itself, and carries what it measured against its threshold."""
    checks = entry_checks(row, RallyRotationConfig())

    assert passes(checks) is False
    culprit = blocking(checks)
    assert expected in culprit.label or expected in culprit.value
    # A rejection with no measurement is the thing this contract exists to prevent.
    assert culprit.value and culprit.limit


def test_an_ineligible_name_is_never_ranked_so_a_thin_field_means_holding_less() -> None:
    config = Runtime(
        risk_on_universe=["AAA", "BBB", "CCC"], defensive_universe=["BIL"],
        max_positions=3, min_base_score=-99,
    )
    daily = {
        "AAA": daily_bars(80, 130),    # eligible
        "BBB": daily_bars(130, 90),    # below its own average
        "CCC": daily_bars(130, 95),    # below its own average
        "BIL": daily_bars(100, 101),
    }
    intraday = {symbol: intraday_bars(frame["close"].iloc[0], frame["close"].iloc[-1])
                for symbol, frame in daily.items()}

    outcome = rank_universe(config, daily, intraday)

    assert [row["symbol"] for row in outcome["ranked"]] == ["AAA"]
    assert outcome["weights"]["BBB"] == 0
    assert outcome["weights"]["CCC"] == 0


# =========================================================================================
# Timing as a flag, not a score bonus
# =========================================================================================


def test_an_entry_must_clear_the_score_floor_but_a_holding_need_not() -> None:
    """Entry and exit are deliberately asymmetric, and always have been.

    The asymmetry used to be demonstrated through the entry-timing gate. That gate is gone, so
    the floor stands in for it here. The two halves used to be two calls -- a pure ``analyze``
    that scored, and a ``refine_weights`` that was handed the book -- and the property could
    only be shown by wiring them together by hand. ``plan`` sees the holdings on its context, so
    the same asymmetry is now visible in one call: identical market data, different book.
    """
    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"],
                     min_base_score=99)
    daily = {"AAA": daily_bars(80, 130), "BIL": daily_bars(100, 101)}
    intraday = {"AAA": intraday_bars(130, 130), "BIL": intraday_bars(101, 101)}
    algorithm = RallyRotationAlgorithm(config)

    empty_book = algorithm.plan(context_for(config, daily, intraday))
    assert empty_book.target_weights.get("AAA", 0) == 0, "below the floor, no new entry"

    holding = replace(
        context_for(config, daily, intraday), positions={"AAA": 10}, equity=10_000.0
    )
    kept = algorithm.plan(holding)

    assert kept.target_weights.get("AAA", 0) > 0, (
        "an eligible holding is kept even when it would not be bought today"
    )


def test_a_kept_holding_survives_into_the_intents_the_pipeline_acts_on() -> None:
    """Being kept is not enough: the keep has to reach the order path.

    ``run_algorithm`` drops zero-weight intents, and ``MODE_TARGET`` reads a symbol absent from
    the intents as "sell it all". So a hold decision that never makes it onto the plan's intents
    is indistinguishable from a decision to exit -- which is exactly the bug this guards: the
    keep was recorded internally and then dropped, force-selling a still-qualifying holding.
    """
    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"],
                     min_base_score=99)
    daily = {"AAA": daily_bars(80, 130), "BIL": daily_bars(100, 101)}
    intraday = {"AAA": intraday_bars(130, 130), "BIL": intraday_bars(101, 101)}
    algorithm = RallyRotationAlgorithm(config)
    context = replace(
        context_for(config, daily, intraday), positions={"AAA": 10}, equity=10_000.0
    )

    plan = algorithm.plan(context)
    # Exactly what run_algorithm forwards to the order path: zero-valued weights removed.
    surviving = {i.symbol for i in plan.intents if i.kind != "weight" or i.value}

    assert "AAA" in surviving, "a kept holding must survive a proposal that omits it"


# =========================================================================================
# Ranking hysteresis
# =========================================================================================


def _ranked(**scores: float) -> list[dict[str, object]]:
    """Candidate rows in score order, ranked, as the hold/rotate pass builds them."""
    rows = [{"symbol": symbol, "base_score": score} for symbol, score in
            sorted(scores.items(), key=lambda item: -item[1])]
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
    return rows


def test_a_challenger_must_win_by_the_replacement_margin() -> None:
    """Displacing an incumbent costs the spread twice, so it needs daylight, not a hair."""
    config = RallyRotationConfig(max_positions=1, entry_rank_max=3, exit_rank_max=5,
                                min_score_delta_to_replace=0.35)

    close = _ranked(HELD=1.00, CLOSE=1.20)
    assert resolve_positions({"HELD"}, close, config) == {"HELD"}

    clear = _ranked(HELD=1.00, CLEAR=1.80)
    assert resolve_positions({"HELD"}, clear, config) == {"CLEAR"}


def test_free_slots_are_filled_before_anything_is_displaced() -> None:
    config = RallyRotationConfig(max_positions=2, entry_rank_max=3, exit_rank_max=5,
                                min_score_delta_to_replace=0.35)
    rows = _ranked(HELD=1.0, NEW=0.2)

    assert resolve_positions({"HELD"}, rows, config) == {"HELD", "NEW"}


def test_an_incumbent_is_held_through_a_rank_it_could_not_have_entered_on() -> None:
    """Enter from the top 2, hold through the top 5.

    Ejecting on the entry rank meant a holding that drifted to 7th of 14 on scores separated
    by 0.15 was sold, and the freed slot refilled from the top with no margin required at all.
    """
    config = RallyRotationConfig(max_positions=2, entry_rank_max=2, exit_rank_max=5,
                                min_score_delta_to_replace=5.0)
    rows = _ranked(A=1.0, B=0.9, C=0.8, HELD=0.7, E=0.6, F=0.5)

    # HELD ranks 4th -- outside entry range, inside hold range, and the margin is unreachable.
    assert "HELD" in resolve_positions({"HELD"}, rows, config)

    # Drop it past exit_rank_max and it is gone.
    far = _ranked(A=1.0, B=0.9, C=0.8, D=0.7, E=0.6, F=0.5, HELD=0.1)
    assert "HELD" not in resolve_positions({"HELD"}, far, config)


# =========================================================================================
# Sizing, volatility target, turnover
# =========================================================================================


def test_a_zero_tilt_sizes_on_score_alone() -> None:
    """Volatility leaves sizing entirely, rather than being neutralised twice.

    The score is already a cross-sectional comparison; dividing by volatility on top of it
    underweighted the leaders, which cost return *and* drawdown on full-coverage replay.
    """
    config = RallyRotationConfig(min_base_score=0.0, risk_on_gross_max=1.0,
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
    config = RallyRotationConfig(min_base_score=0.0, risk_on_gross_max=1.0,
                                volatility_tilt=1.0)
    rows = [
        {"symbol": "CALM", "base_score": 1.0, "annual_volatility": 0.10},
        {"symbol": "WILD", "base_score": 1.0, "annual_volatility": 0.40},
    ]

    weights = score_to_weights(rows, config)

    assert weights["WILD"] == pytest.approx(4 * weights["CALM"], rel=1e-6)


def test_a_negative_tilt_restores_risk_parity() -> None:
    config = RallyRotationConfig(min_base_score=0.0, risk_on_gross_max=1.0,
                                volatility_tilt=-1.0)
    rows = [
        {"symbol": "CALM", "base_score": 1.0, "annual_volatility": 0.10},
        {"symbol": "WILD", "base_score": 1.0, "annual_volatility": 0.40},
    ]

    weights = score_to_weights(rows, config)

    assert weights["CALM"] == pytest.approx(4 * weights["WILD"], rel=1e-6)
    assert sum(weights.values()) == pytest.approx(1.0, rel=1e-6)


def test_sizing_is_proportional_and_uncapped() -> None:
    """No per-name cap and no portfolio volatility overlay: the split is the ranking.

    Both existed and both are gone. A cap on top of a two- or three-name split does not
    diversify anything, it forces the book toward equal weight and discards the ranking the
    whole algorithm exists to produce. The ex-ante volatility overlay scaled the book down
    whenever it got interesting -- over seven sessions in January 2026 it cut exposure to
    ~0.7 and parked a third of the book in bills, which is a drag on a strategy whose job is
    to be in the market when the market is working.
    """
    config = RallyRotationConfig(min_base_score=0.0, risk_on_gross_max=1.0, volatility_tilt=0.0)
    rows = [
        {"symbol": "BIG", "base_score": 3.0, "annual_volatility": 0.5},
        {"symbol": "SMALL", "base_score": 1.0, "annual_volatility": 0.1},
    ]

    weights = score_to_weights(rows, config)

    assert weights["BIG"] == pytest.approx(0.75), "three times the score, three times the weight"
    assert weights["SMALL"] == pytest.approx(0.25)
    assert sum(weights.values()) == pytest.approx(1.0), "fully deployed, whatever the volatility"


def test_a_single_qualifying_name_takes_the_whole_book() -> None:
    """The direct consequence of removing the cap, stated so it cannot surprise anyone later."""
    config = RallyRotationConfig(min_base_score=0.0, risk_on_gross_max=1.0, volatility_tilt=0.0)

    weights = score_to_weights([{"symbol": "ONLY", "base_score": 1.0, "annual_volatility": 0.6}], config)

    assert weights["ONLY"] == pytest.approx(1.0)


def test_trivial_trades_are_left_alone() -> None:
    config = RallyRotationConfig(rebalance_weight_threshold=0.03, minimum_trade_notional=100.0,
                                minimum_trade_nav_fraction=0.005)
    current = {"AAA": 0.30, "BBB": 0.30}

    filtered = apply_turnover_filters({"AAA": 0.31, "BBB": 0.40}, current, 10_000.0, config)

    assert filtered["AAA"] == 0.30, "inside the rebalance threshold"
    assert filtered["BBB"] == 0.40


def test_a_trade_below_the_notional_floor_is_skipped() -> None:
    config = RallyRotationConfig(rebalance_weight_threshold=0.0, minimum_trade_notional=100.0,
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

    raw = base_scores(features, RallyRotationConfig(risk_adjusted_score=False))
    adjusted = base_scores(features, RallyRotationConfig(risk_adjusted_score=True))

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

    scored = base_scores(features, RallyRotationConfig(risk_adjusted_score=True))

    assert math.isfinite(scored["THIN"]["base_score"])
    assert scored["THIN"]["base_score"] == pytest.approx(scored["GOOD"]["base_score"], abs=1e-9)


def test_zscores_of_an_identical_cross_section_are_zero() -> None:
    assert zscores({"A": 0.5, "B": 0.5}, robust=True) == {"A": 0.0, "B": 0.0}


def test_the_defensive_sleeve_is_itself_ranked() -> None:
    config = RallyRotationConfig(defensive_universe=["BIL", "IEF", "AGG"], defensive_max_positions=2)
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

    Kept as a test because the trap is in the interaction, not in either piece.
    """
    from src.algorithms.risk import session_drawdown_breached

    state: dict = {}
    monday = datetime(2026, 6, 5, 15, tzinfo=timezone.utc)
    tuesday = datetime(2026, 6, 8, 15, tzinfo=timezone.utc)

    assert session_drawdown_breached(state, 10_000.0, -0.015, monday) is False
    # Down 20% overnight, and the next daily run still reads a drawdown of exactly zero.
    assert session_drawdown_breached(state, 8_000.0, -0.015, tuesday) is False
    assert state["session_drawdown"] == 0.0

    # Called twice inside one session -- it works as intended.
    assert session_drawdown_breached(state, 7_800.0, -0.015, tuesday) is True


def test_a_single_session_collapse_still_sells_the_holding() -> None:
    """``max_daily_drop`` is the stop that survives, and it reads close-to-close returns."""
    config = RallyRotationConfig(max_daily_drop=0.10)
    held = {"symbol": "AAA", "eligible": 1, "above_moving_average": True,
            "ma_distance": 0.20, "abs_return": 0.30, "fast_return": -0.12,
            "enough_history": True, "nano_return": -0.15}

    checks = exit_checks(held, config)

    assert passes(checks) is False
    culprit = blocking(checks)
    # The gate names itself *and* carries what it measured, which is the whole contract: a
    # rejection with no number attached is what this replaced.
    assert culprit.label == "No crash this session"
    assert "-15" in culprit.value
    assert culprit.limit == "> -10%"


# =========================================================================================
# Wiring
# =========================================================================================


def test_the_algorithm_is_registered_and_declares_what_it_needs() -> None:
    from src.algorithms.registry import get_algorithm_class
    from src.core.config import ALGORITHM_IDS
    from src.core.strategy_models import STRATEGY_LABELS

    assert "rally_rotation" in ALGORITHM_IDS
    assert STRATEGY_LABELS["rally_rotation"] == "Rally Rotation"

    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"])
    algorithm = get_algorithm_class("rally_rotation")(config)
    requirements = algorithm.requirements(config, {"ZZZ": 3})

    assert "ZZZ" in requirements.price_symbols, "held positions are priced too"
    # No intraday window at all: every feature is computed from the daily bars below.
    assert requirements.intraday_lookback_minutes == 0
    assert requirements.daily_lookback_days >= 100
    assert requirements.paper_only is True
    assert requirements.needs_sentiment is False, "sentiment is off until it is phased in"


def test_sentiment_is_requested_only_once_it_is_switched_on() -> None:
    from src.algorithms.registry import get_algorithm_class

    config = Runtime(sentiment_size_scale=0.05)
    algorithm = get_algorithm_class("rally_rotation")(config)

    assert algorithm.requirements(config, {}).needs_sentiment is True


def test_sizing_carries_no_cash_buffer() -> None:
    """Exposure is stated by ``risk_on_gross_max``; funding is checked against buying power.

    The algorithm used to carry the account cash buffer here so a plan built over a retained
    sub-threshold position could still be paid for. Order funding measures that directly now,
    and a buffer applied at sizing time would just under-invest on top of it.
    """
    algorithm = RallyRotationAlgorithm(Runtime())

    assert "cash_buffer" not in algorithm.sizing(Runtime())


def test_every_signal_row_carries_the_audit_trail() -> None:
    """Config-driven decisions are only auditable if each gate's verdict is recorded."""
    config = Runtime(risk_on_universe=["AAA"], defensive_universe=["BIL"])
    algorithm = RallyRotationAlgorithm(config)
    daily = {"AAA": daily_bars(80, 130), "BIL": daily_bars(100, 101)}
    intraday = {"AAA": intraday_bars(100, 130), "BIL": intraday_bars(100, 101)}

    row = algorithm.plan(context_for(config, daily, intraday)).signals["AAA"]

    for key in ("base_score", "rank", "eligible", "checks",
                "annual_volatility", "target_weight", "defensive_weight",
                "data_ok", "data_detail", "universe_coverage", "reason"):
        assert key in row, key


def test_a_close_is_not_blocked_by_the_rebalance_threshold() -> None:
    """A position under the threshold could never clear it, so it was held forever.

    The threshold suppresses small *adjustments* to a name the algorithm still wants. Applied
    to a close it traps the position instead: an 8-position book carried a mean of 12 names,
    5 of them frozen below the threshold and holding a fifth of the equity.
    """
    config = RallyRotationConfig(rebalance_weight_threshold=0.08, minimum_trade_notional=100.0,
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
    config = RallyRotationConfig(rebalance_step=0.5)

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
    assert partial_adjustment({"AAA": 0.40}, {"AAA": 0.20}, RallyRotationConfig())["AAA"] == 0.40


def test_undeployed_gross_is_parked_in_bills_not_left_as_cash() -> None:
    """A funded account holds bills, not idle cash, for whatever the risk sleeve cannot use."""
    config = RallyRotationConfig(risk_on_gross_max=0.96)

    parked = park_residual({"AAA": 0.5}, {"BIL": 0.96}, config)

    assert parked["AAA"] == pytest.approx(0.5)
    assert parked["BIL"] == pytest.approx(0.46)
    assert sum(parked.values()) == pytest.approx(0.96)

    # Fully deployed: nothing to park.
    full = park_residual({"AAA": 0.48, "BBB": 0.48}, {"BIL": 0.96}, config)
    assert "BIL" not in full


def _session(offset: int, hour: int = 10) -> datetime:
    """The ``offset``-th weekday from a fixed Monday, at ``hour`` market time."""
    day = datetime(2026, 1, 5, hour, 15, tzinfo=MARKET_TZ)
    while offset:
        day += timedelta(days=1)
        if day.weekday() < 5:
            offset -= 1
    return day


def test_eligibility_becomes_a_state_rather_than_a_daily_coin_flip() -> None:
    """A name near a floor used to flip in and out on consecutive sessions."""
    config = RallyRotationConfig(eligibility_window=10, entry_min_eligible_days=8,
                                exit_max_eligible_days=3)
    state: dict = {}

    # Alternating in and out: never a settled enough signal to open.
    for day, flag in enumerate([1, 0] * 5):
        history = track_eligibility(state, {"AAA": {"eligible": flag}}, config, _session(day))
    assert qualifying_days(history["AAA"]) == 5
    assert qualifying_days(history["AAA"]) < config.entry_min_eligible_days, "chop does not earn an entry"
    assert qualifying_days(history["AAA"]) > config.exit_max_eligible_days, "nor does it force an exit"

    # A settled signal does.
    for day in range(10, 20):
        history = track_eligibility(state, {"AAA": {"eligible": 1}}, config, _session(day))
    assert qualifying_days(history["AAA"]) == 10

    # The window is bounded, so a sustained breakdown eventually clears the exit bar.
    for day in range(20, 28):
        history = track_eligibility(state, {"AAA": {"eligible": 0}}, config, _session(day))
    assert qualifying_days(history["AAA"]) == 2 <= config.exit_max_eligible_days


def test_a_settling_period_counts_days_rather_than_runs() -> None:
    """The report that motivated this: five fires a session, so "3 days" took 36 minutes.

    Under ``15 10-14 * * 1-5`` the ranking window filled five times faster than the knob says,
    and a name that had led the cross-section since 10:15 was treated as having held the top of
    it for three days.
    """
    config = RallyRotationConfig(eligibility_window=10, entry_min_eligible_days=3, entry_rank_max=3)
    state: dict = {}

    # One session, five fires, top-ranked at every one of them.
    for hour in (10, 11, 12, 13, 14):
        history = track_ranking(state, {"IAU": {"rank": 1}}, config, _session(0, hour))
    assert observed_days(history["IAU"]) == 1, "five fires are one day of evidence"
    assert qualifying_days(history["IAU"]) == 1
    assert qualifying_days(history["IAU"]) < config.entry_min_eligible_days, "not settled yet"

    # Two more sessions at the top, and only now has it served the period.
    for day in (1, 2):
        history = track_ranking(state, {"IAU": {"rank": 1}}, config, _session(day))
    assert observed_days(history["IAU"]) == 3
    assert qualifying_days(history["IAU"]) >= config.entry_min_eligible_days


def test_the_last_run_of_a_session_is_the_one_that_counts() -> None:
    """A day is one observation, and the latest look is the best-informed one."""
    config = RallyRotationConfig(eligibility_window=10, entry_rank_max=3)
    state: dict = {}

    track_ranking(state, {"IAU": {"rank": 1}}, config, _session(0, 10))
    history = track_ranking(state, {"IAU": {"rank": 9}}, config, _session(0, 14))

    assert observed_days(history["IAU"]) == 1
    assert qualifying_days(history["IAU"]) == 0, "it faded by the close, so the day does not count"


def test_a_paused_binding_does_not_backfill_the_days_it_missed() -> None:
    """Absent days are not evidence. A pause slows qualifying; it must not disqualify."""
    config = RallyRotationConfig(eligibility_window=10, entry_min_eligible_days=3, entry_rank_max=3)
    state: dict = {}

    for day in (0, 1):
        track_ranking(state, {"IAU": {"rank": 1}}, config, _session(day))
    # Switched off for a fortnight, then back on.
    history = track_ranking(state, {"IAU": {"rank": 1}}, config, _session(12))

    assert observed_days(history["IAU"]) == 3, "the missed sessions are absent, not zeroed"
    assert qualifying_days(history["IAU"]) == 3


def test_a_cold_state_store_does_not_liquidate_the_book() -> None:
    """With no history a count reads as zero, which must not be read as 'ineligible'."""
    config = RallyRotationConfig(eligibility_window=10, exit_max_eligible_days=3)
    state: dict = {}

    history = track_eligibility(state, {"AAA": {"eligible": 1}}, config, _session(0))

    assert qualifying_days(history["AAA"]) == 1 <= config.exit_max_eligible_days
    assert observed_days(history["AAA"]) < config.eligibility_window, (
        "the window is not full, so the exit rule must not fire yet"
    )


def test_a_run_indexed_window_is_dropped_rather_than_read_as_days() -> None:
    """State written before this change is a bare list with no days attached.

    Reading it as days would let the old five-fires-a-session inflation survive the upgrade, so
    it is discarded. Both gates then see an unfilled window, which holds the book still and makes
    new entries serve the settling period -- the safe direction on both sides.
    """
    config = RallyRotationConfig(eligibility_window=10, exit_max_eligible_days=3)
    state: dict = {"eligible_history": {"AAA": [1] * 10}}

    history = track_eligibility(state, {"AAA": {"eligible": 1}}, config, _session(0))

    assert observed_days(history["AAA"]) == 1, "the run-indexed list is not carried over"
    assert observed_days(history["AAA"]) < config.eligibility_window, "so no exit can fire on it"


def test_a_holding_that_crashes_in_one_session_is_sold_at_once() -> None:
    """The portfolio-level breaker cannot fire at a daily cadence; this one can."""
    config = RallyRotationConfig(max_daily_drop=0.10, exit_threshold_slack=0.05,
                                etf_min_abs_return=0.0, etf_min_fast_return=-0.02)
    healthy = {"enough_history": 1, "ma_distance": 0.08, "abs_return": 0.20,
               "fast_return": 0.05, "nano_return": -0.01}

    stays = passes(exit_checks(healthy, config))
    assert stays

    crashed = {**healthy, "nano_return": -0.12}
    crashed_checks = exit_checks(crashed, config)
    stays, why = passes(crashed_checks), (blocking(crashed_checks).label if blocking(crashed_checks) else "")
    assert stays is False
    assert "crash this session" in why

    # Still inside the limit: everything else about the name is fine, so it is kept.
    dipped = {**healthy, "nano_return": -0.08}
    assert passes(exit_checks(dipped, config)) is True


def test_the_no_trade_band_cannot_push_the_book_over_its_gross_budget() -> None:
    """The 2026-01-08 case: a 100% proposal reached planning as a 106.9% one.

    The band suppresses moves in both directions, but only the trims were funding anything.
    Holding two incumbents above target because their trims were too small to trade, while
    passing a new entry through at full size, produced a plan that could not be paid for --
    16 of 24 sessions that January, and on one of them the entry was dropped outright.
    """
    config = RallyRotationConfig(rebalance_weight_threshold=0.08, minimum_trade_notional=0.0,
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
    config = RallyRotationConfig(rebalance_weight_threshold=0.08, minimum_trade_notional=100.0,
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
    """One clock per decision, counted in elapsed sessions rather than in runs.

    Runs are the wrong unit because the binding states the cron: at five fires a session a
    three-"day" interval came due three times before lunch. Sessions rather than calendar days
    because wall-clock days do not survive a weekend -- a three-day interval set on a Friday
    would be satisfied by Monday, throttling Tuesday-to-Thursday decisions while waving every
    Friday-to-Monday one straight through.
    """
    state: dict = {}
    monday = datetime(2026, 8, 17, 10, 15, tzinfo=MARKET_TZ)

    # A cold state store has never acted, so every clock is due.
    assert action_due(state, "theme_rotation", 3, monday) is True

    record_action(state, "theme_rotation", monday)
    # Four more fires the same session must not advance a clock measured in sessions.
    for hour in (11, 12, 13, 14):
        assert action_due(state, "theme_rotation", 3, monday.replace(hour=hour)) is False, \
            "later the same session"
    assert action_due(state, "theme_rotation", 3, monday + timedelta(days=1)) is False, "Tuesday"
    assert action_due(state, "theme_rotation", 3, monday + timedelta(days=2)) is False, "Wednesday"
    assert action_due(state, "theme_rotation", 3, monday + timedelta(days=3)) is True, "Thursday"

    # The clocks are independent: rotating does not start the intra-theme one.
    assert action_due(state, "intra_theme", 3, monday) is True

    # Zero means every run.
    assert action_due(state, "exit", 0, monday) is True


def test_the_rerank_clock_skips_the_weekend_rather_than_counting_it() -> None:
    """Three sessions after a Thursday is the following Tuesday, not Sunday."""
    state: dict = {}
    thursday = datetime(2026, 8, 20, 10, 15, tzinfo=MARKET_TZ)
    record_action(state, "rerank", thursday)

    assert sessions_since("2026-08-20", thursday + timedelta(days=3)) == 1, "Sunday is one session on"
    assert action_due(state, "rerank", 3, thursday + timedelta(days=3)) is False, "Sunday"
    assert action_due(state, "rerank", 3, thursday + timedelta(days=4)) is False, "Monday"
    assert action_due(state, "rerank", 3, thursday + timedelta(days=5)) is True, "Tuesday"


def test_a_paused_binding_comes_back_due_rather_than_frozen_mid_interval() -> None:
    """The complaint that motivated dropping the run counter: no runs, so no clock.

    Counting runs meant a binding switched off for a fortnight came back believing no time had
    passed, and went on holding a book selected against a cross-section three weeks stale.
    """
    state: dict = {}
    start = datetime(2026, 8, 3, 10, 15, tzinfo=MARKET_TZ)
    record_action(state, "rerank", start)

    assert action_due(state, "rerank", 5, start) is False
    # Fourteen calendar days later, with not one run in between.
    assert action_due(state, "rerank", 5, start + timedelta(days=14)) is True


def test_a_caller_that_does_not_act_does_not_restart_the_clock() -> None:
    """Otherwise a throttled decision that changed nothing would push the next one out."""
    state: dict = {}
    monday = datetime(2026, 8, 17, 10, 15, tzinfo=MARKET_TZ)

    assert action_due(state, "entry", 3, monday) is True
    # ``action_due`` is a question, not a commitment -- only ``record_action`` starts the clock.
    assert action_due(state, "entry", 3, monday) is True
    record_action(state, "entry", monday)
    assert action_due(state, "entry", 3, monday + timedelta(days=1)) is False


def test_the_crash_stop_answers_to_no_clock() -> None:
    """``max_daily_drop`` must fire on the session it happens, whatever exit_interval_days says.

    Otherwise a name can gap 30% across a week of throttled sessions while the algorithm waits
    its turn to look at it.
    """
    config = RallyRotationConfig(max_daily_drop=0.10)

    assert crash_stop({"nano_return": -0.15}, config).ok is False
    assert crash_stop({"nano_return": -0.05}, config).ok is True
    # 0 turns it off entirely.
    assert crash_stop({"nano_return": -0.99}, RallyRotationConfig(max_daily_drop=0.0)).ok is True
