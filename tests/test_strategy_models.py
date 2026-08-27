from __future__ import annotations

import pandas as pd

from src.core.strategy_models import strategy_signal_rows


def _trend_bars(start: float, end: float, periods: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=periods, tz="UTC")
    prices = [start + ((end - start) * index / (periods - 1)) for index in range(periods)]
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
            "volume": [1_000_000 for _date in dates],
        }
    )


def test_builtin_rally_rotation_keeps_original_long_short_template() -> None:
    rows = strategy_signal_rows(
        "rally_rotation",
        {
            "SPY": _trend_bars(100, 130),
            "XBI": _trend_bars(100, 90),
            "BIL": _trend_bars(100, 104),
        },
    )

    by_symbol = {row["symbol"]: row for row in rows}

    assert by_symbol["SPY"]["side"] == "LONG"
    assert by_symbol["XBI"]["side"] == "SHORT"
    assert by_symbol["BIL"]["side"] == "LONG"
    assert by_symbol["SPY"]["score"] == 0.6 * by_symbol["SPY"]["ret_126"] + 0.4 * by_symbol["SPY"]["ret_252"]


def test_rally_rotation_can_apply_sentiment_tilt() -> None:
    social = {
        "SPY": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-05-28T15:00:00Z"], utc=True),
                "mentions": [10],
                "sentiment": [0.5],
                "social_score": [0.5],
            }
        )
    }
    rows = strategy_signal_rows(
        "rally_rotation",
        {"SPY": _trend_bars(100, 130), "BIL": _trend_bars(100, 104)},
        social_by_symbol=social,
        social_weight=0.1,
    )

    spy = {row["symbol"]: row for row in rows}["SPY"]

    assert round(spy["social_score"], 6) == 0.325
    assert round(spy["score"] - spy["price_score"], 6) == 0.0325
    assert "sentiment tilt" in spy["reason"]


def test_a_history_requirement_needs_no_grid_to_be_meaningful() -> None:
    """A lookback in minutes states its own span, so no bar size has to accompany it."""
    from src.core.interfaces import AlgorithmRequirements

    wanted = AlgorithmRequirements(price_symbols=["SPY"], intraday_lookback_minutes=1170)
    assert wanted.intraday_lookback_minutes == 1170
    # Naming a grid stays optional, and means "prefer this fidelity", not "count in these".
    assert wanted.preferred_bar_minutes == 0


def test_algorithms_state_horizons_in_days_not_bars_or_minutes() -> None:
    """No algorithm counts in bars any more, and rally rotation no longer counts in minutes."""
    from src.algorithms.rally_rotation.config import RallyRotationConfig

    # Rally rotation reads daily bars only, so it asks for no intraday window at all rather
    # than one it would not look at.
    assert RallyRotationConfig().required_history_minutes == 0

    # The horizons are stated in market days, which is the only unit a daily-bar score can
    # resolve. Twelve sessions, where the config used to say 4680 minutes.
    assert RallyRotationConfig().macro_days == 12


def test_the_newest_horizon_key_wins_over_every_stale_one() -> None:
    """Four generations of key can sit in one saved config; the newest has to win.

    Bars on a 15-minute grid, then market minutes, then days under a long name, then days under
    a short one. A reader that preferred an older key would silently replay a horizon nobody had
    set since three migrations ago.
    """
    from src.algorithms.rally_rotation.algorithm import RallyRotationAlgorithm

    class _Runtime:
        algorithm_configs = {
            "rally_rotation": {
                "selection_horizon_macro": 320,           # bars           -> 12 days
                "selection_horizon_macro_minutes": 2400,  # minutes        -> 6 days
                "selection_horizon_macro_days": 40,       # the long spelling
                "macro_days": 60,                         # the one that counts
            }
        }

    assert RallyRotationAlgorithm(_Runtime()).tuning(_Runtime()).macro_days == 60

    class _LongDays:
        algorithm_configs = {"rally_rotation": {"selection_horizon_macro": 320,
                                                "selection_horizon_macro_minutes": 2400,
                                                "selection_horizon_macro_days": 40}}

    assert RallyRotationAlgorithm(_LongDays()).tuning(_LongDays()).macro_days == 40

    class _Older:
        algorithm_configs = {"rally_rotation": {"selection_horizon_macro": 320,
                                                "selection_horizon_macro_minutes": 2400}}

    # 2400 minutes / 390 = 6.15 sessions, rounded to the nearest whole day.
    assert RallyRotationAlgorithm(_Older()).tuning(_Older()).macro_days == 6

    class _Oldest:
        algorithm_configs = {"rally_rotation": {"selection_horizon_macro": 320}}

    # 320 bars x 15 minutes = 4800 minutes = 12.3 sessions.
    assert RallyRotationAlgorithm(_Oldest()).tuning(_Oldest()).macro_days == 12


def test_live_signals_are_ordered_by_score_not_by_its_magnitude() -> None:
    """The dashboard list should read in the order the ranking was decided in.

    The previous ordering grouped LONG/SHORT/FLAT and then sorted on ``-abs(score)``, which
    seats the worst name in the universe next to the best: -3.0 sorted ahead of +1.0. For a
    cross-sectional ranker the sign is the entire signal.
    """
    from src.algorithms.base import signal_view_from_plan
    from src.core.interfaces import AlgorithmPlan, intents_from_weights

    plan = AlgorithmPlan(
        intents=intents_from_weights({"GOOD": 0.6}),
        signals={
            "WORST": {"signal": 0, "score": -3.0},
            "GOOD": {"signal": 1, "score": 1.0},
            "MID": {"signal": 0, "score": 0.2},
        },
    )

    view = signal_view_from_plan(plan)

    assert [row.symbol for row in view.rows] == ["GOOD", "MID", "WORST"]
