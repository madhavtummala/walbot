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


def test_builtin_dual_momentum_keeps_original_long_short_template() -> None:
    rows = strategy_signal_rows(
        "dual_momentum",
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


def test_dual_momentum_can_apply_sentiment_tilt() -> None:
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
        "dual_momentum",
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

    wanted = AlgorithmRequirements(price_symbols=["SPY"], history_lookback_minutes=1170)
    assert wanted.history_lookback_minutes == 1170
    # Naming a grid stays optional, and means "prefer this fidelity", not "count in these".
    assert wanted.preferred_bar_minutes == 0


def test_algorithms_state_horizons_in_minutes_not_bars() -> None:
    """No algorithm counts in bars any more, so none of them pins a bar size."""
    from src.algorithms.dual_momentum import DualMomentumConfig
    from src.algorithms.fast_momentum import DefensiveMomentumConfig
    from src.algorithms.invest_spy import InvestSpyConfig

    for config_cls in (DualMomentumConfig, DefensiveMomentumConfig, InvestSpyConfig):
        # 0 means "whatever the feed prefers" -- the grid is the data layer's business now.
        assert config_cls().intraday_bar_minutes == 0, config_cls.__name__
        assert config_cls().required_history_minutes > 0, config_cls.__name__

    # The horizons carried over at their 15-minute wall-clock equivalents.
    assert DefensiveMomentumConfig().micro_momentum_lookback_minutes == 78 * 15
    assert DualMomentumConfig().selection_horizon_macro_minutes == 320 * 15
    assert InvestSpyConfig().meso_momentum_lookback_minutes == 26 * 15


def test_a_config_saved_in_bars_is_read_as_the_minutes_it_meant() -> None:
    """Dashboard tuning saved under the old names must not silently revert to defaults."""
    from src.algorithms.fast_momentum import DefensiveMomentumConfig

    class _Runtime:
        algorithm_configs = {
            "fast_momentum": {
                "intraday_bar_minutes": 15,
                "micro_momentum_lookback_bars": 78,
                "volatility_lookback_bars": 13,
            }
        }

    parsed = DefensiveMomentumConfig.from_runtime_config(_Runtime())
    assert parsed.micro_momentum_lookback_minutes == 1170
    assert parsed.volatility_lookback_minutes == 195


def test_a_minutes_key_wins_over_a_stale_bars_key() -> None:
    """Once saved in minutes, a leftover bars key must not override it."""
    from src.algorithms.dual_momentum import DualMomentumConfig

    class _Runtime:
        algorithm_configs = {
            "dual_momentum": {
                "intraday_bar_minutes": 15,
                "selection_horizon_macro": 320,
                "selection_horizon_macro_minutes": 2400,
            }
        }

    assert DualMomentumConfig.from_runtime_config(_Runtime()).selection_horizon_macro_minutes == 2400
