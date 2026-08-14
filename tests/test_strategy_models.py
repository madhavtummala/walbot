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


def test_defensive_momentum_selects_risk_on_symbols_above_bil_hurdle() -> None:
    rows = strategy_signal_rows(
        "fast_momentum",
        {
            "SPY": _trend_bars(100, 125),
            "XBI": _trend_bars(100, 180),
            "BIL": _trend_bars(100, 104),
            "SHY": _trend_bars(100, 102),
            "AGG": _trend_bars(100, 101),
        },
    )

    active = [row for row in rows if row["signal"] == 1]

    assert {row["symbol"] for row in active} == {"XBI", "SPY"}
    assert all(row["side"] == "LONG" for row in active)
    assert all(row["ret_252"] > row["cash_hurdle"] for row in active)


def test_defensive_momentum_rotates_to_defensive_when_risk_on_fails() -> None:
    rows = strategy_signal_rows(
        "fast_momentum",
        {
            "SPY": _trend_bars(100, 98),
            "XBI": _trend_bars(100, 92),
            "BIL": _trend_bars(100, 104),
            "SHY": _trend_bars(100, 102),
            "AGG": _trend_bars(100, 101),
            "TLT": _trend_bars(100, 95),
        },
    )

    active = [row for row in rows if row["signal"] == 1]

    assert [row["symbol"] for row in active] == ["BIL", "SHY", "AGG"]
    assert all(row["side"] == "LONG" for row in active)
    assert all("defensive" in row["reason"].lower() for row in active)


def test_defensive_momentum_goes_to_cash_without_defensive_symbols() -> None:
    rows = strategy_signal_rows(
        "fast_momentum",
        {
            "VTI": _trend_bars(100, 98),
            "VXUS": _trend_bars(100, 97),
            "IEMG": _trend_bars(100, 95),
            "ACWI": _trend_bars(100, 96),
        },
    )

    assert all(row["signal"] == 0 for row in rows)
    assert all(row["side"] == "FLAT" for row in rows)


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


def test_intraday_requirement_must_declare_its_bar_grid() -> None:
    """A lookback counted in bars is meaningless until the algorithm names the grid."""
    import pytest

    from src.core.interfaces import AlgorithmRequirements

    with pytest.raises(ValueError, match="must declare intraday_bar_minutes"):
        AlgorithmRequirements(price_symbols=["SPY"], intraday_lookback_bars=100)

    # Declining intraday data entirely stays valid, and states no grid.
    assert AlgorithmRequirements(price_symbols=["SPY"]).intraday_bar_minutes == 0


def test_intraday_algorithms_declare_their_own_grid() -> None:
    """The 15-minute assumption lives on each algorithm now, not in the base interface."""
    from src.algorithms.dual_momentum import DualMomentumConfig
    from src.algorithms.fast_momentum import DefensiveMomentumConfig
    from src.algorithms.invest_spy import InvestSpyConfig

    for config_cls in (DualMomentumConfig, DefensiveMomentumConfig, InvestSpyConfig):
        assert config_cls().intraday_bar_minutes == 15, config_cls.__name__
