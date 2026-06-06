from __future__ import annotations

import pandas as pd

from src.algorithms.equities.dual_momentum_optimizer import DualMomentumConfig, rank_dual_momentum_experiments, universe_subsets


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


def test_dual_momentum_experiments_rank_universe_subsets() -> None:
    bars = {
        "WIN": _trend_bars(100, 160),
        "LOSE": _trend_bars(100, 80),
        "BIL": _trend_bars(100, 103),
    }
    config = DualMomentumConfig(long_lookback_days=126, short_lookback_days=21, risk_on_limit=1, defensive_limit=1)

    results = rank_dual_momentum_experiments(
        bars,
        configs=[config],
        subsets=[["WIN", "BIL"], ["LOSE", "BIL"]],
        sort_metric="cagr",
    )

    assert results[0]["symbols"] == ["WIN", "BIL"]
    assert results[0]["metrics"]["cagr"] > results[1]["metrics"]["cagr"]


def test_dual_momentum_experiments_rank_configurations() -> None:
    bars = {
        "WIN": _trend_bars(100, 160),
        "BIL": _trend_bars(100, 103),
    }
    configs = [
        DualMomentumConfig(long_lookback_days=252, short_lookback_days=63, risk_on_limit=1, defensive_limit=1),
        DualMomentumConfig(long_lookback_days=126, short_lookback_days=21, risk_on_limit=1, defensive_limit=1),
    ]

    results = rank_dual_momentum_experiments(bars, configs=configs, subsets=[["WIN", "BIL"]])

    assert len(results) == 2
    assert results[0]["score"] >= results[1]["score"]
    assert {result["config"]["long_lookback_days"] for result in results} == {126, 252}


def test_universe_subsets_caps_combinations() -> None:
    subsets = universe_subsets(["A", "B", "C", "D", "E"], min_size=2, max_size=3, max_subsets=4)

    assert len(subsets) == 4
    assert all(2 <= len(subset) <= 3 for subset in subsets)
