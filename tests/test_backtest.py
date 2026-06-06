from __future__ import annotations

import pandas as pd

from src.execution.backtest import run_backtest


def _bars_from_prices(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": [price * 1.005 for price in prices],
            "volume": [1_000_000 + index * 1_000 for index in range(len(prices))],
        }
    )


def test_backtest_marks_positions_forward_without_alpaca(monkeypatch) -> None:
    monkeypatch.setenv("SYMBOLS", "AAA,BBB")
    monkeypatch.setenv("MOMENTUM_LOOKBACK_DAYS", "5")
    monkeypatch.setenv("SHORT_MOMENTUM_LOOKBACK_DAYS", "3")
    monkeypatch.setenv("LONG_MA_DAYS", "10")
    monkeypatch.setenv("VOLUME_LOOKBACK_DAYS", "5")
    monkeypatch.setenv("MAX_WEIGHT_PER_SYMBOL", "0.60")
    monkeypatch.setenv("MAX_PORTFOLIO_EXPOSURE", "0.90")
    monkeypatch.setenv("MAX_LONGS", "1")
    monkeypatch.setenv("MIN_COMPOSITE_SCORE", "0.0")
    monkeypatch.setenv("TARGET_ANNUAL_VOL", "0.50")
    monkeypatch.setenv("TRANSACTION_COST_BPS", "0")

    bars_by_symbol = {
        "AAA": _bars_from_prices([100 + index for index in range(40)]),
        "BBB": _bars_from_prices([140 - index for index in range(40)]),
    }

    history = run_backtest(starting_equity=100_000.0, bars_by_symbol=bars_by_symbol, social_by_symbol={})

    assert not history.empty
    assert history["equity"].iloc[-1] > history["equity"].iloc[0]
    assert history["shares_AAA"].max() > 0
    assert history["shares_BBB"].max() == 0
