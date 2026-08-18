"""Performance statistics for an equity curve.

All that survives of the old hand-written backtester. ``execution/replay.py`` replaced its
simulation loop -- that loop reimplemented each strategy's scoring and could drift from the
algorithm it claimed to test, which is the whole reason replay exists. These metrics are
strategy-agnostic arithmetic over an equity series, so they outlived it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_performance_metrics(equity_series: pd.Series) -> dict[str, float]:
    returns = equity_series.pct_change().fillna(0)
    trading_days = len(returns)
    if trading_days <= 1:
        return {"cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}

    annual_factor = 252
    total_return = equity_series.iloc[-1] / equity_series.iloc[0]
    cagr = total_return ** (annual_factor / trading_days) - 1
    rolling_max = equity_series.cummax()
    drawdown = equity_series / rolling_max - 1
    max_drawdown = float(drawdown.min())
    vol = float(returns.std())
    sharpe = float((returns.mean() / vol) * np.sqrt(annual_factor)) if vol > 0 else 0.0

    return {
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }
