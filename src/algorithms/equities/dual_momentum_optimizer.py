from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations, product
from typing import Any, Iterable

import pandas as pd

from src.execution.backtest import calculate_performance_metrics
from src.core.strategy_models import DEFENSIVE_MOMENTUM_DEFENSIVE_ORDER, DEFENSIVE_MOMENTUM_DEFENSIVE_SYMBOLS, prepared_strategy_frame


@dataclass(frozen=True)
class DualMomentumConfig:
    long_lookback_days: int = 252
    short_lookback_days: int = 63
    long_weight: float = 0.70
    short_weight: float = 0.30
    risk_on_limit: int = 3
    defensive_limit: int = 3
    max_portfolio_exposure: float = 0.95
    transaction_cost_bps: float = 1.0
    cash_hurdle_symbol: str = "BIL"


def default_dual_momentum_grid() -> list[DualMomentumConfig]:
    return [
        DualMomentumConfig(
            long_lookback_days=long_lookback,
            short_lookback_days=short_lookback,
            risk_on_limit=risk_on_limit,
            defensive_limit=defensive_limit,
        )
        for long_lookback, short_lookback, risk_on_limit, defensive_limit in product(
            (126, 189, 252),
            (21, 63),
            (2, 3, 5),
            (1, 3),
        )
        if short_lookback < long_lookback
    ]


def universe_subsets(
    symbols: Iterable[str],
    *,
    min_size: int = 4,
    max_size: int = 8,
    max_subsets: int = 50,
) -> list[list[str]]:
    unique = sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
    if len(unique) <= max_size:
        return [unique]
    subsets: list[list[str]] = []
    for size in range(max(min_size, 1), min(max_size, len(unique)) + 1):
        for combo in combinations(unique, size):
            subsets.append(list(combo))
            if len(subsets) >= max_subsets:
                return subsets
    return subsets or [unique[:max_size]]


def _price_at(frame: pd.DataFrame, timestamp: pd.Timestamp, column: str) -> float:
    value = frame.loc[timestamp, column]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return float(value)


def _momentum_at(frame: pd.DataFrame, timestamp: pd.Timestamp, config: DualMomentumConfig) -> dict[str, float]:
    history = frame.loc[:timestamp]
    if len(history) <= max(config.long_lookback_days, config.short_lookback_days):
        return {"score": 0.0, "long_return": 0.0, "short_return": 0.0}
    close = pd.to_numeric(history["close"], errors="coerce")
    latest = float(close.iloc[-1])
    long_return = latest / float(close.iloc[-config.long_lookback_days]) - 1
    short_return = latest / float(close.iloc[-config.short_lookback_days]) - 1
    score = config.long_weight * long_return + config.short_weight * short_return
    return {"score": score, "long_return": long_return, "short_return": short_return}


def _weights_for_date(
    frames: dict[str, pd.DataFrame],
    signal_date: pd.Timestamp,
    config: DualMomentumConfig,
) -> dict[str, float]:
    scores = {symbol: _momentum_at(frame, signal_date, config) for symbol, frame in frames.items()}
    cash_hurdle = max(0.0, scores.get(config.cash_hurdle_symbol, {}).get("long_return", 0.0))
    risk_on = [
        (symbol, values)
        for symbol, values in scores.items()
        if symbol not in DEFENSIVE_MOMENTUM_DEFENSIVE_SYMBOLS
        and values["long_return"] > cash_hurdle
        and values["score"] > 0
    ]
    risk_on.sort(key=lambda item: item[1]["score"], reverse=True)
    selected = [symbol for symbol, _values in risk_on[: max(int(config.risk_on_limit), 1)]]

    if not selected:
        defensive = [symbol for symbol in DEFENSIVE_MOMENTUM_DEFENSIVE_ORDER if symbol in frames]
        selected = defensive[: max(int(config.defensive_limit), 0)]

    weights = {symbol: 0.0 for symbol in frames}
    if not selected:
        return weights
    weight = max(min(float(config.max_portfolio_exposure), 1.0), 0.0) / len(selected)
    for symbol in selected:
        weights[symbol] = weight
    return weights


def run_dual_momentum_backtest(
    bars_by_symbol: dict[str, pd.DataFrame],
    symbols: list[str],
    config: DualMomentumConfig,
    *,
    starting_equity: float = 10_000.0,
) -> dict[str, Any]:
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = prepared_strategy_frame(bars_by_symbol.get(symbol, pd.DataFrame()))
        if frame.empty:
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frames[symbol] = frame.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    if not frames:
        raise RuntimeError("No bars were available for the dual momentum experiment.")

    common_dates = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
    required = max(config.long_lookback_days, config.short_lookback_days) + 1
    if len(common_dates) <= required + 1:
        raise RuntimeError("Not enough common trading dates for the dual momentum experiment.")

    cash = float(starting_equity)
    positions = {symbol: 0.0 for symbol in frames}
    records: list[dict[str, Any]] = []
    cost_rate = max(float(config.transaction_cost_bps), 0.0) / 10_000

    for index in range(required, len(common_dates)):
        signal_date = common_dates[index - 1]
        trade_date = common_dates[index]
        weights = _weights_for_date(frames, signal_date, config)
        open_prices = {symbol: _price_at(frame, trade_date, "open") for symbol, frame in frames.items()}
        close_prices = {symbol: _price_at(frame, trade_date, "close") for symbol, frame in frames.items()}
        equity_at_open = cash + sum(positions[symbol] * open_prices[symbol] for symbol in frames)
        target_positions = {
            symbol: (equity_at_open * weights[symbol]) / open_prices[symbol] if open_prices[symbol] > 0 else 0.0
            for symbol in frames
        }
        turnover = 0.0
        for symbol, target in target_positions.items():
            diff = target - positions[symbol]
            if abs(diff) <= 1e-9:
                continue
            notional = abs(diff) * open_prices[symbol]
            cost = notional * cost_rate
            cash -= diff * open_prices[symbol] + cost
            turnover += notional
            positions[symbol] = target

        equity = cash + sum(positions[symbol] * close_prices[symbol] for symbol in frames)
        records.append({"timestamp": trade_date, "equity": equity, "turnover": turnover})

    history = pd.DataFrame(records).set_index("timestamp").sort_index()
    metrics = calculate_performance_metrics(history["equity"])
    return {
        "config": asdict(config),
        "symbols": list(frames),
        "ending_equity": float(history["equity"].iloc[-1]),
        "history": history,
        "metrics": metrics,
    }


def rank_dual_momentum_experiments(
    bars_by_symbol: dict[str, pd.DataFrame],
    configs: list[DualMomentumConfig] | None = None,
    subsets: list[list[str]] | None = None,
    *,
    starting_equity: float = 10_000.0,
    sort_metric: str = "sharpe",
) -> list[dict[str, Any]]:
    configs = configs or default_dual_momentum_grid()
    subsets = subsets or universe_subsets(bars_by_symbol)
    results: list[dict[str, Any]] = []
    for subset in subsets:
        for config in configs:
            try:
                result = run_dual_momentum_backtest(
                    bars_by_symbol,
                    subset,
                    config,
                    starting_equity=starting_equity,
                )
            except RuntimeError:
                continue
            compact = {key: value for key, value in result.items() if key != "history"}
            compact["score"] = float(compact["metrics"].get(sort_metric, 0.0))
            results.append(compact)
    return sorted(results, key=lambda item: item["score"], reverse=True)
