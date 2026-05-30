from __future__ import annotations
import logging
from datetime import datetime
from math import floor

import numpy as np
import pandas as pd

from .config import get_config
from .data import fetch_daily_bars
from .portfolio import compute_target_weights
from .signals import compute_signals_for_universe
from .logging_utils import configure_logging
from .social import load_social_trends_csv, truncate_social_history

logger = logging.getLogger(__name__)


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


def _prepare_history_by_symbol(bars_by_symbol: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    history_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol, df in bars_by_symbol.items():
        if df.empty:
            continue
        work_df = df.copy()
        work_df["timestamp"] = pd.to_datetime(work_df["timestamp"], utc=True)
        history_by_symbol[symbol] = work_df.sort_values("timestamp").set_index("timestamp")
    return history_by_symbol


def _price_at(df: pd.DataFrame, timestamp, column: str) -> float:
    value = df.loc[timestamp, column]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    return float(value)


def _portfolio_value(positions: dict[str, int], prices: dict[str, float], cash: float) -> float:
    return cash + sum(positions.get(symbol, 0) * price for symbol, price in prices.items())


def _rebalance_positions(
    positions: dict[str, int],
    cash: float,
    prices: dict[str, float],
    target_weights: dict[str, float],
    equity: float,
    cash_buffer: float,
    transaction_cost_bps: float,
) -> tuple[dict[str, int], float, float, float]:
    transaction_costs = 0.0
    turnover = 0.0
    investable_equity = equity * max(0.0, min(1.0, 1.0 - cash_buffer))
    target_shares = {
        symbol: floor((investable_equity * max(weight, 0.0)) / prices[symbol])
        if prices[symbol] > 0
        else 0
        for symbol, weight in target_weights.items()
    }

    for symbol, desired_shares in target_shares.items():
        current_shares = positions.get(symbol, 0)
        diff = desired_shares - current_shares
        if diff >= 0:
            continue
        shares_to_sell = abs(diff)
        notional = shares_to_sell * prices[symbol]
        cost = notional * transaction_cost_bps / 10_000
        cash += notional - cost
        transaction_costs += cost
        turnover += notional
        positions[symbol] = desired_shares

    for symbol, desired_shares in target_shares.items():
        current_shares = positions.get(symbol, 0)
        diff = desired_shares - current_shares
        if diff <= 0:
            continue
        notional = diff * prices[symbol]
        cost = notional * transaction_cost_bps / 10_000
        if notional + cost > cash:
            affordable_shares = floor(cash / (prices[symbol] * (1 + transaction_cost_bps / 10_000)))
            diff = max(0, affordable_shares)
            notional = diff * prices[symbol]
            cost = notional * transaction_cost_bps / 10_000
        if diff == 0:
            continue
        cash -= notional + cost
        transaction_costs += cost
        turnover += notional
        positions[symbol] = current_shares + diff

    return positions, cash, transaction_costs, turnover


def run_backtest(
    starting_equity: float = 10_000.0,
    end_date: datetime | None = None,
    bars_by_symbol: dict[str, pd.DataFrame] | None = None,
    social_by_symbol: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    config = get_config()
    if bars_by_symbol is None:
        from .alpaca_client import create_data_client

        data_client = create_data_client(config)
        bars_by_symbol = fetch_daily_bars(
            config.symbols,
            config.momentum_lookback_days,
            ma_days=config.long_ma_days,
            extra_buffer_days=config.history_extra_buffer_days,
            alpaca_data_client=data_client,
            end_date=end_date,
            data_feed=config.alpaca_data_feed,
        )
    if social_by_symbol is None:
        social_by_symbol = load_social_trends_csv(config.social_trends_csv, config.symbols)

    history_by_symbol = _prepare_history_by_symbol(bars_by_symbol)
    if not history_by_symbol:
        raise RuntimeError("No historical bars were available for the backtest.")
    if len(history_by_symbol) != len(config.symbols):
        logger.warning("Some symbols did not return bars: %s", set(config.symbols) - set(history_by_symbol.keys()))

    common_dates = sorted(
        set.intersection(*(set(df.index) for df in history_by_symbol.values()))
    )
    if not common_dates:
        raise RuntimeError("No common trading dates were available across the backtest universe.")
    required_history = max(config.momentum_lookback_days, config.long_ma_days) + 1

    positions = {symbol: 0 for symbol in history_by_symbol.keys()}
    cash = starting_equity
    history_records: list[dict[str, float | datetime]] = []

    for index in range(1, len(common_dates)):
        signal_date = common_dates[index - 1]
        trade_date = common_dates[index]
        if any(len(df.loc[:signal_date]) < required_history for df in history_by_symbol.values()):
            continue

        open_prices = {
            symbol: _price_at(df, trade_date, "open")
            for symbol, df in history_by_symbol.items()
        }
        close_prices = {
            symbol: _price_at(df, trade_date, "close")
            for symbol, df in history_by_symbol.items()
        }
        equity_at_open = _portfolio_value(positions, open_prices, cash)

        snapshot = {
            symbol: df.loc[:signal_date].reset_index()
            for symbol, df in history_by_symbol.items()
        }
        social_snapshot = truncate_social_history(social_by_symbol, signal_date)
        signals = compute_signals_for_universe(
            snapshot,
            config.momentum_lookback_days,
            config.long_ma_days,
            short_lookback_days=config.short_momentum_lookback_days,
            volume_lookback_days=config.volume_lookback_days,
            social_by_symbol=social_snapshot,
            social_lookback_days=config.social_lookback_days,
            price_momentum_weight=config.price_momentum_weight,
            social_momentum_weight=config.social_momentum_weight,
            volume_momentum_weight=config.volume_momentum_weight,
            min_composite_score=config.min_composite_score,
        )
        weights = compute_target_weights(
            signals,
            config.max_weight_per_symbol,
            max_portfolio_exposure=config.max_portfolio_exposure,
            max_longs=config.max_longs,
            target_annual_vol=config.target_annual_vol,
        )
        positions, cash, transaction_costs, turnover = _rebalance_positions(
            positions,
            cash,
            open_prices,
            weights,
            equity_at_open,
            config.cash_buffer,
            config.transaction_cost_bps,
        )
        invested = sum(positions.get(symbol, 0) * close_prices[symbol] for symbol in history_by_symbol.keys())
        equity = invested + cash
        position_values = {
            symbol: positions.get(symbol, 0) * close_prices[symbol]
            for symbol in history_by_symbol.keys()
            if abs(positions.get(symbol, 0) * close_prices[symbol]) > 0.005
        }

        history_records.append(
            {
                "timestamp": trade_date,
                "signal_timestamp": signal_date,
                "equity": equity,
                "cash": cash,
                "invested": invested,
                "positions": position_values,
                "turnover": turnover,
                "transaction_costs": transaction_costs,
                **{f"weight_{symbol}": weights.get(symbol, 0.0) for symbol in history_by_symbol.keys()},
                **{f"shares_{symbol}": positions.get(symbol, 0) for symbol in history_by_symbol.keys()},
                **{f"score_{symbol}": signals[symbol].get("score", 0.0) for symbol in history_by_symbol.keys()},
            }
        )

    history_df = pd.DataFrame(history_records).set_index("timestamp").sort_index()
    if history_df.empty:
        raise RuntimeError("Backtest did not produce any equity history.")

    metrics = calculate_performance_metrics(history_df["equity"])
    logger.info("Backtest results")
    logger.info("  start equity: %.2f", starting_equity)
    logger.info("  ending equity: %.2f", history_df["equity"].iloc[-1])
    logger.info("  CAGR: %.2f%%", metrics["cagr"] * 100)
    logger.info("  Max drawdown: %.2f%%", metrics["max_drawdown"] * 100)
    logger.info("  Sharpe ratio: %.2f", metrics["sharpe"])

    return history_df


def main() -> None:
    configure_logging()
    history_df = run_backtest()
    print(history_df.tail())


if __name__ == "__main__":
    main()
