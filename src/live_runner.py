from __future__ import annotations
import logging
from .config import get_config
from .connectors import fetch_latest_market_quotes, fetch_latest_news_sentiment, merge_social_frames, news_records_to_social_frames
from .controls import load_controls
from .alpaca_client import (
    create_data_client,
    create_trading_client,
    get_account_equity,
    get_positions,
    get_latest_price,
    is_market_open,
)
from .data import fetch_daily_bars
from .logging_utils import configure_logging, log_signals, log_portfolio, log_orders, log_position_changes
from .orders import sync_positions_to_targets
from .portfolio import compute_target_weights
from .signals import compute_signals_for_universe
from .social import load_social_trends_csv
from .strategy_models import STRATEGY_LABELS, strategy_signal_rows, weights_from_strategy_rows
from .fast_momentum import DefensiveMomentumConfig, build_defensive_momentum_targets

logger = logging.getLogger(__name__)


def _max_live_exposure(config) -> float:
    return min(
        max(float(config.max_portfolio_exposure), 0.0),
        max(1.0 - max(float(config.cash_buffer), 0.0), 0.0),
        1.0,
    )


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sizing_equity(config, account_equity: float) -> float:
    cap = max(_as_float(getattr(config, "algorithm_equity_cap", 0.0)), 0.0)
    return min(account_equity, cap) if cap > 0 else account_equity


def _template_signal_map(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    return {
        str(row["symbol"]): {
            "signal": int(row.get("signal", 0)),
            "score": _as_float(row.get("score")),
            "price_score": _as_float(row.get("ret_N")),
            "social_score": 0.0,
            "volume_score": _as_float(row.get("volume_score")),
            "ret_N": _as_float(row.get("ret_N")),
            "sma_long": _as_float(row.get("sma_long")),
        }
        for row in rows
    }


def run_once(account_id: str | None = None) -> None:
    controls = load_controls()
    strategy = str(controls.get("active_strategy") or "momentum_social").lower()
    if strategy == "none":
        logger.warning("No algorithm strategy is selected. Exiting without sending orders.")
        return

    config = (
        get_config(account_id=account_id, strategy_id=strategy)
        if account_id
        else get_config(strategy_id=strategy)
    )
    configure_logging()

    logger.info(
        "Starting live runner for account %s with trading endpoint %s",
        config.account_id,
        config.alpaca_base_url,
    )

    if config.kill_switch:
        logger.warning("Kill switch is enabled. Exiting without sending orders.")
        return

    if not controls["algorithm_enabled"]:
        logger.warning("Algorithm trading is disabled in dashboard controls. Exiting without sending orders.")
        return
    logger.info("Active strategy: %s", STRATEGY_LABELS.get(strategy, strategy))

    trading_client = create_trading_client(config)
    if not is_market_open(trading_client):
        logger.warning("Market is closed according to Alpaca clock. Exiting without sending orders.")
        return

    data_client = create_data_client(config)

    account_equity = get_account_equity(trading_client)
    equity = _sizing_equity(config, account_equity)
    if equity < account_equity:
        logger.info("Algorithm sizing equity capped at %.2f from account equity %.2f", equity, account_equity)
    current_positions = get_positions(trading_client)

    defensive_config = DefensiveMomentumConfig.from_runtime_config(config)
    if strategy == "fast_momentum":
        price_symbols = sorted(set(defensive_config.symbols) | set(current_positions))
    else:
        price_symbols = sorted(set(config.symbols) | set(current_positions))
    latest_quotes = fetch_latest_market_quotes(price_symbols, config, data_client=data_client)
    latest_prices = {}
    for symbol in price_symbols:
        quote = latest_quotes.get(symbol)
        latest_prices[symbol] = (
            float(quote["price"])
            if quote and quote.get("price")
            else get_latest_price(symbol, data_client, data_feed=config.alpaca_data_feed)
        )

    if strategy == "fast_momentum":
        if "paper-api.alpaca.markets" not in str(config.alpaca_base_url):
            logger.warning(
                "Fast Momentum is restricted to Alpaca paper trading by default. "
                "Configured endpoint %s is not paper; exiting without orders.",
                config.alpaca_base_url,
            )
            return
        target_weights, signals, metadata = build_defensive_momentum_targets(
            config,
            data_client,
            current_positions,
            latest_prices,
            equity,
        )
        logger.info("Fast Momentum metadata: %s", metadata)
        log_signals(signals, latest_prices)
        log_portfolio(target_weights, equity)
        order_results = sync_positions_to_targets(
            trading_client,
            latest_prices,
            current_positions,
            target_weights,
            equity,
            cash_buffer=0.0,
            min_trade_dollars=defensive_config.per_trade_value_min,
            rebalance_threshold=0.0,
        )
        log_orders(order_results)
        log_position_changes(order_results)
        return

    bars_by_symbol = fetch_daily_bars(
        config.symbols,
        config.momentum_lookback_days,
        ma_days=config.long_ma_days,
        extra_buffer_days=config.history_extra_buffer_days,
        alpaca_data_client=data_client,
        data_feed=config.alpaca_data_feed,
        include_latest=True,
        config=config,
    )
    if strategy == "momentum_social":
        social_by_symbol = merge_social_frames(
            load_social_trends_csv(config.social_trends_csv, config.symbols),
            news_records_to_social_frames(fetch_latest_news_sentiment(config.symbols, config)),
        )
        signals = compute_signals_for_universe(
            bars_by_symbol,
            config.momentum_lookback_days,
            config.long_ma_days,
            short_lookback_days=config.short_momentum_lookback_days,
            volume_lookback_days=config.volume_lookback_days,
            social_by_symbol=social_by_symbol,
            social_lookback_days=config.social_lookback_days,
            price_momentum_weight=config.price_momentum_weight,
            social_momentum_weight=config.social_momentum_weight,
            volume_momentum_weight=config.volume_momentum_weight,
            min_composite_score=config.min_composite_score,
        )
        target_weights = compute_target_weights(
            signals,
            config.max_weight_per_symbol,
            max_portfolio_exposure=_max_live_exposure(config),
            max_longs=config.max_longs,
            target_annual_vol=config.target_annual_vol,
        )
    else:
        rows = strategy_signal_rows(strategy, bars_by_symbol)
        signals = _template_signal_map(rows)
        target_weights = weights_from_strategy_rows(
            rows,
            config.symbols,
            max_longs=config.max_longs,
            max_weight_per_symbol=config.max_weight_per_symbol,
            max_portfolio_exposure=_max_live_exposure(config),
        )
    log_signals(signals, latest_prices)
    log_portfolio(target_weights, equity)
    order_results = sync_positions_to_targets(
        trading_client,
        latest_prices,
        current_positions,
        target_weights,
        equity,
        cash_buffer=config.cash_buffer,
        min_trade_dollars=config.min_trade_dollars,
        rebalance_threshold=config.rebalance_threshold,
    )
    log_orders(order_results)
    log_position_changes(order_results)


def main() -> None:
    run_once()


if __name__ == "__main__":
    main()
