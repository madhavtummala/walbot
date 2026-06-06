from __future__ import annotations
import logging
from ..algorithms.registry import get_algorithm_class
from ..core.config import get_config
from ..connectors import fetch_latest_market_quotes, fetch_latest_news_sentiment, merge_social_frames, news_records_to_social_frames
from ..api.controls import load_controls
from ..brokerages.alpaca_client import (
    create_data_client,
    create_trading_client,
    get_account_equity,
    get_positions,
    get_latest_price,
    is_market_open,
)
from ..data import fetch_daily_bars
from ..common.logging_utils import configure_logging, log_signals, log_portfolio, log_orders, log_position_changes
from ..core.interfaces import AlgorithmContext
from ..core.orders import sync_positions_to_targets
from ..data.social import load_social_trends_csv
from ..core.strategy_models import STRATEGY_LABELS

logger = logging.getLogger(__name__)


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sizing_equity(config, account_equity: float) -> float:
    cap = max(_as_float(getattr(config, "algorithm_equity_cap", 0.0)), 0.0)
    return min(account_equity, cap) if cap > 0 else account_equity


def _load_sentiment_frames(config, symbols: list[str]):
    return merge_social_frames(
        load_social_trends_csv(config.social_trends_csv, symbols),
        news_records_to_social_frames(fetch_latest_news_sentiment(symbols, config)),
    )


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

    algorithm = get_algorithm_class(strategy).from_config(config)
    requirements = algorithm.requirements(config, current_positions)
    if requirements.paper_only and "paper-api.alpaca.markets" not in str(config.alpaca_base_url):
        logger.warning(
            "%s is restricted to Alpaca paper trading by default. Configured endpoint %s is not paper; exiting without orders.",
            STRATEGY_LABELS.get(strategy, strategy),
            config.alpaca_base_url,
        )
        return

    price_symbols = sorted(set(requirements.price_symbols or config.symbols) | set(current_positions))
    latest_quotes = fetch_latest_market_quotes(price_symbols, config, data_client=data_client)
    latest_prices = {}
    for symbol in price_symbols:
        quote = latest_quotes.get(symbol)
        latest_prices[symbol] = (
            float(quote["price"])
            if quote and quote.get("price")
            else get_latest_price(symbol, data_client, data_feed=config.alpaca_data_feed)
        )

    bars_by_symbol = {}
    if requirements.daily_lookback_days:
        bars_by_symbol = fetch_daily_bars(
            config.symbols,
            requirements.daily_lookback_days,
            ma_days=requirements.daily_ma_days,
            extra_buffer_days=requirements.daily_extra_buffer_days,
            alpaca_data_client=data_client,
            data_feed=config.alpaca_data_feed,
            include_latest=requirements.include_latest_daily,
            config=config,
        )
    sentiment_by_symbol = _load_sentiment_frames(config, config.symbols) if requirements.needs_sentiment else {}
    decision = algorithm.decide(
        AlgorithmContext(
            config=config,
            bars_by_symbol=bars_by_symbol,
            sentiment_by_symbol=sentiment_by_symbol,
            positions=current_positions,
            latest_prices=latest_prices,
            equity=equity,
            account_id=config.account_id,
            extra={"data_client": data_client},
        )
    )
    logger.info("%s metadata: %s", STRATEGY_LABELS.get(strategy, strategy), decision.metadata)
    log_signals(decision.signals, latest_prices)
    log_portfolio(decision.target_weights, equity)
    order_results = sync_positions_to_targets(
        trading_client,
        latest_prices,
        current_positions,
        decision.target_weights,
        equity,
        cash_buffer=decision.cash_buffer,
        min_trade_dollars=decision.min_trade_dollars,
        rebalance_threshold=decision.rebalance_threshold,
        require_approval=config.require_trade_approval,
        approval_timeout_seconds=config.trade_approval_timeout_seconds,
        approval_poll_seconds=config.trade_approval_poll_seconds,
    )
    log_orders(order_results)
    log_position_changes(order_results)


def main() -> None:
    run_once()


if __name__ == "__main__":
    main()
