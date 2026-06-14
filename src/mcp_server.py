from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.algorithms.registry import get_algorithm_class
from src.api.api_payloads import strategy_signals_payload
from src.brokerages import BROKERAGE_REGISTRY
from src.brokerages.alpaca_client import create_data_client, get_latest_price
from src.connectors import (
    fetch_latest_market_quotes,
    fetch_latest_news_sentiment,
    merge_social_frames,
    news_records_to_social_frames,
)
from src.core.config import get_config, get_account_broker_type
from src.core.interfaces import AlgorithmContext, AlgorithmDecision, OrderRequest
from src.core.orders import plan_position_orders
from src.data import fetch_daily_bars
from src.data.social import load_social_trends_csv

logger = logging.getLogger(__name__)


def _algorithm_pipeline(algorithm: str) -> dict[str, Any]:
    """Run the algorithm pipeline: config -> brokerage -> data -> decide -> plan orders.

    Returns a result dict with keys: decision, latest_prices, current_positions, equity,
    config, brokerage, planned_orders, data_client.
    On error returns a dict with ``status`` set to ``"error"``.
    """
    config = get_config(strategy_id=algorithm)

    broker_type = get_account_broker_type(config.account_id)
    broker_cls = BROKERAGE_REGISTRY.get(broker_type)
    if broker_cls is None:
        return {"algorithm": algorithm, "status": "error", "reason": f"Unknown brokerage: {broker_type}"}
    brokerage = broker_cls(config)

    account_state = brokerage.get_account_state()
    equity = float(account_state.get("equity", 0.0))
    cap = max(float(getattr(config, "algorithm_equity_cap", 0.0) or 0), 0.0)
    equity = min(equity, cap) if cap > 0 else equity
    current_positions = brokerage.get_positions()
    data_client = create_data_client(config)

    algorithm_instance = get_algorithm_class(algorithm).from_config(config)
    requirements = algorithm_instance.requirements(config, current_positions)

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

    sentiment_by_symbol = {}
    if requirements.needs_sentiment:
        sentiment_by_symbol = merge_social_frames(
            load_social_trends_csv(config.social_trends_csv, config.symbols),
            news_records_to_social_frames(fetch_latest_news_sentiment(config.symbols, config)),
        )

    decision = algorithm_instance.decide(
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

    planned_orders = plan_position_orders(
        latest_prices,
        current_positions,
        decision.target_weights,
        equity,
        cash_buffer=decision.cash_buffer,
        min_trade_dollars=decision.min_trade_dollars,
        rebalance_threshold=decision.rebalance_threshold,
    )

    return {
        "algorithm": algorithm,
        "config": config,
        "brokerage": brokerage,
        "equity": equity,
        "current_positions": current_positions,
        "latest_prices": latest_prices,
        "decision": decision,
        "planned_orders": planned_orders,
        "data_client": data_client,
    }


def _server(name: str, host: str, port: int):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised in container builds with mcp installed.
        raise RuntimeError("The MCP runtime is not installed. Install the 'mcp' package to use --mcp mode.") from exc

    try:
        return FastMCP(name, host=host, port=port)
    except TypeError:
        return FastMCP(name)


def create_mcp_server(host: str = "0.0.0.0", port: int = 8001):
    mcp = _server("walbot", host, port)

    @mcp.tool()
    def get_live_signals(algorithm: str = "fast_momentum") -> dict[str, Any]:
        """Return active positions from the live strategy-signal payload (filters out inactive symbols)."""
        data = strategy_signals_payload(strategy=algorithm)
        active = [row for row in data.get("leaders", []) if row.get("signal") == "LONG"]
        data["leaders"] = active
        return data

    @mcp.tool()
    def get_portfolio_preview(algorithm: str = "fast_momentum") -> dict[str, Any]:
        """Preview which positions the algorithm would keep, add, or drop given current holdings."""
        result = _algorithm_pipeline(algorithm)
        if result.get("status") == "error":
            return result

        decision: AlgorithmDecision = result["decision"]
        latest_prices = result["latest_prices"]
        current_positions = result["current_positions"]
        equity = result["equity"]
        config = result["config"]

        current_weights: dict[str, float] = {}
        for symbol, shares in current_positions.items():
            price = latest_prices.get(symbol, 0.0)
            if price > 0:
                current_weights[symbol] = round((shares * price) / max(equity, 1.0), 4)

        target_weights = decision.target_weights
        target_set = {s for s, w in target_weights.items() if w > 0}
        current_held = set(current_weights)

        all_symbols = sorted(set(config.symbols) | current_held | set(target_weights))
        positions_detail = []
        for symbol in all_symbols:
            cw = current_weights.get(symbol, 0.0)
            tw = round(target_weights.get(symbol, 0.0), 4)
            in_current = symbol in current_held
            in_target = symbol in target_set

            if in_current and in_target:
                status, reason = "retained", "Retained"
            elif in_current and not in_target:
                status, reason = "dropped", "Dropped"
            elif not in_current and in_target:
                status, reason = "new", "New"
            else:
                status, reason = "none", "None"

            positions_detail.append(
                {"symbol": symbol, "current_weight": cw, "target_weight": tw, "status": status, "reason": reason}
            )

        return {
            "strategy": algorithm,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "equity": equity,
            "current_weights": {s: w for s, w in sorted(current_weights.items()) if w > 0},
            "target_weights": {s: w for s, w in sorted(target_weights.items()) if w != 0},
            "positions": positions_detail,
        }

    @mcp.tool()
    def get_planned_orders(algorithm: str = "fast_momentum") -> dict[str, Any]:
        """Return the buy/sell orders the algorithm would submit to reach its target weights."""
        result = _algorithm_pipeline(algorithm)
        if result.get("status") == "error":
            return result

        return {
            "strategy": algorithm,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "equity": result["equity"],
            "planned_orders": result["planned_orders"],
        }

    @mcp.tool()
    def place_orders(algorithm: str = "fast_momentum") -> dict[str, Any]:
        """Run the algorithm's decision pipeline and submit portfolio rebalance orders directly (no additional approval)."""
        result = _algorithm_pipeline(algorithm)
        if result.get("status") == "error":
            return result

        if result["config"].kill_switch:
            logger.warning("Kill switch is enabled. Skipping order placement for %s.", algorithm)
            return {"algorithm": algorithm, "status": "skipped", "reason": "Kill switch is enabled"}

        brokerage = result["brokerage"]
        planned_orders = result["planned_orders"]
        decision: AlgorithmDecision = result["decision"]

        order_results: list[dict[str, Any]] = []
        for planned_order in planned_orders:
            symbol = str(planned_order["symbol"])
            side = str(planned_order["action"])
            quantity = int(planned_order["quantity"])
            logger.info("Submitting %s order for %s qty=%s via %s", side, symbol, quantity, get_account_broker_type(result["config"].account_id))
            submit_result = brokerage.submit_order(
                OrderRequest(
                    symbol=symbol,
                    action=side,
                    quantity=quantity,
                )
            )
            order_results.append(
                {
                    **planned_order,
                    "order_id": submit_result.get("order_id", "unknown"),
                }
            )

        return {
            "algorithm": algorithm,
            "equity": result["equity"],
            "target_weights": dict(decision.target_weights),
            "order_results": order_results,
        }

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Walbot MCP tools.")
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", default=int(os.getenv("MCP_PORT", "8001")), type=int)
    parser.add_argument("--transport", default=os.getenv("MCP_TRANSPORT", "sse"))
    args = parser.parse_args()

    mcp = create_mcp_server(host=args.host, port=args.port)
    try:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    except TypeError:
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
