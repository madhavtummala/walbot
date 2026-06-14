from __future__ import annotations
import logging
from math import ceil, floor
from uuid import uuid4

from src.core.interfaces import Brokerage, OrderRequest
from src.notifications.service import request_trade_approval

logger = logging.getLogger(__name__)


def _shares_for_target_weight(investable_equity: float, target_weight: float, price: float) -> int:
    target_dollar = investable_equity * target_weight
    raw_shares = target_dollar / price
    return floor(raw_shares) if raw_shares >= 0 else ceil(raw_shares)


def plan_position_orders(
    latest_prices: dict[str, float],
    current_positions: dict[str, int],
    target_weights: dict[str, float],
    equity: float,
    cash_buffer: float = 0.02,
    min_trade_dollars: float = 50.0,
    rebalance_threshold: float = 0.02,
) -> list[dict[str, str | int | float]]:
    """Build ordered market orders needed to move current positions toward target weights."""
    planned_orders: list[dict[str, str | int | float]] = []
    investable_equity = equity * max(0.0, min(1.0, 1.0 - cash_buffer))
    min_rebalance_dollars = max(min_trade_dollars, equity * max(rebalance_threshold, 0.0))

    symbols = sorted(set(target_weights) | set(current_positions))
    for symbol in symbols:
        target_weight = target_weights.get(symbol, 0.0)
        price = latest_prices.get(symbol)
        if price is None or price <= 0:
            logger.warning("Skipping %s because latest price is invalid: %s", symbol, price)
            continue

        target_shares = _shares_for_target_weight(investable_equity, target_weight, price)
        current_shares = current_positions.get(symbol, 0)
        diff = target_shares - current_shares
        trade_dollars = abs(diff) * price

        if diff == 0 or trade_dollars < min_rebalance_dollars:
            logger.info(
                "No trade required for %s: target_shares=%s current_shares=%s drift_dollars=%.2f",
                symbol,
                target_shares,
                current_shares,
                trade_dollars,
            )
            continue

        side = "buy" if diff > 0 else "sell"
        quantity = abs(diff)
        opens_or_increases_short = side == "sell" and target_shares < 0
        planned_orders.append(
            {
                "symbol": symbol,
                "action": side,
                "quantity": quantity,
                "target_weight": target_weight,
                "target_shares": target_shares,
                "current_shares": current_shares,
                "trade_dollars": trade_dollars,
                "latest_price": price,
                "position_intent": (
                    "sell_short"
                    if opens_or_increases_short
                    else "buy_to_cover"
                    if side == "buy" and current_shares < 0
                    else "sell_to_close"
                    if side == "sell"
                    else "buy_to_open"
                ),
                "opens_short": opens_or_increases_short,
                "short_shares_after": abs(target_shares) if target_shares < 0 else 0,
            }
        )

    return sorted(planned_orders, key=lambda order: 0 if order["action"] == "sell" else 1)


def _approval_skips(planned_orders: list[dict[str, str | int | float]], approval_id: str) -> list[dict[str, str | int | float]]:
    return [
        {
            **planned_order,
            "action": "skip",
            "quantity": 0,
            "approval_id": approval_id,
            "approval_status": "not_approved",
        }
        for planned_order in planned_orders
    ]


def sync_positions_to_targets(
    brokerage: Brokerage,
    latest_prices: dict[str, float],
    current_positions: dict[str, int],
    target_weights: dict[str, float],
    equity: float,
    cash_buffer: float = 0.02,
    min_trade_dollars: float = 50.0,
    rebalance_threshold: float = 0.02,
    require_approval: bool = False,
    approval_timeout_seconds: int = 300,
    approval_poll_seconds: int = 5,
) -> list[dict[str, str | int | float]]:
    """Send market orders to move current positions toward target weights."""
    planned_orders = plan_position_orders(
        latest_prices,
        current_positions,
        target_weights,
        equity,
        cash_buffer=cash_buffer,
        min_trade_dollars=min_trade_dollars,
        rebalance_threshold=rebalance_threshold,
    )
    if not planned_orders:
        return []

    if require_approval:
        approval_id = uuid4().hex[:10]
        approved = request_trade_approval(
            planned_orders,
            approval_id=approval_id,
            timeout_seconds=approval_timeout_seconds,
            poll_seconds=approval_poll_seconds,
        )
        if not approved:
            logger.warning("Trade approval %s was denied or timed out; skipping %s planned order(s)", approval_id, len(planned_orders))
            return _approval_skips(planned_orders, approval_id)

    order_results: list[dict[str, str | int | float]] = []
    for desired_order in planned_orders:
        symbol = str(desired_order["symbol"])
        side = str(desired_order["action"])
        quantity = int(desired_order["quantity"])
        if bool(desired_order.get("opens_short")):
            feasibility = brokerage.validate_short_sale_feasibility(
                symbol,
                quantity=quantity,
                target_shares=int(desired_order["target_shares"]),
                latest_price=float(desired_order["latest_price"]),
            )
            if not feasibility["shortable"]:
                logger.warning("Skipping short sale for %s: %s", symbol, feasibility["reason"])
                order_results.append(
                    {
                        **desired_order,
                        "action": "skip",
                        "quantity": 0,
                        "approval_status": "short_sale_not_feasible",
                        "reason": feasibility["reason"],
                    }
                )
                continue
        logger.info("Submitting %s order for %s qty=%s", side, symbol, quantity)
        result = brokerage.submit_order(
            OrderRequest(symbol=symbol, action=side, quantity=quantity)
        )
        order_results.append(
            {
                **desired_order,
                "symbol": symbol,
                "action": side,
                "quantity": quantity,
                "order_id": result.get("order_id", "unknown"),
            }
        )

    return order_results
