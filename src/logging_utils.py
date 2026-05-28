from __future__ import annotations
import logging
from typing import Any


def configure_logging() -> None:
    """Configure root logging to the console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = [console_handler]


def log_position_changes(order_results: list[dict[str, Any]]) -> None:
    logger = logging.getLogger(__name__)
    submitted_orders = [
        order
        for order in order_results
        if int(order.get("quantity") or 0) > 0
        and str(order.get("action") or "").lower() not in {"hold", "skip"}
        and order.get("order_id")
    ]
    if not submitted_orders:
        logger.warning("No position changes were submitted.")
        return

    logger.warning("Position changes submitted: %s order(s)", len(submitted_orders))
    for order in submitted_orders:
        details = [
            f"{str(order.get('action')).upper()}",
            str(order.get("symbol")),
            f"qty={order.get('quantity')}",
        ]
        if order.get("current_shares") is not None and order.get("target_shares") is not None:
            details.append(f"current={order.get('current_shares')}")
            details.append(f"target={order.get('target_shares')}")
        if order.get("trade_dollars") is not None:
            details.append(f"trade_dollars={float(order.get('trade_dollars') or 0.0):.2f}")
        if order.get("limit_price") is not None:
            details.append(f"limit={float(order.get('limit_price') or 0.0):.2f}")
        logger.warning("  %s order_id=%s", " ".join(details), order.get("order_id"))


def log_signals(signals: dict[str, dict[str, float | int]], prices: dict[str, float]) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Computed signals for universe")
    for symbol, info in signals.items():
        price = prices.get(symbol, float("nan"))
        logger.info(
            "%s signal=%s score=%.3f price_score=%.3f social=%.3f volume=%.3f ret_N=%s close=%.2f sma_long=%.2f",
            symbol,
            info.get("signal"),
            info.get("score", 0.0),
            info.get("price_score", 0.0),
            info.get("social_score", 0.0),
            info.get("volume_score", 0.0),
            info.get("ret_N"),
            price,
            info.get("sma_long"),
        )


def log_portfolio(target_weights: dict[str, float], equity: float) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Target portfolio weights computed")
    logger.info("Account equity: %.2f", equity)
    for symbol, weight in target_weights.items():
        logger.info("  %s -> %.2f%%", symbol, weight * 100)


def log_orders(order_results: list[dict[str, str | int | float]]) -> None:
    logger = logging.getLogger(__name__)
    if not order_results:
        logger.info("No orders were generated.")
        return

    logger.info("Order summary")
    for order in order_results:
        logger.info(
            "  %s %s qty=%s target=%s current=%s order_id=%s",
            order.get("action"),
            order.get("symbol"),
            order.get("quantity"),
            order.get("target_shares"),
            order.get("current_shares"),
            order.get("order_id", "n/a"),
        )
