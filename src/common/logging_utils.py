from __future__ import annotations
import logging
import os
from typing import Any



class _UvicornAccessDebugFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO:
            return True

        access_logger = logging.getLogger("uvicorn.access")
        if access_logger.getEffectiveLevel() > logging.DEBUG:
            return False

        record.levelno = logging.DEBUG
        record.levelname = logging.getLevelName(logging.DEBUG)
        return True


def demote_uvicorn_access_logs_to_debug() -> None:
    """Show Uvicorn access logs only when debug logging is enabled."""
    access_logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(log_filter, _UvicornAccessDebugFilter) for log_filter in access_logger.filters):
        access_logger.addFilter(_UvicornAccessDebugFilter())


def _resolve_log_level(level: int | str | None = None) -> int:
    if isinstance(level, int):
        return level
    raw_level = str(level or os.getenv("TRADING_LOG_LEVEL", "WARNING")).strip().upper()
    if raw_level == "WARN":
        raw_level = "WARNING"
    resolved = logging.getLevelName(raw_level)
    return resolved if isinstance(resolved, int) else logging.WARNING


def configure_logging(level: int | str | None = None) -> None:
    """Configure root logging to the console."""
    resolved_level = _resolve_log_level(level)
    logger = logging.getLogger()
    logger.setLevel(resolved_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(resolved_level)
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
        if order.get("target_weight") is not None:
            details.append(f"target_weight={float(order.get('target_weight') or 0.0) * 100:.2f}%")
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
