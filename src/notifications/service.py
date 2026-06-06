from __future__ import annotations

import logging
from typing import Any

from src.core.config import load_connectors_config

from .base import NotificationMessage
from .registry import get_notification_connector_class

logger = logging.getLogger(__name__)


def notification_provider_configs(raw_config: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    raw = raw_config if raw_config is not None else load_connectors_config()
    connectors = raw.get("connectors", raw) if isinstance(raw, dict) else {}
    notifications = connectors.get("notifications", {}) if isinstance(connectors, dict) else {}
    providers = notifications.get("providers", {}) if isinstance(notifications, dict) else {}
    return {
        str(name).strip().lower(): config if isinstance(config, dict) else {}
        for name, config in providers.items()
    } if isinstance(providers, dict) else {}


def _submitted_orders(order_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        order
        for order in order_results
        if int(order.get("quantity") or 0) > 0
        and str(order.get("action") or "").lower() not in {"hold", "skip"}
        and order.get("order_id")
    ]


def _format_order_line(order: dict[str, Any]) -> str:
    details = [
        str(order.get("action") or "").upper(),
        str(order.get("symbol") or ""),
        f"qty={order.get('quantity')}",
    ]
    if order.get("underlying"):
        details.append(f"underlying={order.get('underlying')}")
    if order.get("target_weight") is not None:
        details.append(f"target_weight={float(order.get('target_weight') or 0.0) * 100:.2f}%")
    if order.get("current_shares") is not None and order.get("target_shares") is not None:
        details.append(f"current={order.get('current_shares')}")
        details.append(f"target={order.get('target_shares')}")
    if order.get("trade_dollars") is not None:
        details.append(f"trade_dollars={float(order.get('trade_dollars') or 0.0):.2f}")
    if order.get("notional") is not None:
        details.append(f"notional={float(order.get('notional') or 0.0):.2f}")
    if order.get("limit_price") is not None:
        details.append(f"limit={float(order.get('limit_price') or 0.0):.2f}")
    if order.get("estimated_premium") is not None:
        details.append(f"premium={float(order.get('estimated_premium') or 0.0):.2f}")
    details.append(f"order_id={order.get('order_id')}")
    return " ".join(details)


def format_portfolio_change_message(order_results: list[dict[str, Any]]) -> str | None:
    submitted_orders = _submitted_orders(order_results)
    if not submitted_orders:
        return None

    noun = "order" if len(submitted_orders) == 1 else "orders"
    lines = [
        "Walbot 🤖 💰 💸",
        f"Portfolio changes submitted: {len(submitted_orders)} {noun}",
    ]
    lines.extend(_format_order_line(order) for order in submitted_orders)
    return "\n".join(lines)


def format_trade_approval_message(planned_orders: list[dict[str, Any]], approval_id: str) -> str:
    noun = "order" if len(planned_orders) == 1 else "orders"
    total_dollars = sum(float(order.get("trade_dollars") or order.get("notional") or 0.0) for order in planned_orders)
    lines = [
        "Walbot approval requested",
        f"Approval ID: {approval_id}",
        f"Planned: {len(planned_orders)} {noun}, ${total_dollars:.2f}",
        "Reply with:",
        f"/approve {approval_id}",
        f"/deny {approval_id}",
    ]
    lines.extend(_format_order_line({**order, "order_id": "pending"}) for order in planned_orders)
    return "\n".join(lines)


def send_notification(message: NotificationMessage, *, provider: str | None = None) -> bool:
    provider_configs = notification_provider_configs()
    if provider:
        provider_configs = {
            str(provider).strip().lower(): provider_configs.get(str(provider).strip().lower(), {})
        }
    sent_any = False
    for provider_name, provider_config in provider_configs.items():
        try:
            connector = get_notification_connector_class(provider_name)(provider_config)
        except KeyError:
            logger.warning("Unsupported notification provider %s", provider_name)
            continue
        sent_any = connector.send(message) or sent_any
    return sent_any


def send_telegram_message(text: str) -> bool:
    return send_notification(NotificationMessage(text=text), provider="telegram")


def request_trade_approval(
    planned_orders: list[dict[str, Any]],
    *,
    approval_id: str,
    timeout_seconds: int = 300,
    poll_seconds: int = 5,
) -> bool:
    provider_config = notification_provider_configs().get("telegram", {})
    try:
        connector = get_notification_connector_class("telegram")(provider_config)
    except KeyError:
        return False
    if not hasattr(connector, "request_approval"):
        return False
    message = format_trade_approval_message(planned_orders, approval_id)
    return bool(
        connector.request_approval(
            NotificationMessage(
                text=message,
                subject="Trade approval requested",
                metadata={"event_type": "trade_approval", "approval_id": approval_id, "orders": planned_orders},
            ),
            approval_id=approval_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )
    )


def notify_portfolio_changes(order_results: list[dict[str, Any]]) -> bool:
    message = format_portfolio_change_message(order_results)
    if not message:
        return False
    return send_notification(
        NotificationMessage(
            text=message,
            subject="Portfolio changes submitted",
            metadata={"event_type": "portfolio_changes", "orders": _submitted_orders(order_results)},
        )
    )
