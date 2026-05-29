from __future__ import annotations

import json
import logging
import os
from typing import Any
from urllib import parse, request

from .config import load_connectors_config

logger = logging.getLogger(__name__)


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _notification_timeout() -> float:
    try:
        return max(float(os.getenv("TELEGRAM_NOTIFICATION_TIMEOUT_SECONDS", "5")), 0.1)
    except ValueError:
        return 5.0


def _env_ref(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], default)
    return str(value)


def _direct_or_env(section: dict[str, Any], key: str, env_key: str, fallback_env: str = "") -> str:
    if section.get(key):
        return _env_ref(section.get(key), "")
    env_name = str(section.get(env_key) or "").strip()
    if env_name:
        return os.getenv(env_name, "")
    return os.getenv(fallback_env, "") if fallback_env else ""


def _telegram_connector_config() -> dict[str, Any]:
    raw = load_connectors_config()
    connectors = raw.get("connectors", raw) if isinstance(raw, dict) else {}
    notifications = connectors.get("notifications", {}) if isinstance(connectors, dict) else {}
    providers = notifications.get("providers", {}) if isinstance(notifications, dict) else {}
    telegram = providers.get("telegram", {}) if isinstance(providers, dict) else {}
    return telegram if isinstance(telegram, dict) else {}


def _telegram_settings() -> dict[str, Any]:
    connector = _telegram_connector_config()
    return {
        "enabled": bool(connector.get("enabled", True)) and _env_enabled("TELEGRAM_NOTIFICATIONS_ENABLED"),
        "bot_token": _direct_or_env(connector, "bot_token", "bot_token_env", "TELEGRAM_BOT_TOKEN"),
        "chat_id": _direct_or_env(connector, "chat_id", "chat_id_env", "TELEGRAM_CHAT_ID"),
        "api_root": str(connector.get("api_root") or os.getenv("TELEGRAM_API_ROOT", "https://api.telegram.org")),
        "timeout_seconds": connector.get("timeout_seconds"),
    }


def _telegram_timeout(settings: dict[str, Any]) -> float:
    raw_timeout = settings.get("timeout_seconds")
    if raw_timeout is None or raw_timeout == "":
        return _notification_timeout()
    try:
        return max(float(raw_timeout), 0.1)
    except ValueError:
        return 5.0


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
        "Trading Bot 🤖 💰 💸",
        f"Portfolio changes submitted: {len(submitted_orders)} {noun}",
    ]
    lines.extend(_format_order_line(order) for order in submitted_orders)
    return "\n".join(lines)


def send_telegram_message(text: str) -> bool:
    settings = _telegram_settings()
    token = str(settings["bot_token"]).strip()
    chat_id = str(settings["chat_id"]).strip()
    if not token or not chat_id or not settings["enabled"]:
        return False

    api_root = str(settings["api_root"]).strip().rstrip("/")
    timeout = _telegram_timeout(settings)
    url = f"{api_root}/bot{token}/sendMessage"
    payload = parse.urlencode({"chat_id": chat_id, "text": text}).encode("utf-8")
    telegram_request = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with request.urlopen(telegram_request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except Exception as exc:
        logger.warning("Unable to send Telegram portfolio notification: %s", exc)
        return False

    try:
        result = json.loads(body)
    except json.JSONDecodeError:
        return True
    if not result.get("ok", True):
        logger.warning("Telegram portfolio notification failed: %s", result.get("description") or result)
        return False
    return True


def notify_portfolio_changes(order_results: list[dict[str, Any]]) -> bool:
    message = format_portfolio_change_message(order_results)
    if not message:
        return False
    return send_telegram_message(message)
