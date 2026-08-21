"""Account state: balances, holdings, income and broker activity.

Split out of the single ``api_payloads`` module, which had grown to 1253 lines covering nine
unrelated domains. The public names are unchanged and still importable from ``api_payloads``.
"""


from __future__ import annotations

from ...brokerages.alpaca_client import create_trading_client
from src.api.controls import load_controls

import os
import logging
from typing import Any


from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from ...brokerages.providers.paper import PaperBrokerage
from ...core.config import (
    UnknownAccountError,
    get_account_broker_type,
    get_config,
    load_accounts_config,
    save_accounts_config,
)
from ...common.config_utils import json_number
from ...data.order_journal import load_order_journal

logger = logging.getLogger(__name__)


#: How far back the account page totals income. Just under a year: comparable to a trailing
#: yield, small enough to stay one request, and inside Schwab's transactions window, which
#: rejects a range of exactly 365 days as "more than a year".
DIVIDEND_ACTIVITY_DAYS = 364

#: Fields a deployment target carries. Secrets are deliberately absent: accounts reference the
#: *names* of environment variables, so the dashboard can wire up a target without ever handling
#: an API key. Setting the secret stays a deploy-time action on the host.
ACCOUNT_FIELDS = ("label", "broker", "base_url", "data_feed", "api_key_env", "api_secret_env")


def positions_payload(account_id: str = "") -> dict[str, Any]:
    """Live holdings and P/L for one account.

    Per account rather than per algorithm: the broker reports a single blended position per
    symbol, so two bindings trading the same account cannot be told apart here.
    """
    try:
        config = get_config(account_id=account_id) if account_id else get_config()
    except UnknownAccountError as error:
        # Named but not configured. Reported as itself rather than served from another
        # account, which is the only honest answer and the one this used to get wrong.
        return {
            "account_id": account_id, "account_label": account_id, "equity": None,
            "cash": None, "day_pl": None, "day_pl_percent": None, "total_pl": None,
            "dividend_pl": None, "dividend_rows": [], "rows": [], "error": str(error),
        }
    payload: dict[str, Any] = {
        "account_id": config.account_id,
        "account_label": config.account_label,
        "equity": None,
        "cash": None,
        "day_pl": None,
        "day_pl_percent": None,
        "total_pl": None,
        # Income is reported beside price P/L rather than inside it. They answer different
        # questions -- what the holdings are worth versus what they paid out -- and a cash
        # sleeve earns almost entirely through the second one, which an "open P/L" figure
        # alone shows as flat.
        "dividend_pl": None,
        "dividend_rows": [],
        "rows": [],
        "error": "",
    }
    # Routed by broker. Only the Alpaca path below is Alpaca-specific; anything else asking
    # create_trading_client would silently report the *Alpaca* account's money under another
    # account's name, which is what a Schwab account used to show.
    broker = get_account_broker_type(config.account_id)
    if broker == "paper":
        return {**payload, **_paper_positions(config), **_dividend_pl(config)}
    if broker != "alpaca":
        return {**payload, **_brokerage_positions(config, broker), **_dividend_pl(config)}
    try:
        client = create_trading_client(config)
        account = client.get_account()
        equity = float(getattr(account, "equity", 0.0) or 0.0)
        last_equity = float(getattr(account, "last_equity", 0.0) or 0.0)
        payload["equity"] = equity
        payload["cash"] = float(getattr(account, "cash", 0.0) or 0.0)
        if last_equity:
            payload["day_pl"] = equity - last_equity
            payload["day_pl_percent"] = (equity - last_equity) / last_equity
        from ...brokerages.providers.alpaca import AlpacaBrokerage
        rows = AlpacaBrokerage(config).get_position_details()
        payload["rows"] = rows
        payload["total_pl"] = sum(float(row["unrealized_pl"]) for row in rows) if rows else 0.0
    except Exception as error:  # noqa: BLE001 - a broker outage must not blank the dashboard
        logger.warning("Could not load positions for %s: %s", config.account_id, error)
        payload["error"] = str(error)
    payload.update(_dividend_pl(config))
    return payload


#: How far back the account page totals income. Just under a year: comparable to a trailing
#: yield, small enough to stay one request, and inside Schwab's transactions window, which
#: rejects a range of exactly 365 days as "more than a year".
DIVIDEND_ACTIVITY_DAYS = 364


def _dividend_pl(config: Any) -> dict[str, Any]:
    """Income received, through the brokerage interface rather than per-broker branching.

    Kept apart from ``total_pl`` deliberately. A distribution is cash that arrived, not a
    change in what the holdings are worth, and folding the two together is what made a T-bill
    sleeve look like it earned nothing at all.
    """
    from datetime import datetime, timedelta, timezone

    from ...core.pipeline import resolve_brokerage

    try:
        brokerage = resolve_brokerage(config)
        end = datetime.now(timezone.utc).date()
        rows = brokerage.get_dividend_activity(end - timedelta(days=DIVIDEND_ACTIVITY_DAYS), end)
    except Exception as error:  # noqa: BLE001 - income is a detail, not the whole page
        logger.warning("Could not read dividend activity for %s: %s", config.account_id, error)
        return {"dividend_pl": None, "dividend_rows": []}
    return {
        "dividend_pl": float(sum(float(row.get("amount") or 0.0) for row in rows)),
        "dividend_rows": rows[:40],
    }


#: Fields a deployment target carries. Secrets are deliberately absent: accounts reference the
#: *names* of environment variables, so the dashboard can wire up a target without ever handling
#: an API key. Setting the secret stays a deploy-time action on the host.
ACCOUNT_FIELDS = ("label", "broker", "base_url", "data_feed", "api_key_env", "api_secret_env")


def _account_items(raw: dict[str, Any]) -> dict[str, Any]:
    accounts = raw.get("accounts") if isinstance(raw.get("accounts"), dict) else {}
    items = accounts.get("items") if isinstance(accounts.get("items"), dict) else {}
    return items


def accounts_payload() -> dict[str, Any]:
    """Every deployment target, with whether its credentials are actually present."""
    raw = load_accounts_config()
    items = _account_items(raw)
    controls = load_controls()
    deployed: dict[str, list[str]] = {}
    for binding in controls.get("bindings") or []:
        deployed.setdefault(str(binding.get("account_id") or ""), []).append(str(binding.get("strategy") or ""))

    rows = []
    for account_id, section in items.items():
        section = section if isinstance(section, dict) else {}
        key_env = str(section.get("api_key_env") or "")
        secret_env = str(section.get("api_secret_env") or "")
        missing = [name for name in (key_env, secret_env) if name and not os.getenv(name)]
        rows.append(
            {
                "id": str(account_id),
                "label": str(section.get("label") or account_id),
                "broker": str(section.get("broker") or "alpaca"),
                "base_url": str(section.get("base_url") or ""),
                "data_feed": str(section.get("data_feed") or ""),
                "api_key_env": key_env,
                "api_secret_env": secret_env,
                "credentials_ready": not missing,
                "missing_env": missing,
                "deployments": sorted(set(deployed.get(str(account_id), []))),
            }
        )
    rows.sort(key=lambda row: row["id"])
    return {"default": str(raw.get("default") or ""), "rows": rows}


def save_account_payload(body: dict[str, Any]) -> dict[str, Any]:
    account_id = str(body.get("id") or "").strip()[:80]
    if not account_id:
        raise ValueError("A deployment target needs an id.")
    if not account_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Target id may only contain letters, numbers, dashes, and underscores.")

    raw = load_accounts_config()
    accounts = raw.setdefault("accounts", {})
    if not isinstance(accounts, dict):
        accounts = {}
        raw["accounts"] = accounts
    items = accounts.setdefault("items", {})
    if not isinstance(items, dict):
        items = {}
        accounts["items"] = items

    section = items.get(account_id) if isinstance(items.get(account_id), dict) else {}
    for field_name in ACCOUNT_FIELDS:
        if field_name in body:
            section[field_name] = str(body.get(field_name) or "")
    section.setdefault("broker", "alpaca")
    items[account_id] = section
    if not raw.get("default"):
        raw["default"] = account_id
    save_accounts_config(raw)
    return accounts_payload()


def delete_account_payload(account_id: str) -> dict[str, Any]:
    account_id = str(account_id or "").strip()
    raw = load_accounts_config()
    items = _account_items(raw)
    if account_id not in items:
        raise ValueError(f"No deployment target named {account_id}.")
    if len(items) <= 1:
        raise ValueError("Keep at least one deployment target.")

    controls = load_controls()
    in_use = [b for b in (controls.get("bindings") or []) if str(b.get("account_id")) == account_id]
    if in_use:
        # Deleting a target out from under a running deployment would leave it pointed at an
        # account that no longer resolves, which fails at order time rather than here.
        raise ValueError(f"{account_id} still has {len(in_use)} deployment(s). Remove them first.")

    items.pop(account_id)
    if str(raw.get("default") or "") == account_id:
        raw["default"] = next(iter(items), "")
    save_accounts_config(raw)
    return accounts_payload()


def _paper_positions(config: Any) -> dict[str, Any]:
    """Holdings and P/L from the local paper book.

    ``day_pl`` stays None: the book has no notion of yesterday's close, and inventing one
    would put a number on the dashboard that nothing backs.
    """
    try:
        brokerage = PaperBrokerage(config)
        state = brokerage.get_account_state()
        rows = brokerage.get_position_details()
    except Exception as error:  # noqa: BLE001 - a corrupt book must not blank the page
        logger.warning("Could not read the paper book for %s: %s", config.account_id, error)
        return {"error": str(error)}
    return {
        "equity": float(state.get("equity") or 0.0),
        "cash": float(state.get("cash") or 0.0),
        "total_pl": sum(float(row["unrealized_pl"]) for row in rows) if rows else 0.0,
        "rows": rows,
    }


def _brokerage_positions(config: Any, broker: str) -> dict[str, Any]:
    """Holdings and cash for any non-Alpaca brokerage, through the shared interface."""
    from ...core.pipeline import resolve_brokerage

    try:
        brokerage = resolve_brokerage(config)
        state = brokerage.get_account_state()
        rows = brokerage.get_position_details()
    except Exception as error:  # noqa: BLE001 - an unreachable broker must not blank the page
        logger.warning("Could not read %s positions for %s: %s", broker, config.account_id, error)
        return {"error": str(error)}

    return {
        "equity": float(state.get("equity") or 0.0),
        "cash": float(state.get("cash") or 0.0),
        "total_pl": sum(float(row["unrealized_pl"]) for row in rows) if rows else None,
        "rows": rows,
    }


def _paper_activity(config: Any, limit: int) -> dict[str, Any]:
    """Order history for the local paper book, from the bot's own journal.

    The paper brokerage fills immediately and keeps no order log, so the journal written at
    submission is the whole record -- which is complete here, since nothing but this bot can
    trade a local book.
    """
    rows = []
    for entry in load_order_journal(account_id=config.account_id, limit=limit):
        rows.append(
            {
                "symbol": entry.get("symbol", ""),
                "side": entry.get("side", ""),
                "status": entry.get("status", ""),
                "qty": entry.get("quantity"),
                "filled_qty": entry.get("quantity") if entry.get("status") == "submitted" else 0.0,
                "filled_avg_price": entry.get("price") or None,
                "submitted_at": entry.get("submitted_at", ""),
            }
        )
    return {"rows": rows}


def account_activity_payload(account_id: str = "", limit: int = 40) -> dict[str, Any]:
    """Recent broker orders for one account.

    Read straight from the brokerage rather than a local mirror: the broker is the only source
    that knows about fills, partial fills, and cancels after submission. The local paper book
    is the exception -- there is no broker, so the bot's own journal is the record.
    """
    try:
        config = get_config(account_id=account_id) if account_id else get_config()
    except UnknownAccountError as error:
        return {"account_id": account_id, "rows": [], "error": str(error)}
    payload: dict[str, Any] = {"account_id": config.account_id, "rows": [], "error": ""}
    broker = get_account_broker_type(config.account_id)
    if broker == "paper":
        return {**payload, **_paper_activity(config, limit)}
    if broker != "alpaca":
        # Only the bot's own journal is available for a non-Alpaca broker here; its order feed
        # would need its own client, and reporting Alpaca's would name the wrong account.
        return {**payload, **_paper_activity(config, limit)}
    try:
        client = create_trading_client(config)
        try:
            orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit))
        except Exception:  # noqa: BLE001 - older SDKs reject the filter; fall back to open orders
            orders = client.get_orders()
        rows = []
        for order in orders:
            submitted = getattr(order, "submitted_at", None) or getattr(order, "created_at", None)
            rows.append(
                {
                    "symbol": str(getattr(order, "symbol", "")),
                    "side": _enum_value(getattr(order, "side", "")),
                    "status": _enum_value(getattr(order, "status", "")),
                    "qty": json_number(getattr(order, "qty", None)),
                    "filled_qty": json_number(getattr(order, "filled_qty", None)),
                    "filled_avg_price": json_number(getattr(order, "filled_avg_price", None)),
                    "submitted_at": submitted.isoformat() if hasattr(submitted, "isoformat") else str(submitted or ""),
                }
            )
        rows.sort(key=lambda row: row["submitted_at"], reverse=True)
        payload["rows"] = rows[:limit]
    except Exception as error:  # noqa: BLE001 - a broker outage must not blank the page
        logger.warning("Could not load activity for %s: %s", config.account_id, error)
        payload["error"] = str(error)
    return payload


#: Broker payload fields arrive as strings, and an unparseable one must serialise as null
#: rather than as a zero that reads like a real quantity.


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")
