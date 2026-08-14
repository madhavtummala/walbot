"""A record of the orders this bot placed, tagged with the algorithm that placed them.

The brokerage cannot answer "what did Fast Momentum do this week": it reports one blended
stream per account, and a hand-placed order looks exactly like an algorithm's. So the bot
writes its own line as it submits, which is the only place that attribution exists.

Kept in the same key/value state store as the other runtime state, capped at a few hundred
entries -- this is the "what just happened" panel, not an audit ledger.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from .state_store import load_state, save_state

JOURNAL_KEY = "algorithm_order_journal"
JOURNAL_LIMIT = 400


def _entry(strategy: str, account_id: str, order: dict[str, Any], recorded_at: str) -> dict[str, Any]:
    action = str(order.get("action") or "")
    status = str(order.get("status") or "")
    if not status:
        # A skipped short sale never reaches the brokerage, so it carries no status of its own.
        status = "skipped" if action == "skip" else "planned"
    quantity = order.get("quantity")
    return {
        "submitted_at": recorded_at,
        "strategy": strategy,
        "account_id": account_id,
        "symbol": str(order.get("symbol") or ""),
        "side": action,
        "quantity": float(quantity) if isinstance(quantity, (int, float)) else 0.0,
        "notional": float(order.get("notional") or 0.0),
        "price": float(order.get("latest_price") or 0.0),
        "status": status,
        "order_id": str(order.get("order_id") or ""),
        "reason": str(order.get("reason") or ""),
    }


def record_orders(
    strategy: str,
    account_id: str,
    order_results: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append one run's order results to the journal and return the entries written.

    Never raises: a failure to journal must not take down a run that already placed real
    orders, so callers can treat this as fire and forget.
    """
    orders = [order for order in (order_results or []) if isinstance(order, dict)]
    if not orders:
        return []
    recorded_at = datetime.now(timezone.utc).isoformat()
    entries = [_entry(str(strategy), str(account_id), order, recorded_at) for order in orders]
    try:
        journal = load_state(JOURNAL_KEY, [])
        if not isinstance(journal, list):
            journal = []
        save_state(JOURNAL_KEY, (journal + entries)[-JOURNAL_LIMIT:])
    except Exception:  # noqa: BLE001 - journalling is never worth failing a live run over
        return entries
    return entries


def load_order_journal(
    strategy: str = "",
    account_id: str = "",
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Most recent entries first, optionally narrowed to one algorithm or account."""
    journal = load_state(JOURNAL_KEY, [])
    if not isinstance(journal, list):
        return []
    rows = [row for row in journal if isinstance(row, dict)]
    if strategy:
        rows = [row for row in rows if str(row.get("strategy") or "") == strategy]
    if account_id:
        rows = [row for row in rows if str(row.get("account_id") or "") == account_id]
    return list(reversed(rows))[: max(1, int(limit))]
