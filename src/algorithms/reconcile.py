"""Making a broker's resting orders match what a run decided they should be.

Plain functions, not a base class: an algorithm whose output is an order book rather than a
target portfolio overrides ``execute`` and calls :func:`reconcile_orders` from there.

The broker is the source of truth, never our own memory -- every run reads the working orders
back before deciding anything, since a remembered book cannot survive a partial fill or a
manual cancellation. Reconciliation is idempotent, so a cron cadence can run it as often as it
likes. And order identity is a role (``DesiredOrder.key``), not a submission, so an order
re-priced repeatedly across a session stays one thing; broker order ids are recorded against
the key in algorithm state.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..core.interfaces import DesiredOrder

logger = logging.getLogger(__name__)

#: Where an algorithm records the broker order id standing behind each desired order's key.
ORDER_IDS_KEY = "order_ids"


def broker_supports_oco(account_id: str) -> bool:
    """Whether this account's broker will hold an OCO pair itself.

    Read off the class without instantiating it (and authenticating), since this is a property
    of the venue, not of a live session. Unknown accounts fall back to ``False``, the
    conservative reading.
    """
    from ..brokerages.registry import get_brokerage_class
    from ..core.config import get_account_broker_type

    try:
        return bool(get_brokerage_class(get_account_broker_type(account_id)).supports_oco)
    except Exception:
        return False


def reconcile_orders(
    desired: List[DesiredOrder],
    brokerage: Any,
    recorded: Dict[str, str],
    *,
    persist: Any = None,
) -> Dict[str, Any]:
    """Cancel what is no longer wanted, then place or re-price the rest.

    ``recorded`` maps each desired order's key to the broker order id last known to stand
    behind it. Returns ``{"results", "order_ids", "working"}``; the caller persists
    ``order_ids`` and reports ``results``.

    Cancels run first so a replacement is never submitted alongside the order it replaces.

    ``persist`` is called with the id map after every change to it. An order that exists at the
    broker but not in our state is the one condition this system cannot recover from -- so the
    write cannot be batched behind work that can still fail. Without it, a broker error on the
    third leg discards the ids of the two already submitted, and the next run, reading empty
    state, submits the whole book a second time.

    The invariant every branch below keeps: **an id is dropped only once the broker has
    confirmed the order is gone.** A working-order listing is a snapshot that can be stale or
    partial, so "absent from the listing" is never on its own taken as "no longer exists".
    """
    try:
        working = brokerage.get_orders("WORKING")
    except NotImplementedError as exc:
        # Refusing here is the safe direction: proceeding would resubmit the whole desired book
        # every run, since nothing would ever look already-present.
        raise NotImplementedError(
            f"{type(brokerage).__name__} cannot list working orders, which order reconciliation "
            "requires. An algorithm that rests orders needs a brokerage that holds them, so it "
            "cannot be backtested against the paper book."
        ) from exc

    working_by_id = {str(o.get("order_id")): o for o in working if o.get("order_id")}
    results: List[Dict[str, Any]] = []
    wanted = {order.key: order for order in desired}
    # Seeded from what was already recorded rather than built up from scratch, so a key this
    # pass never resolves is carried forward instead of being forgotten. Forgetting an id
    # orphans a live order: nothing tracks it, so no later run can ever cancel it.
    order_ids: Dict[str, str] = {key: order_id for key, order_id in recorded.items() if order_id}

    def commit() -> None:
        if persist is not None:
            persist(dict(order_ids))

    for key, order_id in list(recorded.items()):
        if key in wanted or not order_id:
            continue
        existing = working_by_id.get(order_id, {})
        # Cancelled even when the listing does not show it. ``cancel_order`` is contracted to
        # treat an already-gone order as a success, so the call is safe either way -- and it is
        # the only way to be sure a stale id is not still live at the broker.
        if not _cancel(brokerage, order_id, key):
            results.append(_result(
                key, "cancel_failed", order_id,
                symbol=str(existing.get("symbol", "")),
                action=str(existing.get("action", "")),
                status="unreconciled",
            ))
            continue
        order_ids.pop(key, None)
        commit()
        results.append(_result(
            key, "cancelled", order_id,
            symbol=str(existing.get("symbol", "")),
            action=str(existing.get("action", "")),
            quantity=float(existing.get("quantity", 0.0) or 0.0),
        ))

    for key, desired_order in wanted.items():
        existing_id = recorded.get(key, "")
        existing = working_by_id.get(existing_id) if existing_id else None
        if existing is None:
            if existing_id:
                # Recorded but not listed. Most often it filled or was cancelled, but a partial
                # listing looks identical -- and submitting beside an order that is still live
                # would leave two working orders for one role. Cancel first: a no-op if it is
                # genuinely gone, and the thing that prevents a duplicate if it is not.
                if not _cancel(brokerage, existing_id, key):
                    results.append(_result(
                        key, "cancel_failed", existing_id,
                        symbol=desired_order.request.symbol,
                        action=desired_order.request.action,
                        status="unreconciled",
                    ))
                    continue
                order_ids.pop(key, None)
                commit()
            results.append(_submit(key, desired_order, brokerage, order_ids))
            commit()
        elif _needs_replacement(desired_order, existing):
            results.append(_replace(key, desired_order, existing_id, brokerage, order_ids))
            commit()
        else:
            order_ids[key] = existing_id
            results.append(_result(
                key, "unchanged", existing_id,
                symbol=desired_order.request.symbol,
                action=desired_order.request.action,
                quantity=float(desired_order.request.quantity),
            ))

    commit()
    return {"results": results, "order_ids": order_ids, "working": working}


def _cancel(brokerage: Any, order_id: str, key: str) -> bool:
    """Cancel ``order_id``, reporting whether the broker confirmed it is gone.

    ``False`` means the id must stay recorded: the order may still be live, and an id we stop
    tracking is one no later run can cancel. Not letting the exception escape matters as much --
    it would abandon the rest of the book mid-pass, which is how orders get orphaned.
    """
    try:
        brokerage.cancel_order(order_id)
        return True
    except Exception as exc:  # noqa: BLE001 - one stuck cancel must not end the pass
        logger.warning("Could not cancel %s (order %s): %s; keeping it recorded", key, order_id, exc)
        return False


def _submit(key: str, desired: DesiredOrder, brokerage: Any, order_ids: Dict[str, str]) -> Dict[str, Any]:
    try:
        result = brokerage.submit_order(desired.request)
    except Exception as exc:
        # One rejected order must not abandon the rest of the book.
        logger.warning("Order %s rejected: %s", key, exc)
        return _result(
            key, "rejected", "", symbol=desired.request.symbol,
            action=desired.request.action, quantity=float(desired.request.quantity),
            status="rejected", reason=str(exc),
        )
    order_id = str(result.get("order_id", "") or "")
    if not order_id:
        # An accepted order we cannot name is worse than a rejected one: only the rejection is
        # recoverable. Recording "" would make the next run look up an empty id, find nothing,
        # and submit again -- every run, stacking live orders each time.
        logger.error(
            "Broker accepted %s (%s %s %s) without returning an order id; it cannot be tracked "
            "or cancelled and must be reconciled by hand",
            key, desired.request.action, desired.request.quantity, desired.request.symbol,
        )
        order_ids.pop(key, None)
        return _result(
            key, "untracked", "", symbol=desired.request.symbol,
            action=desired.request.action, quantity=float(desired.request.quantity),
            status="untracked", reason="broker returned no order id",
        )
    order_ids[key] = order_id
    logger.info(
        "Placed %s: %s %s %s @ %s",
        key, desired.request.action, desired.request.quantity,
        desired.request.symbol, desired.request.limit_price or "market",
    )
    return _result(
        key, "submitted", order_ids[key], symbol=desired.request.symbol,
        action=desired.request.action, quantity=float(desired.request.quantity),
        status="submitted", limit_price=desired.request.limit_price,
        stop_price=desired.request.stop_price, order_type=desired.request.order_type,
    )


def _replace(
    key: str, desired: DesiredOrder, order_id: str, brokerage: Any, order_ids: Dict[str, str]
) -> Dict[str, Any]:
    """Re-price in place; fall back to cancel-and-resubmit where the venue refuses.

    Replace is tried first because it is atomic -- the order is never absent from the book.
    Not every venue allows it in every order state (e.g. Alpaca while still ``accepted``), so
    the fallback cancels and resubmits, and records which route it took.
    """
    try:
        result = brokerage.replace_order(order_id, desired.request)
    except Exception as exc:
        logger.info("Order %s could not be re-priced in place (%s); resubmitting", key, exc)
        # Guarded: a cancel that fails for a reason other than "already gone" (a network error,
        # a broker outage) must not escape and abandon the rest of the book. The old order may
        # still be resting, so it keeps its id and this run leaves the price where it was --
        # a stale price being strictly better than an untracked order plus a duplicate.
        if not _cancel(brokerage, order_id, key):
            order_ids[key] = order_id
            return _result(
                key, "cancel_failed", order_id, symbol=desired.request.symbol,
                action=desired.request.action, quantity=float(desired.request.quantity),
                status="unreconciled", reason="could not re-price or cancel; order left as it was",
            )
        outcome = _submit(key, desired, brokerage, order_ids)
        if outcome.get("reconciled") == "submitted":
            return {**outcome, "reconciled": "resubmitted", "previous_order_id": order_id}
        # Both routes failed; the old order is confirmed cancelled, so there is nothing to record.
        order_ids.pop(key, None)
        return outcome
    order_ids[key] = str(result.get("order_id", order_id))
    return _result(
        key, "replaced", order_ids[key], symbol=desired.request.symbol,
        action=desired.request.action, quantity=float(desired.request.quantity),
        status="submitted", previous_order_id=order_id,
        limit_price=desired.request.limit_price, stop_price=desired.request.stop_price,
        order_type=desired.request.order_type,
    )


def _result(key: str, reconciled: str, order_id: str, **fields: Any) -> Dict[str, Any]:
    """One reconciliation outcome, in the shape the journal and logs already read.

    ``action`` stays the trade side (buy/sell); what the reconciler *did* is a separate axis
    and lives in ``reconciled``.
    """
    return {"key": key, "reconciled": reconciled, "order_id": order_id, **fields}


def _needs_replacement(desired: DesiredOrder, existing: Dict[str, Any]) -> bool:
    """Whether the working order differs from what is wanted by enough to re-price.

    Quantity, side and order type are exact; prices are compared against
    ``replace_tolerance`` since a wide-spread instrument shouldn't churn on a cent of drift.
    """
    request = desired.request
    if str(existing.get("action", "")).lower() != request.action.lower():
        return True
    if float(existing.get("quantity", 0.0)) != float(request.quantity):
        return True
    if str(existing.get("order_type", "")).lower() != request.order_type.lower():
        return True
    return _moved(request.limit_price, existing.get("limit_price"), desired.replace_tolerance) or _moved(
        request.stop_price, existing.get("stop_price"), desired.replace_tolerance
    )


def _moved(wanted: float | None, current: Any, tolerance: float) -> bool:
    """Whether a price differs by more than ``tolerance`` as a fraction of the wanted price."""
    if wanted is None:
        return False
    current_price = float(current or 0.0)
    if current_price <= 0:
        return True
    return abs(float(wanted) - current_price) > max(float(wanted) * max(tolerance, 0.0), 0.005)
