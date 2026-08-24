"""Making a broker's resting orders match what a run decided they should be.

Plain functions, not a base class. An algorithm whose output is an order book rather than a
target portfolio needs a different ``execute`` -- but "different ``execute``" is a method
override, which :class:`BaseAlgorithm` already permits and which ``bursty_dca`` already does for
``state_after``. Forking the class hierarchy for one algorithm would buy nothing that overriding
one hook does not, while adding a second algorithm shape everyone reading the package has to
learn. If a second order-book algorithm ever appears, it calls :func:`reconcile_orders` too.

Three properties this module exists to hold:

**The broker is the source of truth, not our memory.** Every run reads the working orders back
before deciding anything. Reconstructing the book from what we remember submitting cannot
survive a partial fill, a manual cancellation in the broker's own UI, or a run that died between
placing an order and recording it.

**Reconciliation is idempotent, which is what makes a cron cadence safe.** Running twice in a
minute is the same as running once; skipping an hour converges on the next fire. The caller is
never mid-transaction -- it is only ever asserting a desired state.

**Order identity is a role, not a submission.** ``DesiredOrder.key`` names what an order is
*for* ("this symbol's entry bid"), so an order re-priced five times across a session stays one
thing. Broker order ids are recorded against the key in algorithm state; the key outlives them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..core.interfaces import DesiredOrder

logger = logging.getLogger(__name__)

#: Where an algorithm records the broker order id standing behind each desired order's key.
#: Nested under one key so it cannot collide with whatever the strategy keeps beside it.
ORDER_IDS_KEY = "order_ids"


def broker_supports_oco(account_id: str) -> bool:
    """Whether this account's broker will hold an OCO pair itself.

    Read off the class without instantiating it, exactly as ``bursty_dca`` reads
    ``supports_fractional_shares``: this is a property of the venue, not of a live session, and
    building the brokerage would authenticate it just to answer a question about its capabilities.
    That matters because the caller is ``plan``, which must stay free of brokerage objects.

    An unknown account falls back to ``False`` -- the conservative reading, since two independent
    legs work everywhere and an OCO does not.
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
) -> Dict[str, Any]:
    """Cancel what is no longer wanted, then place or re-price the rest.

    ``recorded`` maps each desired order's key to the broker order id last known to stand behind
    it. Returns ``{"results", "order_ids", "working"}`` -- the caller persists ``order_ids`` and
    reports ``results``.

    Cancels run first so a replacement is never submitted alongside the order it replaces, which
    for a bracket would leave two stops on one position.
    """
    try:
        working = brokerage.get_orders("WORKING")
    except NotImplementedError as exc:
        # A brokerage that cannot list orders cannot be reconciled against. Refusing here is the
        # safe direction: proceeding would submit the whole desired book on every run, since
        # nothing would ever look already-present.
        raise NotImplementedError(
            f"{type(brokerage).__name__} cannot list working orders, which order reconciliation "
            "requires. An algorithm that rests orders needs a brokerage that holds them, so it "
            "cannot be backtested against the paper book."
        ) from exc

    working_by_id = {str(o.get("order_id")): o for o in working if o.get("order_id")}
    results: List[Dict[str, Any]] = []
    wanted = {order.key: order for order in desired}
    order_ids: Dict[str, str] = {}

    for key, order_id in recorded.items():
        if key in wanted or order_id not in working_by_id:
            # Not ours to cancel, or already gone -- filled, expired, or cancelled by hand. An id
            # that has left the working set is simply dropped from the record.
            continue
        existing = working_by_id[order_id]
        brokerage.cancel_order(order_id)
        results.append(_result(
            key, "cancelled", order_id,
            symbol=str(existing.get("symbol", "")),
            action=str(existing.get("action", "")),
            quantity=float(existing.get("quantity", 0.0) or 0.0),
        ))

    for key, desired_order in wanted.items():
        existing_id = recorded.get(key, "")
        existing = working_by_id.get(existing_id)
        if existing is None:
            results.append(_submit(key, desired_order, brokerage, order_ids))
        elif _needs_replacement(desired_order, existing):
            results.append(_replace(key, desired_order, existing_id, brokerage, order_ids))
        else:
            order_ids[key] = existing_id
            results.append(_result(
                key, "unchanged", existing_id,
                symbol=desired_order.request.symbol,
                action=desired_order.request.action,
                quantity=float(desired_order.request.quantity),
            ))

    return {"results": results, "order_ids": order_ids, "working": working}


def _submit(key: str, desired: DesiredOrder, brokerage: Any, order_ids: Dict[str, str]) -> Dict[str, Any]:
    try:
        result = brokerage.submit_order(desired.request)
    except Exception as exc:
        # One rejected order must not abandon the rest of the book: a stop that cannot be placed
        # is exactly when the remaining orders matter most.
        logger.warning("Order %s rejected: %s", key, exc)
        return _result(
            key, "rejected", "", symbol=desired.request.symbol,
            action=desired.request.action, quantity=float(desired.request.quantity),
            status="rejected", reason=str(exc),
        )
    order_ids[key] = str(result.get("order_id", ""))
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
    """Re-price in place, falling back to cancel-and-resubmit where the venue refuses.

    Replace is tried first because it is atomic: the order is never absent from the book, so
    nothing can slip through the gap and a crash cannot leave the position unprotected.

    But not every venue will replace every order. Alpaca refuses while an order is still in
    ``accepted`` -- "cannot replace order in accepted status" -- which for a strategy whose whole
    mechanism is re-pricing a resting bid every five minutes would mean the bid never moves at
    all. A brief gap is a far smaller cost than an order frozen at the morning's price, so the
    fallback cancels and resubmits, and says which route it took.
    """
    try:
        result = brokerage.replace_order(order_id, desired.request)
    except Exception as exc:
        logger.info("Order %s could not be re-priced in place (%s); resubmitting", key, exc)
        brokerage.cancel_order(order_id)
        outcome = _submit(key, desired, brokerage, order_ids)
        if outcome.get("reconciled") == "submitted":
            return {**outcome, "reconciled": "resubmitted", "previous_order_id": order_id}
        # Both routes failed. The old order was cancelled, so there is nothing left to point at:
        # recording its id would have the next run believe a dead order is still working.
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
    """One reconciliation outcome, in the shape the journal and the logs already read.

    ``action`` stays the trade side -- buy or sell -- because that is what
    ``src/data/order_journal.py`` files under ``side`` and what the logs print. What the
    reconciler *did* is a separate axis and lives in ``reconciled``; collapsing the two put
    "unchanged" in the journal's side column, which reads as a trade that never happened.
    """
    return {"key": key, "reconciled": reconciled, "order_id": order_id, **fields}


def _needs_replacement(desired: DesiredOrder, existing: Dict[str, Any]) -> bool:
    """Whether the working order differs from what is wanted by enough to be worth re-pricing.

    Quantity and side are exact: those are not prices and any difference is a different order.
    Prices are compared against ``replace_tolerance`` because options spreads are wide and every
    replace costs a round trip -- rewriting the book because a mark moved a cent would churn all
    day and buy nothing.
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
