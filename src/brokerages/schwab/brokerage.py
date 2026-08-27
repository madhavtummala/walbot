from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List

import pandas as pd

from ..base import BaseBrokerage
from .client import TRADER_BASE, SchwabSession, account_hash
from ...core.interfaces import OrderRequest

logger = logging.getLogger(__name__)

_INSTRUCTIONS = {
    ("buy", False): "BUY",
    ("sell", False): "SELL",
    ("buy", True): "BUY_TO_COVER",
    ("sell", True): "SELL_SHORT",
}

#: Options are opened and closed rather than bought and sold, and Schwab wants to be told which.
#: Defaults follow this codebase's only options strategy -- long premium -- so a bare buy opens
#: and a bare sell closes; ``extra["position_intent"]`` overrides for the short side.
_OPTION_INSTRUCTIONS = {
    ("buy", "open"): "BUY_TO_OPEN",
    ("buy", "close"): "BUY_TO_CLOSE",
    ("sell", "open"): "SELL_TO_OPEN",
    ("sell", "close"): "SELL_TO_CLOSE",
}

_ORDER_TYPES = {
    "market": "MARKET",
    "limit": "LIMIT",
    "stop": "STOP",
    "stop_limit": "STOP_LIMIT",
}

_DURATIONS = {"day": "DAY", "gtc": "GOOD_TILL_CANCEL"}

_STRATEGY_TYPES = {"single": "SINGLE", "oco": "OCO", "trigger": "TRIGGER"}

#: Statuses that mean an order is finished. Everything else -- ``PENDING_ACTIVATION`` on a fresh
#: submission, ``AWAITING_PARENT_ORDER`` on an untriggered bracket leg, ``QUEUED`` out of hours --
#: is still live and must be treated as resting. Enumerated the terminal side rather than the live
#: side deliberately: a status Schwab adds later should read as "still live", which makes a
#: reconciler leave it alone, not as "gone", which makes it place a duplicate.
_TERMINAL_STATUSES = frozenset({
    "FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED", "REPLACED",
})


def _instruction_for(request: OrderRequest) -> str:
    """The Schwab verb for this order, which differs by asset class.

    Equities net long and short, so the verb encodes direction. Options are positional, so it
    encodes whether the leg opens or closes exposure -- Schwab rejects the order outright if the
    two disagree with the account's actual position.
    """
    action = request.action.lower()
    intent = str((request.extra or {}).get("position_intent", "")).lower()
    if request.asset_type == "option":
        if "open" in intent:
            side = "open"
        elif "close" in intent:
            side = "close"
        else:
            side = "open" if action == "buy" else "close"
        return _OPTION_INSTRUCTIONS[(action, side)]
    shorting = "short" in intent or "cover" in intent
    return _INSTRUCTIONS[(action, shorting)]


def _order_payload(request: OrderRequest) -> Dict[str, Any]:
    """One :class:`OrderRequest` as Schwab's order JSON, children and all.

    Recursive, because Schwab nests the same order shape under ``childOrderStrategies`` for both
    OCO pairs and trigger brackets. Writing the bracket as one payload rather than as a sequence
    of submissions is the whole point: the exchange then owns the invariant that the target and
    the stop cannot both fill.

    A plain equity market order must serialise exactly as it did before this function existed --
    ``tests/test_schwab.py`` asserts the payload field by field, and that test is the contract.
    """
    quantity = float(request.quantity)
    if quantity != int(quantity):
        raise ValueError(
            f"Schwab does not accept fractional quantities (got {quantity} for {request.symbol})"
        )

    payload: Dict[str, Any] = {
        "orderType": _ORDER_TYPES[request.order_type],
        "session": "NORMAL",
        "duration": _DURATIONS.get(request.time_in_force.lower(), "DAY"),
        "orderStrategyType": _STRATEGY_TYPES[request.strategy],
        "orderLegCollection": [
            {
                "instruction": _instruction_for(request),
                "quantity": int(quantity),
                "instrument": {
                    "symbol": request.symbol.upper(),
                    "assetType": "OPTION" if request.asset_type == "option" else "EQUITY",
                },
            }
        ],
    }
    # Schwab names the limit price ``price``, not ``limitPrice``, and rejects the field outright
    # on a market order rather than ignoring it.
    if request.limit_price is not None and request.order_type in ("limit", "stop_limit"):
        payload["price"] = _tick(request.limit_price)
    if request.stop_price is not None and request.order_type in ("stop", "stop_limit"):
        payload["stopPrice"] = _tick(request.stop_price)
    if request.children:
        # A trigger with two children must nest them under an OCO, not list them as siblings.
        # Verified against the live API: listing them flat is *accepted*, and Schwab stores two
        # independent SINGLE orders -- so after the target fills the stop stays live and can sell
        # a position that no longer exists. The whole point of the bracket is that it cannot.
        payload["childOrderStrategies"] = (
            [_oco_payload(request.children)]
            if request.strategy == "trigger" and len(request.children) > 1
            else [_order_payload(child) for child in request.children]
        )
    return payload


def _oco_payload(children: tuple[OrderRequest, ...]) -> Dict[str, Any]:
    """An OCO wrapper, which carries no legs of its own -- only the pair it governs."""
    return {
        "orderStrategyType": "OCO",
        "childOrderStrategies": [_order_payload(child) for child in children],
    }


def _tick(price: float) -> float:
    """Round to a penny.

    Schwab rejects a price carrying more precision than the instrument's tick, and the arithmetic
    upstream -- a delta-translated limit, a percentage stop -- routinely produces one. Options
    below $3 actually trade in half-cents, so rounding up to a penny is conservative on the buy
    side and costs at most half a tick on the sell.
    """
    return round(float(price), 2)


def _rejection_reason(order: Dict[str, Any]) -> str:
    """Why Schwab refused an order tree, gathered from wherever in it the explanation sits.

    A rejected bracket carries no ``statusDescription`` on the parent: the reason is written
    against the leg that actually offended, and the siblings each get a bare "Order Rejected due
    to Order: <id>" pointing at it. Reading only the top level therefore reports nothing at all
    for the one case where the caller most needs a reason, so this walks the tree and keeps every
    distinct description it finds, in the order encountered.
    """
    reasons: List[str] = []
    def walk(node: Dict[str, Any]) -> None:
        description = str(node.get("statusDescription") or "").strip()
        if description and description not in reasons:
            reasons.append(description)
        for child in node.get("childOrderStrategies") or []:
            walk(child)
    walk(order)
    return "; ".join(reasons)


class SchwabBrokerage(BaseBrokerage):
    """Charles Schwab Trader API.

    Schwab does not accept fractional share quantities on equity orders, so sizing is whole
    shares only -- see ``supports_fractional_shares``.
    """

    supports_fractional_shares = False
    supports_options = True
    #: Schwab holds the OCO pair itself, via ``orderStrategyType``, so a bracket is one order and
    #: the venue owns the "never both" invariant.
    supports_oco = True

    def __init__(self, config: Dict[str, Any], session: SchwabSession | None = None):
        super().__init__(config)
        self.session = session or SchwabSession(config)
        self._account_hash = ""
        self._account_number = str(getattr(config, "schwab_account_number", "") or "")

    @property
    def account_hash(self) -> str:
        if not self._account_hash:
            self._account_hash = account_hash(self.session, self._account_number)
        return self._account_hash

    def _account_payload(self, fields: str = "") -> Dict[str, Any]:
        params = {"fields": fields} if fields else None
        payload = self.session.get(f"{TRADER_BASE}/accounts/{self.account_hash}", params=params)
        # Schwab wraps the account in ``securitiesAccount``; tolerate a bare body as well.
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        return payload.get("securitiesAccount", payload) if isinstance(payload, dict) else {}

    def get_account_state(self) -> Dict[str, Any]:
        account = self._account_payload()
        balances = account.get("currentBalances", {}) or {}
        # Schwab reports total account value as liquidationValue; equity is margin-only.
        equity = balances.get("liquidationValue", balances.get("equity", 0.0))
        cash = balances.get("cashBalance", balances.get("cashAvailableForTrading", 0.0))
        buying_power = balances.get("buyingPower", balances.get("cashAvailableForTrading", cash))
        return {
            "equity": float(equity or 0.0),
            "cash": float(cash or 0.0),
            "buying_power": float(buying_power or 0.0),
            "is_market_open": self.is_market_open(),
        }

    def get_positions(self) -> Dict[str, float]:
        account = self._account_payload(fields="positions")
        positions: Dict[str, float] = {}
        for row in account.get("positions", []) or []:
            symbol = str((row.get("instrument") or {}).get("symbol", "")).upper()
            if not symbol:
                continue
            quantity = float(row.get("longQuantity", 0.0) or 0.0) - float(row.get("shortQuantity", 0.0) or 0.0)
            if quantity:
                positions[symbol] = quantity
        return positions

    def get_marks(self, symbols) -> Dict[str, float]:
        """Schwab prices each position on the account payload, so no quote call is needed."""
        wanted = {str(symbol).upper() for symbol in symbols}
        account = self._account_payload(fields="positions")
        marks: Dict[str, float] = {}
        for row in account.get("positions", []) or []:
            symbol = str((row.get("instrument") or {}).get("symbol", "")).upper()
            if symbol not in wanted:
                continue
            quantity = float(row.get("longQuantity", 0.0) or 0.0) - float(row.get("shortQuantity", 0.0) or 0.0)
            market_value = float(row.get("marketValue", 0.0) or 0.0)
            if quantity and market_value:
                marks[symbol] = market_value / quantity
        return marks

    def get_position_details(self) -> List[Dict[str, Any]]:
        account = self._account_payload(fields="positions")
        rows: List[Dict[str, Any]] = []
        for pos in account.get("positions", []) or []:
            symbol = str((pos.get("instrument") or {}).get("symbol", "")).upper()
            if not symbol:
                continue
            qty = float(pos.get("longQuantity", 0.0) or 0.0) - float(pos.get("shortQuantity", 0.0) or 0.0)
            if not qty:
                continue
            market_value = float(pos.get("marketValue", 0.0) or 0.0)
            avg_entry = float(pos.get("averagePrice", 0.0) or 0.0)
            unrealized_pl = float(pos.get("currentDayProfitLoss", 0.0) or 0.0)
            unrealized_plpc = (market_value / (avg_entry * abs(qty)) - 1.0) if avg_entry and qty else 0.0
            rows.append({
                "symbol": symbol,
                "qty": qty,
                "avg_entry_price": avg_entry,
                "market_value": market_value,
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
            })
        rows.sort(key=lambda row: abs(row["market_value"]), reverse=True)
        return rows

    def get_dividend_activity(self, start=None, end=None) -> List[Dict[str, Any]]:
        """Cash distributions credited to this account, from ``/accounts/{hash}/transactions``.

        Schwab files dividends and cash interest under one ``DIVIDEND_OR_INTEREST`` type, which
        is the right granularity here: both are income the account received rather than price
        appreciation, and a T-bill fund's payment is literally interest.
        """
        from datetime import datetime, time, timedelta, timezone

        end_dt = (
            datetime.combine(end, time.max, tzinfo=timezone.utc)
            if end else datetime.now(timezone.utc)
        )
        start_dt = (
            datetime.combine(start, time.min, tzinfo=timezone.utc)
            if start else end_dt - timedelta(days=365)
        )
        payload = self.session.get(
            f"{TRADER_BASE}/accounts/{self.account_hash}/transactions",
            params={
                "startDate": start_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "endDate": end_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "types": "DIVIDEND_OR_INTEREST",
            },
        )
        rows: List[Dict[str, Any]] = []
        for item in payload or []:
            if not isinstance(item, dict):
                continue
            # The symbol lives on the transferItem describing the instrument, when there is
            # one at all -- account-level cash interest has no instrument.
            symbol = ""
            for leg in item.get("transferItems", []) or []:
                candidate = str((leg.get("instrument") or {}).get("symbol", "")).upper()
                if candidate:
                    symbol = candidate
                    break
            rows.append(
                {
                    "symbol": symbol,
                    "date": str(item.get("tradeDate") or item.get("time") or "")[:10],
                    "amount": float(item.get("netAmount") or 0.0),
                    "description": str(item.get("description") or item.get("type") or ""),
                }
            )
        rows.sort(key=lambda row: row["date"], reverse=True)
        return rows

    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        payload = (
            _oco_payload(request.children)
            if request.strategy == "oco"
            else _order_payload(request)
        )
        logger.info(
            "Submitting Schwab %s %s order for %s qty=%s",
            payload["orderStrategyType"], payload.get("orderType", "-"),
            request.symbol, int(request.quantity),
        )
        response = self.session.post(
            f"{TRADER_BASE}/accounts/{self.account_hash}/orders",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        order_id = _order_id_from(response)
        return {
            "order_id": order_id,
            "client_order_id": request.client_order_id or "",
            "status": self._confirm(order_id, request),
            "symbol": request.symbol.upper(),
            "qty": int(request.quantity),
        }

    def _confirm(self, order_id: str, request: OrderRequest) -> str:
        """The status Schwab actually holds for a just-submitted order, not the one 201 implies.

        A 201 with a ``Location`` header means "accepted for processing", not "resting". Schwab
        runs its own validation afterwards and can move the order straight to ``REJECTED``,
        reporting why in ``statusDescription`` -- observed live with a sub-penny option price,
        which this client cannot produce (see :func:`_tick`) but a hand-built payload can.

        **A rejection takes the whole tree with it.** The bad leg was a bracket's stop, and the
        entry and the profit target were rejected alongside it, each citing the offending order
        id. So reporting the parent as accepted would not merely overstate one leg -- it would
        claim a position was protected when nothing at all had been placed.

        One read, no retry loop: the rejection may not have landed yet, and an order still shown
        as pending here is reported as accepted, exactly as before. That is not a gap the caller
        has to cover, because :func:`..algorithms.reconcile.reconcile_orders` reads the working
        set at the top of every run and resubmits anything that is no longer resting -- a
        rejection this misses self-heals on the next fire rather than persisting as a phantom.
        """
        if not order_id or order_id == "unknown":
            return "accepted"
        try:
            order = self.session.get(
                f"{TRADER_BASE}/accounts/{self.account_hash}/orders/{order_id}"
            ) or {}
        except Exception as exc:
            # The order is placed either way; failing to read it back is not a reason to tell the
            # caller the submission failed.
            logger.warning("Could not read Schwab order %s back after submitting: %s", order_id, exc)
            return "accepted"
        status = str(order.get("status") or "").upper()
        if status not in _TERMINAL_STATUSES or status == "FILLED":
            return "accepted"
        logger.error(
            "Schwab %s order %s for %s was %s on submission: %s",
            str(order.get("orderStrategyType") or "-"), order_id, request.symbol, status,
            _rejection_reason(order) or "no reason given",
        )
        return status.lower()

    def get_orders(self, status: str = "WORKING") -> List[Dict[str, Any]]:
        """Orders in ``status``, flattened so a bracket's legs are listed alongside plain orders.

        Flattened because a reconciler asks "is the stop still working", and under a trigger
        bracket the stop is a child of the entry rather than a top-level order. Each row keeps
        ``parent_order_id`` so the tree is still recoverable.

        ``status="WORKING"`` means "still live", which at Schwab is a dozen different words --
        see :data:`_TERMINAL_STATUSES`. Filtering on the literal ``WORKING`` misses an untriggered
        bracket leg (``AWAITING_PARENT_ORDER``) and a freshly placed order (``PENDING_ACTIVATION``),
        which would have a reconciler conclude nothing is resting and submit the whole book again.
        """
        # ``fromEnteredTime``/``toEnteredTime`` are mandatory -- Schwab answers 400 without them,
        # rather than defaulting to a recent window. Sixty days back covers any GTC bracket this
        # algorithm could still have resting, since nothing it opens is held past a few sessions.
        now = pd.Timestamp.now(tz="UTC")
        orders = self.session.get(
            f"{TRADER_BASE}/accounts/{self.account_hash}/orders",
            params={
                "fromEnteredTime": _schwab_time(now - pd.Timedelta(days=60)),
                "toEnteredTime": _schwab_time(now),
            },
        ) or []
        rows: List[Dict[str, Any]] = []
        for order in orders:
            rows.extend(_flatten_order(order, parent_order_id=""))
        if str(status).upper() in ("WORKING", "OPEN"):
            rows = [row for row in rows if row["status"] not in _TERMINAL_STATUSES]
        elif status:
            rows = [row for row in rows if row["status"] == str(status).upper()]
        return rows

    def cancel_order(self, order_id: str) -> None:
        """Cancel one order. An order that is already gone is a success, not an error.

        Schwab answers 400 or 404 for an order that filled or was cancelled a moment ago, and
        that race is routine for a reconciler working from a snapshot a few seconds old. Raising
        would abort the rest of the pass over something that is already true.
        """
        try:
            self.session.delete(f"{TRADER_BASE}/accounts/{self.account_hash}/orders/{order_id}")
            logger.info("Cancelled Schwab order %s", order_id)
        except Exception as exc:
            logger.info("Schwab order %s was not cancellable (already filled or gone): %s", order_id, exc)

    def replace_order(self, order_id: str, request: OrderRequest) -> Dict[str, Any]:
        """Re-price in one call, so the order is never absent from the book in between.

        **The replacement payload may not carry children.** Schwab answers
        ``400 "Replacing order cannot have child orders."`` to a PUT that repeats the tree,
        verified live -- so a trigger parent is re-priced by sending its own leg alone. Schwab
        rebuilds the bracket underneath the replacement from the children's *current* prices, and
        issues new ids for every node in it. The legs are stripped here rather than at the callers
        so that a request built once for :meth:`submit_order` can be handed to this method
        unchanged, which is exactly what the reconciler does.
        """
        payload = (
            _oco_payload(request.children)
            if request.strategy == "oco"
            else _order_payload(replace(request, children=()))
        )
        response = self.session.put(
            f"{TRADER_BASE}/accounts/{self.account_hash}/orders/{order_id}",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        new_id = _order_id_from(response)
        logger.info("Replaced Schwab order %s with %s for %s", order_id, new_id, request.symbol)
        return {
            # Schwab issues a new id for the replacement; falling back to the old one keeps the
            # caller's bookkeeping pointing at a real order either way.
            "order_id": new_id if new_id != "unknown" else str(order_id),
            "client_order_id": request.client_order_id or "",
            # Confirmed for the same reason a submission is: a replacement is a fresh order to
            # Schwab's validation, and it can be rejected on exactly the grounds a first
            # submission can -- leaving the caller believing it re-priced something that is gone.
            "status": self._confirm(new_id if new_id != "unknown" else str(order_id), request),
            "symbol": request.symbol.upper(),
            "qty": int(request.quantity),
        }

    def cancel_all_orders(self) -> None:
        for order in self.get_orders("WORKING"):
            if order.get("order_id"):
                self.cancel_order(str(order["order_id"]))

    def is_market_open(self) -> bool:
        """Whether the equity market is open, per ``/marketdata/v1/markets``."""
        from .client import MARKETDATA_BASE

        try:
            payload = self.session.get(f"{MARKETDATA_BASE}/markets", params={"markets": "equity"}) or {}
        except Exception as exc:  # A failed lookup must not read as "open".
            logger.warning("Could not read Schwab market hours: %s", exc)
            return False

        now = pd.Timestamp.now(tz="UTC")
        for product in (payload.get("equity") or {}).values():
            if not product.get("isOpen", False):
                continue
            for window in (product.get("sessionHours") or {}).get("regularMarket", []):
                start = pd.to_datetime(window.get("start"), utc=True, errors="coerce")
                end = pd.to_datetime(window.get("end"), utc=True, errors="coerce")
                if pd.notna(start) and pd.notna(end) and start <= now <= end:
                    return True
        return False

    def validate_short_sale_feasibility(
        self, symbol: str, quantity: float, target_shares: float, latest_price: float
    ) -> Dict[str, Any]:
        """Schwab exposes no pre-trade shortability check, so shorts are not auto-approved."""
        return {
            "shortable": False,
            "reason": "Schwab does not expose a pre-trade shortability check",
        }


def _schwab_time(moment: Any) -> str:
    """Schwab's order-window format: ISO-8601 with milliseconds and a literal ``Z``.

    It rejects both a bare date and an offset like ``+00:00``, so the format is spelled out here
    rather than left to ``isoformat``.
    """
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def _flatten_order(order: Dict[str, Any], *, parent_order_id: str) -> List[Dict[str, Any]]:
    """One Schwab order tree as flat rows, in the shape :meth:`Brokerage.get_orders` promises.

    An OCO wrapper contributes no row of its own: it carries no legs, so there is nothing to
    reconcile against. Only orders that name an instrument are returned.
    """
    rows: List[Dict[str, Any]] = []
    order_id = str(order.get("orderId") or "")
    legs = order.get("orderLegCollection") or []
    if legs:
        leg = legs[0]
        instrument = leg.get("instrument") or {}
        instruction = str(leg.get("instruction") or "").upper()
        rows.append({
            "order_id": order_id,
            "parent_order_id": parent_order_id,
            "symbol": str(instrument.get("symbol") or "").upper(),
            "asset_type": "option" if str(instrument.get("assetType") or "") == "OPTION" else "equity",
            # Schwab's verb encodes open/close as well as direction; the caller wants direction.
            "action": "buy" if instruction.startswith("BUY") else "sell",
            "instruction": instruction,
            "quantity": float(leg.get("quantity") or 0.0),
            "filled_quantity": float(order.get("filledQuantity") or 0.0),
            "order_type": str(order.get("orderType") or "").lower(),
            "limit_price": float(order.get("price") or 0.0),
            "stop_price": float(order.get("stopPrice") or 0.0),
            "status": str(order.get("status") or "").upper(),
        })
    for child in order.get("childOrderStrategies") or []:
        rows.extend(_flatten_order(child, parent_order_id=order_id or parent_order_id))
    return rows


def _order_id_from(response: Any) -> str:
    """Schwab returns 201 with an empty body; the new order id is the last Location segment."""
    if isinstance(response, dict):
        location = str(response.get("location") or "")
        if location:
            return location.rstrip("/").rsplit("/", 1)[-1]
        if response.get("orderId") is not None:
            return str(response["orderId"])
    return "unknown"
