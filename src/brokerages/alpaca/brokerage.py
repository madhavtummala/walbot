from __future__ import annotations

import logging
from typing import Dict, Any, List
from ..base import BaseBrokerage
from ...core.interfaces import OrderRequest
from ...core.options import is_osi_symbol, to_osi_form
from src.brokerages.alpaca.client import (
    DIVIDEND_ACTIVITY_TYPES,
    build_order_request,
    build_replace_request,
    create_trading_client,
    get_account_activities,
    get_open_orders,
    get_position_marks,
    get_positions,
    is_market_open,
    validate_short_sale_feasibility as _alpaca_short_check,
)

logger = logging.getLogger(__name__)


class AlpacaBrokerage(BaseBrokerage):
    supports_fractional_shares = True
    supports_options = True

    #: Verified against the live API, not assumed: Alpaca answers "complex orders not supported
    #: for options trading" to any OCO or bracket on a contract. Single-leg limit, stop and
    #: stop-limit orders are all accepted, so a bracket is still expressible -- as two
    #: independent orders whose mutual exclusivity the *caller* must maintain. See
    #: ``broker_supports_oco``, which is how the algorithm learns to split it.
    supports_oco = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = create_trading_client(config)
        self._config = config

    def get_dividend_activity(self, start=None, end=None) -> List[Dict[str, Any]]:
        from datetime import datetime, time, timezone

        after = datetime.combine(start, time.min, tzinfo=timezone.utc) if start else None
        rows: List[Dict[str, Any]] = []
        for item in get_account_activities(self._config, DIVIDEND_ACTIVITY_TYPES, after=after):
            stamp = str(item.get("date") or item.get("transaction_time") or "")[:10]
            if end and stamp and stamp > end.isoformat():
                continue
            rows.append(
                {
                    "symbol": str(item.get("symbol") or ""),
                    "date": stamp,
                    "amount": float(item.get("net_amount") or 0.0),
                    "description": str(item.get("description") or item.get("activity_type") or ""),
                }
            )
        rows.sort(key=lambda row: row["date"], reverse=True)
        return rows

    def get_account_state(self) -> Dict[str, Any]:
        account = self.client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "is_market_open": is_market_open(self.client)
        }

    def get_positions(self) -> Dict[str, float]:
        return get_positions(self.client)

    def get_marks(self, symbols) -> Dict[str, float]:
        marks = get_position_marks(self.client)
        return {symbol: marks[symbol] for symbol in symbols if symbol in marks}

    def get_position_details(self) -> List[Dict[str, Any]]:
        rows = []
        for position in self.client.get_all_positions():
            rows.append({
                "symbol": str(getattr(position, "symbol", "")),
                "qty": float(getattr(position, "qty", 0.0) or 0.0),
                "avg_entry_price": float(getattr(position, "avg_entry_price", 0.0) or 0.0),
                "market_value": float(getattr(position, "market_value", 0.0) or 0.0),
                "unrealized_pl": float(getattr(position, "unrealized_pl", 0.0) or 0.0),
                "unrealized_plpc": float(getattr(position, "unrealized_plpc", 0.0) or 0.0),
            })
        rows.sort(key=lambda row: abs(row["market_value"]), reverse=True)
        return rows

    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        order = self.client.submit_order(order_data=build_order_request(request))
        return _order_result(order)

    def get_orders(self, status: str = "WORKING") -> List[Dict[str, Any]]:
        """Open orders, in the shape :meth:`Brokerage.get_orders` promises.

        Alpaca calls the resting set ``open``; Schwab calls it ``WORKING``. The caller speaks
        Schwab's word because that is what the interface settled on, so it is translated here.
        """
        wanted = "open" if str(status).upper() in ("WORKING", "OPEN", "") else str(status).lower()
        return [_order_row(order) for order in get_open_orders(self.client, wanted)]

    def cancel_order(self, order_id: str) -> None:
        """Cancel one order. Already-gone is success, not failure.

        A reconciler works from a snapshot seconds old, so racing a fill is routine; raising
        would abort the rest of its pass over something that is already true.
        """
        try:
            self.client.cancel_order_by_id(order_id)
        except Exception as exc:
            logger.info("Alpaca order %s was not cancellable (already filled or gone): %s", order_id, exc)

    def replace_order(self, order_id: str, request: OrderRequest) -> Dict[str, Any]:
        """Re-price in place, so the order is never absent from the book in between."""
        order = self.client.replace_order_by_id(
            order_id, order_data=build_replace_request(request)
        )
        return _order_result(order)

    def cancel_all_orders(self) -> None:
        self.client.cancel_orders()

    def validate_short_sale_feasibility(
        self, symbol: str, quantity: int, target_shares: int, latest_price: float
    ) -> Dict[str, Any]:
        return _alpaca_short_check(self.client, symbol, quantity, target_shares, latest_price)


def _order_result(order: Any) -> Dict[str, Any]:
    """A submitted or replaced order, in the shape ``submit_order`` promises."""
    return {
        "order_id": str(order.id),
        "client_order_id": str(order.client_order_id),
        "status": str(getattr(order, "status", "")),
        "symbol": str(order.symbol),
        "qty": int(float(order.qty or 0)),
    }


def _order_row(order: Any) -> Dict[str, Any]:
    """One resting order, in the shape ``get_orders`` promises.

    The symbol is re-spelled to the padded OSI form the rest of the codebase uses, so a
    reconciler comparing a working order against a desired one is comparing like with like --
    otherwise every Alpaca option order looks different from the contract that asked for it and
    is replaced on every single run.
    """
    symbol = str(getattr(order, "symbol", "")).upper()
    option = is_osi_symbol(symbol)
    return {
        "order_id": str(order.id),
        # Always empty: Alpaca refuses complex orders on options, so nothing this codebase
        # submits there has a parent. The key is present because the interface promises it.
        "parent_order_id": "",
        "symbol": to_osi_form(symbol, padded=True) if option else symbol,
        "asset_type": "option" if option else "equity",
        "action": str(getattr(order, "side", "")).split(".")[-1].lower(),
        "quantity": float(getattr(order, "qty", 0.0) or 0.0),
        "filled_quantity": float(getattr(order, "filled_qty", 0.0) or 0.0),
        "order_type": str(getattr(order, "order_type", "")).split(".")[-1].lower(),
        "limit_price": float(getattr(order, "limit_price", 0.0) or 0.0),
        "stop_price": float(getattr(order, "stop_price", 0.0) or 0.0),
        "status": str(getattr(order, "status", "")).split(".")[-1].upper(),
    }
