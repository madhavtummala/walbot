from __future__ import annotations

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


class AlpacaBrokerage(BaseBrokerage):
    supports_fractional_shares = True
    supports_options = True

    #: Alpaca rejects OCO/bracket orders on options; a bracket must be split into two
    #: independent orders. See ``broker_supports_oco``.
    supports_oco = False

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = create_trading_client(config)
        self._config = config

    @property
    def is_paper(self) -> bool:
        """Alpaca separates paper from live by endpoint, so the base URL is the whole answer.

        Read from the config this brokerage actually authenticated with, rather than from a
        config some other layer happens to hold -- which is the distinction that makes this
        trustworthy where the old string sniff was not.
        """
        return "paper-api.alpaca.markets" in str(getattr(self._config, "alpaca_base_url", "") or "")

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
        return self._sorted_by_date_desc(rows)

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
                "current_price": float(getattr(position, "current_price", 0.0) or 0.0),
                "market_value": float(getattr(position, "market_value", 0.0) or 0.0),
                "unrealized_pl": float(getattr(position, "unrealized_pl", 0.0) or 0.0),
                "unrealized_plpc": float(getattr(position, "unrealized_plpc", 0.0) or 0.0),
            })
        return self._sorted_by_market_value(rows)

    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        order = self.client.submit_order(order_data=build_order_request(request))
        return _order_result(order)

    def get_orders(self, status: str = "WORKING") -> List[Dict[str, Any]]:
        """Open orders, in the shape :meth:`Brokerage.get_orders` promises.

        Alpaca calls the resting set ``open``; the interface speaks Schwab's word ``WORKING``.
        """
        wanted = "open" if str(status).upper() in ("WORKING", "OPEN", "") else str(status).lower()
        return [_order_row(order) for order in get_open_orders(self.client, wanted)]

    def cancel_order(self, order_id: str) -> None:
        self._cancel_ignoring_gone(order_id, lambda: self.client.cancel_order_by_id(order_id))

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
    return BaseBrokerage._order_receipt(
        order.id, order.client_order_id, getattr(order, "status", ""), order.symbol,
        int(float(order.qty or 0)),
    )


def _order_row(order: Any) -> Dict[str, Any]:
    """One resting order, in the shape ``get_orders`` promises.

    The symbol is re-spelled to the padded OSI form the rest of the codebase uses, so an
    option order compares equal to the contract that requested it.
    """
    symbol = str(getattr(order, "symbol", "")).upper()
    option = is_osi_symbol(symbol)
    return {
        "order_id": str(order.id),
        # Always empty: Alpaca refuses complex orders on options, so nothing here has a parent.
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
