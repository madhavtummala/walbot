from __future__ import annotations

from typing import Dict, Any, List
from ..base import BaseBrokerage
from ...core.interfaces import OrderRequest
from src.brokerages.alpaca_client import (
    DIVIDEND_ACTIVITY_TYPES,
    create_trading_client,
    get_account_activities,
    get_position_marks,
    get_positions,
    submit_market_order,
    submit_option_limit_order,
    is_market_open,
    validate_short_sale_feasibility as _alpaca_short_check,
)

class AlpacaBrokerage(BaseBrokerage):
    supports_fractional_shares = True

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
        if request.order_type == "market":
            order = submit_market_order(
                self.client,
                symbol=request.symbol,
                side=request.action,
                qty=request.quantity
            )
        elif request.order_type == "limit":
            position_intent = (request.extra or {}).get("position_intent", "buy_to_open")
            time_in_force = (request.extra or {}).get("time_in_force", "day")
            order = submit_option_limit_order(
                self.client,
                symbol=request.symbol,
                side=request.action,
                qty=request.quantity,
                limit_price=request.limit_price,
                position_intent=position_intent,
                time_in_force=time_in_force,
            )
        else:
            raise NotImplementedError(f"Order type {request.order_type} not implemented for Alpaca")

        return {
            "order_id": str(order.id),
            "client_order_id": str(order.client_order_id),
            "status": str(order.status),
            "symbol": str(order.symbol),
            "qty": int(float(order.qty))
        }

    def cancel_all_orders(self) -> None:
        self.client.cancel_orders()

    def validate_short_sale_feasibility(
        self, symbol: str, quantity: int, target_shares: int, latest_price: float
    ) -> Dict[str, Any]:
        return _alpaca_short_check(self.client, symbol, quantity, target_shares, latest_price)
