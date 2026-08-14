from __future__ import annotations

from typing import Dict, Any, List, Optional
from ..base import BaseBrokerage
from ...core.interfaces import OrderRequest
from src.brokerages.alpaca_client import (
    create_trading_client,
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
        # Assuming config is a dict with keys like 'alpaca_api_key', etc.
        # or it might be a Config object. alpaca_client functions expect Config object or similar.
        # For now, let's pass config as is.
        self.client = create_trading_client(config)

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

    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        if request.order_type == "market":
            order = submit_market_order(
                self.client,
                symbol=request.symbol,
                side=request.action,
                qty=request.quantity
            )
        elif request.order_type == "limit" and "to_open" in (request.extra or {}).get("position_intent", ""):
            # Handle options/limit orders if needed
            order = submit_option_limit_order(
                self.client,
                symbol=request.symbol,
                side=request.action,
                qty=request.quantity,
                limit_price=request.limit_price,
                position_intent=request.extra.get("position_intent")
            )
        else:
            # Fallback or other order types
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
