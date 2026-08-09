from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import uuid4

from ..base import BaseBrokerage
from ...core.interfaces import OrderRequest
from ...data.state_store import load_state, save_state

logger = logging.getLogger(__name__)

STATE_KEY = "paper_brokerage"
DEFAULT_STARTING_CASH = 100_000.0


class PaperBrokerage(BaseBrokerage):
    """A local, fill-immediately brokerage backed by the state store.

    Lets an algorithm be run, refined, and "traded" before any real broker is configured, and
    gives the position-aware half of a strategy something to exercise in tests without
    credentials or a network. Fills are assumed complete at the price supplied with the order,
    which is the same simplification a backtest makes.
    """

    supports_fractional_shares = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        starting_cash = float(getattr(config, "paper_starting_cash", DEFAULT_STARTING_CASH) or DEFAULT_STARTING_CASH)
        self.state = load_state(STATE_KEY, {"cash": starting_cash, "positions": {}, "prices": {}})

    def _save(self) -> None:
        save_state(STATE_KEY, self.state)

    def _market_value(self) -> float:
        prices = self.state.get("prices", {})
        return sum(shares * float(prices.get(symbol, 0.0)) for symbol, shares in self.state["positions"].items())

    def get_account_state(self) -> Dict[str, Any]:
        cash = float(self.state.get("cash", 0.0))
        equity = cash + self._market_value()
        return {
            "equity": equity,
            "cash": cash,
            "buying_power": max(cash, 0.0),
            "is_market_open": True,
        }

    def get_positions(self) -> Dict[str, float]:
        return {symbol: shares for symbol, shares in self.state.get("positions", {}).items() if shares}

    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        price = float((request.extra or {}).get("latest_price") or self.state.get("prices", {}).get(request.symbol, 0.0))
        if price <= 0:
            raise ValueError(f"Paper brokerage needs a price for {request.symbol} to fill an order")

        signed = request.quantity if request.action == "buy" else -request.quantity
        positions = dict(self.state.get("positions", {}))
        positions[request.symbol] = positions.get(request.symbol, 0.0) + signed
        if abs(positions[request.symbol]) < 1e-9:
            positions.pop(request.symbol, None)

        self.state["positions"] = positions
        self.state["cash"] = float(self.state.get("cash", 0.0)) - signed * price
        self.state.setdefault("prices", {})[request.symbol] = price
        self._save()

        logger.info("Paper fill: %s %s qty=%s @ %.2f", request.action, request.symbol, request.quantity, price)
        return {
            "order_id": f"paper-{uuid4().hex[:8]}",
            "client_order_id": request.client_order_id or "",
            "status": "filled",
            "symbol": request.symbol,
            "qty": request.quantity,
        }

    def cancel_all_orders(self) -> None:
        """No-op: paper orders fill immediately, so nothing is ever open."""

    def mark_prices(self, latest_prices: Dict[str, float]) -> None:
        """Update marks so equity reflects current prices rather than last fill prices."""
        self.state.setdefault("prices", {}).update({s: float(p) for s, p in latest_prices.items() if p > 0})
        self._save()

    def validate_short_sale_feasibility(
        self, symbol: str, quantity: float, target_shares: float, latest_price: float
    ) -> Dict[str, Any]:
        return {"shortable": True, "reason": "paper brokerage allows shorts"}
