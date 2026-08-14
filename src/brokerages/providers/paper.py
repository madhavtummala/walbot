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


def _state_key(account_id: str) -> str:
    return f"{STATE_KEY}:{account_id}" if account_id else STATE_KEY


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
        self.state_key = _state_key(str(getattr(config, "account_id", "") or ""))
        # Books are per account, so two local accounts are two separate portfolios rather
        # than one shared pile that neither of them explains.
        self.state = load_state(self.state_key, None)
        if self.state is None:
            # Fall back to the single unnamespaced book written before accounts were a thing.
            self.state = load_state(STATE_KEY, {"cash": starting_cash, "positions": {}, "prices": {}})

    def _save(self) -> None:
        save_state(self.state_key, self.state)

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
        held = positions.get(request.symbol, 0.0)
        positions[request.symbol] = held + signed
        if abs(positions[request.symbol]) < 1e-9:
            positions.pop(request.symbol, None)

        self._update_basis(request.symbol, held=held, signed=signed, price=price)
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

    def _update_basis(self, symbol: str, *, held: float, signed: float, price: float) -> None:
        """Track average entry price, so the book can report P/L rather than just holdings.

        Adding to a position averages the new fill in; reducing one leaves the basis alone,
        because a partial sale does not change what the remaining shares cost. Closing out
        (or crossing through zero) starts fresh from the crossing price.
        """
        basis = dict(self.state.get("basis", {}))
        after = held + signed
        if abs(after) < 1e-9:
            basis.pop(symbol, None)
        elif held == 0 or (held > 0) != (after > 0):
            basis[symbol] = price
        elif (signed > 0) == (held > 0):
            previous = float(basis.get(symbol, price))
            basis[symbol] = ((abs(held) * previous) + (abs(signed) * price)) / abs(after)
        self.state["basis"] = basis

    def book(self) -> Dict[str, Any]:
        """Holdings with marks and cost basis, for callers that report rather than trade."""
        prices = self.state.get("prices", {})
        basis = self.state.get("basis", {})
        rows = []
        for symbol, shares in self.get_positions().items():
            price = float(prices.get(symbol, 0.0))
            entry = float(basis.get(symbol, 0.0))
            rows.append(
                {
                    "symbol": symbol,
                    "qty": float(shares),
                    "avg_entry_price": entry,
                    "market_value": float(shares) * price,
                    "unrealized_pl": (price - entry) * float(shares) if entry and price else 0.0,
                    "unrealized_plpc": (price / entry - 1.0) if entry and price else 0.0,
                }
            )
        rows.sort(key=lambda row: abs(row["market_value"]), reverse=True)
        return {**self.get_account_state(), "rows": rows}

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
