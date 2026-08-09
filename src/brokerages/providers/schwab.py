from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd

from ..base import BaseBrokerage
from ..schwab_client import TRADER_BASE, SchwabSession, account_hash
from ...core.interfaces import OrderRequest

logger = logging.getLogger(__name__)

_INSTRUCTIONS = {
    ("buy", False): "BUY",
    ("sell", False): "SELL",
    ("buy", True): "BUY_TO_COVER",
    ("sell", True): "SELL_SHORT",
}


class SchwabBrokerage(BaseBrokerage):
    """Charles Schwab Trader API.

    Schwab does not accept fractional share quantities on equity orders, so sizing is whole
    shares only -- see ``supports_fractional_shares``.
    """

    supports_fractional_shares = False

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

    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        if request.order_type != "market":
            raise NotImplementedError(f"Order type {request.order_type} is not implemented for Schwab")

        quantity = float(request.quantity)
        if quantity != int(quantity):
            raise ValueError(
                f"Schwab does not accept fractional quantities (got {quantity} for {request.symbol})"
            )

        intent = str((request.extra or {}).get("position_intent", ""))
        shorting = "short" in intent or "cover" in intent
        instruction = _INSTRUCTIONS[(request.action.lower(), shorting)]

        payload = {
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": int(quantity),
                    "instrument": {"symbol": request.symbol.upper(), "assetType": "EQUITY"},
                }
            ],
        }

        logger.info("Submitting Schwab %s order for %s qty=%s", instruction, request.symbol, int(quantity))
        response = self.session.post(
            f"{TRADER_BASE}/accounts/{self.account_hash}/orders",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        return {
            "order_id": _order_id_from(response),
            "client_order_id": request.client_order_id or "",
            "status": "accepted",
            "symbol": request.symbol.upper(),
            "qty": int(quantity),
        }

    def cancel_all_orders(self) -> None:
        orders = self.session.get(
            f"{TRADER_BASE}/accounts/{self.account_hash}/orders",
            params={"status": "WORKING"},
        ) or []
        for order in orders:
            order_id = order.get("orderId")
            if order_id is None:
                continue
            self.session.delete(f"{TRADER_BASE}/accounts/{self.account_hash}/orders/{order_id}")
            logger.info("Cancelled Schwab order %s", order_id)

    def is_market_open(self) -> bool:
        """Whether the equity market is open, per ``/marketdata/v1/markets``."""
        from ..schwab_client import MARKETDATA_BASE

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


def _order_id_from(response: Any) -> str:
    """Schwab returns 201 with an empty body; the new order id is the last Location segment."""
    if isinstance(response, dict):
        location = str(response.get("location") or "")
        if location:
            return location.rstrip("/").rsplit("/", 1)[-1]
        if response.get("orderId") is not None:
            return str(response["orderId"])
    return "unknown"
