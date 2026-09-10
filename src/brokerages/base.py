from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from ..core.interfaces import Brokerage, OrderRequest

logger = logging.getLogger(__name__)


class BaseBrokerage(Brokerage):
    """What every venue provides: account state, positions, and order submission.

    Deliberately thin -- no strategy, sizing, or market data beyond what an order needs.
    """

    #: Whether this venue accepts fractional quantities. Declared here rather than left as an
    #: undeclared attribute, so a provider that forgets it fails loudly instead of silently
    #: becoming whole-share.
    supports_fractional_shares: bool = False

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__.lower().replace("brokerage", "")

    def cash_equivalent_symbols(self) -> List[str]:
        """The account's configured cash-like holdings, in preference order."""
        return [str(symbol).upper() for symbol in (getattr(self.config, "cash_equivalents", None) or [])]

    def get_account_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_positions(self) -> Dict[str, float]:
        raise NotImplementedError

    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        raise NotImplementedError

    def cancel_all_orders(self) -> None:
        raise NotImplementedError

    # -- shared by concrete brokerages ---------------------------------------------------

    @staticmethod
    def _order_receipt(order_id: Any, client_order_id: Any, status: Any, symbol: str, qty: Any) -> Dict[str, Any]:
        """The dict shape ``submit_order``/``replace_order`` promise."""
        return {
            "order_id": str(order_id),
            "client_order_id": str(client_order_id or ""),
            "status": str(status),
            "symbol": str(symbol),
            "qty": qty,
        }

    def _cancel_ignoring_gone(self, order_id: str, cancel: Callable[[], None]) -> bool:
        """Cancel via ``cancel``; already-gone (filled or cancelled) is success, not failure.

        A reconciler works from a snapshot seconds old, so racing a fill is routine. Returns
        whether ``cancel`` succeeded, for callers that log on success too.
        """
        try:
            cancel()
            return True
        except Exception as exc:
            logger.info(
                "%s order %s was not cancellable (already filled or gone): %s",
                self.name.capitalize(), order_id, exc,
            )
            return False

    @staticmethod
    def _sorted_by_market_value(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Position rows, largest notional first."""
        return sorted(rows, key=lambda row: abs(row["market_value"]), reverse=True)

    @staticmethod
    def _sorted_by_date_desc(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Activity rows, most recent first."""
        return sorted(rows, key=lambda row: row["date"], reverse=True)
