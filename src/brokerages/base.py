from __future__ import annotations

from typing import Dict, Any, List
from ..core.interfaces import Brokerage, OrderRequest

class BaseBrokerage(Brokerage):
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
