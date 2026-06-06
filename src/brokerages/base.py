from __future__ import annotations

from typing import Dict, Any, Optional
from ..core.interfaces import Brokerage, OrderRequest

class BaseBrokerage(Brokerage):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__.lower().replace("brokerage", "")

    def get_account_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_positions(self) -> Dict[str, int]:
        raise NotImplementedError

    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        raise NotImplementedError

    def cancel_all_orders(self) -> None:
        raise NotImplementedError
