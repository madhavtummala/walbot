from __future__ import annotations

from typing import Any, Dict, List

from ..core.interfaces import Brokerage, OrderRequest


class BaseBrokerage(Brokerage):
    """What every venue has to provide, and the one capability the order path branches on.

    Deliberately thin. A brokerage answers questions about an account and submits orders; it
    holds no strategy, no sizing rule and no market data beyond what an order needs.
    """

    #: Whether this venue accepts fractional quantities. Declared here rather than left to each
    #: provider, because it is the most consequential brokerage fact in the system: it sets the
    #: smallest trade that can reach the market, which is what makes a $100/month DCA budget
    #: against a $600 share accrue for six months before it can trade at all. It used to be an
    #: undeclared attribute the callers read with ``getattr(broker, ..., False)`` -- so a
    #: provider that forgot it silently became whole-share, the safe-looking default that is
    #: wrong for two of the three venues here.
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
