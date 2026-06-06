from __future__ import annotations

from typing import Dict, Type
from .market.yfinance import YFinanceConnector
from .base import BaseMarketConnector

MARKET_CONNECTOR_REGISTRY: Dict[str, Type[BaseMarketConnector]] = {
    "yfinance": YFinanceConnector,
}

def register_market_connector(name: str, cls: Type[BaseMarketConnector]):
    MARKET_CONNECTOR_REGISTRY[name] = cls
