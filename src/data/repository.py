from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Type
import pandas as pd
from ..core.interfaces import MarketDataConnector, MarketDataRequest, SentimentDataConnector, SentimentDataRequest
from ..connectors.base import BaseMarketConnector

logger = logging.getLogger(__name__)

class DataRepository:
    def __init__(self, config: Dict[str, Any], connector_registry: Dict[str, Type[BaseMarketConnector]]):
        self.config = config
        self.registry = connector_registry
        self.connectors: Dict[str, BaseMarketConnector] = {}

    def _category_config(self, category: str) -> dict[str, Any]:
        data_sources = self.config.get("data_sources", {})
        if not isinstance(data_sources, dict):
            return {}
        section = data_sources.get(category, {})
        if not section and category == "market_data":
            section = data_sources.get("eod_market_data", {})
        return section if isinstance(section, dict) else {}

    def _provider_configs(self, category: str) -> dict[str, Any]:
        providers = self._category_config(category).get("providers", {})
        return providers if isinstance(providers, dict) else {}

    def _get_connector(self, name: str, category: str) -> BaseMarketConnector:
        if name not in self.connectors:
            if name not in self.registry:
                raise ValueError(f"Connector {name} not registered")
            
            provider_config = self._provider_configs(category).get(name, {})
            self.connectors[name] = self.registry[name](provider_config)
            
        return self.connectors[name]

    def fetch_market_bars(self, request: MarketDataRequest) -> Dict[str, pd.DataFrame]:
        if request.provider:
            provider_order = [request.provider]
        else:
            provider_order = list(self._provider_configs(request.category))

        if not provider_order:
            logger.warning(f"No providers configured for category {request.category}")
            return {}

        errors = []
        for provider_name in provider_order:
            try:
                connector = self._get_connector(provider_name, request.category)
                return connector.fetch_bars(request)
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}. Trying next...")
                errors.append(f"{provider_name}: {e}")
                continue

        logger.error(f"All providers failed for {request.category}: {errors}")
        return {}
