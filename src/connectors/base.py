from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, timezone, timedelta

from ..core.interfaces import MarketDataConnector, MarketDataRequest, SentimentDataConnector, SentimentDataRequest
from ..data.duckdb_store import DataStore

logger = logging.getLogger(__name__)

class BaseConnector(ABC):
    def __init__(self, config: Dict[str, Any], store: Optional[DataStore] = None):
        self.config = config
        self.store = store or DataStore()
        self.name = self.__class__.__name__.lower().replace("connector", "")

    def _get_cache_ttl(self, category: str) -> int:
        if "intraday" in category:
            return 900  # 15 mins
        return 86400  # 1 day

class BaseMarketConnector(BaseConnector, MarketDataConnector):
    def fetch_bars(self, request: MarketDataRequest) -> Dict[str, pd.DataFrame]:
        results = {}
        symbols_to_fetch = []

        # 1. Check Cache
        if not request.force_refresh:
            for symbol in request.symbols:
                # Approximate start/end if not provided
                start = request.start or (datetime.now(timezone.utc) - timedelta(days=7))
                end = request.end or datetime.now(timezone.utc)
                
                cached_df = self.store.get_market_bars(symbol, request.timeframe, self.name, start, end)
                if not cached_df.empty:
                    # Logic to check if we have enough bars... simplified for now
                    results[symbol.upper()] = cached_df
                else:
                    symbols_to_fetch.append(symbol)
        else:
            symbols_to_fetch = request.symbols

        if not symbols_to_fetch:
            return results

        # 2. Execute Actual Fetch
        try:
            new_request = MarketDataRequest(
                symbols=symbols_to_fetch,
                timeframe=request.timeframe,
                category=request.category,
                lookback_bars=request.lookback_bars,
                start=request.start,
                end=request.end,
                provider=self.name,
                extra=request.extra
            )
            fetched_data = self._execute_fetch(new_request)
            
            # 3. Cache results
            ttl = self._get_cache_ttl(request.category)
            for symbol, df in fetched_data.items():
                self.store.write_market_bars(symbol, request.timeframe, request.category, self.name, df, ttl_seconds=ttl)
                results[symbol.upper()] = df
                
            return results
        except Exception as e:
            logger.error(f"Error fetching data from {self.name}: {e}")
            raise

    @abstractmethod
    def _execute_fetch(self, request: MarketDataRequest) -> Dict[str, pd.DataFrame]:
        pass
