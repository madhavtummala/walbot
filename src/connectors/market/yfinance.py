from __future__ import annotations

import logging
import pandas as pd
import yfinance as yf
from typing import Dict
from ..base import BaseMarketConnector
from ..utils import normalize_intraday_frame, filter_bar_range
from ...core.interfaces import MarketDataRequest

logger = logging.getLogger(__name__)

class YFinanceConnector(BaseMarketConnector):
    """Yahoo Finance connector implementing standardized fetch logic."""

    def _execute_fetch(self, request: MarketDataRequest) -> Dict[str, pd.DataFrame]:
        results = {}
        interval = "1d" if request.category == "eod_market_data" or request.timeframe == "1d" else request.timeframe
        
        # Determine period or use start/end dates
        lookback_bars = request.lookback_bars or 30
        period = f"{max(int(lookback_bars * 2), 30)}d"
        
        for symbol in [s.upper() for s in request.symbols]:
            try:
                kwargs = {"interval": interval, "auto_adjust": False, "progress": False, "threads": False}
                if request.start is not None or request.end is not None:
                    kwargs.update({"start": request.start, "end": request.end})
                else:
                    kwargs["period"] = period
                
                raw = yf.download(symbol, **kwargs)
                bars = filter_bar_range(
                    normalize_intraday_frame(raw),
                    request.start,
                    request.end,
                )
                
                if request.lookback_bars:
                    bars = bars.tail(request.lookback_bars).reset_index(drop=True)
                
                results[symbol] = bars
            except Exception as exc:
                logger.warning("Skipping yfinance bars for %s after provider error: %s", symbol, exc)
                results[symbol] = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        
        return results
