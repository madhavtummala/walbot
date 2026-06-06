from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

@dataclass(frozen=True)
class MarketDataRequest:
    symbols: List[str]
    timeframe: str
    category: str = "market_data"  # intraday_market_data, eod_market_data
    lookback_bars: Optional[int] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    provider: Optional[str] = None
    force_refresh: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SentimentDataRequest:
    symbols: List[str]
    lookback_days: Optional[int] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    provider: Optional[str] = None
    force_refresh: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AlgorithmContext:
    config: Any
    bars_by_symbol: Dict[str, pd.DataFrame] = field(default_factory=dict)
    sentiment_by_symbol: Dict[str, pd.DataFrame] = field(default_factory=dict)
    positions: Dict[str, int] = field(default_factory=dict)
    latest_prices: Dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    account_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    extra: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AlgorithmRequirements:
    price_symbols: List[str] = field(default_factory=list)
    daily_lookback_days: Optional[int] = None
    daily_ma_days: int = 0
    daily_extra_buffer_days: int = 0
    include_latest_daily: bool = True
    needs_sentiment: bool = False
    paper_only: bool = False

@dataclass(frozen=True)
class AlgorithmDecision:
    target_weights: Dict[str, float] = field(default_factory=dict)
    signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    cash_buffer: float = 0.0
    min_trade_dollars: float = 0.0
    rebalance_threshold: float = 0.0

class MarketDataConnector(ABC):
    @abstractmethod
    def fetch_bars(self, request: MarketDataRequest) -> Dict[str, pd.DataFrame]:
        pass

class SentimentDataConnector(ABC):
    @abstractmethod
    def fetch_sentiment(self, request: SentimentDataRequest) -> List[Dict[str, Any]]:
        pass

@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    action: str  # BUY, SELL
    quantity: int
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

class Brokerage(ABC):
    @abstractmethod
    def get_account_state(self) -> Dict[str, Any]:
        """Return equity, cash, and buying power."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, int]:
        """Return current share counts by symbol."""
        pass

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        """Execute a trade and return the order result."""
        pass

    @abstractmethod
    def cancel_all_orders(self) -> None:
        pass

class AlgorithmPlugin(ABC):
    algorithm_id: str = ""

    def requirements(self, config: Any, current_positions: Dict[str, int]) -> AlgorithmRequirements:
        return AlgorithmRequirements()

    @abstractmethod
    def generate_signals(self, context: AlgorithmContext) -> List[Dict[str, Any]]:
        pass

    def decide(self, context: AlgorithmContext) -> AlgorithmDecision:
        return AlgorithmDecision()
