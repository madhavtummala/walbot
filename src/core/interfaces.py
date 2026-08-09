from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
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

@dataclass(frozen=True)
class PortfolioSnapshot:
    """What is currently held, independent of where it is held.

    Algorithms take this rather than a ``Brokerage`` so the position-aware half of a strategy
    runs identically against a live account, a backtest's simulated portfolio, or the local
    paper brokerage when no broker is configured.
    """

    positions: Dict[str, float] = field(default_factory=dict)
    equity: float = 0.0

    def weights(self, latest_prices: Dict[str, float]) -> Dict[str, float]:
        """Current holdings expressed as portfolio weights, skipping unpriced symbols."""
        if self.equity <= 0:
            return {}
        return {
            symbol: (shares * latest_prices[symbol]) / self.equity
            for symbol, shares in self.positions.items()
            if latest_prices.get(symbol, 0.0) > 0
        }


@dataclass(frozen=True)
class AlgorithmResult:
    """Step 1 output: what the algorithm proposes from market data alone.

    Deliberately free of positions, equity, and brokerage state so it can be produced without
    an account, handed to an agent for validation, and passed back into step 2 unchanged.
    ``latest_prices`` travels with the result because step 2 does no data fetching of its own.
    """

    strategy: str
    target_weights: Dict[str, float] = field(default_factory=dict)
    signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    latest_prices: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class SignalView:
    """Dashboard-ready view of an algorithm's current signals."""

    leaders: List[Dict[str, Any]] = field(default_factory=list)
    summary: List[Dict[str, str]] = field(default_factory=list)
    wired: bool = True

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
    quantity: float  # Whole shares unless the brokerage supports fractional quantities.
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

class Brokerage(ABC):
    #: Whether this brokerage accepts fractional share quantities. Defaults to ``False`` so a
    #: brokerage that has not opted in is sized in whole shares only.
    supports_fractional_shares: bool = False

    @abstractmethod
    def get_account_state(self) -> Dict[str, Any]:
        """Return equity, cash, and buying power."""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, float]:
        """Return current share counts by symbol (fractional if the brokerage supports it)."""
        pass

    @abstractmethod
    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        """Execute a trade and return the order result."""
        pass

    @abstractmethod
    def cancel_all_orders(self) -> None:
        pass

    def is_market_open(self) -> bool:
        """Return whether the market is currently open for trading."""
        return bool(self.get_account_state().get("is_market_open", False))

    def validate_short_sale_feasibility(
        self, symbol: str, quantity: float, target_shares: float, latest_price: float
    ) -> Dict[str, Any]:
        """Check whether the brokerage allows a short sale for the given symbol and quantity.

        Returns a dict with ``"shortable"`` (bool) and ``"reason"`` (str).
        The default assumes shorting is allowed.
        """
        return {"shortable": False, "reason": "short sale not confirmed by brokerage"}

class AlgorithmPlugin(ABC):
    algorithm_id: str = ""

    def requirements(self, config: Any, current_positions: Dict[str, int]) -> AlgorithmRequirements:
        return AlgorithmRequirements()

    @abstractmethod
    def generate_signals(self, context: AlgorithmContext) -> List[Dict[str, Any]]:
        pass

    def decide(self, context: AlgorithmContext) -> AlgorithmDecision:
        return AlgorithmDecision()

    def analyze(self, context: AlgorithmContext) -> AlgorithmDecision:
        """Step 1: propose weights from market data alone.

        ``context.positions`` and ``context.equity`` are empty here by construction. Anything
        that depends on what is currently held belongs in ``refine``. Defaults to ``decide``
        so an algorithm with no position-aware logic needs no changes.
        """
        return self.decide(context)

    def refine(
        self,
        target_weights: Dict[str, float],
        signals: Dict[str, Dict[str, Any]],
        snapshot: PortfolioSnapshot,
        latest_prices: Dict[str, float],
        config: Any,
    ) -> Dict[str, float]:
        """Step 2: adjust proposed weights for what is already held.

        This is where hysteresis lives -- stickiness, turnover thresholds, per-trade minimums,
        exposure caps. ``target_weights`` may have been edited by a reviewing agent, so honour
        them as the intent rather than re-deriving from ``signals``. Defaults to a passthrough.
        """
        return dict(target_weights)

    def sizing(self, config: Any) -> Dict[str, float]:
        """Order-sizing knobs for step 2, defaulting to the account config.

        An algorithm that already bakes cash into its weights (via a gross-exposure cap) must
        override ``cash_buffer`` to 0, or the buffer is applied twice and it under-invests.
        """
        return {
            "cash_buffer": float(getattr(config, "cash_buffer", 0.0) or 0.0),
            "min_trade_dollars": float(getattr(config, "min_trade_dollars", 0.0) or 0.0),
            "rebalance_threshold": float(getattr(config, "rebalance_threshold", 0.0) or 0.0),
        }

    def signal_view(self, config: Any, *, data_client: Any = None) -> SignalView:
        """Return dashboard-ready signals for this algorithm. Default: not wired."""
        return SignalView(wired=False)
