from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

@dataclass(frozen=True)
class AlgorithmContext:
    """Everything ``analyze`` is allowed to read.

    ``analyze`` must be a pure function of this context: no fetching, no clock, no state
    store. That is what lets the same call be driven by the live runner against today's data
    and by the backtester against a point-in-time slice, instead of the backtester having to
    reimplement the strategy -- which is what it used to do, and what let the two drift.
    """

    config: Any
    #: Daily OHLCV per symbol, already truncated to what the algorithm may see. Kept distinct
    #: because the moving-average and regime gates are statements about daily closes.
    bars_by_symbol: Dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Timestamped OHLCV per symbol covering ``AlgorithmRequirements.history_lookback_minutes``,
    #: at whatever resolution was available -- possibly blending fine bars recently with
    #: coarser ones further back. Read it through ``src/data/bars.py``, never by position.
    history_bars_by_symbol: Dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Per-symbol sentiment score, plus the universe-wide average.
    sentiment_scores: Dict[str, float] = field(default_factory=dict)
    market_sentiment: float = 0.0
    positions: Dict[str, int] = field(default_factory=dict)
    latest_prices: Dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    account_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgorithmRequirements:
    """What data the algorithm needs, so the caller can load it once and hand it over.

    Declaring the need here rather than fetching inside ``analyze`` is what makes an
    algorithm replayable: the backtester satisfies the same declaration from cached history.
    """

    price_symbols: List[str] = field(default_factory=list)
    daily_lookback_days: Optional[int] = None
    daily_ma_days: int = 0
    daily_extra_buffer_days: int = 0
    include_latest_daily: bool = True
    #: How far back the algorithm needs timestamped bars, in minutes the market was open --
    #: 4800 is about twelve sessions, not three and a bit days. Stated in minutes rather than
    #: bars because a bar count only means a span of time once the grid is known, and the grid
    #: is now the feed's business: Schwab serves 1/5/10/15/30-minute bars where yfinance served
    #: 15. The lookback resolves against whichever resolution the cache holds -- down to daily,
    #: at the far end of a long window. See ``src/data/bars.py``.
    history_lookback_minutes: int = 0
    #: The finest grid the algorithm would like, when it has a preference. 0 takes the
    #: configured default. A provider that cannot serve it supplies its nearest coarser grid
    #: rather than failing, because the horizons no longer depend on getting an exact one.
    preferred_bar_minutes: int = 0
    needs_sentiment: bool = False
    paper_only: bool = False

#: An algorithm's intent list is the complete portfolio: a held symbol absent from it is exited.
MODE_TARGET = "target"
#: Only the listed symbols are touched; every other holding is left alone.
MODE_INCREMENTAL = "incremental"

INTENT_KINDS = ("weight", "notional", "shares")


@dataclass(frozen=True)
class Intent:
    """One symbol's worth of what an algorithm wants, in whichever unit it thinks in.

    Allocation strategies speak ``weight`` ("hold 44% BBC"); DCA speaks ``notional``
    ("buy $200 of VTI"). Step 2 resolves either against the portfolio, which is what lets
    DCA be an ordinary algorithm rather than a separate submission path.
    """

    symbol: str
    kind: str = "weight"
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in INTENT_KINDS:
            raise ValueError(f"Unknown intent kind: {self.kind!r} (expected one of {INTENT_KINDS})")


def intents_from_weights(target_weights: Dict[str, float]) -> List[Intent]:
    """Weight-mode convenience: the representation every allocation algorithm already produces."""
    return [Intent(symbol=symbol, kind="weight", value=float(weight)) for symbol, weight in target_weights.items()]


def weights_from_intents(intents: List[Intent]) -> Dict[str, float]:
    """Inverse of :func:`intents_from_weights`, skipping intents that are not weights."""
    return {intent.symbol: intent.value for intent in intents if intent.kind == "weight"}


@dataclass(frozen=True)
class AlgorithmDecision:
    target_weights: Dict[str, float] = field(default_factory=dict)
    signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    min_trade_dollars: float = 0.0
    rebalance_threshold: float = 0.0
    #: Left empty by weight-based algorithms, which populate ``target_weights`` instead.
    intents: List[Intent] = field(default_factory=list)
    mode: str = MODE_TARGET

    def resolved_intents(self) -> List[Intent]:
        """The decision as intents, deriving them from ``target_weights`` when unset."""
        if self.intents:
            return list(self.intents)
        return intents_from_weights(self.target_weights)

@dataclass(frozen=True)
class PortfolioSnapshot:
    """What is currently held, independent of where it is held.

    Algorithms take this rather than a ``Brokerage`` so the position-aware half of a strategy
    runs identically against a live account, a backtest's simulated portfolio, or the local
    paper brokerage when no broker is configured.
    """

    positions: Dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    #: Settled cash, and what the broker will actually let this account spend right now. The
    #: two differ wherever margin does, which is why order funding reads ``buying_power`` and
    #: never infers it from equity: sizing a portfolio and paying for it are different sums.
    cash: float = 0.0
    buying_power: float = 0.0

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
    intents: List[Intent] = field(default_factory=list)
    mode: str = MODE_TARGET

    def __post_init__(self) -> None:
        # ``intents`` is what step 2 consumes, but weight-mode callers (the MCP agent, the
        # dashboard) construct results from ``target_weights``. Keep the two consistent
        # whichever one was supplied so neither view of the result can go stale.
        if not self.intents and self.target_weights:
            object.__setattr__(self, "intents", intents_from_weights(self.target_weights))
        elif self.intents and not self.target_weights:
            object.__setattr__(self, "target_weights", weights_from_intents(self.intents))


@dataclass(frozen=True)
class SignalView:
    """Dashboard-ready view of an algorithm's current signals."""

    leaders: List[Dict[str, Any]] = field(default_factory=list)
    summary: List[Dict[str, str]] = field(default_factory=list)
    wired: bool = True

@dataclass(frozen=True)
class CashDividend:
    """One cash distribution, as the event it actually is.

    Deliberately not a yield or an adjustment factor. A yield is a derived, point-in-time
    summary; back-projecting one across history invents payments that were never made, which
    matters most for the funds whose whole return *is* the distribution -- SGOV and BIL pay
    the T-bill rate, and that went from ~0% in 2022 to over 5% and back to ~4%.

    Two dates, because they answer different questions. ``ex_date`` decides *entitlement*:
    hold the shares before it and the payment is yours. ``payable_date`` is when the cash
    actually lands, which is what a ledger should credit -- typically a few days later, and
    that gap is real cash drag rather than a rounding detail.
    """

    symbol: str
    ex_date: date
    amount: float
    payable_date: Optional[date] = None
    record_date: Optional[date] = None
    special: bool = False
    source: str = ""


class DividendProvider(ABC):
    """A source of cash-distribution history.

    Separate from the market-data providers on purpose. Dividends are corporate actions, not
    prices: they are discrete, they are append-only once published, and they must never be
    folded into ``market_bars``. Keeping the two apart is what lets the price cache stay a
    faithful record of what the market printed while the ledger still books real income.
    """

    #: Provider name as it appears in configuration and in ``CashDividend.source``.
    name: str = ""

    @abstractmethod
    def fetch_dividends(
        self, symbols: List[str], start: date, end: date
    ) -> List[CashDividend]:
        """Every cash distribution for ``symbols`` with an ex-date in ``[start, end]``.

        Ordering is not guaranteed. A symbol that pays nothing simply contributes no rows,
        which is a fact about the symbol rather than a failure -- GLD and IBIT never appear.
        """
        raise NotImplementedError


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

    def get_buying_power(self) -> float:
        """What this account may spend right now, per the broker's own accounting.

        Read rather than derived. Settlement rules, margin, and pending orders all move this
        number, and the broker is the only party that knows the answer -- reconstructing it
        from equity and a settlement model can only drift away from what the broker will
        actually accept.
        """
        return float(self.get_account_state().get("buying_power", 0.0) or 0.0)

    def get_marks(self, symbols: List[str]) -> Dict[str, float]:
        """Last known price per symbol, from the broker's own position marks.

        Needed because order funding has to value holdings the running algorithm may never
        price: DCA buys VTI and has no reason to ever look up SGOV, but SGOV is exactly what
        would fund the buy. Defaults to empty, which leaves those holdings unusable for
        funding rather than mispriced.
        """
        return {}

    def cash_equivalent_symbols(self) -> List[str]:
        """Symbols this account treats as cash. See ``BaseBrokerage`` for the config-backed one."""
        return []

    def get_cash_equivalents(self) -> Dict[str, Dict[str, float]]:
        """Held cash-equivalent positions with marks, as ``{symbol: {shares, price, value}}``.

        What order funding may liquidate when buying power alone cannot cover a batch.
        Unpriced holdings are dropped rather than valued at zero, so a missing mark shows up
        as "could not be funded" instead of quietly shrinking the batch.
        """
        symbols = [str(symbol).upper() for symbol in self.cash_equivalent_symbols()]
        if not symbols:
            return {}
        positions = self.get_positions()
        held = {symbol: float(positions.get(symbol, 0.0) or 0.0) for symbol in symbols}
        held = {symbol: shares for symbol, shares in held.items() if shares > 0}
        if not held:
            return {}
        marks = self.get_marks(sorted(held))
        return {
            symbol: {"shares": shares, "price": float(marks[symbol]), "value": shares * float(marks[symbol])}
            # Ordered by the config, so "prefer BIL over SGOV" is expressible by listing it first.
            for symbol, shares in ((s, held[s]) for s in symbols if s in held)
            if float(marks.get(symbol, 0.0) or 0.0) > 0
        }

    def get_dividend_activity(
        self, start: Optional[date] = None, end: Optional[date] = None
    ) -> List[Dict[str, Any]]:
        """Cash distributions this account actually received.

        Read from the broker rather than reconstructed from the dividend ledger: the ledger
        says what the *market* paid per share, only the broker knows what *this* account was
        credited, after the withholding and the odd-lot rounding that make the two differ.

        Rows are ``{symbol, date, amount, description}`` with ``amount`` in account currency.
        Defaults to empty so a brokerage that cannot report income degrades to showing none
        rather than failing the account page.
        """
        return []

    def validate_short_sale_feasibility(
        self, symbol: str, quantity: float, target_shares: float, latest_price: float
    ) -> Dict[str, Any]:
        """Check whether the brokerage allows a short sale for the given symbol and quantity.

        Returns a dict with ``"shortable"`` (bool) and ``"reason"`` (str).
        The default assumes shorting is allowed.
        """
        return {"shortable": False, "reason": "short sale not confirmed by brokerage"}

@dataclass(frozen=True)
class Schedule:
    """When an algorithm is allowed to act.

    Declared in code on the algorithm class rather than in config, because cadence is a
    property of the strategy and not of the deployment: Fast Momentum is meaningless at a
    weekly cadence, and running DCA hourly only burns API calls. Two algorithms that want
    different cadences cannot share one config knob, which is what the old
    ``algorithm_market_data_refresh_minutes`` forced them to do.

    Cadence never controls how much DCA spends -- accrual is wall-clock (see
    ``algorithms/dca/accrual.py``), so this decides only how often a symbol gets the
    *opportunity* to deploy what it has already accrued.
    """

    #: Minutes between opportunities within a session. Anything at or above the session
    #: length yields exactly one run per trading day.
    refresh_minutes: int = 60
    #: Spread runs deterministically within the bucket, so every deployment does not hit the
    #: market data provider on the same minute.
    jitter_minutes: int = 5
    #: Market-local window, as ``HH:MM``.
    start_time: str = "08:30"
    end_time: str = "15:00"
    #: Weekdays the algorithm may run at all, ``0`` being Monday. This is the part
    #: ``refresh_minutes`` cannot express: bucketing happens inside a session, so without it
    #: the coarsest possible cadence would be daily.
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    #: How often the runtime loop wakes to test the schedule. Not a cadence in itself.
    check_seconds: int = 60


#: Once per trading day, at the open. Shared by algorithms whose inputs are daily bars, where
#: a second look in the same session cannot produce a different answer.
DAILY_AT_OPEN = Schedule(refresh_minutes=390, jitter_minutes=15)

_WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def schedule_minutes(value: str, default: int = 0) -> int:
    """Parse ``HH:MM`` into minutes past midnight, falling back to ``default``."""
    try:
        hour, minute = str(value or "").strip().split(":", 1)
        parsed = (int(hour) * 60) + int(minute)
    except (TypeError, ValueError):
        return default
    return parsed if 0 <= parsed <= (23 * 60) + 59 else default


def describe_schedule(schedule: Schedule) -> str:
    """Render a schedule for the dashboard, e.g. ``"Mondays at 08:30"``.

    Deliberately not shown in the live signal view: cadence is set per binding on the
    dashboard and every algorithm runs inside the trading session regardless, so a Schedule
    row beside the signals restated the deployment the reader had just configured -- and, when
    it read the class instead of the binding, restated it wrongly.
    """
    days = tuple(sorted(set(schedule.weekdays)))
    if not days:
        return "Never"
    if days == (0, 1, 2, 3, 4):
        day_text = "Weekdays"
    elif len(days) == 1:
        day_text = f"{_WEEKDAY_NAMES[days[0]]}s"
    else:
        day_text = ", ".join(_WEEKDAY_NAMES[day][:3] for day in days)

    cadence = schedule.refresh_minutes
    session_minutes = schedule_minutes(schedule.end_time) - schedule_minutes(schedule.start_time)
    if cadence >= max(session_minutes, 1):
        return f"{day_text} at {schedule.start_time}"
    return f"{day_text}, every {cadence}m from {schedule.start_time} to {schedule.end_time}"


class AlgorithmPlugin(ABC):
    algorithm_id: str = ""

    #: Cadence, declared per algorithm. See :class:`Schedule`.
    schedule: Schedule = Schedule()

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
        intents: List[Intent],
        signals: Dict[str, Dict[str, Any]],
        snapshot: PortfolioSnapshot,
        latest_prices: Dict[str, float],
        config: Any,
        as_of: datetime,
    ) -> List[Intent]:
        """Step 2: adjust the proposed intents for what is already held.

        This is where hysteresis lives -- stickiness, turnover thresholds, per-trade minimums,
        exposure caps. ``intents`` may have been edited by a reviewing agent, so honour them as
        the intent rather than re-deriving from ``signals``. Defaults to a passthrough.

        ``as_of`` is the moment step 1 described, and is the *only* clock step 2 may read.
        Everything stateful here -- a session drawdown breaker, a re-entry cooldown, a value
        averaging path -- is a statement about elapsed time, and reading the wall clock instead
        makes those answers wrong in a replay, where every step would be "now". It is a
        required argument rather than an optional one for exactly that reason: a default would
        put the wall clock back one forgetful call site at a time.
        """
        return list(intents)

    def settle(
        self,
        config: Any,
        order_results: List[Dict[str, Any]],
        intents: List[Intent],
    ) -> None:
        """Record what actually reached the market, after step 2 has submitted.

        Only algorithms that carry state between runs need this. DCA draws its accrued budget
        down here rather than at proposal time, so a run that proposes but never submits --
        a denied approval, a rejected order -- keeps its budget instead of spending it on
        nothing. Defaults to doing nothing.
        """
        return None

    def sizing(self, config: Any) -> Dict[str, float]:
        """Order-sizing knobs for step 2, defaulting to the account config.

        Deliberately no cash buffer. How much of the book to deploy is a strategy decision,
        already stated by whatever gross-exposure cap the algorithm applies to its own weights;
        holding cash back to *fund* a batch is an account decision, applied once against
        buying power in ``pipeline.place_orders``. Fusing the two is what made every
        exposure-capped algorithm override this to zero to avoid under-investing twice.
        """
        return {
            "min_trade_dollars": float(getattr(config, "min_trade_dollars", 0.0) or 0.0),
            "rebalance_threshold": float(getattr(config, "rebalance_threshold", 0.0) or 0.0),
        }

    def signal_view(self, config: Any, *, data_client: Any = None) -> SignalView:
        """Return dashboard-ready signals for this algorithm. Default: not wired."""
        return SignalView(wired=False)
