from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import pandas as pd

#: Every cron expression is evaluated in this zone -- the market's own, so "0 11 * * 1-5"
#: means the NYSE morning and DST shifts move scheduled runs with the session rather than
#: away from it. See :mod:`src.core.cron`.
MARKET_TZ = ZoneInfo("America/New_York")

@dataclass(frozen=True)
class AlgorithmContext:
    """Everything ``plan`` is allowed to read.

    ``plan`` must be a pure function of this context: no fetching, no clock, no brokerage, no
    state store. Every input it needs -- bars, holdings, equity, memory of previous runs --
    arrives here as plain data, which is what lets the same call be driven by the live runner
    against today's account and by the backtester against a point-in-time slice, instead of
    the backtester having to reimplement the strategy.
    """

    config: Any
    #: Daily OHLCV per symbol, already truncated to what the algorithm may see. Kept distinct
    #: because the moving-average and regime gates are statements about daily closes.
    daily_bars_by_symbol: Dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Timestamped OHLCV per symbol covering ``AlgorithmRequirements.intraday_lookback_minutes``,
    #: at whatever resolution was available -- possibly blending fine bars recently with
    #: coarser ones further back. Read it through ``src/data/bars.py``, never by position.
    intraday_bars_by_symbol: Dict[str, pd.DataFrame] = field(default_factory=dict)
    #: Per-symbol sentiment score, plus the universe-wide average.
    sentiment_scores: Dict[str, float] = field(default_factory=dict)
    market_sentiment: float = 0.0
    positions: Dict[str, int] = field(default_factory=dict)
    latest_prices: Dict[str, float] = field(default_factory=dict)
    equity: float = 0.0
    account_id: str = ""
    #: What this binding remembered from its last run -- an accrued budget, a re-entry
    #: cooldown, an eligibility history. Loaded by the context builder when
    #: ``AlgorithmRequirements.needs_state`` asks for it, so the algorithm reads memory the
    #: same way it reads bars: as data it was handed, never as a store it goes to itself.
    #: Keyed per binding, since two accounts running one algorithm are two separate books.
    state: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlgorithmRequirements:
    """What data the algorithm needs, so the caller can load it once and hand it over.

    Declaring the need here rather than fetching inside ``plan`` is what makes an algorithm
    replayable: the backtester satisfies the same declaration from cached history.
    """

    price_symbols: List[str] = field(default_factory=list)
    daily_lookback_days: Optional[int] = None
    daily_ma_days: int = 0
    daily_extra_buffer_days: int = 0
    #: How far back the algorithm needs intraday bars, in minutes the market was open --
    #: 4800 is about twelve sessions, not three and a bit days. Stated in minutes rather than
    #: bars because a bar count only means a span of time once the grid is known, and the grid
    #: is now the feed's business: Schwab serves 1/5/10/15/30-minute bars where yfinance served
    #: 15. The lookback resolves against whichever resolution the cache holds -- down to daily,
    #: at the far end of a long window. See ``src/data/bars.py``.
    intraday_lookback_minutes: int = 0
    #: The finest grid the algorithm would like, when it has a preference. 0 takes the
    #: configured default. A provider that cannot serve it supplies its nearest coarser grid
    #: rather than failing, because the horizons no longer depend on getting an exact one.
    preferred_bar_minutes: int = 0
    needs_sentiment: bool = False
    #: Whether ``AlgorithmContext.state`` should be loaded for this run. Only algorithms that
    #: carry something between runs pay for the read.
    needs_state: bool = False
    paper_only: bool = False

#: An algorithm's intent list is the complete portfolio: a held symbol absent from it is exited.
MODE_TARGET = "target"
#: Only the listed symbols are touched; every other holding is left alone.
MODE_INCREMENTAL = "incremental"

INTENT_KINDS = ("weight", "notional", "shares", "option")


@dataclass(frozen=True)
class Intent:
    """One symbol's worth of what an algorithm wants, in whichever unit it thinks in.

    Allocation strategies speak ``weight`` ("hold 44% BBC"); DCA speaks ``notional``
    ("buy $200 of VTI").  ``option`` intents carry contract metadata in ``extra`` so
    the pipeline can route them to the options order path.
    """

    symbol: str
    kind: str = "weight"
    value: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)

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
class AlgorithmPlan:
    """What an algorithm wants to do, and everything needed to act on it.

    One object for one run. There used to be two -- a ``decision`` from step 1 and a
    ``result`` that carried it to step 2 -- which existed only because the halves ran at
    different moments and had to survive the gap between them. ``plan`` runs once against a
    context that already holds positions and equity, so there is no gap, no wire format, and
    nothing that can go stale in between.

    ``latest_prices`` travels with the plan because ``execute`` does no data fetching of its
    own: a symbol it cannot price is one the plan never priced.
    """

    strategy: str = ""
    intents: List[Intent] = field(default_factory=list)
    signals: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    latest_prices: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    mode: str = MODE_TARGET
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    #: What ``AlgorithmContext.state`` should become -- committed by ``execute`` once orders
    #: have actually been placed, and never before. A plan that is only ever looked at (the
    #: dashboard, an agent inspecting a proposal) leaves the stored state untouched, so
    #: viewing what an algorithm would do cannot change what it does next.
    state: Dict[str, Any] = field(default_factory=dict)

    @property
    def target_weights(self) -> Dict[str, float]:
        """The weight intents as a mapping. Derived, so it can never disagree with them."""
        return weights_from_intents(self.intents)


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


#: What a run decided about one symbol. Five outcomes, and every algorithm's decision is one
#: of them however differently it arrived there -- which is what lets one dashboard render all
#: of them without knowing which algorithm it is looking at.
ACTION_ENTER = "enter"      #: Opening a position that was not held.
ACTION_HOLD = "hold"        #: Keeping one that was, whether or not it would be bought today.
ACTION_EXIT = "exit"        #: Closing one that was held.
ACTION_BLOCKED = "blocked"  #: Wanted, but a gate said no. ``checks`` says which.
ACTION_IDLE = "idle"        #: Neither held nor wanted this run.

ACTIONS = (ACTION_ENTER, ACTION_HOLD, ACTION_EXIT, ACTION_BLOCKED, ACTION_IDLE)


@dataclass(frozen=True)
class Check:
    """One gate, as this run measured it.

    The unit the dashboard explains a decision in. A gate that is only a boolean can say *that*
    a name was rejected; carrying the measured value beside the threshold it had to clear says
    *by how much*, which is the difference between "not eligible" and "-6.2% against a -3.0%
    band". Both strings are pre-formatted by the algorithm, because only it knows whether a
    number is a percentage, a dollar amount or a count of runs.
    """

    label: str
    ok: bool
    #: What this run measured, formatted. "-6.2%", "$450", "2 of 5 runs".
    value: str = ""
    #: What it had to be, formatted with its comparison. "> -3.0%", "<= 55%".
    limit: str = ""
    #: Whether failing this is what actually decided the outcome. Several gates can fail at
    #: once; the blocking one is the answer to "why", and the rest are context.
    blocking: bool = False


@dataclass(frozen=True)
class SignalRow:
    """One symbol's decision, and the evidence for it.

    Deliberately a fixed schema rather than a bag of whatever numbers an algorithm happened to
    compute. The dashboard used to sniff for known keys -- ``peak``, ``vol_5d``, ``rank`` -- to
    decide which columns to draw, so adding a signal meant editing the frontend, and the only
    account of *why* anything happened was one free-text ``reason`` string.
    """

    symbol: str
    action: str = ACTION_IDLE
    #: One line stating the decision in the algorithm's own terms: "Rank 2 - held",
    #: "Accruing ($40 of $500)", "Below exit band".
    headline: str = ""
    #: The few numbers worth showing beside the decision, each pre-formatted and labelled by
    #: the algorithm. Declared rather than detected, so a column is something an algorithm asks
    #: for rather than something the deck guesses at.
    metrics: List[Dict[str, str]] = field(default_factory=list)
    #: Every gate this run applied to this symbol, passed and failed alike. Passed ones matter:
    #: they are how a reader tells "cleared everything but the vol ceiling" from "failed at the
    #: first hurdle".
    checks: List[Check] = field(default_factory=list)


@dataclass(frozen=True)
class SignalView:
    """Dashboard-ready view of an algorithm's current decisions."""

    rows: List[SignalRow] = field(default_factory=list)
    summary: List[Dict[str, str]] = field(default_factory=list)

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

    def get_position_details(self) -> List[Dict[str, Any]]:
        """Full position data per symbol: qty, avg_entry_price, market_value, unrealized_pl, unrealized_plpc.

        The single source for the accounts page.  Defaults to reading ``get_positions`` +
        ``get_marks`` and filling the rest as zeros, which is what the non-Alpaca path did
        by hand before this existed.
        """
        holdings = self.get_positions()
        if not holdings:
            return []
        marks = self.get_marks(sorted(holdings))
        return [
            {
                "symbol": symbol,
                "qty": float(shares),
                "avg_entry_price": 0.0,
                "market_value": float(shares) * float(marks.get(symbol, 0.0)),
                "unrealized_pl": 0.0,
                "unrealized_plpc": 0.0,
            }
            for symbol, shares in holdings.items()
        ]

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

