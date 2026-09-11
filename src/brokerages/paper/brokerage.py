from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from ..base import BaseBrokerage
from ...core.interfaces import MARKET_TZ, OrderRequest
from ...data.state_store import load_state, save_state, state_lock

logger = logging.getLogger(__name__)

STATE_KEY = "paper_brokerage"
DEFAULT_STARTING_CASH = 100_000.0


def _state_key(account_id: str) -> str:
    return f"{STATE_KEY}:{account_id}" if account_id else STATE_KEY


class PaperBrokerage(BaseBrokerage):
    """A local, fill-immediately brokerage backed by the state store.

    Lets an algorithm be run, refined, and "traded" before any real broker is configured, and
    gives the position-aware half of a strategy something to exercise in tests without
    credentials or a network. Fills are assumed complete at the price supplied with the order,
    which is the same simplification a backtest makes.
    """

    #: Whole shares only. Sizing reads this off the instance (``pipeline.place_orders``) and
    #: Bursty DCA off the class, so flipping it moves every path -- live paper runs, backtests
    #: (which execute through this same class) -- to integer quantities. Enforced again in
    #: ``submit_order`` the way Schwab does, so anything bypassing the sizer is refused rather
    #: than quietly filled.
    supports_fractional_shares = False

    #: Nothing here reaches a venue -- the book is a local ledger in ``app_state``.
    is_paper = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        starting_cash = float(getattr(config, "paper_starting_cash", DEFAULT_STARTING_CASH) or DEFAULT_STARTING_CASH)
        #: Half-spread plus slippage, charged on every fill: a buy pays above the mark and a
        #: sell receives below it. Zero here, and set by the caller -- ``replay`` takes it as an
        #: argument -- rather than read from ``config.transaction_cost_bps``, because a paper
        #: account is a rehearsal of a live one and quietly moving its fills would change what
        #: an existing book reports. A backtest, which needs to price churn to mean anything,
        #: asks for it explicitly.
        self.cost_bps = 0.0
        self.state_key = _state_key(str(getattr(config, "account_id", "") or ""))
        # Books are per account, so two local accounts are two separate portfolios rather
        # than one shared pile that neither of them explains.
        self.state = load_state(self.state_key, None)
        if self.state is None:
            # Fall back to the single unnamespaced book written before accounts were a thing.
            self.state = load_state(STATE_KEY, {"cash": starting_cash, "positions": {}, "prices": {}})

    @contextmanager
    def _transaction(self):
        """Run a mutation against freshly-read state, with this book held for the whole of it.

        The book is loaded once in ``__init__`` and lives on the instance, which is fine for
        reading and wrong for writing: the scheduler runs a thread per binding and several
        bindings may be deployed on one account, so two instances of this class can hold the
        same book at once. Each would fill its orders against the balance it read at
        construction and write the whole blob back, and the second write would discard the
        first's cash and positions -- fills silently vanishing from a book that is supposed to
        rehearse a live one.

        Re-reading inside the lock is the half that matters most: taking the lock around a
        balance read minutes ago protects nothing.
        """
        with state_lock(self.state_key):
            stored = load_state(self.state_key, None)
            if stored is not None:
                self.state = stored
            yield
            save_state(self.state_key, self.state)

    def _market_value(self) -> float:
        prices = self.state.get("prices", {})
        return sum(shares * float(prices.get(symbol, 0.0)) for symbol, shares in self.state["positions"].items())

    def credit_dividends(self, as_of: Optional[date] = None) -> Dict[str, Any]:
        """Book distributions that have gone ex since this account was last credited.

        A real account receives this cash whether or not anything asked it to, so the paper
        book has to as well -- otherwise every dividend payer is quietly marked down by its
        own yield and the paper equity curve stops matching the live one it exists to
        rehearse. Prices stay raw; the income arrives as cash, exactly as the statement shows.

        Idempotent by watermark: each payment is credited once, and re-running on the same day
        is a no-op rather than a second payday.
        """
        as_of = as_of or datetime.now(timezone.utc).date()
        with self._transaction():
            return self._credit_dividends(as_of)

    def _credit_dividends(self, as_of: date) -> Dict[str, Any]:
        """The body of :meth:`credit_dividends`, run with the book locked and freshly read."""
        from src.data.dividends import read_dividends

        positions = {s: v for s, v in self.state.get("positions", {}).items() if v}
        if not positions:
            self.state["dividends_credited_through"] = as_of.isoformat()
            return {"credited": 0.0, "events": 0}

        watermark = self.state.get("dividends_credited_through")
        # First run on an existing book starts from today rather than back-paying history the
        # account never actually held through.
        start = date.fromisoformat(watermark) if watermark else as_of

        frame = read_dividends(sorted(positions), start=start, end=as_of)
        credited = 0.0
        events = 0
        paid: list[Dict[str, Any]] = list(self.state.get("dividend_activity", []))
        for row in frame.to_dict(orient="records"):
            # DuckDB hands a DATE column back as a pandas Timestamp, which will not compare
            # against a ``date``; normalise before the watermark test rather than after.
            ex_date = row["ex_date"]
            ex_date = ex_date.date() if hasattr(ex_date, "date") else ex_date
            if watermark and ex_date <= start:
                continue  # already counted under a previous watermark
            shares = float(positions.get(str(row["symbol"]).upper(), 0.0))
            if abs(shares) < 1e-9:
                continue
            value = shares * float(row["amount"])
            credited += value
            events += 1
            paid.append(
                {
                    "symbol": str(row["symbol"]).upper(),
                    "ex_date": str(ex_date),
                    "amount_per_share": float(row["amount"]),
                    "shares": shares,
                    "cash": value,
                }
            )

        self.state["cash"] = float(self.state.get("cash", 0.0)) + credited
        self.state["dividend_income"] = float(self.state.get("dividend_income", 0.0)) + credited
        # Bounded so a long-lived paper book does not grow an unbounded blob in app_state.
        self.state["dividend_activity"] = paid[-200:]
        self.state["dividends_credited_through"] = as_of.isoformat()
        return {"credited": credited, "events": events}

    def get_dividend_activity(self, start=None, end=None) -> list:
        """What ``credit_dividends`` booked. No broker exists here, so this book is the record."""
        rows = []
        for item in self.state.get("dividend_activity", []) or []:
            stamp = str(item.get("ex_date") or "")
            if start and stamp and stamp < start.isoformat():
                continue
            if end and stamp and stamp > end.isoformat():
                continue
            rows.append(
                {
                    "symbol": str(item.get("symbol") or ""),
                    "date": stamp,
                    "amount": float(item.get("cash") or 0.0),
                    "description": f"{item.get('shares', 0)} sh x {item.get('amount_per_share', 0)}",
                }
            )
        rows.sort(key=lambda row: row["date"], reverse=True)
        return rows

    def get_account_state(self) -> Dict[str, Any]:
        cash = float(self.state.get("cash", 0.0))
        equity = cash + self._market_value()
        return {
            "equity": equity,
            "cash": cash,
            "buying_power": max(cash, 0.0),
            "is_market_open": True,
            "last_equity": self._opening_equity(equity),
            # Surfaced separately so income is legible rather than buried in equity.
            "dividend_income": float(self.state.get("dividend_income", 0.0)),
        }

    def _opening_equity(self, equity: float) -> float:
        """What this book was worth when today's session began.

        A real broker reports this and the paper book could not, so it was the one account on
        the page showing no day's move. There is nothing to derive it from after the fact --
        the book stores a position and a cash balance, not a history -- so the session's first
        read of a new market day stamps it.

        A write from a read, which is worth being explicit about: the alternative is a
        scheduled job whose only purpose is to touch this value, and a book nobody looked at
        would still need one. The stamp is idempotent within a day and self-correcting across
        one, so the cost of the impurity is a single write per account per session.

        The first read of a brand-new day reports equity equal to itself, so the day's move
        starts at zero rather than at whatever the book last happened to be worth.
        """
        today = datetime.now(MARKET_TZ).date().isoformat()
        if str(self.state.get("opening_day") or "") == today:
            return float(self.state.get("opening_equity", equity) or equity)
        with self._transaction():
            # Re-checked inside the lock: two bindings reading this book at once must not each
            # decide they are the first of the day and stamp a different number.
            if str(self.state.get("opening_day") or "") != today:
                self.state["opening_day"] = today
                self.state["opening_equity"] = equity
        return float(self.state.get("opening_equity", equity) or equity)

    def get_positions(self) -> Dict[str, float]:
        return {symbol: shares for symbol, shares in self.state.get("positions", {}).items() if shares}

    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
        # This book fills at the mark, immediately, and keeps no resting orders -- so it cannot
        # represent a contract, a limit that waits, or a stop that watches. Refusing outright is
        # the only honest answer: filling an option order here would buy shares of whatever the
        # symbol string happened to be and report it as a filled contract.
        if request.asset_type != "equity":
            raise NotImplementedError(
                f"Paper brokerage cannot trade {request.asset_type} contracts "
                f"(got {request.symbol}); options require a live options-capable brokerage"
            )
        if request.strategy != "single":
            raise NotImplementedError(
                f"Paper brokerage cannot hold a {request.strategy.upper()} order: it fills "
                "immediately and keeps no resting orders"
            )
        quantity = float(request.quantity)
        if quantity != int(quantity):
            raise ValueError(
                f"Paper brokerage fills whole shares only (got {quantity} for {request.symbol})"
            )
        # Validation above reads nothing from the book, so it stays outside the lock. Everything
        # from the price lookup down is a read-modify-write of cash and positions.
        with self._transaction():
            return self._fill(request)

    def _fill(self, request: OrderRequest) -> Dict[str, Any]:
        """The body of :meth:`submit_order`, run with the book locked and freshly read."""
        price = float((request.extra or {}).get("latest_price") or self.state.get("prices", {}).get(request.symbol, 0.0))
        if price <= 0:
            raise ValueError(f"Paper brokerage needs a price for {request.symbol} to fill an order")

        signed = request.quantity if request.action == "buy" else -request.quantity
        # Cross the spread in the direction that costs money. Applied to the fill rather than
        # deducted as a fee so it shows up in the cost basis and the equity curve the same way
        # it does live, instead of as a separate line nothing reads.
        fill_price = price * (1.0 + (self.cost_bps / 10_000.0) * (1.0 if signed > 0 else -1.0))
        cash = float(self.state.get("cash", 0.0))
        if signed > 0 and signed * fill_price > cash + 1e-9:
            # A real broker refuses an order it cannot fund, and ``submit_planned_orders``
            # already models that -- it catches the refusal, records the leg as rejected and
            # carries on with the rest of the batch. Without this the book simply went
            # negative: the backtest carried its own cash clamp to compensate, and a scheduled
            # live paper run could spend money the account did not have.
            raise ValueError(
                f"Insufficient buying power for {request.symbol}: "
                f"{signed * fill_price:.2f} required, {cash:.2f} available"
            )
        positions = dict(self.state.get("positions", {}))
        held = positions.get(request.symbol, 0.0)
        after = held + signed
        # Asked here rather than only in ``submit_planned_orders``, which is one caller of
        # several: the reconciler and the MCP tools reach ``submit_order`` directly, and a check
        # that only some callers perform is not a check. Without it a sell with nothing held was
        # booked as a short that *credited* cash, with no margin requirement and no size limit,
        # so the book could short indefinitely and manufacture its own buying power -- a paper
        # account that permits what the live one refuses flatters a backtest in exactly the
        # direction that hurts.
        if signed < 0 and after < -1e-9:
            feasibility = self.validate_short_sale_feasibility(
                request.symbol, quantity=abs(signed), target_shares=after, latest_price=price
            )
            if not feasibility.get("shortable"):
                raise ValueError(f"Short sale refused for {request.symbol}: {feasibility.get('reason', '')}")

        positions[request.symbol] = after
        if abs(positions[request.symbol]) < 1e-9:
            positions.pop(request.symbol, None)

        self._update_basis(request.symbol, held=held, signed=signed, price=fill_price)
        self.state["positions"] = positions
        self.state["cash"] = cash - signed * fill_price
        # The *mark* is the clean price, not what this order paid: marking the book at its own
        # fill price would let a round trip look flat while the spread was being paid twice.
        self.state.setdefault("prices", {})[request.symbol] = price

        logger.info("Paper fill: %s %s qty=%s @ %.4f", request.action, request.symbol, request.quantity, fill_price)
        return {
            "order_id": f"paper-{uuid4().hex[:8]}",
            "client_order_id": request.client_order_id or "",
            "status": "filled",
            "symbol": request.symbol,
            "qty": request.quantity,
        }

    def _update_basis(self, symbol: str, *, held: float, signed: float, price: float) -> None:
        """Track average entry price, so the book can report P/L rather than just holdings.

        Adding to a position averages the new fill in; reducing one leaves the basis alone,
        because a partial sale does not change what the remaining shares cost. Closing out
        (or crossing through zero) starts fresh from the crossing price.
        """
        basis = dict(self.state.get("basis", {}))
        after = held + signed
        if abs(after) < 1e-9:
            basis.pop(symbol, None)
        elif held == 0 or (held > 0) != (after > 0):
            basis[symbol] = price
        elif (signed > 0) == (held > 0):
            previous = float(basis.get(symbol, price))
            basis[symbol] = ((abs(held) * previous) + (abs(signed) * price)) / abs(after)
        self.state["basis"] = basis

    def book(self) -> Dict[str, Any]:
        """Holdings with marks and cost basis, for callers that report rather than trade."""
        prices = self.state.get("prices", {})
        basis = self.state.get("basis", {})
        rows = []
        for symbol, shares in self.get_positions().items():
            price = float(prices.get(symbol, 0.0))
            entry = float(basis.get(symbol, 0.0))
            rows.append(
                {
                    "symbol": symbol,
                    "qty": float(shares),
                    "avg_entry_price": entry,
                    "current_price": price,
                    "market_value": float(shares) * price,
                    "unrealized_pl": (price - entry) * float(shares) if entry and price else 0.0,
                    "unrealized_plpc": (price / entry - 1.0) if entry and price else 0.0,
                }
            )
        rows.sort(key=lambda row: abs(row["market_value"]), reverse=True)
        return {**self.get_account_state(), "rows": rows}

    def cancel_all_orders(self) -> None:
        """No-op: paper orders fill immediately, so nothing is ever open."""

    def get_marks(self, symbols) -> Dict[str, float]:
        """The book's own marks, which ``mark_prices`` keeps current."""
        prices = self.state.get("prices", {})
        return {symbol: float(prices[symbol]) for symbol in symbols if float(prices.get(symbol, 0.0) or 0.0) > 0}

    def get_position_details(self) -> list:
        return self.book().get("rows", [])

    def mark_prices(self, latest_prices: Dict[str, float]) -> None:
        """Update marks so equity reflects current prices rather than last fill prices."""
        with self._transaction():
            self.state.setdefault("prices", {}).update(
                {s: float(p) for s, p in latest_prices.items() if p > 0}
            )

    def validate_short_sale_feasibility(
        self, symbol: str, quantity: float, target_shares: float, latest_price: float
    ) -> Dict[str, Any]:
        return {"shortable": True, "reason": "paper brokerage allows shorts"}
