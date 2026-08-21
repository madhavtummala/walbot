from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional
from uuid import uuid4

from ..base import BaseBrokerage
from ...core.interfaces import OrderRequest
from ...data.state_store import load_state, save_state

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

    supports_fractional_shares = True

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

    def _save(self) -> None:
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
        from src.data.dividends import read_dividends

        as_of = as_of or datetime.now(timezone.utc).date()
        positions = {s: v for s, v in self.state.get("positions", {}).items() if v}
        if not positions:
            self.state["dividends_credited_through"] = as_of.isoformat()
            self._save()
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
        self._save()
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
            # Surfaced separately so income is legible rather than buried in equity.
            "dividend_income": float(self.state.get("dividend_income", 0.0)),
        }

    def get_positions(self) -> Dict[str, float]:
        return {symbol: shares for symbol, shares in self.state.get("positions", {}).items() if shares}

    def submit_order(self, request: OrderRequest) -> Dict[str, Any]:
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
        positions[request.symbol] = held + signed
        if abs(positions[request.symbol]) < 1e-9:
            positions.pop(request.symbol, None)

        self._update_basis(request.symbol, held=held, signed=signed, price=fill_price)
        self.state["positions"] = positions
        self.state["cash"] = cash - signed * fill_price
        # The *mark* is the clean price, not what this order paid: marking the book at its own
        # fill price would let a round trip look flat while the spread was being paid twice.
        self.state.setdefault("prices", {})[request.symbol] = price
        self._save()

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
        self.state.setdefault("prices", {}).update({s: float(p) for s, p in latest_prices.items() if p > 0})
        self._save()

    def validate_short_sale_feasibility(
        self, symbol: str, quantity: float, target_shares: float, latest_price: float
    ) -> Dict[str, Any]:
        return {"shortable": True, "reason": "paper brokerage allows shorts"}
