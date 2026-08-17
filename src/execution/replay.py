"""Backtest by replaying an algorithm, rather than by reimplementing it.

Every algorithm already exposes the two-step contract the live runner uses: ``analyze``
proposes from market data alone, ``refine`` adjusts for what is held, ``settle`` records what
filled. If ``analyze`` is a pure function of its ``AlgorithmContext`` -- which is the whole
point of ``AlgorithmRequirements`` -- then a backtest is that same loop with the context built
from history and the brokerage replaced by a fill simulator.

This replaces two hand-written backtests that each reimplemented a strategy's scoring logic
and could silently drift from the algorithm they claimed to test. Adding an algorithm now
costs nothing here: declare its data needs in ``requirements()`` and it is backtestable.

History coverage is reported rather than assumed. The cache accumulates bars from live runs,
so depth grows over time, and a window it does not reach would otherwise score every symbol
near zero and look merely unprofitable instead of unsupported.
"""

from __future__ import annotations

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..brokerages.providers.paper import PaperBrokerage
from ..core.interfaces import AlgorithmPlugin, AlgorithmResult, CashDividend, OrderRequest
from ..core.market_context import ContextSource, build_algorithm_context
from ..core.orders import round_shares
from ..core.pipeline import place_orders
from ..data.bars import TRADING_MINUTES_PER_DAY, calendar_days_for, coverage_minutes, read_history
from ..data.state_store import ephemeral_state

logger = logging.getLogger(__name__)


def dividend_schedule(
    symbols: list[str],
    trade_dates: list[pd.Timestamp],
    dividends: dict[str, list[CashDividend]] | None = None,
) -> dict[pd.Timestamp, list[tuple[str, CashDividend]]]:
    """Map each trade date to the distributions going ex on it.

    An ex-date that is not itself a trade date attaches to the next one, so a payment is never
    dropped merely because the replay steps on a coarser grid than the calendar.
    """
    if dividends is None:
        from ..data.dividends import read_dividends

        frame = read_dividends(symbols)
        dividends = {}
        for row in frame.to_dict(orient="records"):
            dividends.setdefault(str(row["symbol"]).upper(), []).append(
                CashDividend(
                    symbol=str(row["symbol"]).upper(),
                    ex_date=row["ex_date"],
                    amount=float(row["amount"]),
                    payable_date=row.get("payable_date"),
                    record_date=row.get("record_date"),
                    special=bool(row.get("special", False)),
                    source=str(row.get("source", "")),
                )
            )

    ordered = sorted(trade_dates)
    if not ordered:
        return {}
    out: dict[pd.Timestamp, list[tuple[str, CashDividend]]] = {}
    for symbol in symbols:
        for item in dividends.get(str(symbol).upper(), []):
            ex = pd.Timestamp(item.ex_date, tz="UTC")
            # A payment that went ex before the replay opened was earned by whoever held the
            # shares then, which was nobody here. Without this every historical distribution
            # piles onto the first trade date -- currently harmless only because the book
            # starts empty, and quietly wrong the moment it does not.
            if ex < ordered[0]:
                continue
            landing = [date for date in ordered if date >= ex]
            if not landing:
                continue
            out.setdefault(landing[0], []).append((str(symbol).upper(), item))
    return out


def _payable_on(
    ex_trade_date: pd.Timestamp,
    dividend: CashDividend,
    trade_dates: list[pd.Timestamp],
) -> pd.Timestamp:
    """The trade date the cash actually lands on, never earlier than the ex-date itself."""
    if dividend.payable_date is None:
        return ex_trade_date
    payable = pd.Timestamp(dividend.payable_date, tz="UTC")
    landing = [date for date in sorted(trade_dates) if date >= payable]
    # Past the end of the replay the payment still belongs to the holder, so it settles on the
    # final date rather than vanishing.
    return max(landing[0], ex_trade_date) if landing else max(trade_dates[-1], ex_trade_date)


@dataclass
class Coverage:
    """How much of what the algorithm asked for the cache could actually supply.

    Counted in minutes rather than symbols now that horizons are stated that way: a symbol
    the cache holds two sessions of, against a twelve-session horizon, is not "covered" in any
    useful sense, and the old per-symbol count called it so.
    """

    history_requested: int = 0
    history_supplied: int = 0
    missing_symbols: set[str] = field(default_factory=set)

    @property
    def history_ratio(self) -> float:
        return min(self.history_supplied / self.history_requested, 1.0) if self.history_requested else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "history_requested_minutes": self.history_requested,
            "history_supplied_minutes": self.history_supplied,
            "history_ratio": round(self.history_ratio, 4),
            "missing_symbols": sorted(self.missing_symbols),
            "complete": not self.missing_symbols and self.history_ratio >= 1.0,
        }


def _slice_daily(history: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Daily bars up to and including ``as_of``, with the index restored to a column.

    Algorithms read ``timestamp`` as a column, the way the live fetchers return it.
    """
    sliced: dict[str, pd.DataFrame] = {}
    for symbol, frame in history.items():
        window = frame.loc[:as_of]
        if not window.empty:
            sliced[symbol] = window.reset_index()
    return sliced


class HistoryCache:
    """Every symbol's cached bars, read once for the whole replay and sliced per date.

    Daily bars have always been loaded once and sliced -- see :func:`_slice_daily` -- but the
    intraday window was re-read from the store on every trade date, for every symbol, for every
    configured provider. A twelve-month Fast Momentum backtest spent 97 of its 99 seconds on
    21,000 database round trips, which put it past the dashboard's request timeout and surfaced
    as an aborted fetch rather than as anything explicable.

    The window one date needs is a sub-range of the window the whole replay needs, so a single
    read per symbol answers all of them. Slicing reproduces ``read_history``'s own window --
    ``[end - calendar_days_for(lookback), end]`` -- so a date sees exactly the rows it saw
    before, including the blend of fine bars recently and daily bars further back.
    """

    def __init__(
        self,
        symbols: list[str],
        trade_dates: list[pd.Timestamp],
        *,
        providers: list[str],
        lookback_minutes: int,
    ) -> None:
        self.lookback_minutes = lookback_minutes
        # ``None`` reads across every provider, which is the usual case; a named provider is
        # tried first so a deliberate preference wins when it can answer in full.
        self.providers: list[str | None] = [*providers, None]
        self.window_days = calendar_days_for(lookback_minutes) if lookback_minutes > 0 else 0
        self._frames: dict[tuple[str, str | None], tuple[pd.DataFrame, Any]] = {}
        if lookback_minutes <= 0 or not trade_dates:
            return

        ordered = sorted(trade_dates)
        end = ordered[-1] + pd.Timedelta(hours=20)
        # Enough to cover the replay itself plus the lookback behind its first date.
        span_days = (ordered[-1] - ordered[0]).days + self.window_days + 1
        span_minutes = int(span_days * TRADING_MINUTES_PER_DAY)

        for symbol in symbols:
            for provider in self.providers:
                try:
                    bars = read_history(symbol, lookback_minutes=span_minutes, end=end, provider=provider)
                except Exception as exc:
                    logger.warning("History cache read failed provider=%s symbol=%s: %s", provider, symbol, exc)
                    continue
                if bars.empty:
                    continue
                stamps = pd.DatetimeIndex(pd.to_datetime(bars["timestamp"], utc=True))
                self._frames[(symbol, provider)] = (bars, stamps)
                if coverage_minutes(bars) >= span_minutes:
                    # This provider reaches the whole replay, so it reaches every date inside
                    # it: there is nothing a later provider could add.
                    break

    def _window(self, symbol: str, provider: str | None, as_of: pd.Timestamp) -> pd.DataFrame:
        """The rows a read for ``as_of`` would have returned, without going to the store."""
        entry = self._frames.get((symbol, provider))
        if entry is None:
            return pd.DataFrame()
        bars, stamps = entry
        end = as_of + pd.Timedelta(hours=20)
        floor = end - pd.Timedelta(days=self.window_days)
        # A tz-aware DatetimeIndex against tz-aware bounds: the comparison stays in pandas so
        # a provider frame carrying object-dtype timestamps cannot silently mismatch.
        left = int(stamps.searchsorted(floor, side="left"))
        right = int(stamps.searchsorted(end, side="right"))
        return bars.iloc[left:right] if right > left else pd.DataFrame()

    def bars_as_of(
        self, symbols: list[str], as_of: pd.Timestamp, coverage: Coverage
    ) -> dict[str, pd.DataFrame]:
        """Each symbol's history window ending at ``as_of``, measuring what it could supply."""
        bars_by_symbol: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            coverage.history_requested += self.lookback_minutes
            # Deepest wins, not first. Taking the first provider with *any* rows let a two-day
            # sliver from the preferred provider shadow months of history from another, and the
            # symbol then scored as flat with nothing in the output saying why.
            #
            # Each window is measured once and the number carried, rather than re-measuring the
            # incumbent on every comparison: measuring a window rebuilds its frame and its
            # market-minute axis, which made this bookkeeping cost more than the scoring it
            # feeds.
            best, best_covered = pd.DataFrame(), 0
            for provider in self.providers:
                window = self._window(symbol, provider, as_of)
                covered = coverage_minutes(window)
                if covered > best_covered:
                    best, best_covered = window, covered
                if best_covered >= self.lookback_minutes:
                    break
            if best.empty:
                coverage.missing_symbols.add(symbol)
                continue
            bars_by_symbol[symbol] = best
            coverage.history_supplied += min(best_covered, self.lookback_minutes)
        return bars_by_symbol

class ReplayContextSource(ContextSource):
    """Satisfies the same requirements from cached history, as of one past date.

    The counterpart to ``LiveContextSource``. It exists so the backtester stops assembling its
    own ``AlgorithmContext``: the shape of a context, and what each requirement means, is now
    decided in one place for both paths.

    Every method answers strictly as of ``as_of``. That is where the backtest's honesty lives
    -- daily bars are sliced at the signal date, prices are that date's closes, and the cache
    read ends there -- so a decision physically cannot see data it could not have had.
    """

    def __init__(
        self,
        as_of: pd.Timestamp,
        daily_history: dict[str, pd.DataFrame],
        *,
        history: HistoryCache,
        coverage: Coverage,
    ) -> None:
        self.as_of = as_of
        self.daily_history = daily_history
        self.history = history
        self.coverage = coverage

    def timestamp(self) -> datetime:
        return self.as_of.to_pydatetime()

    def latest_prices(self, symbols: list[str], config) -> dict[str, float]:
        return {
            symbol: float(frame.loc[self.as_of, "close"])
            for symbol, frame in self.daily_history.items()
            if self.as_of in frame.index
        }

    def daily_bars(self, symbols: list[str], requirements, config) -> dict[str, Any]:
        # Already loaded once for the whole replay and sliced per date, rather than refetched:
        # the depth an algorithm needs is the same on every date, only the right edge moves.
        return _slice_daily(self.daily_history, self.as_of)

    def history_bars(self, symbols: list[str], requirements, config) -> dict[str, Any]:
        # Sliced out of the read the whole replay shares, for the same reason the daily bars
        # above are: only the right edge of the window moves from one date to the next.
        return self.history.bars_as_of(sorted(self.daily_history), self.as_of, self.coverage)



def _open_in_cash_equivalent(
    book: PaperBrokerage, symbol: str, closes: dict[str, float], fractional: bool
) -> None:
    """Park the whole opening balance in ``symbol`` instead of holding it as raw cash.

    A funded account does not sit at 0% waiting to be invested -- the balance is in T-bills
    earning the bill rate until something needs it. Opening in cash therefore flatters nothing
    but understates income, and, more importantly, never exercises the liquidation a real
    rebalance has to perform to get at that money. Starting here measures both.
    """
    price = float(closes.get(symbol, 0.0) or 0.0)
    if price <= 0:
        logger.warning("No %s price on the first trade date; the book opens in cash instead", symbol)
        return
    cash = float(book.state.get("cash", 0.0))
    # Truncated, never rounded: rounding a whole balance up asks for a fraction of a cent more
    # than the account holds, and the book refuses the order rather than overdraw.
    quantity = round_shares(cash / price, supports_fractional_shares=fractional)
    if quantity <= 0:
        return
    book.submit_order(
        OrderRequest(symbol=symbol, action="buy", quantity=quantity, extra={"latest_price": price})
    )
    logger.info("Book opens in %s: %s shares at %.2f", symbol, quantity, price)


def replay(
    algorithm: AlgorithmPlugin,
    config: Any,
    *,
    daily_history: dict[str, pd.DataFrame],
    trade_dates: list[pd.Timestamp],
    should_run,
    starting_equity: float,
    history_providers: list[str] | None = None,
    fractional: bool = True,
    dividends: dict[str, list[CashDividend]] | None = None,
    history: "HistoryCache | None" = None,
    open_in: str = "",
) -> tuple[pd.DataFrame, Coverage]:
    """Step ``algorithm`` through ``trade_dates``, trading at each date's close.

    ``daily_history`` is keyed by symbol and indexed by timestamp. ``should_run(date)`` gates
    a date against the algorithm's ``Schedule``; a date that does not run still marks the
    portfolio to market, so the equity curve is continuous.

    Signals are computed from bars up to the *previous* date and executed at this date's
    close, so no decision can see the price it trades at.

    ``history`` lets a caller replaying the same symbols over the same dates many times -- a
    parameter sweep -- read the bar store once for all of them instead of once per run. It is
    a pure cache of what the store holds, so sharing it cannot make two runs differ.

    ``open_in`` names a symbol to park the opening balance in -- ``"SGOV"`` for an account
    whose idle cash really sits in T-bills. Its bars must be in ``daily_history``, or the book
    opens in cash as usual.
    """
    requirements = algorithm.requirements(config, {})
    coverage = Coverage()
    records: list[dict[str, Any]] = []

    # One read per symbol for the whole replay, sliced per date below.
    history = history or HistoryCache(
        sorted(daily_history),
        trade_dates,
        providers=history_providers or ["yfinance"],
        lookback_minutes=requirements.history_lookback_minutes,
    )

    schedule = dividend_schedule(sorted(daily_history), trade_dates, dividends)
    pending: list[tuple[pd.Timestamp, str, float]] = []
    dividend_income = 0.0
    contributed = 0.0
    # Carried between dates rather than reset: on a date the algorithm does not run, the book
    # is still positioned the way the last decision left it, which is what attributing a day's
    # P&L to a mode has to mean.
    allocation_mode = ""

    # The book is a real ``PaperBrokerage`` on a throwaway state store, not the configured
    # local paper account: that account is for live scheduled testing and must never be moved
    # by a backtest. ``ephemeral_state`` redirects every read and write to an in-memory dict,
    # so the class, the fills and the accounting are shared while the balances are not.
    with ephemeral_state():
        book = PaperBrokerage(config)
        book.state.update({"cash": float(starting_equity), "positions": {}, "prices": {}})

        for index, trade_date in enumerate(trade_dates):
            closes = {
                symbol: float(frame.loc[trade_date, "close"])
                for symbol, frame in daily_history.items()
                if trade_date in frame.index
            }
            # Seeded before anything reads the book, so the opening balance is already in the
            # sleeve on the first date rather than a day late.
            if index == 0 and open_in:
                _open_in_cash_equivalent(book, open_in.upper(), closes, fractional)
            positions = book.get_positions()

            # Entitlement is settled before this date trades: owning the shares through the
            # previous close is what earns the payment, so a position opened today does not.
            for symbol, amount in schedule.get(trade_date, ()):
                held = positions.get(symbol, 0.0)
                if abs(held) > 1e-9:
                    pending.append((_payable_on(trade_date, amount, trade_dates), symbol, held * amount.amount))

            # Cash lands on the payable date, not the ex-date. The gap is a real few days of
            # drag rather than a rounding detail, so the replay waits it out too.
            still_pending: list[tuple[pd.Timestamp, str, float]] = []
            paid_today = 0.0
            for payable, symbol, value in pending:
                if payable <= trade_date:
                    paid_today += value
                else:
                    still_pending.append((payable, symbol, value))
            pending = still_pending
            if paid_today:
                book.state["cash"] = float(book.state.get("cash", 0.0)) + paid_today
                dividend_income += paid_today

            book.mark_prices(closes)
            account = book.get_account_state()
            equity = float(account["equity"])
            order_count = 0
            turnover = 0.0

            # index 0 has no prior bar to form a signal from, so it only seeds the clock.
            if index > 0 and should_run(trade_date):
                signal_date = trade_dates[index - 1]
                context = build_algorithm_context(
                    config,
                    requirements,
                    positions={s: int(v) for s, v in positions.items()},
                    equity=equity,
                    source=ReplayContextSource(
                        signal_date, daily_history, history=history, coverage=coverage
                    ),
                )
                decision = algorithm.analyze(context)
                # The regime the algorithm believes it is in. Recorded so a result can be
                # attributed to the states that produced it -- "which mode earned this, and how
                # long was it in each" is not answerable from an equity curve alone.
                allocation_mode = str(decision.metadata.get("allocation_mode", "") or allocation_mode)

                # Step 2 exactly as the live runner performs it -- the same sizing, the same
                # rebalance threshold and minimum trade size, the same brokerage call. The
                # replay used to size intents itself with a simpler rule, so a backtest was
                # quietly testing execution logic the runtime does not use.
                #
                # ``latest_prices`` is deliberately this date's closes rather than the signal
                # date's: the decision is made on yesterday's data and filled at today's price.
                result = AlgorithmResult(
                    strategy=getattr(algorithm, "algorithm_id", ""),
                    intents=[i for i in decision.resolved_intents() if i.kind != "weight" or i.value],
                    signals=decision.signals,
                    latest_prices=closes,
                    metadata={**decision.metadata, "requirements": requirements},
                    mode=decision.mode,
                    # The signal date, not the wall clock. ``as_of`` is the only clock step 2
                    # reads, so leaving it to default to ``now`` would tell every stateful
                    # guard in ``refine`` that the whole backtest happened this instant.
                    as_of=context.timestamp,
                )
                outcome = place_orders(
                    result,
                    config,
                    book,
                    # A historical result is stale by construction; freshness is a live guard.
                    max_result_age_seconds=0,
                    algorithm=algorithm,
                )
                filled = [o for o in outcome["order_results"] if o.get("status") == "submitted"]
                order_count = len(filled)
                for order in filled:
                    value = abs(float(order.get("quantity", 0.0)) * float(order.get("latest_price", 0.0)))
                    turnover += value
                    if str(order.get("action", "")).lower() == "buy":
                        contributed += value

                book.mark_prices(closes)
                account = book.get_account_state()

            positions = book.get_positions()
            records.append(
                {
                    "timestamp": trade_date,
                    "equity": float(account["equity"]),
                    "cash": float(account["cash"]),
                    "invested": float(account["equity"]) - float(account["cash"]),
                    "positions": {s: positions.get(s, 0.0) * p for s, p in closes.items()
                                  if abs(positions.get(s, 0.0) * p) > 0.005},
                    "dca_contributions": contributed,
                    # Reported in its own right rather than only inside equity: income and
                    # price appreciation are different things, and a strategy that earns its
                    # return by holding T-bills should be legible as doing exactly that.
                    "dividend_income": dividend_income,
                    "dividends_paid": paid_today,
                    "turnover": turnover,
                    "order_count": order_count,
                    "allocation_mode": allocation_mode,
                }
            )

    if coverage.missing_symbols:
        logger.warning(
            "Backtest history coverage %.0f%%; no cached bars for %s",
            coverage.history_ratio * 100,
            ", ".join(sorted(coverage.missing_symbols)),
        )
    return pd.DataFrame(records).set_index("timestamp").sort_index(), coverage
