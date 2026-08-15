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
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..core.interfaces import (
    MODE_TARGET,
    AlgorithmContext,
    AlgorithmPlugin,
    CashDividend,
    Intent,
    PortfolioSnapshot,
)
from ..data.bars import coverage_minutes
from ..data.bars import read_history
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


def _read_history(
    symbols: list[str],
    as_of: pd.Timestamp,
    *,
    providers: list[str],
    lookback_minutes: int,
    coverage: Coverage,
) -> dict[str, pd.DataFrame]:
    """Bars from the accumulating cache covering the window ending at ``as_of``.

    The cache is written through on every live fetch, so it deepens over time, and
    ``read_history`` blends whatever resolutions it holds -- fine bars where they exist,
    daily further back. A symbol whose window it cannot fill is measured, not silently
    returned short.
    """
    end = as_of + pd.Timedelta(hours=20)
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        coverage.history_requested += lookback_minutes
        # Deepest wins, not first. Taking the first provider with *any* rows let a two-day
        # sliver from the preferred provider shadow months of history from another, and the
        # symbol then scored as flat with nothing in the output saying why. A provider of
        # ``None`` reads across all of them, which is the usual case.
        best = pd.DataFrame()
        for provider in [*providers, None]:
            try:
                bars = read_history(symbol, lookback_minutes=lookback_minutes, end=end, provider=provider)
            except Exception as exc:
                logger.warning("History cache read failed provider=%s symbol=%s: %s", provider, symbol, exc)
                continue
            if coverage_minutes(bars) > coverage_minutes(best):
                best = bars
            if coverage_minutes(best) >= lookback_minutes:
                break
        if best.empty:
            coverage.missing_symbols.add(symbol)
            continue
        bars_by_symbol[symbol] = best
        coverage.history_supplied += min(coverage_minutes(best), lookback_minutes)
    return bars_by_symbol


def _target_shares(
    intents: list[Intent],
    mode: str,
    positions: dict[str, float],
    prices: dict[str, float],
    equity: float,
) -> dict[str, float]:
    """Convert intents of any kind into desired share counts.

    Mirrors what step 2 does live, minus the brokerage: ``weight`` is a share of equity,
    ``notional`` a dollar increment, ``shares`` an absolute count. In target mode a held
    symbol with no intent is exited; in incremental mode it is left alone.
    """
    desired = dict(positions) if mode != MODE_TARGET else {symbol: 0.0 for symbol in positions}
    for intent in intents:
        price = float(prices.get(intent.symbol, 0.0) or 0.0)
        if price <= 0:
            continue
        if intent.kind == "weight":
            desired[intent.symbol] = (intent.value * equity) / price
        elif intent.kind == "notional":
            desired[intent.symbol] = positions.get(intent.symbol, 0.0) + (intent.value / price)
        else:
            desired[intent.symbol] = float(intent.value)
    return desired


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
) -> tuple[pd.DataFrame, Coverage]:
    """Step ``algorithm`` through ``trade_dates``, trading at each date's close.

    ``daily_history`` is keyed by symbol and indexed by timestamp. ``should_run(date)`` gates
    a date against the algorithm's ``Schedule``; a date that does not run still marks the
    portfolio to market, so the equity curve is continuous.

    Signals are computed from bars up to the *previous* date and executed at this date's
    close, so no decision can see the price it trades at.
    """
    requirements = algorithm.requirements(config, {})
    providers = history_providers or ["yfinance"]
    coverage = Coverage()

    cash = starting_equity
    positions: dict[str, float] = {}
    contributed = 0.0
    dividend_income = 0.0
    records: list[dict[str, Any]] = []

    # Distributions are booked as the cash events they are rather than being folded into the
    # price series. Marking at a back-adjusted close used to make a payment look like price
    # appreciation, which meant a replay holding SGOV for four years booked +0.5% against an
    # actual +18% -- the income simply never existed. Prices here stay raw and the cash shows
    # up in the ledger, which is also what the brokerage statement says.
    schedule = dividend_schedule(sorted(daily_history), trade_dates, dividends)
    pending: list[tuple[pd.Timestamp, str, float]] = []

    # One throwaway state store for the whole replay: algorithms that carry state between runs
    # keep it across dates, but nothing touches the live account's state.
    with ephemeral_state():
        for index, trade_date in enumerate(trade_dates):
            closes = {
                symbol: float(frame.loc[trade_date, "close"])
                for symbol, frame in daily_history.items()
                if trade_date in frame.index
            }

            # Entitlement is settled before this date trades: owning the shares through the
            # previous close is what earns the payment, so a position opened today does not.
            for symbol, amount in schedule.get(trade_date, ()):
                held = positions.get(symbol, 0.0)
                if abs(held) > 1e-9:
                    payable = _payable_on(trade_date, amount, trade_dates)
                    pending.append((payable, symbol, held * amount.amount))

            # Cash lands on the payable date, not the ex-date. The gap is a real few days of
            # drag rather than a rounding detail, so the replay waits it out too.
            still_pending: list[tuple[pd.Timestamp, str, float]] = []
            paid_today = 0.0
            for payable, symbol, value in pending:
                if payable <= trade_date:
                    cash += value
                    dividend_income += value
                    paid_today += value
                else:
                    still_pending.append((payable, symbol, value))
            pending = still_pending

            market_value = sum(positions.get(s, 0.0) * p for s, p in closes.items())
            equity = cash + market_value
            order_count = 0
            turnover = 0.0

            # index 0 has no prior bar to form a signal from, so it only seeds the clock.
            if index > 0 and should_run(trade_date):
                signal_date = trade_dates[index - 1]
                daily = _slice_daily(daily_history, signal_date)
                history = (
                    _read_history(
                        sorted(daily_history),
                        signal_date,
                        providers=providers,
                        lookback_minutes=requirements.history_lookback_minutes,
                        coverage=coverage,
                    )
                    if requirements.history_lookback_minutes
                    else {}
                )
                signal_closes = {
                    symbol: float(frame.loc[signal_date, "close"])
                    for symbol, frame in daily_history.items()
                    if signal_date in frame.index
                }

                context = AlgorithmContext(
                    config=config,
                    bars_by_symbol=daily,
                    history_bars_by_symbol=history,
                    positions={s: int(v) for s, v in positions.items()},
                    latest_prices=signal_closes,
                    equity=equity,
                    account_id=getattr(config, "account_id", ""),
                    timestamp=signal_date.to_pydatetime(),
                )
                decision = algorithm.analyze(context)
                snapshot = PortfolioSnapshot(positions=dict(positions), equity=equity)
                intents = algorithm.refine(
                    decision.resolved_intents(), decision.signals, snapshot, closes, config
                )

                desired = _target_shares(intents, decision.mode, positions, closes, equity)
                fills: list[dict[str, Any]] = []
                for symbol, want in sorted(desired.items()):
                    price = closes.get(symbol)
                    if not price or price <= 0:
                        continue
                    delta = (want if fractional else float(int(want))) - positions.get(symbol, 0.0)
                    value = delta * price
                    if abs(value) < 0.01:
                        continue
                    if value > cash:  # never spend money the account does not have
                        delta = cash / price
                        value = delta * price
                    if delta > 0 and value <= 0:
                        continue
                    positions[symbol] = positions.get(symbol, 0.0) + delta
                    cash -= value
                    turnover += abs(value)
                    order_count += 1
                    if value > 0:
                        contributed += value
                    fills.append(
                        {"symbol": symbol, "status": "submitted",
                         "quantity": abs(delta), "latest_price": price,
                         "side": "buy" if delta > 0 else "sell"}
                    )
                # Lets a stateful algorithm draw down what actually filled, exactly as live.
                algorithm.settle(config, fills, intents)

                market_value = sum(positions.get(s, 0.0) * p for s, p in closes.items())

            positions = {s: v for s, v in positions.items() if abs(v) > 1e-9}
            records.append(
                {
                    "timestamp": trade_date,
                    "equity": cash + market_value,
                    "cash": cash,
                    "invested": market_value,
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
                }
            )

    if coverage.missing_symbols:
        logger.warning(
            "Backtest history coverage %.0f%%; no cached bars for %s",
            coverage.history_ratio * 100,
            ", ".join(sorted(coverage.missing_symbols)),
        )
    return pd.DataFrame(records).set_index("timestamp").sort_index(), coverage
