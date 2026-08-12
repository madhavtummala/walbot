"""Backtest by replaying an algorithm, rather than by reimplementing it.

Every algorithm already exposes the two-step contract the live runner uses: ``analyze``
proposes from market data alone, ``refine`` adjusts for what is held, ``settle`` records what
filled. If ``analyze`` is a pure function of its ``AlgorithmContext`` -- which is the whole
point of ``AlgorithmRequirements`` -- then a backtest is that same loop with the context built
from history and the brokerage replaced by a fill simulator.

This replaces two hand-written backtests that each reimplemented a strategy's scoring logic
and could silently drift from the algorithm they claimed to test. Adding an algorithm now
costs nothing here: declare its data needs in ``requirements()`` and it is backtestable.

Intraday coverage is reported rather than assumed. The cache accumulates intraday bars from
live runs, so depth grows over time, and a window it does not reach would otherwise score
every symbol near zero and look merely unprofitable instead of unsupported.
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
    Intent,
    PortfolioSnapshot,
)
from ..data.duckdb_store import read_market_bars
from ..data.state_store import ephemeral_state

logger = logging.getLogger(__name__)

INTRADAY_CATEGORY = "intraday_market_data"


@dataclass
class Coverage:
    """How much of what the algorithm asked for the cache could actually supply."""

    intraday_requested: int = 0
    intraday_supplied: int = 0
    missing_symbols: set[str] = field(default_factory=set)

    @property
    def intraday_ratio(self) -> float:
        return self.intraday_supplied / self.intraday_requested if self.intraday_requested else 1.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "intraday_requested": self.intraday_requested,
            "intraday_supplied": self.intraday_supplied,
            "intraday_ratio": round(self.intraday_ratio, 4),
            "missing_symbols": sorted(self.missing_symbols),
            "complete": not self.missing_symbols and self.intraday_ratio >= 1.0,
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


def _read_intraday(
    symbols: list[str],
    as_of: pd.Timestamp,
    *,
    providers: list[str],
    lookback_bars: int,
    bar_minutes: int,
    coverage: Coverage,
) -> dict[str, pd.DataFrame]:
    """Intraday bars from the accumulating cache, as of ``as_of``.

    The cache is written through on every live intraday fetch, so it deepens over time. A
    symbol the cache cannot cover is counted, not silently returned empty.
    """
    timeframe = f"{int(bar_minutes)}m"
    end = as_of + pd.Timedelta(hours=20)
    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        coverage.intraday_requested += 1
        for provider in providers:
            try:
                bars = read_market_bars(
                    INTRADAY_CATEGORY, provider, symbol, timeframe,
                    lookback_bars=lookback_bars, end=end,
                )
            except Exception as exc:
                logger.warning("Intraday cache read failed provider=%s symbol=%s: %s", provider, symbol, exc)
                continue
            if not bars.empty:
                bars_by_symbol[symbol] = bars
                coverage.intraday_supplied += 1
                break
        else:
            coverage.missing_symbols.add(symbol)
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
    intraday_providers: list[str] | None = None,
    fractional: bool = True,
) -> tuple[pd.DataFrame, Coverage]:
    """Step ``algorithm`` through ``trade_dates``, trading at each date's close.

    ``daily_history`` is keyed by symbol and indexed by timestamp. ``should_run(date)`` gates
    a date against the algorithm's ``Schedule``; a date that does not run still marks the
    portfolio to market, so the equity curve is continuous.

    Signals are computed from bars up to the *previous* date and executed at this date's
    close, so no decision can see the price it trades at.
    """
    requirements = algorithm.requirements(config, {})
    providers = intraday_providers or ["yfinance"]
    coverage = Coverage()

    cash = starting_equity
    positions: dict[str, float] = {}
    contributed = 0.0
    records: list[dict[str, Any]] = []

    # One throwaway state store for the whole replay: algorithms that carry state between runs
    # keep it across dates, but nothing touches the live account's state.
    with ephemeral_state():
        for index, trade_date in enumerate(trade_dates):
            closes = {
                symbol: float(frame.loc[trade_date, "close"])
                for symbol, frame in daily_history.items()
                if trade_date in frame.index
            }
            market_value = sum(positions.get(s, 0.0) * p for s, p in closes.items())
            equity = cash + market_value
            order_count = 0
            turnover = 0.0

            # index 0 has no prior bar to form a signal from, so it only seeds the clock.
            if index > 0 and should_run(trade_date):
                signal_date = trade_dates[index - 1]
                daily = _slice_daily(daily_history, signal_date)
                intraday = (
                    _read_intraday(
                        sorted(daily_history),
                        signal_date,
                        providers=providers,
                        lookback_bars=requirements.intraday_lookback_bars,
                        bar_minutes=requirements.intraday_bar_minutes,
                        coverage=coverage,
                    )
                    if requirements.intraday_lookback_bars
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
                    intraday_bars_by_symbol=intraday,
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
                    "turnover": turnover,
                    "order_count": order_count,
                }
            )

    if coverage.missing_symbols:
        logger.warning(
            "Backtest intraday coverage %.0f%%; no cached bars for %s",
            coverage.intraday_ratio * 100,
            ", ".join(sorted(coverage.missing_symbols)),
        )
    return pd.DataFrame(records).set_index("timestamp").sort_index(), coverage
