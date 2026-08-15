"""Read a bar frame by elapsed market time rather than by position.

Every intraday horizon in this project used to be a count of bars, and every bar was assumed
to be 15 minutes because that was yfinance's floor. ``closes.iloc[-bars - 1]`` therefore
meant a different span on every grid: a 78-bar lookback is three sessions at 15m, one session
at 5m, and a fifth of one at 1m. With Schwab serving 1/5/10/15/30-minute bars in real time,
that coupling is a bug waiting to happen -- the tuned horizons would silently rescale the
moment the feed changed.

So horizons are stated in minutes and resolved here, against whatever observations exist.

Those are *trading* minutes, not calendar ones. A market is shut for two thirds of every day
and all weekend, and a momentum horizon means "this much of the market being open", not "this
much elapsed time" -- 4800 minutes is twelve sessions, not three and a bit days. Every lookup
therefore walks an axis of elapsed market time, built by advancing one bar's worth per bar and
never counting the gap across a close. That is the property a bar count had for free and the
reason this module exists rather than a bare ``timestamp <= now - delta`` comparison.

The other half is resolution independence: one frame may blend fine bars recently with daily
bars further back (what :func:`read_history` below returns), and a horizon has to mean the
same thing across the seam. A daily bar advances one session; a 5-minute bar advances five
minutes; the axis makes both comparable.
"""

from __future__ import annotations

from datetime import datetime

import math
from typing import Any, Iterable, Sequence

import pandas as pd

from src.data.duckdb_store import (
    DUCKDB_STATE_PATH,
    MARKET_BAR_COLUMNS,
    _now,
    available_intervals,
    read_bars,
)

#: The grid every tuned threshold in this project was fitted on. Volatility is reported
#: against it so a change of feed resolution does not silently rescale the numbers.
REFERENCE_INTERVAL_MINUTES = 15

#: A regular US equities session. The unit that makes a daily bar and an intraday one
#: commensurable: one daily bar advances the market clock by exactly this much.
TRADING_MINUTES_PER_DAY = 390

_DAILY_MINUTES = 1440


#: Column holding the derived total-return series. Never stored -- ``attach_total_return``
#: computes it when a frame is loaded, so nothing in ``market_bars`` is ever a derived number.
TOTAL_RETURN_COLUMN = "total_return"


def total_return_index(
    timestamps: pd.Series,
    closes: pd.Series,
    dividends: pd.Series | None,
) -> pd.Series:
    """Compound raw closes with the distributions paid inside the window.

    Scaled to end at the final raw close, so the series is a real price at the right-hand
    edge -- the end a caller is most likely to quote or size against -- and drifts below the
    raw price going back, by exactly the income paid since. Anchoring at the *start* instead
    would leave the newest value a synthetic number that matches no print.

    Note what this does *not* do: it never rewrites a stored price. The result lives in a
    frame for the length of one call.
    """
    prices = pd.to_numeric(closes, errors="coerce")
    if prices.empty:
        return prices
    stamps = pd.to_datetime(timestamps, utc=True, errors="coerce")

    paid = pd.Series(0.0, index=prices.index)
    if dividends is not None and len(dividends):
        ex_dates = pd.to_datetime(pd.Series(dividends.index), utc=True, errors="coerce")
        for ex_date, amount in zip(ex_dates, dividends.to_numpy()):
            if pd.isna(ex_date) or not amount:
                continue
            # The first bar at or after the ex-date is the one that opens already trading
            # without the payment, so that is where the return has to be put back.
            on_or_after = stamps[stamps >= ex_date]
            if on_or_after.empty:
                continue
            paid.loc[on_or_after.index[0]] += float(amount)

    previous = prices.shift(1)
    step = ((prices + paid) / previous).where(previous > 0, 1.0).fillna(1.0)
    index = step.cumprod()
    final = float(index.iloc[-1])
    if final <= 0:
        return prices
    return index * (float(prices.iloc[-1]) / final)


def attach_total_return(
    bars: pd.DataFrame,
    symbol: str | None = None,
    dividends: pd.Series | None = None,
) -> pd.DataFrame:
    """Return ``bars`` with a :data:`TOTAL_RETURN_COLUMN` derived from raw closes.

    Looks the distributions up by ``symbol`` when they are not supplied. A symbol that pays
    nothing gets a column equal to ``close``, so downstream shape never depends on dividend
    policy.
    """
    if bars is None or getattr(bars, "empty", True) or "close" not in bars:
        return bars
    if dividends is None and symbol:
        from src.data.dividends import dividends_by_symbol

        try:
            dividends = dividends_by_symbol([symbol]).get(str(symbol).upper())
        except Exception:  # a missing ledger must not take the price path down with it
            dividends = None
    work = bars.copy()
    work[TOTAL_RETURN_COLUMN] = total_return_index(
        work["timestamp"], work["close"], dividends
    )
    # Adding this column changes what ``signal_price`` returns, so a frame that had already
    # been through ``_frame`` is no longer equivalent to one built from it now. Clearing the
    # marker forces a rebuild rather than serving a frame whose ``close`` predates the column.
    work.attrs.pop(_PREPARED, None)
    return work


def signal_price(bars: pd.DataFrame) -> pd.Series:
    """The series every horizon is measured on: total return, not raw price.

    Momentum compares symbols against each other, and a raw-price series quietly ranks every
    dividend payer below where it belongs. Measured on the live ``dual_momentum`` universe,
    ranking on raw price picks a different top-4 on 17.3% of rebalance days, and the tilt runs
    monotonically with yield -- EWJ, VGK and IEMG systematically under-selected, GLD, IBIT and
    USO over-selected. The eligibility gate shows the same one-sided pattern: it flips for
    5-6% of payer-days and never once for a non-payer.

    The preference order matters. :data:`TOTAL_RETURN_COLUMN` is derived here from raw prices
    plus the dividend ledger; ``adjusted_close`` is honoured only when a *feed* supplied one
    (Alpaca sends ``Adjustment.ALL``); ``close`` is the floor. Nothing consulted here is a
    number this project wrote back into its own price cache.
    """
    for column in (TOTAL_RETURN_COLUMN, "adjusted_close"):
        if column in bars:
            series = pd.to_numeric(bars[column], errors="coerce")
            if series.notna().any():
                return series.fillna(pd.to_numeric(bars.get("close"), errors="coerce"))
    return pd.to_numeric(bars.get("close", pd.Series(dtype=float)), errors="coerce")


#: Marks a frame that has already been through :func:`_frame`. Reading it back is what makes
#: ``_frame`` cheap to call defensively: the horizon helpers take raw bars from algorithms but
#: also hand frames to each other, and rebuilding an already-clean frame was the single largest
#: cost in a backtest -- 22,344 rebuilds, half the total runtime, all of them idempotent.
_PREPARED = "_walbot_prepared_bars"


def _frame(bars: Any) -> pd.DataFrame:
    """A clean, ascending ``timestamp``/``close`` view of whatever was handed in.

    ``close`` here means the *signal* price -- see :func:`signal_price`.
    """
    if not isinstance(bars, pd.DataFrame) or bars.empty or "close" not in bars:
        return pd.DataFrame(columns=["timestamp", "close"])
    if bars.attrs.get(_PREPARED):
        return bars
    work = pd.DataFrame({"timestamp": bars["timestamp"], "close": signal_price(bars)})
    if "interval_minutes" in bars:
        work["interval_minutes"] = bars["interval_minutes"].to_numpy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = (
        work.dropna(subset=["timestamp", "close"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    # Set after building, so the flag can only ever describe a frame this function produced.
    work.attrs[_PREPARED] = True
    return work


def _prepared(bars: Any) -> tuple[pd.DataFrame, pd.Series]:
    """The frame and its market-minute axis -- the two things every horizon lookup needs.

    Returned together because they are always wanted together and the axis is derived from the
    frame, so computing them at separate call sites guarantees rebuilding one of them.
    """
    work = _frame(bars)
    return work, _market_minutes_axis(work)


def bar_interval_minutes(bars: Any, default: int = REFERENCE_INTERVAL_MINUTES) -> int:
    """The frame's dominant resolution, in minutes.

    Prefers an explicit ``interval_minutes`` column when the store supplied one; otherwise
    reads the median spacing, so an overnight gap or a blended tail does not inflate it.
    """
    work = _frame(bars)
    if "interval_minutes" in work and not work["interval_minutes"].dropna().empty:
        return max(int(work["interval_minutes"].dropna().mode().iloc[0]), 1)
    if len(work) < 2:
        return default
    gaps = work["timestamp"].diff().dropna()
    if gaps.empty:
        return default
    minutes = float(gaps.median().total_seconds()) / 60.0
    return max(int(round(minutes)), 1) if minutes > 0 else default


def _market_minutes_axis(work: pd.DataFrame) -> pd.Series:
    """Elapsed *market* minutes at each bar, measured from the frame's first observation.

    Each bar advances the clock by its own resolution, capped at one session -- so a 5-minute
    bar is worth five minutes, a daily bar is worth a full session, and the overnight gap
    between two bars is worth nothing at all. Without the cap, an ordinary Friday-to-Monday
    gap would consume more of a lookback window than a week of trading does.
    """
    if work.empty:
        return pd.Series(dtype=float)
    if "interval_minutes" in work and not work["interval_minutes"].dropna().empty:
        steps = pd.to_numeric(work["interval_minutes"], errors="coerce").fillna(REFERENCE_INTERVAL_MINUTES)
    else:
        steps = pd.Series(float(bar_interval_minutes(work)), index=work.index)
    steps = steps.clip(upper=TRADING_MINUTES_PER_DAY)
    # The elapsed wall-clock gap, but never more than the bar itself claims to represent.
    gaps = work["timestamp"].diff().dt.total_seconds().div(60.0)
    gaps = gaps.fillna(0.0).clip(upper=steps).clip(lower=0.0)
    return gaps.cumsum()


def _close_at_offset(work: pd.DataFrame, axis: pd.Series, minutes_back: float) -> float:
    """The close ``minutes_back`` market-minutes before the newest bar."""
    target = float(axis.iloc[-1]) - float(minutes_back)
    at_or_before = work.loc[axis <= target]
    if at_or_before.empty:
        return float(work["close"].iloc[0])
    return float(at_or_before["close"].iloc[-1])


def latest_timestamp(bars: Any) -> pd.Timestamp | None:
    work = _frame(bars)
    return None if work.empty else pd.Timestamp(work["timestamp"].iloc[-1])


def close_at_minutes_ago(bars: Any, minutes: int) -> float:
    """Close as of ``minutes`` of market time ago, or the earliest close if that is older."""
    work = _frame(bars)
    if work.empty:
        return 0.0
    return _close_at_offset(work, _market_minutes_axis(work), minutes)


def return_over_minutes(bars: Any, minutes: int, *, offset_minutes: float = 0.0) -> float:
    """Simple return across the trailing ``minutes`` of market time, 0.0 when unmeasurable.

    The wall-clock replacement for ``closes.iloc[-bars - 1]``: identical for a frame on a
    matching grid, and unchanged in meaning when the grid changes underneath it.
    ``offset_minutes`` ends the window early, which is how a rolling path is sampled.

    Returns 0.0 rather than a shorter measurement when the window reaches past the start of
    the data. Quietly measuring a shorter span would understate some symbols and not others,
    and the cross-sectional score would read that as a market fact rather than a data gap.
    """
    work, axis = _prepared(bars)
    return _return_over(work, axis, minutes, offset_minutes)


def _return_over(
    work: pd.DataFrame, axis: pd.Series, minutes: int, offset_minutes: float = 0.0
) -> float:
    """:func:`return_over_minutes` against an already-prepared frame and axis.

    Exists so a caller sampling the same bars repeatedly -- ``return_path`` takes nine
    readings -- pays for the frame and the axis once rather than once per sample.
    """
    if work.empty or minutes <= 0:
        return 0.0
    span = float(axis.iloc[-1])
    if span < minutes + offset_minutes:
        return 0.0
    start_close = _close_at_offset(work, axis, minutes + offset_minutes)
    end_close = _close_at_offset(work, axis, offset_minutes)
    return (end_close / start_close) - 1.0 if start_close > 0 else 0.0


def return_path(bars: Any, horizon_minutes: int, *, span_minutes: int) -> list[float]:
    """The trailing ``horizon_minutes`` return, sampled across ``span_minutes``, oldest first.

    One sample per bar, so the number of points follows the feed's resolution while the span
    stays fixed in market time. Leading samples the data cannot reach are filled with the
    first real reading rather than with zero, so a thin cache biases the smoothing toward the
    data that exists instead of toward flat.
    """
    work, axis = _prepared(bars)
    step = bar_interval_minutes(work)
    points = max(int(round(max(span_minutes, step) / step)), 1)
    if work.empty or horizon_minutes <= 0:
        return [0.0] * points

    values = [
        _return_over(work, axis, horizon_minutes, step * index)
        for index in range(points - 1, -1, -1)
    ]
    first_real = next((index for index, value in enumerate(values) if value != 0.0), None)
    if first_real:
        values[:first_real] = [values[first_real]] * first_real
    return values


def ema(values: Iterable[float], span_minutes: int, step_minutes: int) -> float:
    """Last value of an EMA over ``values`` (oldest first), with the span given in minutes.

    Expressing the span in market time is what keeps smoothing constant across feeds: 45
    minutes is three points on a 15m grid and nine on a 5m one, and both describe the same
    three-quarters of an hour of memory.
    """
    data = [float(value) for value in values]
    if not data:
        return 0.0
    span = max(int(round(max(span_minutes, 1) / max(step_minutes, 1))), 1)
    if span <= 1 or len(data) == 1:
        return data[-1]
    alpha = 2.0 / (span + 1.0)
    result = data[0]
    for value in data[1:]:
        result = (alpha * value) + ((1 - alpha) * result)
    return result


def realized_volatility(
    bars: Any,
    window_minutes: int,
    *,
    reference_minutes: int = REFERENCE_INTERVAL_MINUTES,
) -> float:
    """Standard deviation of bar returns over the trailing window, stated per reference bar.

    Rescaled by ``sqrt(reference / interval)`` because the raw figure depends on the grid:
    the same price path measured on 5-minute bars gives roughly 1/sqrt(3) of its 15-minute
    standard deviation. Every tuned threshold in this project (``max_intraday_volatility``
    at 0.06 and 0.08, and the sigma behind ``volatility_tilt``) was fitted against 15-minute
    bars, so the number is reported in those units and the thresholds keep their meaning on
    any feed.
    """
    work = _frame(bars)
    if work.empty or window_minutes <= 0:
        return 0.0
    interval = bar_interval_minutes(work)
    if "interval_minutes" in work:
        # A blended frame mixes grids, and differencing across the seam would treat one daily
        # step as a single bar return. Measure on the dominant resolution alone.
        work = work.loc[work["interval_minutes"] == interval].reset_index(drop=True)
        if work.empty:
            return 0.0
    axis = _market_minutes_axis(work)
    window = work.loc[axis > float(axis.iloc[-1]) - window_minutes]
    returns = window["close"].pct_change().dropna()
    if returns.empty:
        return 0.0
    value = float(returns.std())
    if math.isnan(value):
        return 0.0
    return value * math.sqrt(max(reference_minutes, 1) / max(interval, 1))


def coverage_minutes(bars: Any) -> int:
    """How much market time the frame spans, for coverage reporting.

    Market time, not wall-clock: a week of cached 15-minute bars covers about 1950 minutes of
    tradable history, and reporting the 10,080 calendar minutes it sits across would call a
    twelve-session horizon comfortably covered when it is not.
    """
    work = _frame(bars)
    if len(work) < 2:
        return 0
    return max(int(_market_minutes_axis(work).iloc[-1]), 0)


def calendar_days_for(minutes: int) -> int:
    """Calendar days to request so ``minutes`` of market time is actually covered.

    Weekends and holidays, plus slack: five trading days occupy seven calendar ones.
    """
    sessions = max(math.ceil(max(minutes, 1) / TRADING_MINUTES_PER_DAY), 1)
    return max(int(sessions * 1.6) + 4, 7)


def finest_interval(frames: Sequence[Any]) -> int:
    """The finest resolution present across several symbols' frames.

    Scores are cross-sectional, so horizons must be sampled on one common step or two symbols
    on different feeds would be compared over different spans.
    """
    intervals = [bar_interval_minutes(frame) for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    return min(intervals) if intervals else REFERENCE_INTERVAL_MINUTES


# =========================================================================================
# Reads that blend resolutions
# =========================================================================================
# ``read_history`` lives here rather than in ``duckdb_store`` because it is a statement about
# bars, not about storage: it walks the market-time axis defined in this module and returns a
# total-return series. Keeping it in the store meant the store imported this module, which
# imports the dividend ledger, which imports the store -- a cycle that only stayed harmless
# because all three imports were deferred into function bodies.

def read_history(
    symbol: str,
    *,
    lookback_minutes: int,
    end: datetime | None = None,
    max_interval_minutes: int | None = None,
    provider: str | None = None,
    db_path: str = DUCKDB_STATE_PATH,
) -> pd.DataFrame:
    """The best available view of ``symbol`` over the trailing window, blending resolutions.

    ``lookback_minutes`` is market time -- minutes the exchange was open -- because that is
    what a momentum horizon means; the calendar span to search is derived from it. Resolutions
    are walked finest first, each taking what it covers, with the next coarser tier filling
    the stretch still missing. So a window longer than the intraday cache reaches is served by
    daily bars at its far end rather than coming back truncated.

    That blend is the point of stating horizons in minutes: a lookup only needs *an*
    observation near the moment it asks about, not one on a particular grid.
    """
    end_ts = pd.Timestamp(end or _now())
    floor = end_ts - pd.Timedelta(days=calendar_days_for(lookback_minutes))
    intervals = [
        interval
        for interval in available_intervals(symbol, provider=provider, db_path=db_path)
        if max_interval_minutes is None or interval <= int(max_interval_minutes)
    ]
    if not intervals:
        return pd.DataFrame(columns=MARKET_BAR_COLUMNS + ["interval_minutes"])

    frames: list[pd.DataFrame] = []
    # Filled from the newest end backwards; each coarser tier only supplies what is still
    # uncovered, so a fine bar always wins over a coarse one for the same instant. The walk
    # stops as soon as the window is covered in market minutes, which is what keeps a coarse
    # tier out of a frame the fine bars already answer in full.
    uncovered_end = end_ts
    covered = 0
    for interval in intervals:
        if uncovered_end <= floor or covered >= lookback_minutes:
            break
        # Attached here rather than inside ``read_bars``: the store returns what the market
        # printed, and the total-return series is derived for the signal layer that wants it.
        bars = attach_total_return(
            read_bars(
                symbol,
                interval_minutes=interval,
                provider=provider,
                start=floor,
                end=uncovered_end,
                db_path=db_path,
            ),
            symbol,
        )
        if bars.empty:
            continue
        frames.append(bars)
        covered += coverage_minutes(bars)
        uncovered_end = pd.Timestamp(bars["timestamp"].min()) - pd.Timedelta(minutes=1)

    if not frames:
        return pd.DataFrame(columns=MARKET_BAR_COLUMNS + ["interval_minutes"])
    merged = pd.concat(frames, ignore_index=True)
    return (
        merged.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="first")
        .reset_index(drop=True)
    )
