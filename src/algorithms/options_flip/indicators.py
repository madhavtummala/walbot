"""ATR, VWAP, the opening range, and moving-average slope.

The measurements the regime gate and the level model are stated in. None of them existed in this
codebase before; the strategy previously reasoned in raw percentages and in a mean absolute move
from the open, neither of which says how big a move is *for this symbol on this day*.

**Everything here is a per-symbol, absolute measurement.** Nothing is ranked against a universe,
so a two-symbol run and a forty-symbol run compute the same numbers -- which is the defect that
``docs/rally-rotation-simplification.md`` diagnoses in a cross-sectional score, and the reason
that algorithm's *features* port here while its ``base_scores`` does not.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def average_true_range(daily_bars: pd.DataFrame, window: int = 14) -> float:
    """Wilder's true range, averaged over ``window`` sessions, in price units.

    True range rather than the high-low spread, because a gap is part of the move a position is
    exposed to: a symbol that opens 2% below yesterday's close and then trades in a quiet 0.5%
    range has moved 2.5%, and a range that ignores the gap reports 0.5%.

    Returned in dollars rather than as a fraction, because that is what the entry and target
    levels are built from and converting back and forth is where sign errors live.
    """
    if daily_bars is None or daily_bars.empty or len(daily_bars) < 2:
        return 0.0
    frame = daily_bars.tail(max(window, 1) + 1)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    prior_close = frame["close"].astype(float).shift(1)
    true_range = pd.concat([
        high - low,
        (high - prior_close).abs(),
        (low - prior_close).abs(),
    ], axis=1).max(axis=1).dropna()
    return float(true_range.tail(window).mean()) if len(true_range) else 0.0


def session_vwap(intraday_today: pd.DataFrame) -> float:
    """Volume-weighted average price for the session so far.

    The reference the regime gate reads "is the buyer or the seller in control today" from. A
    price above VWAP means the average share traded today changed hands below where it is now.

    Falls back to the last close when volume is missing, which is better than returning zero --
    zero would read as "price is above VWAP" to every comparison downstream.
    """
    if intraday_today is None or intraday_today.empty:
        return 0.0
    frame = intraday_today
    typical = (frame["high"].astype(float) + frame["low"].astype(float)
               + frame["close"].astype(float)) / 3.0
    volume = frame["volume"].astype(float) if "volume" in frame else pd.Series(dtype=float)
    total = float(volume.sum()) if len(volume) else 0.0
    if total <= 0:
        return float(frame["close"].astype(float).iloc[-1])
    return float((typical * volume).sum() / total)


def opening_range(intraday_today: pd.DataFrame, minutes: int = 30) -> dict[str, float]:
    """High, low and width of the first ``minutes`` of the session.

    The width, divided by ATR, is one of the day-shape features the level model buckets on: a
    session that has already used half its usual range in thirty minutes is not the same day as
    one that has barely moved, and the two have different pullback distributions.
    """
    if intraday_today is None or intraday_today.empty:
        return {"high": 0.0, "low": 0.0, "width": 0.0, "close": 0.0}
    first_minute = int(intraday_today["minute"].iloc[0])
    window = intraday_today[intraday_today["minute"] < first_minute + max(minutes, 5)]
    if window.empty:
        window = intraday_today.head(1)
    high = float(window["high"].astype(float).max())
    low = float(window["low"].astype(float).min())
    return {"high": high, "low": low, "width": max(high - low, 0.0),
            "close": float(window["close"].astype(float).iloc[-1])}


def moving_average(closes: pd.Series, window: int) -> float:
    """Simple moving average, or 0.0 when the history is shorter than the window.

    Zero rather than a shorter average on purpose: a name below "the 50-day average" computed
    from thirty bars looks like a market fact and is a data gap. Callers treat 0.0 as unknown.
    """
    if closes is None or len(closes) < window or window < 1:
        return 0.0
    return float(closes.tail(window).mean())


def ma_slope(closes: pd.Series, window: int, lookback: int = 5) -> float:
    """Change in the ``window``-day average over ``lookback`` sessions, as a fraction of itself.

    The slope rather than the level, because "price above a falling average" and "price above a
    rising average" are different regimes and the level cannot tell them apart.
    """
    if closes is None or len(closes) < window + lookback:
        return 0.0
    now = float(closes.tail(window).mean())
    then = float(closes.iloc[-(lookback + 1) - window + 1:-lookback].mean())
    return ((now / then) - 1.0) if then > 0 else 0.0


def quote_age_seconds(contract: Any, now_ms: float) -> float:
    """How old the contract's quote is, in seconds. ``-1.0`` when the provider did not say.

    Checked because this codebase has already been burned by a stale quote: an expired cache row
    served IBIT at $36.15 on a day the venue was quoting $45.04, flagged ``current: True``, to an
    algorithm that sizes real orders from it. An age the provider does not publish is reported as
    unknown rather than assumed fresh.
    """
    stamp = float(getattr(contract, "quote_time_ms", 0) or 0)
    if stamp <= 0:
        return -1.0
    return max((float(now_ms) - stamp) / 1000.0, 0.0)
