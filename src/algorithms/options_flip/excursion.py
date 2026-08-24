"""How far the underlying travels against you before it goes your way, and what to bid on it.

The one piece of arithmetic the whole strategy rests on. A trader placing this order by hand
estimates "how low could it get today", bids there, and walks the bid in as the afternoon wears
on. Both halves of that are estimable from history rather than by eye.

**The estimate.** For each past session, measure the excursion from the open:
``(open - low) / open`` downward and ``(high - open) / open`` upward. Condition the sample on
sessions that *agreed with today's direction* -- that is what makes it a forecast rather than an
average. On genuinely trending days the pullback distribution is much tighter than the
unconditional one, so an unconditional bid is systematically too low on exactly the days you
most wanted to be filled.

**The knob.** A quantile of that distribution is a fill probability: bidding at the q-th
quantile of past downside excursions would have been reached on ``1 - q`` of them. So the config
asks for ``target_fill_probability`` and the maths returns a price, instead of asking for a
discount percentage that means nothing on its own and has to be re-tuned per symbol.

**The prediction is a price, not an offset.** The excursion is measured from the open, so the
level it predicts is ``open × (1 - expected)`` -- a fixed price for the session, which the bid
can sit at all morning however far the market wanders above it. Applying the fraction to the
*current* price instead makes the bid chase a rally while claiming to wait for a pullback; see
:func:`predicted_extreme`.

**The walk-in.** The bid is then interpolated between that predicted level and the market, on a
decay that reaches zero at the close -- so it starts where the model thinks the low will be and
converges on whatever the market is actually offering as the day runs out. The trader's "place it
closer late in the day, intending to execute" is not a special case here; it is the interpolation
running out of runway. If the market trades *through* the prediction the bid collapses to the
market, because the dip being waited for has happened.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

#: Below this many observations the conditional sample is not a distribution, it is anecdote.
#: The caller falls back to the unconditional one rather than trading on four data points.
MIN_CONDITIONAL_SAMPLE = 12


def excursions(
    daily_bars: pd.DataFrame, *, direction: str, lookback: int, horizon: int = 1
) -> pd.Series:
    """Past sessions' excursion from the open, over ``horizon`` sessions.

    For a long (``call``) the adverse move is downward, so this returns ``(open - low) / open``;
    for a ``put`` it is ``(high - open) / open``. Both come back positive -- they are distances,
    and the sign is already carried by the direction.

    ``horizon`` is how many sessions the extreme is taken over, and it exists because the entry
    and the exit ask different questions. An entry waits for a dip *today*, so horizon 1. An exit
    is held for up to ``max_hold_sessions``, and the move available over two sessions is much
    larger than over one -- measured on these symbols, the median upside runs 0.99% in a day and
    1.74% in two. Pricing a two-day target off a one-day excursion sets it far too low, and the
    position spends its life chasing a target it should have cleared on the first morning.
    """
    if daily_bars is None or daily_bars.empty:
        return pd.Series(dtype=float)
    span = max(int(horizon), 1)
    frame = daily_bars.tail(max(lookback, 1) + span - 1)
    # A zero-range bar is not a session. The bar store appends the latest quote as a synthetic
    # daily bar with open == high == low == close, so the most recent row is routinely one of
    # these -- and left in, it contributes a 0% excursion that drags the quantile down on exactly
    # the sample the current run depends on.
    frame = frame[frame["high"].astype(float) > frame["low"].astype(float)]
    if frame.empty:
        return pd.Series(dtype=float)
    opens = frame["open"].astype(float)
    if direction == "put":
        # The extreme reached over the next ``span`` sessions, aligned back onto the open it is
        # measured from. At span 1 this is the plain daily high.
        extreme = frame["high"].astype(float).rolling(span).max().shift(-(span - 1))
        raw = (extreme - opens) / opens
    else:
        extreme = frame["low"].astype(float).rolling(span).min().shift(-(span - 1))
        raw = (opens - extreme) / opens
    return raw.replace([float("inf"), float("-inf")], pd.NA).dropna().clip(lower=0.0)


def agreeing_mask(daily_bars: pd.DataFrame, *, direction: str, lookback: int) -> pd.Series:
    """Which of those sessions closed in the direction being traded.

    The conditioning step, and the reason the estimate is a forecast at all: it answers "how far
    back did it pull on days that went on to do what today is expected to do".
    """
    if daily_bars is None or daily_bars.empty:
        return pd.Series(dtype=bool)
    frame = daily_bars.tail(max(lookback, 1))
    # Same zero-range exclusion as :func:`excursions`, so the two line up row for row -- the mask
    # is reindexed onto that series and a mismatch silently drops observations.
    frame = frame[frame["high"].astype(float) > frame["low"].astype(float)]
    if frame.empty:
        return pd.Series(dtype=bool)
    change = (frame["close"].astype(float) - frame["open"].astype(float)) / frame["open"].astype(float)
    return change < 0 if direction == "put" else change > 0


def expected_excursion(
    daily_bars: pd.DataFrame, *, direction: str, lookback: int, fill_probability: float,
    horizon: int = 1,
) -> dict[str, Any]:
    """The excursion to budget for today, as a fraction of the open.

    Returns the estimate alongside the sample it came from, because the deck has to be able to
    say *why* a bid sits where it does -- "1.4%, the 40th percentile of 31 agreeing sessions" is
    an explanation; "1.4%" is a number.
    """
    series = excursions(daily_bars, direction=direction, lookback=lookback, horizon=horizon)
    if series.empty:
        return {"excursion": 0.0, "sample": 0, "conditional": False, "quantile": 0.0}

    probability = min(max(float(fill_probability), 0.0), 1.0)
    # Bidding at the q-th quantile fills on (1 - q) of comparable sessions, so a *higher* desired
    # fill probability means a *shallower* bid, hence the inversion.
    quantile = 1.0 - probability

    agreeing = agreeing_mask(daily_bars, direction=direction, lookback=lookback)
    conditional = series[agreeing.reindex(series.index, fill_value=False)]
    use_conditional = len(conditional) >= MIN_CONDITIONAL_SAMPLE
    sample = conditional if use_conditional else series

    return {
        "excursion": float(sample.quantile(quantile)),
        "sample": int(len(sample)),
        "conditional": bool(use_conditional),
        "quantile": quantile,
    }


def session_fraction_remaining(now: pd.Timestamp, *, open_time: pd.Timestamp, close_time: pd.Timestamp) -> float:
    """How much of the regular session is left, as a fraction in ``[0, 1]``."""
    total = (close_time - open_time).total_seconds()
    if total <= 0:
        return 0.0
    remaining = (close_time - now).total_seconds()
    return min(max(remaining / total, 0.0), 1.0)


def predicted_extreme(session_open: float, expected: float, *, direction: str) -> float:
    """Today's predicted low (or high, for a put), as an **absolute price**.

    The distinction that matters, and the one an earlier version of this module got wrong. The
    excursion is measured from the *open* -- ``(open - low) / open`` -- so the level it predicts
    is ``open × (1 - expected)``, a fixed price for the session. Applying that same fraction to
    the *current* price instead makes the prediction drift upward all morning: with the open at
    100 and a 1% expected dip, the predicted low is 99.00, but a price that has run to 104 would
    put the bid at 102.96 -- chasing a rally while claiming to wait for a pullback.

    A prediction about the day is a level, not an offset from wherever the last print happened
    to land.
    """
    if session_open <= 0:
        return 0.0
    move = max(float(expected), 0.0)
    return session_open * (1.0 + move) if direction == "put" else session_open * (1.0 - move)


def entry_underlying_target(
    current: float,
    predicted: float,
    *,
    direction: str,
    fraction_remaining: float,
    decay_power: float = 0.5,
) -> float:
    """Where to bid: the predicted extreme early on, converging to the market by the close.

    Interpolates between the two on ``fraction_remaining ** decay_power`` -- the whole budget at
    the open, none of it at the close -- so the walk-in falls out of the arithmetic rather than
    needing a rule of its own.

    Never on the wrong side of the market: a call bids at or below the current price, a put at or
    above it. That single clamp also handles the case the old "subtract what has been observed"
    logic existed for. If the market has already traded through the prediction, ``current`` is
    past ``predicted`` and the target collapses to ``current`` -- the dip happened, so take the
    market rather than waiting for a second one.
    """
    if current <= 0 or predicted <= 0:
        return 0.0
    decay = math.pow(min(max(fraction_remaining, 0.0), 1.0), max(decay_power, 0.0))
    target = current - (current - predicted) * decay
    return min(target, current) if direction != "put" else max(target, current)


def observed_excursion(intraday_bars: pd.DataFrame, *, direction: str, session_open: float) -> float:
    """How far today has already moved against the trade, as a fraction of today's open.

    Derived from bars rather than accumulated in state on purpose. A low-water mark carried
    between runs would be wrong after any missed fire and unrecoverable after a restart; the bars
    hold the same fact and reconstruct it every time.
    """
    if intraday_bars is None or intraday_bars.empty or session_open <= 0:
        return 0.0
    if direction == "put":
        extreme = float(intraday_bars["high"].astype(float).max())
        return max((extreme - session_open) / session_open, 0.0)
    extreme = float(intraday_bars["low"].astype(float).min())
    return max((session_open - extreme) / session_open, 0.0)


def target_price(current_price: float, budget: float, *, direction: str) -> float:
    """A price offset from the current one, by a fraction. Used by the *exit* only.

    The entry does not use this -- see :func:`entry_underlying_target`, which anchors to the
    session open instead. The exit legitimately reasons from the current mark: a profit target is
    a move to be captured from here, and a position may be held across sessions, which makes
    "today's open" meaningless for it.

    A buy waits below the market for a call and above it for a put, because the adverse move for
    a long put is the underlying rising. The *option* order is a buy either way -- the direction
    lives in the contract, not in the side.
    """
    if current_price <= 0:
        return 0.0
    move = current_price * max(budget, 0.0)
    return current_price + move if direction == "put" else current_price - move


def option_price_for(
    underlying_target: float,
    *,
    underlying_now: float,
    option_mark: float,
    delta: float,
) -> float:
    """Translate an underlying price into a contract price, first order, through delta.

    Predicting the underlying and translating is deliberate. A contract's own intraday series is
    thin, wide, has no history before today, and drifts down all day on theta for reasons that
    have nothing to do with direction -- so its "low" is not the thing being predicted. The
    underlying has decades of clean daily bars, and the chain hands us the delta to convert with.

    ``delta`` must keep its sign -- negative for a put. That is what makes one formula serve both
    sides: the underlying rising is a gain for a call and a loss for a put, and the sign of delta
    already says so. Taking its magnitude would price a put's adverse move as a gain.

    Gamma is ignored. Over the fraction of a percent this budget spans, the second-order term is
    smaller than the tick, and ignoring it errs toward bidding slightly low -- which for a buy is
    the safe side.
    """
    if underlying_now <= 0 or option_mark <= 0:
        return 0.0
    move = (underlying_target - underlying_now) * float(delta)
    return max(option_mark + move, 0.01)
