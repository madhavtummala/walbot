"""Whether to trade a symbol today, and which way.

Two readings in series, and they are not symmetric. The multi-day trend *proposes* a direction;
pre-market can only *refuse* it. That asymmetry is the whole design:

* A trend is measured over days of full-volume trading and is the only thing here entitled to
  assert a direction.
* Pre-market is thin. A few hundred shares can move it a percent, and inferring a fresh
  direction from that is how you end up long into a gap that fills by 10 a.m. But it is
  genuinely informative about *disagreement* -- an overnight move against a trend is a real
  reason to stand aside for a session, and standing aside costs nothing.

So a contradicting or absent pre-market means no trade today, never a trade the other way.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...core.interfaces import Check
from ...core.options import CALL, PUT

#: No direction. Distinct from an error: the trend was read and had nothing to say.
FLAT = ""


def trend_direction(daily_bars: pd.DataFrame, config: Any) -> tuple[str, list[Check]]:
    """The multi-day reading: ``call``, ``put``, or flat, with the gates it applied.

    Two conditions that must agree -- position against a moving average, and the sign of the
    recent return. Requiring both is what keeps a one-day bounce inside a downtrend from reading
    as bullish, and the checks come back so the deck can say which of the two disagreed.
    """
    ma_period = max(int(config.trend_ma_period), 1)
    if daily_bars is None or len(daily_bars) < ma_period:
        have = 0 if daily_bars is None else len(daily_bars)
        return FLAT, [Check(
            label=f"{ma_period}-day history",
            ok=False,
            value=f"{have} bars",
            limit=f"≥ {ma_period} bars",
            blocking=True,
        )]

    closes = daily_bars["close"].astype(float)
    last = float(closes.iloc[-1])
    average = float(closes.tail(ma_period).mean())
    lookback = max(int(config.trend_lookback_days), 1)
    recent = (last / float(closes.iloc[-lookback - 1]) - 1.0) if len(closes) > lookback else 0.0

    above = last > average
    strong = abs(recent) >= abs(float(config.trend_min_return))
    if above and recent > 0 and strong:
        direction = CALL
    elif not above and recent < 0 and strong:
        direction = PUT
    else:
        direction = FLAT

    return direction, [
        Check(
            label=f"Price vs {ma_period}-day average",
            ok=True,
            value=f"${last:,.2f} {'above' if above else 'below'} ${average:,.2f}",
            limit="either side sets the direction",
        ),
        Check(
            label=f"{lookback}-day move has conviction",
            ok=strong,
            value=f"{recent:+.2%}",
            limit=f"|move| ≥ {abs(float(config.trend_min_return)):.2%}",
            blocking=not strong,
        ),
        Check(
            label="Trend and position agree",
            ok=direction != FLAT,
            value=f"{'above' if above else 'below'} average, moving {recent:+.2%}",
            limit="both pointing the same way",
            blocking=strong and direction == FLAT,
        ),
    ]


def premarket_confirms(
    premarket: dict[str, Any] | None, direction: str, typical_move: float, config: Any
) -> tuple[bool, list[Check]]:
    """Whether this morning's pre-market supports ``direction``. A veto, never a vote.

    The gap is scored **relative to the symbol's own typical daily move** rather than as a raw
    percentage. A 0.3% overnight move is a shrug on a small-cap and a substantial gap on a
    broad-market ETF; one absolute threshold across a mixed symbol list would be far too strict
    for one and meaningless for the other.

    An absent or too-thin pre-market fails closed. That is not a judgement about direction -- it
    is the absence of the confirmation this strategy requires before it will trade at all.
    """
    if not direction:
        return False, []
    if not premarket:
        return False, [Check(
            label="Pre-market confirms the trend",
            ok=False,
            value="no pre-market data",
            limit="required before entering",
            blocking=True,
        )]

    bars = int(premarket.get("bars", 0) or 0)
    min_bars = max(int(config.premarket_min_bars), 0)
    if bars < min_bars:
        return False, [Check(
            label="Pre-market has enough prints",
            ok=False,
            value=f"{bars} one-minute bars",
            limit=f"≥ {min_bars} bars",
            blocking=True,
        )]

    change = float(premarket.get("change_pct", 0.0) or 0.0)
    # Normalising by the typical daily move turns the threshold into "a meaningful fraction of a
    # normal day", which is comparable across symbols. With no volatility estimate the raw
    # percentage is used rather than dividing by zero.
    score = change / typical_move if typical_move > 0 else change
    wanted = 1.0 if direction == CALL else -1.0
    threshold = float(config.premarket_confirm_min)
    confirms = (score * wanted) >= threshold

    return confirms, [Check(
        label=f"Pre-market confirms the {'call' if direction == CALL else 'put'} trend",
        ok=confirms,
        value=f"{change:+.2%} overnight ({score * wanted:+.2f} normalised, {bars} bars)",
        limit=f"≥ {threshold:+.2f} in the trend's direction",
        blocking=not confirms,
    )]


def typical_daily_move(daily_bars: pd.DataFrame, window: int = 20) -> float:
    """The symbol's ordinary day, as the mean absolute open-to-close move.

    Used only to normalise the pre-market gap. Mean absolute rather than standard deviation
    because it is being compared against a single observed move, not a variance.
    """
    if daily_bars is None or len(daily_bars) < 2:
        return 0.0
    frame = daily_bars.tail(max(window, 2))
    opens = frame["open"].astype(float)
    moves = ((frame["close"].astype(float) - opens) / opens).abs()
    return float(moves.mean()) if len(moves) else 0.0
