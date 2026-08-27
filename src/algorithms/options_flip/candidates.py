"""How strongly one symbol is trending, in its own sigma.

**Nothing here compares symbols.** Each is scored from its own bars alone, so a one-symbol run
and a forty-symbol run give a symbol the same number, and adding or removing a name cannot change
what the others do.

That was not always true. This used to z-score the set and keep the top few, which is right for a
*rotation* -- it holds a fixed number of names and must pick which -- and wrong here, where every
qualifying symbol can be traded or none of them. A z-score's mean is zero by construction, so the
best name always scored positive and the rest always negative, whatever the market did: a symbol
up 13% over twenty days was excluded for the sole reason that another was up 23%.

The measure is the move divided by what that horizon's move would be at one standard deviation:

    strength = sum over horizons of  w_h x return_h / (annual_vol x sqrt(days_h / 252))

So +2.0 is a two-sigma move, and it means the same on a quiet symbol as on a violent one. The
horizon ladder and its weights are borrowed from Rally Rotation because a slow-weighted ladder is
a reasonable way to read a trend; nothing about its universe or its ranking is.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


#: Trading sessions in a year, for annualising.
TRADING_DAYS = 252


def scoring_parameters() -> Any:
    """The horizon ladder and its weights. Nothing about a universe, and no ranking.

    Borrowed from Rally Rotation because a 1/2/3/12-session ladder weighted toward the slow end
    is a reasonable way to read a trend and there is no reason to invent a second one. Its
    ``risk_on_universe`` is never read and its ``base_scores`` is not called.
    """
    from ..rally_rotation.config import RallyRotationConfig

    return RallyRotationConfig()


def trend_strength(daily_bars: Any, params: Any) -> float:
    """One symbol's trend, in its own sigma. Absolute: no other symbol is consulted.

    Returns 0.0 when the history is too short for the slowest horizon, which reads as "no lean"
    rather than as an error -- the common case early in a backtest, and not a reason to refuse.
    """
    import math

    if daily_bars is None or getattr(daily_bars, "empty", True):
        return 0.0
    closes = daily_bars["close"].astype(float)
    ladder = (
        (int(params.nano_days), float(params.w_nano)),
        (int(params.micro_days), float(params.w_micro)),
        (int(params.meso_days), float(params.w_meso)),
        (int(params.macro_days), float(params.w_macro)),
    )
    longest = max(days for days, _weight in ladder)
    if len(closes) <= longest + 1:
        return 0.0
    returns = closes.pct_change().dropna()
    annual_vol = float(returns.tail(max(int(params.vol_estimation_days), 20)).std()) * math.sqrt(
        TRADING_DAYS
    )
    if annual_vol <= 0:
        return 0.0

    total = 0.0
    weights = 0.0
    for days, weight in ladder:
        if days < 1 or len(closes) <= days or weight <= 0:
            continue
        move = float(closes.iloc[-1]) / float(closes.iloc[-1 - days]) - 1.0
        # What this horizon's move would be at one sigma, if returns scaled with the square root
        # of time. The ratio is dimensionless, so one threshold works across every symbol.
        horizon_sigma = annual_vol * math.sqrt(days / TRADING_DAYS)
        if horizon_sigma <= 0:
            continue
        total += weight * (move / horizon_sigma)
        weights += weight
    return (total / weights) if weights > 0 else 0.0
