"""How strongly one symbol is trending, in its own sigma.

Nothing here compares symbols -- each is scored from its own bars alone, unlike a cross-sectional
rotation score. The measure is the move divided by what that horizon's move would be at one
standard deviation, so +2.0 means the same on a quiet symbol as on a violent one::

    strength = sum over horizons of  w_h x return_h / (annual_vol x sqrt(days_h / 252))

The horizon ladder and its weights are borrowed from Rally Rotation's config; nothing about its
universe or ranking is.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def scoring_parameters() -> Any:
    """The horizon ladder and its weights, borrowed from Rally Rotation's config."""
    from ..rally_rotation.config import RallyRotationConfig

    return RallyRotationConfig()


def trend_strength(daily_bars: Any, params: Any) -> float:
    """One symbol's trend, in its own sigma. 0.0 when history is too short to score."""
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
        horizon_sigma = annual_vol * math.sqrt(days / TRADING_DAYS)
        if horizon_sigma <= 0:
            continue
        total += weight * (move / horizon_sigma)
        weights += weight
    return (total / weights) if weights > 0 else 0.0
