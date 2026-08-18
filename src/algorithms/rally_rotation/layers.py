"""Two layers, in order: absolute eligibility, then sizing.

Relative strength selects and absolute momentum permits -- each layer here answers one of
those questions and nothing else.

There were four, and briefly a fifth. The market-regime gate and the entry-timing gate are
gone: both had been switched off in the deployed configuration long enough to be measured, and
both measured as subtracting. What remains of the regime layer is :func:`universe_data_ok`,
which is not a market view at all -- it refuses to trade a cache too thin to judge.

A theme layer -- correlation clusters over the ETFs, selected and rotated as units -- is also
gone. It was aimed at a real problem, that a cross-sectional score treats near-substitutes as
independent choices, but a universe of sector and thematic ETFs already *is* one instrument per
exposure, so the clustering mostly restated the universe with more machinery: a theme map, a
per-theme cap, a per-theme rank, an intra-theme replacement margin and a separate allocation
path, all to decide what ranking the names already said. Selection is per ETF again.
"""

from __future__ import annotations

from .config import EPSILON, TRADING_DAYS, RallyRotationConfig
from .scoring import _closes



import logging
import math
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)




def universe_data_ok(
    scored: dict[str, dict[str, Any]],
    config: RallyRotationConfig,
) -> dict[str, Any]:
    """Whether enough of the universe can be judged at all. Not a view on the market.

    All that survives of the breadth regime gate. That gate asked "is enough of the universe
    in an uptrend", which duplicated the question :func:`eligibility` already asks per name --
    and answered it worse, by vetoing names that had passed every test the strategy makes of
    them because other, unrelated names were weak. It was measured at ``breadth_min: 0.0`` for
    long enough to be sure it was only ever subtracting, so it is gone.

    This check is a different kind of thing and is kept for that reason: too little usable
    history is not a bearish reading, it is an unusable one. A cold or truncated cache would
    otherwise read as "nothing is above its average", which is a bear market that never
    happened. Below the coverage floor the algorithm holds the defensive sleeve and says why.
    """
    risk_on = [scored[symbol] for symbol in config.risk_on_universe if symbol in scored]
    usable = [row for row in risk_on if row.get("enough_history")]
    coverage = len(usable) / len(risk_on) if risk_on else 0.0
    ok = bool(usable) and coverage >= config.min_universe_coverage
    return {
        "data_ok": ok,
        "coverage": coverage,
        "detail": "" if ok else (
            f"only {len(usable)} of {len(risk_on)} risk-on names have "
            f"{config.etf_ma_days} daily bars"
        ),
    }


# =========================================================================================
# Layer 1: absolute eligibility


def eligibility(row: dict[str, Any], config: RallyRotationConfig) -> tuple[bool, str]:
    """Whether one ETF may be held at all, independent of how it ranks.

    This is the absolute-momentum half of dual momentum. A name failing here is not ranked,
    so a thin qualifying set means holding less rather than lowering the bar.
    """
    if not row.get("has_daily"):
        return False, "No daily history"
    if not row.get("enough_history"):
        return False, f"Only {int(row.get('daily_bars', 0))} of {config.etf_ma_days} daily bars"
    if not row.get("above_moving_average"):
        return False, f"Below its {config.etf_ma_days}-day average"
    if float(row.get("abs_return", 0.0)) <= config.etf_min_abs_return:
        return False, f"{config.etf_abs_return_days}-day return below {config.etf_min_abs_return:+.0%}"
    if float(row.get("fast_return", 0.0)) <= config.etf_min_fast_return:
        return False, f"{config.etf_fast_return_days}-day return below {config.etf_min_fast_return:+.0%}"
    # Volatility ceiling: reject names that have become too volatile.
    vol_ceiling = max(config.vol_ceiling, 0.0)
    if vol_ceiling and float(row.get("annual_volatility", 0.0)) > vol_ceiling:
        return False, f"Annualized vol {float(row.get('annual_volatility', 0.0)):.0%} exceeds ceiling {vol_ceiling:.0%}"
    # Rising volatility: reject if short-term vol is running above long-term vol.
    vol_rising_thresh = max(config.vol_rising_threshold, 0.0)
    if vol_rising_thresh:
        vol_5d = float(row.get("vol_5d", 0.0))
        vol_20d = float(row.get("annual_volatility", 0.0))
        if vol_20d > 0 and vol_5d > vol_20d * (1.0 + vol_rising_thresh):
            return False, f"5d vol {vol_5d:.0%} rising above 20d vol {vol_20d:.0%}"
    # Trend filter: reject names with short-term rally but medium-term downtrend.
    trend_ma_days = max(config.trend_ma_days, 0)
    if trend_ma_days and float(row.get("trend_ma_distance", 0.0)) < 0:
        return False, f"Below {trend_ma_days}-day trend MA"
    trend_return_days = max(config.trend_return_days, 0)
    trend_min_ret = float(config.trend_min_return)
    if trend_return_days and trend_min_ret:
        trend_ret = float(row.get("trend_return", 0.0))
        if trend_ret <= trend_min_ret:
            return False, f"{trend_return_days}-day return {trend_ret:+.2%} below {trend_min_ret:+.2%}"
    return True, ""


def crash_stop(row: dict[str, Any], config: RallyRotationConfig) -> tuple[bool, str]:
    """The one exit that answers to no clock: a single session down ``max_daily_drop``.

    Separated from :func:`hold_eligibility` because the two exits are on different cadences.
    The band-based tests below are a considered judgement and can wait for
    ``exit_interval_days``; this one cannot, or a name can gap 30% over a week of throttled
    sessions while the algorithm politely waits its turn to look.
    """
    drop = max(config.max_daily_drop, 0.0)
    session_return = float(row.get("nano_return", 0.0))
    if drop and session_return <= -drop:
        return False, f"Fell {abs(session_return):.0%} in one session"
    return True, ""


def climax_top(row: dict[str, Any], config: RallyRotationConfig) -> tuple[bool, str]:
    """Detect a potential climax top: price far above MA + range expansion + volume spike.

    This fires on every run like crash_stop, outside the normal eligibility cadence.
    The logic: when price is far above its 100-day MA (at a peak, not a discount),
    the same volume spike + wide range pattern that would be a buy-the-dip signal
    near the MA becomes an exit signal. The combination means "exhausted buying pressure
    at a blowoff top."
    """
    # Only fire when far above MA (at a peak, not near MA where same pattern is bullish)
    climax_ma_min = max(config.climax_ma_distance_min, 0.0)
    if climax_ma_min <= 0:
        return True, ""
    ma_distance = float(row.get("ma_distance", 0.0))
    if ma_distance < climax_ma_min:
        return True, ""

    # Range expansion: today's range is multiple of 20d average range
    range_limit = max(config.range_expansion_limit, 0.0)
    if range_limit <= 0:
        return True, ""
    range_expansion = float(row.get("range_expansion", 0.0))
    if range_expansion < range_limit:
        return True, ""

    # Volume spike confirms exhaustion
    vol_ratio_min = max(config.climax_volume_ratio_min, 0.0)
    if vol_ratio_min > 0:
        volume_ratio = float(row.get("volume_ratio", 0.0))
        if volume_ratio < vol_ratio_min:
            return True, ""

    return False, f"Climax top: {ma_distance:.0%} above MA, range {range_expansion:.1f}x, vol {float(row.get('volume_ratio', 0.0)):.1f}x"


def hold_eligibility(row: dict[str, Any], config: RallyRotationConfig) -> tuple[bool, str]:
    """Whether a name already held may stay, on floors widened by ``exit_threshold_slack``.

    The counterpart to :func:`eligibility`, which decides entry. Sharing one threshold for
    both makes every floor a coin-flip boundary for anything sitting near it, and the round
    trip costs the spread twice.
    """
    if not row.get("enough_history", 1):
        return False, "History no longer sufficient"
    # The crash stop comes first: a name that has just fallen this far is not a candidate for
    # any of the slower, band-based judgements below.
    ok, why = crash_stop(row, config)
    if not ok:
        return False, why
    slack = max(config.exit_threshold_slack, 0.0)
    if float(row.get("ma_distance", 0.0)) < -slack:
        return False, f"More than {slack:.0%} below its {config.etf_ma_days}-day average"
    if float(row.get("abs_return", 0.0)) <= config.etf_min_abs_return - slack:
        return False, f"{config.etf_abs_return_days}-day return below the exit band"
    if float(row.get("fast_return", 0.0)) <= config.etf_min_fast_return - slack:
        return False, f"{config.etf_fast_return_days}-day return below the exit band"
    return True, ""


# =========================================================================================
# Layer 2: sizing and portfolio volatility


def score_to_weights(rows: list[dict[str, Any]], config: RallyRotationConfig) -> dict[str, float]:
    """Split ``risk_on_gross_max`` between the selected names, in proportion to score.

    The score enters as its excess over ``min_base_score``, so a name that only just clears the
    quality floor gets a small position rather than an equal one, tilted by ``volatility_tilt``.

    No per-name cap. A cap on top of a two-name split does not diversify anything -- it just
    forces the pair toward equal weight and throws away the ranking the whole algorithm exists
    to produce. It also came with a water-filling routine to re-spread the overflow, because
    the first version dropped it and left the book quietly under-invested: one selected name
    produced a 50% position and 50% idle cash rather than the gross the configuration asked
    for. Both are gone; the proportional split cannot leave a residual.
    """
    raw: dict[str, float] = {}
    for row in rows:
        excess = max(float(row.get("base_score", 0.0)) - config.min_base_score, 0.0)
        volatility = float(row.get("annual_volatility", 0.0))
        # sigma ** tilt: negative divides (risk parity), zero ignores it, positive leans in.
        scale = (volatility + EPSILON) ** config.volatility_tilt if config.volatility_tilt else 1.0
        raw[str(row["symbol"])] = excess * scale

    gross = max(config.risk_on_gross_max, 0.0)
    total = sum(raw.values())
    if total <= EPSILON:
        # Every candidate sits exactly at the floor: equal-weight rather than divide by zero.
        return {symbol: gross / len(raw) for symbol in raw} if raw else {}
    return {symbol: gross * value / total for symbol, value in raw.items()}


def defensive_weights(
    scored: dict[str, dict[str, Any]],
    config: RallyRotationConfig,
) -> dict[str, float]:
    """Where the book sits when risk-on is not permitted.

    Ranked by medium-term absolute return so the defensive sleeve is itself chosen rather
    than fixed. There is no per-name cap on the risk sleeve any more; there was one, and it limited
    single-name risk in the risky sleeve, and applying it here would force idle cash for no
    reason when the whole point is to be in T-bills.
    """
    candidates = [scored[symbol] for symbol in config.defensive_universe if symbol in scored]
    if not candidates:
        return {}
    candidates.sort(key=lambda row: float(row.get("abs_return", 0.0)), reverse=True)
    chosen = candidates[: max(config.defensive_max_positions, 1)]
    share = max(config.risk_on_gross_max, 0.0) / len(chosen)
    return {str(row["symbol"]): share for row in chosen}


def park_residual(
    weights: dict[str, float],
    defensive_book: dict[str, float],
    config: RallyRotationConfig,
) -> dict[str, float]:
    """Put whatever the risk sleeve could not deploy into the defensive sleeve, not into cash.

    A real funded account never holds idle cash: the balance sits in T-bills until something
    needs it, which is the same reason the backtester opens the book in a cash equivalent
    rather than in cash. Without this the book was a three-way split -- risk assets, T-bills,
    and a raw cash slice earning nothing -- when the intent is binary: deployed in the names
    that qualify, and in bills for the rest.

    Over 2023 that slice averaged 10.3% of equity and reached 15% in some months.
    """
    gross = max(config.risk_on_gross_max, 0.0)
    deployed = sum(value for value in weights.values() if value > 0)
    residual = gross - deployed
    if residual <= EPSILON or not defensive_book:
        return dict(weights)

    # Spread across the defensive sleeve in its own proportions, so a multi-name sleeve keeps
    # the ranking ``defensive_weights`` gave it.
    total = sum(defensive_book.values())
    if total <= EPSILON:
        return dict(weights)
    combined = dict(weights)
    for symbol, share in defensive_book.items():
        combined[symbol] = combined.get(symbol, 0.0) + residual * share / total
    return combined


def sentiment_adjusted(
    weights: dict[str, float],
    sentiment_scores: dict[str, float],
    config: RallyRotationConfig,
) -> dict[str, float]:
    """A bounded size modifier, never a reason to hold something price logic rejected.

    Capped at the +-10% the spec asks for by construction: a clipped sentiment of +-2 times a
    0.05 scale. Sentiment cannot create a position, only nudge one that already qualified.
    """
    if not config.sentiment_size_scale:
        return dict(weights)
    adjusted: dict[str, float] = {}
    for symbol, weight in weights.items():
        clip = max(config.sentiment_clip, 0.0)
        score = max(-clip, min(clip, float(sentiment_scores.get(symbol, 0.0))))
        modifier = 1.0 + (config.sentiment_size_scale * score)
        low = 1.0 - (config.sentiment_size_scale * clip)
        high = 1.0 + (config.sentiment_size_scale * clip)
        adjusted[symbol] = weight * max(low, min(high, modifier))
    return adjusted


# =========================================================================================
# Assembly
