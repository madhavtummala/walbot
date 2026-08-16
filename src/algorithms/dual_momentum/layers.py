"""The four gates, in order: regime, absolute eligibility, entry timing, then sizing.

Relative strength selects and absolute momentum permits -- each layer here answers one of
those questions and nothing else.
"""

from __future__ import annotations

from .config import EPSILON, TRADING_DAYS, DualMomentumConfig
from .scoring import _closes



import logging
import math
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)




def market_regime(
    scored: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
) -> dict[str, Any]:
    """The raw risk-on gate: benchmark trend, benchmark absolute momentum, and breadth.

    Raw because the hysteresis that turns this into a *state* needs to count consecutive
    observations, and counting requires memory that ``analyze`` is not allowed to have.
    """
    benchmark = scored.get(config.benchmark, {})
    risk_on = [scored[symbol] for symbol in config.risk_on_universe if symbol in scored]
    above = [row for row in risk_on if row.get("above_moving_average")]
    breadth = len(above) / len(risk_on) if risk_on else 0.0

    # No usable benchmark history is not a bearish reading, it is an unusable one. Say so,
    # and stay risk-off: this gate is the algorithm's whole premise, so guessing is worse
    # than declining to act.
    if not benchmark.get("enough_history"):
        return {
            "risk_on": False,
            "trend_ok": False,
            "momentum_ok": False,
            "breadth": breadth,
            "breadth_ok": False,
            "data_ok": False,
            "detail": (
                f"no usable {config.benchmark} history "
                f"({int(benchmark.get('daily_bars', 0))} of {config.benchmark_ma_days} daily bars)"
            ),
        }

    trend_ok = bool(benchmark.get("above_moving_average"))
    momentum_ok = float(benchmark.get("abs_return", 0.0)) > 0.0
    breadth_ok = breadth >= config.breadth_min
    reasons = []
    if not trend_ok:
        reasons.append(f"{config.benchmark} below its {config.benchmark_ma_days}-day average")
    if not momentum_ok:
        reasons.append(f"{config.benchmark} {config.benchmark_return_days}-day return is negative")
    if not breadth_ok:
        reasons.append(f"breadth {breadth:.0%} below {config.breadth_min:.0%}")

    return {
        "risk_on": trend_ok and momentum_ok and breadth_ok,
        "trend_ok": trend_ok,
        "momentum_ok": momentum_ok,
        "breadth": breadth,
        "breadth_ok": breadth_ok,
        "data_ok": True,
        "detail": "; ".join(reasons) or "benchmark trend, momentum and breadth all pass",
    }


# =========================================================================================
# Layer 2: absolute eligibility


def eligibility(row: dict[str, Any], config: DualMomentumConfig) -> tuple[bool, str]:
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
    return True, ""


# =========================================================================================
# Layer 3: entry timing


def timing(row: dict[str, Any], config: DualMomentumConfig) -> tuple[bool, str]:
    """Whether *now* is a moment to open or add, given the name already qualifies.

    Two independent ways in: accelerating volatility-normalised momentum, or a pullback
    inside an intact uptrend. Deliberately a flag rather than a score term -- as a bonus, a
    deep enough dip could outvote the trend horizons and promote a weak ETF.
    """
    if float(row.get("momentum_change", 0.0)) > config.momentum_change_enter:
        return True, "Momentum accelerating"

    z = row.get("z", {}) if isinstance(row.get("z"), dict) else {}
    pullback = (
        float(z.get("macro", 0.0)) >= config.pullback_macro_z_min
        and float(z.get("meso", 0.0)) >= config.pullback_meso_z_min
        and float(z.get("nano", 0.0)) <= config.pullback_nano_z_max
        and float(row.get("micro_return", 0.0)) >= config.pullback_micro_return_min
    )
    if pullback:
        return True, "Pullback in uptrend"
    return False, "Waiting for entry timing"


# =========================================================================================
# Layer 4: sizing and portfolio volatility


def covariance_matrix(
    daily_bars_by_symbol: dict[str, Any],
    symbols: list[str],
    config: DualMomentumConfig,
) -> dict[str, dict[str, float]]:
    """Annualised covariance of daily returns over ``vol_estimation_days``.

    Returned as nested dicts rather than a frame because it travels to step 2 inside the
    signal rows, and step 2 only ever receives JSON-shaped signals.
    """
    frame = pd.DataFrame(
        {
            symbol: _closes(daily_bars_by_symbol.get(symbol)).pct_change().dropna().tail(config.vol_estimation_days)
            for symbol in symbols
        }
    ).dropna(how="all")
    if frame.empty or len(frame) < 2:
        return {symbol: {symbol: 0.0} for symbol in symbols}
    covariance = frame.cov() * TRADING_DAYS
    return {
        row: {
            column: float(value) if pd.notna(value) else 0.0
            for column, value in covariance.loc[row].items()
        }
        for row in covariance.index
    }


def portfolio_volatility(weights: dict[str, float], covariance: dict[str, dict[str, float]]) -> float:
    """Ex-ante annualised volatility, sqrt(w' Sigma w)."""
    variance = 0.0
    for left, left_weight in weights.items():
        if not left_weight:
            continue
        row = covariance.get(left, {})
        for right, right_weight in weights.items():
            if not right_weight:
                continue
            variance += left_weight * right_weight * float(row.get(right, 0.0))
    return math.sqrt(variance) if variance > 0 else 0.0


def volatility_scale(
    weights: dict[str, float],
    covariance: dict[str, dict[str, float]],
    config: DualMomentumConfig,
) -> dict[str, Any]:
    """Scale factor that pulls ex-ante volatility back toward the target.

    Scaling only engages once the estimate exceeds ``high_vol_trigger`` times the target, so
    a portfolio already inside its budget is not re-sized on every 15-minute tick -- turnover
    is a cost, and this overlay is a risk-budgeting device rather than a source of return.
    """
    estimate = portfolio_volatility(weights, covariance)
    target = max(config.target_portfolio_vol, 0.0)
    if estimate <= EPSILON or target <= 0:
        return {"scale": 1.0, "portfolio_volatility": estimate, "engaged": False, "below_floor": False}
    if estimate <= target * max(config.high_vol_trigger, 1.0):
        return {"scale": 1.0, "portfolio_volatility": estimate, "engaged": False, "below_floor": False}

    raw = target / (estimate + EPSILON)
    floor = max(config.vol_scale_floor, 0.0)
    return {
        "scale": min(1.0, max(raw, floor)),
        "portfolio_volatility": estimate,
        "engaged": True,
        # Too volatile to hold even at the floor: the honest answer is the defensive sleeve.
        "below_floor": raw < floor,
    }


def score_to_weights(rows: list[dict[str, Any]], config: DualMomentumConfig) -> dict[str, float]:
    """Score-over-volatility weights, capped per name and in total.

    The score enters as its excess over ``min_base_score``, so a name that only just clears
    the quality floor gets a small position rather than an equal one.
    """
    raw: dict[str, float] = {}
    for row in rows:
        excess = max(float(row.get("base_score", 0.0)) - config.min_base_score, 0.0)
        volatility = float(row.get("annual_volatility", 0.0))
        # sigma ** tilt: negative divides (risk parity), zero ignores it, positive leans in.
        scale = (volatility + EPSILON) ** config.volatility_tilt if config.volatility_tilt else 1.0
        raw[str(row["symbol"])] = excess * scale

    total = sum(raw.values())
    if total <= EPSILON:
        # Every candidate sits exactly at the floor: equal-weight rather than divide by zero.
        share = min(config.name_weight_max, config.risk_on_gross_max / len(raw)) if raw else 0.0
        return {symbol: share for symbol in raw}

    gross = max(config.risk_on_gross_max, 0.0)
    weights = {symbol: min(gross * value / total, config.name_weight_max) for symbol, value in raw.items()}
    total_weight = sum(weights.values())
    if total_weight > gross > 0:
        weights = {symbol: weight * gross / total_weight for symbol, weight in weights.items()}
    return weights


def defensive_weights(
    scored: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
) -> dict[str, float]:
    """Where the book sits when risk-on is not permitted.

    Ranked by medium-term absolute return so the defensive sleeve is itself chosen rather
    than fixed, and deliberately *not* subject to ``name_weight_max``: that cap limits
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


def sentiment_adjusted(
    weights: dict[str, float],
    sentiment_scores: dict[str, float],
    config: DualMomentumConfig,
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
