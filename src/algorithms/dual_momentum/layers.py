"""Two layers, in order: absolute eligibility, then sizing.

Relative strength selects and absolute momentum permits -- each layer here answers one of
those questions and nothing else.

There were four. The market-regime gate and the entry-timing gate are gone: both had been
switched off in the deployed configuration for long enough to be measured, and both were
measured as subtracting. What remains of the regime layer is :func:`universe_data_ok`, which
is not a market view at all -- it refuses to trade a cache too thin to judge.
"""

from __future__ import annotations

from .config import EPSILON, TRADING_DAYS, DualMomentumConfig
from .scoring import _closes



import logging
import math
from typing import Any

import pandas as pd


logger = logging.getLogger(__name__)




def universe_data_ok(
    scored: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
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


def crash_stop(row: dict[str, Any], config: DualMomentumConfig) -> tuple[bool, str]:
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


def theme_ranks(theme_score: dict[str, float]) -> dict[str, int]:
    """Themes ordered by score, best first. 1 is the strongest theme.

    A theme's own rank, which the algorithm did not previously have. Entry was gated on
    ``entry_rank_max`` against the *ETF* rank of a theme's best member, which asks a different
    question entirely: with fifteen ETFs across eight themes, "top 3 ETF" and "top 3 theme"
    diverge as soon as one theme owns several of the leaders -- metals holding both SLV and
    GLD at ranks 1 and 2 pushed every other theme's best name down the ETF ladder and out of
    entry range, whatever its own standing.
    """
    ordered = sorted(theme_score, key=lambda theme: -float(theme_score.get(theme, 0.0)))
    return {theme: position for position, theme in enumerate(ordered, start=1)}


def hold_eligibility(row: dict[str, Any], config: DualMomentumConfig) -> tuple[bool, str]:
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

    return _water_fill(raw, max(config.risk_on_gross_max, 0.0), max(config.name_weight_max, 0.0))


def _water_fill(raw: dict[str, float], gross: float, cap: float) -> dict[str, float]:
    """Split ``gross`` in proportion to ``raw``, capping each name and re-spreading the excess.

    The cap used to be applied once and the overflow simply dropped, so the book was quietly
    under-invested whenever it bound: one selected name produced a 50% position and 50% idle
    cash rather than the 96% gross the configuration asks for. Measured over 2023, step 2 asked
    for a mean gross of 80% on risk-on days -- 49.9% with one name held, 78.1% with two, 88.2%
    with three -- and the shortfall earned nothing.

    Re-spreading is iterative because capping one name raises everyone else's share, which can
    push the next name over the cap in turn. Terminates when no name exceeds the cap, or when
    every name is capped -- at which point the remainder genuinely cannot be deployed within
    the per-name limit, and the caller parks it in the defensive sleeve.
    """
    weights: dict[str, float] = {}
    uncapped = {symbol: value for symbol, value in raw.items() if value > 0}
    remaining = gross
    while uncapped and remaining > EPSILON:
        total = sum(uncapped.values())
        if total <= EPSILON:
            break
        over = [s for s, v in uncapped.items() if remaining * v / total > cap]
        if not over:
            for symbol, value in uncapped.items():
                weights[symbol] = remaining * value / total
            return weights
        for symbol in over:
            weights[symbol] = cap
            remaining -= cap
            uncapped.pop(symbol)
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


def theme_of(symbol: str, config: DualMomentumConfig) -> str:
    """The theme ``symbol`` belongs to, or the symbol itself when it is unmapped."""
    return str(config.themes.get(str(symbol).upper(), str(symbol).upper()))


def limit_per_theme(
    ranked: list[dict[str, Any]],
    config: DualMomentumConfig,
    already: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Take ``ranked`` best-first, skipping names whose theme is already full.

    ``already`` counts themes held outside this list -- incumbents the caller is keeping --
    so the cap applies to the whole book rather than to each half of it separately.
    """
    cap = max(config.max_positions_per_theme, 0)
    if not cap:
        return list(ranked)
    counts = dict(already or {})
    kept: list[dict[str, Any]] = []
    for row in ranked:
        theme = theme_of(str(row.get("symbol", "")), config)
        if counts.get(theme, 0) >= cap:
            continue
        counts[theme] = counts.get(theme, 0) + 1
        kept.append(row)
    return kept


def theme_allocation(
    themes: set[str],
    rows: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
    current: dict[str, float] | None = None,
) -> dict[str, float]:
    """Budget each selected theme, then split it across that theme's eligible members.

    Two speeds on purpose. *Which* themes are held is a slow, confirmed decision made in
    ``refine``; *how much of each ETF inside a theme* is a fast one, recomputed from today's
    scores with no confirmation at all. Rotating between QQQM and XSD is a sizing question,
    not a change of view, and it should not have to clear the same bar as dropping growth for
    energy.

    A theme is scored by its best eligible member, and the split inside it is proportional to
    each member's score excess -- so a theme whose second name is nearly as strong holds both,
    and one carried by a single name concentrates there.

    "Fast" was previously "free": which members filled a theme's slots was decided on today's
    scores with no reference at all to what was already held, so two near-tied siblings could
    trade places on any session that reordered them by a hair. Measured over 2023 that path was
    13.9% of all turnover. ``current`` and ``intra_theme_delta_to_replace`` put a bar on it --
    a sitting name keeps its slot unless a sibling beats it by that margin. The knob was
    documented in the config and in the dashboard's explainers but read by nothing, so this is
    the rule that was already described rather than a new one.
    """
    held = {str(symbol).upper() for symbol, weight in (current or {}).items() if weight > 0}
    incumbency = max(config.intra_theme_delta_to_replace, 0.0)

    def excess(row: dict[str, Any]) -> float:
        return max(float(row.get("base_score", 0.0)) - config.min_base_score, 0.0)

    def standing(row: dict[str, Any]) -> float:
        """Score for the purpose of *keeping a slot*, not for sizing one.

        The margin is added to the incumbent rather than subtracted from the challenger so a
        name near the quality floor cannot be pushed below zero and stop counting entirely.
        """
        return excess(row) + (incumbency if str(row.get("symbol", "")).upper() in held else 0.0)

    def tilt(row: dict[str, Any]) -> float:
        volatility = float(row.get("annual_volatility", 0.0))
        return (volatility + EPSILON) ** config.volatility_tilt if config.volatility_tilt else 1.0

    members: dict[str, list[dict[str, Any]]] = {}
    for symbol, row in rows.items():
        theme = theme_of(symbol, config)
        if theme not in themes or not int(row.get("eligible", 0)):
            continue
        members.setdefault(theme, []).append(dict(row, symbol=symbol))
    if not members:
        return {}

    per_theme = max(config.max_positions_per_theme, 0)
    for theme, rowset in members.items():
        # Slots go by ``standing`` -- score plus the incumbent's margin -- while the split
        # across whoever wins them stays proportional to raw score, so holding a slot does not
        # also earn a bigger position.
        rowset.sort(key=standing, reverse=True)
        kept = rowset[:per_theme] if per_theme else list(rowset)
        kept.sort(key=excess, reverse=True)
        members[theme] = kept

    # Theme budgets, from the strongest member of each, water-filled so a capped theme's
    # overflow reaches the others instead of becoming cash.
    raw = {theme: excess(rowset[0]) * tilt(rowset[0]) for theme, rowset in members.items()}
    if sum(raw.values()) <= EPSILON:
        share = min(config.name_weight_max, config.risk_on_gross_max / len(raw))
        budgets = {theme: share for theme in raw}
    else:
        budgets = _water_fill(raw, max(config.risk_on_gross_max, 0.0), max(config.name_weight_max, 0.0))

    weights: dict[str, float] = {}
    for theme, budget in budgets.items():
        rowset = members[theme]
        inner = {str(row["symbol"]): excess(row) * tilt(row) for row in rowset}
        total = sum(inner.values())
        if total <= EPSILON:
            for symbol in inner:
                weights[symbol] = budget / len(inner)
            continue
        for symbol, value in inner.items():
            weights[symbol] = budget * value / total
    return weights


def park_residual(
    weights: dict[str, float],
    defensive_book: dict[str, float],
    config: DualMomentumConfig,
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
