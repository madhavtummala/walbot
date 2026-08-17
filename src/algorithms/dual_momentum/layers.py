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
    """The raw risk-on gate: is enough of the universe itself in an uptrend?

    Breadth alone, measured over the names the algorithm could actually buy. There is no
    benchmark test here any more, on purpose. Dual momentum's absolute-momentum leg is
    ``eligibility`` -- each name must clear its own 100-day average and its own 60/20-day
    return floors -- and ``analyze_universe`` already holds the defensive sleeve when nothing
    qualifies. "Defensive when there is no strong risk-on option" is therefore already the
    behaviour, and a benchmark overlay on top of it can only subtract: it vetoes names that
    passed every test the strategy asks of them because one unrelated ETF is weak.

    Raw because the hysteresis that turns this into a *state* needs to count consecutive
    observations, and counting requires memory that ``analyze`` is not allowed to have.
    """
    risk_on = [scored[symbol] for symbol in config.risk_on_universe if symbol in scored]
    usable = [row for row in risk_on if row.get("enough_history")]
    # Measured over the names that can be judged, not over the whole list: a symbol with a
    # short cache is unknown, and counting it as "not above its average" is a bearish reading
    # of a data gap.
    above = [row for row in usable if row.get("above_moving_average")]
    breadth = len(above) / len(usable) if usable else 0.0
    coverage = len(usable) / len(risk_on) if risk_on else 0.0

    # Too little usable history is not a bearish reading, it is an unusable one. Say so, and
    # stay risk-off: guessing is worse than declining to act.
    if not usable or coverage < config.min_universe_coverage:
        return {
            "risk_on": False,
            "breadth": breadth,
            "breadth_ok": False,
            "coverage": coverage,
            "data_ok": False,
            "detail": (
                f"only {len(usable)} of {len(risk_on)} risk-on names have "
                f"{config.etf_ma_days} daily bars"
            ),
        }

    breadth_ok = breadth >= config.breadth_min
    return {
        "risk_on": breadth_ok,
        "breadth": breadth,
        "breadth_ok": breadth_ok,
        "coverage": coverage,
        "data_ok": True,
        "detail": (
            f"breadth {breadth:.0%} at or above {config.breadth_min:.0%}"
            if breadth_ok
            else f"breadth {breadth:.0%} below {config.breadth_min:.0%}"
        ),
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
    drop = max(config.max_daily_drop, 0.0)
    session_return = float(row.get("nano_return", 0.0))
    if drop and session_return <= -drop:
        return False, f"Fell {abs(session_return):.0%} in one session"
    slack = max(config.exit_threshold_slack, 0.0)
    if float(row.get("ma_distance", 0.0)) < -slack:
        return False, f"More than {slack:.0%} below its {config.etf_ma_days}-day average"
    if float(row.get("abs_return", 0.0)) <= config.etf_min_abs_return - slack:
        return False, f"{config.etf_abs_return_days}-day return below the exit band"
    if float(row.get("fast_return", 0.0)) <= config.etf_min_fast_return - slack:
        return False, f"{config.etf_fast_return_days}-day return below the exit band"
    return True, ""


# =========================================================================================
# Layer 3: entry timing


def timing(row: dict[str, Any], config: DualMomentumConfig) -> tuple[bool, str]:
    """Whether *now* is a moment to open or add, given the name already qualifies.

    Two independent ways in: accelerating volatility-normalised momentum, or a pullback
    inside an intact uptrend. Deliberately a flag rather than a score term -- as a bonus, a
    deep enough dip could outvote the trend horizons and promote a weak ETF.

    Switchable via ``require_entry_timing``, because what it does is clear but whether it pays
    is not. Both entry paths are shaped to catch a *spike*, and neither fires during a steady
    advance:

    * ``momentum_change`` is the one-session return minus the three-session return, each
      divided by the square root of its own length. In any sustained trend the three-session
      return is about three times the one-session return, and dividing by sqrt(3) leaves the
      slower term ~1.7x larger -- so the difference is structurally negative and the gate asks
      the last session to be more than 58% of the last three sessions' move.
    * the pullback path needs ``nano_z <= -0.75``, a dip a grind-up rally never supplies.

    In November 2023 this blocked all 13 eligible names for eight consecutive sessions while
    the S&P rose 7.98% -- the single worst month of that year against SPY.

    It does not follow that removing it is an improvement, and measurement says it is not: over
    2023 as a whole, switching it off *lost* 8pp (+15.8% to +7.8%) and raised turnover from 74x
    to 85x. Blocking entries is costly when the trend is real and valuable when it is not, and
    over that year the second effect dominated. Treat the November finding as "this gate cannot
    distinguish a trend from a spike", not as "delete it".
    """
    if not config.require_entry_timing:
        return True, "Entry timing off"
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
    """
    def excess(row: dict[str, Any]) -> float:
        return max(float(row.get("base_score", 0.0)) - config.min_base_score, 0.0)

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
        rowset.sort(key=excess, reverse=True)
        if per_theme:
            members[theme] = rowset[:per_theme]

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
