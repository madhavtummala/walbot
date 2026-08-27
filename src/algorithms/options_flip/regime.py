"""Is today eligible for a bullish pullback trade on this symbol?

The first of three gates, and the only one entitled to assert a direction. It is a *conjunction*:
every condition must hold, because each one rules out a different way the thesis can already be
wrong, and passing four out of five means the fifth is telling you something.

**The trend readings come from Rally Rotation's features, not from its score.** ``base_scores``
ranks names against each other and needs a universe to rank within; with two symbols the top one
scores positive by construction, which is exactly the defect
``docs/rally-rotation-simplification.md`` diagnoses. What ports is the per-symbol, absolute
material -- a return over a horizon, a position against a moving average, an annualised
volatility to divide by -- and that is what this module reads.

**Stated in sigma and in ATR, never in raw percent.** The same document's finding: VEA held the
book on a 20-day move of +3.0% against 17% annualised volatility, which is 0.6 sigma and is
noise, while XBI was locked out by -8.4% that was -1.7 sigma for a 31%-volatility ETF. A gap
threshold in percent has the identical defect, so gaps are measured against ATR.

**Earnings are not checked.** Every symbol this strategy trades is an ETF or a trust, which do
not report. A calendar gate would be a permanently-true check, and a permanently-true check on
a deck teaches a reader to stop reading the column.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...core.interfaces import Check
from .indicators import average_true_range, ma_slope, moving_average, opening_range, session_vwap


def bull_regime(
    daily_bars: pd.DataFrame,
    intraday_today: pd.DataFrame,
    *,
    price: float,
    config: Any,
) -> tuple[bool, dict[str, Any], list[Check]]:
    """``(eligible, readings, checks)`` -- whether the bull thesis holds for this symbol today.

    **Two gates were removed here and the reason is the same for both: they cost opportunity and
    bought nothing measurable.**

    *The moving-average slope* is a lagging confirmation that was being used as leading
    permission. A 20-day mean turns only after a move has largely happened: over 2026-08-17 to
    08-20 IBIT ran 36.40 -> 41.19, closing 13% above its own average, while the slope was still
    negative and the strategy sat out every session of it. It first armed on 08-21 at 43.67,
    having missed the whole advance. Measured independently it rejected 34.6% of opportunities
    and *exclusively* rejected 0.32% -- it was not adding a distinct filter, it was adding delay.
    ``Above the trend`` confirms the same thing without waiting for the mean to catch up.

    *The broad-market check* asked a question this strategy has already answered. Candidates come
    from a cross-sectional ranking, so a name is selected precisely for outperforming its peers;
    a symbol making new highs while the index chops is the trade, not a disqualifier. It blocked
    IBIT on 08-25 -- up 23% in a week -- because SPYM sat under its own average. 25.6% rejected,
    0.43% exclusively.
    """
    readings: dict[str, Any] = {}
    checks: list[Check] = []
    if daily_bars is None or daily_bars.empty:
        return False, readings, [Check(
            label="Bull regime", ok=False, value="no daily history",
            limit="daily bars required", blocking=True,
        )]

    closes = daily_bars["close"].astype(float)
    fast = moving_average(closes, int(config.regime_fast_ma_days))
    slow = moving_average(closes, int(config.regime_slow_ma_days))
    # Kept as a reading. It is genuine information about the trend's shape and worthless as a
    # gate, for the reason in this function's docstring.
    slope = ma_slope(closes, int(config.regime_fast_ma_days))
    atr = average_true_range(daily_bars, int(config.atr_days))
    readings.update({"fast_ma": fast, "slow_ma": slow, "slope": slope, "atr": atr})

    # The 20/50 stack was a gate and is now a reading. Measured independently across 1,872
    # opportunities it rejected 70.5% and *exclusively* rejected 0.00%: every session it refused
    # was already refused by "price above the fast average", which is the same statement made
    # once. Two names for one filter is not two filters.
    checks.append(Check(
        label="Trend",
        ok=True,
        value=(
            f"${price:,.2f} / {int(config.regime_fast_ma_days)}d ${fast:,.2f} / "
            f"{int(config.regime_slow_ma_days)}d ${slow:,.2f}"
            if fast > 0 and slow > 0 else "not enough history for both averages"
        ),
        limit=(
            f"reported, not gated — {int(config.regime_fast_ma_days)}d slope {slope:+.2%} over "
            f"5 sessions; the fast average carries the test"
        ),
    ))

    above_fast = bool(price > fast > 0)
    checks.append(Check(
        label="Above the trend",
        ok=above_fast,
        value=f"${price:,.2f} vs {int(config.regime_fast_ma_days)}d ${fast:,.2f}",
        limit=f"price > {int(config.regime_fast_ma_days)}d average",
        blocking=not above_fast,
    ))


    # ── the gap, in ATR: an extreme open is a different distribution, not a better one ──
    prior_close = float(closes.iloc[-2]) if len(closes) >= 2 else 0.0
    session_open = (
        float(intraday_today["open"].astype(float).iloc[0])
        if intraday_today is not None and not intraday_today.empty else price
    )
    gap_atr = ((session_open - prior_close) / atr) if atr > 0 and prior_close > 0 else 0.0
    readings["gap_atr"] = gap_atr
    # Downside only. An up-gap is followed by a *smaller* pullback (corr -0.156 IBIT, -0.118
    # GLD), so it is directionally favourable and merely makes the entry less likely to fill --
    # which the touch probability already prices. A gap down is what breaks the bull thesis.
    calm_open = gap_atr >= -float(config.max_gap_down_atr)
    checks.append(Check(
        label="Open not a gap down",
        ok=calm_open,
        value=f"{gap_atr:+.2f} ATR ({(session_open / prior_close - 1.0) if prior_close else 0:+.2%})",
        limit=f"gap ≥ -{float(config.max_gap_down_atr):.2f} ATR (up-gaps are allowed)",
        blocking=not calm_open,
    ))

    # ── VWAP: is today's average buyer under water or in front? ────────────────────────
    vwap = session_vwap(intraday_today)
    opening = opening_range(intraday_today, int(config.opening_range_minutes))
    readings.update({"vwap": vwap, "opening_range": opening})
    # Above VWAP, or below it but recovering off the opening-range low -- the spec's "above, or
    # recovering toward". A symbol pinned under VWAP *and* under its opening low is not pulling
    # back within an uptrend, it is falling.
    above = price > vwap > 0
    recovering = bool(vwap > 0 and price > opening["low"] > 0 and price <= vwap)
    vwap_ok = above or recovering
    checks.append(Check(
        label="Holding VWAP",
        ok=vwap_ok,
        value=(
            f"${price:,.2f} vs VWAP ${vwap:,.2f}"
            + (" — above" if above else (" — recovering off the opening low" if recovering
                                         else " — below, and below the opening low"))
            if vwap > 0 else "no intraday volume yet"
        ),
        limit="above VWAP, or recovering toward it",
        blocking=not vwap_ok,
    ))


    eligible = all(not check.blocking for check in checks)
    return eligible, readings, checks
