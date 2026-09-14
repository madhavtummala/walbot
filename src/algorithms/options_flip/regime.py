"""Is today eligible for a pullback trade on this symbol, in the direction already proposed?

The first of three gates, and -- since the bear side arrived -- a pure veto: it is handed a
direction and answers only whether that thesis is intact today. It no longer asserts one. Exactly
one measure proposes a direction (``trend_strength``, against ``min_trend_strength`` on the long
side and ``min_bear_trend_strength`` on the short), and the gap between those two thresholds is
the neutral band. That split is what makes "failed the bull test" structurally different from
"passed the bear test": a veto cannot promote a symbol into the opposite direction, so no
combination of these checks can turn a merely-not-bullish name into a short.

A conjunction, since each condition rules out a different way the thesis can already be wrong. Readings are absolute
and per-symbol (borrowed from Rally Rotation's features, not its cross-sectional score), and
stated in sigma/ATR rather than raw percent so one threshold works across symbols of different
volatility.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...core.interfaces import Check
from ...core.options import CALL, PUT
from .indicators import (
    average_true_range,
    directional_volume,
    ma_slope,
    moving_average,
    opening_range,
    session_vwap,
)


def regime(
    daily_bars: pd.DataFrame,
    intraday_today: pd.DataFrame,
    *,
    price: float,
    config: Any,
    direction: str = CALL,
    for_exit: bool = False,
) -> tuple[bool, dict[str, Any], list[Check]]:
    """``(eligible, readings, checks)`` -- whether ``direction``'s thesis holds today.

    Every reading below is sign-free and computed once for both sides; only the three blocking
    comparisons flip. They are reflected rather than re-derived, which is deliberate: the short
    side has none of the out-of-sample work behind it that the long side's thresholds carry (see
    the docstring's three removed gates), so it starts as the honest mirror and earns its own
    numbers from the walk-forward rather than being guessed at here.

    **Three gates were removed here and the reason is the same for all three: they cost
    opportunity and bought nothing measurable.**

    *The same-day checks* (``Holding VWAP``, ``Open not a gap down``) are demoted to readings
    when ``for_exit`` is set -- re-checked on a held position (see ``_sell_ok``), not on a new
    entry. Measured on the full August walk-forward: of four large exits this pair forced, three
    were an intact uptrend's one ordinary red day, sold at the day's low the session before it
    resumed (GLD 380C Aug 13, GLD 390C Aug 18, IBIT 41C Aug 28 -- each recovered or kept climbing
    within 1-2 sessions). A same-day measure answers "should I buy into today's weakness", which
    is the right question on entry; it does not answer "has the multi-day thesis this position
    was opened on broken", which is what a held position needs asked. Kept as blocking on entry,
    where intraday caution is still correct.

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
    bearish = direction == PUT
    readings: dict[str, Any] = {}
    checks: list[Check] = []
    if daily_bars is None or daily_bars.empty:
        return False, readings, [Check(
            label="Bear regime" if bearish else "Bull regime", ok=False, value="no daily history",
            limit="daily bars to measure the trend against", blocking=True,
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
            f" — {int(config.regime_fast_ma_days)}d slope {slope:+.2%} over 5 sessions"
            if fast > 0 and slow > 0 else "not enough history for both averages"
        ),
        # Deliberately no ``limit``: this is a reading, and ``limit`` means "what it had to be".
        # The deck prefixes that field with "needs", so prose there rendered as
        # "needs reported, not gated" -- a requirement stated for a check that requires nothing.
        gate=False,
    ))

    # A strict crossing, and the one check here that would partition the space if it were the
    # measure proposing direction -- every symbol sits on one side of its own average, so
    # "not above" would read as "below" with no room in between. It is safe as written only
    # because it is a veto: a symbol has to have *already* cleared the trend-strength threshold
    # in this direction to get here, and the two disagree on about 1% of sessions.
    # The deadband makes the crossing hysteretic rather than strict. On entry it is measured on
    # the side the thesis needs (be decisively there); on exit it is measured on the *opposite*
    # side (be decisively wrong before the thesis is declared broken), so a position is not closed
    # by the same cent-wide flicker that would have been too weak to open it.
    # Exit only -- see ``exit_trend_band_atr``. On the way in the crossing is a veto on a
    # direction ``trend_strength`` has already proposed, so there is no flicker for a band to
    # absorb; on the way out it decides whether a thesis is broken, and there is.
    band_atr = float(getattr(config, "exit_trend_band_atr", 0.0)) if for_exit else 0.0
    band = band_atr * atr if atr > 0 else 0.0
    threshold = (fast + band if bearish else fast - band) if for_exit else fast
    side_ok = bool(fast > 0 and (price < threshold if bearish else price > threshold))
    checks.append(Check(
        label="Below the trend" if bearish else "Above the trend",
        ok=side_ok,
        value=(
            f"${price:,.2f} vs {int(config.regime_fast_ma_days)}d ${fast:,.2f}"
            + (f" — {(price - fast) / atr:+.2f} ATR" if atr > 0 else "")
        ),
        limit=(
            f"price {'<' if bearish else '>'} ${threshold:,.2f}"
            + (f" ({int(config.regime_fast_ma_days)}d average "
               f"{'+' if threshold > fast else '-'} {abs(band_atr):.2f} ATR"
               f"{', conceding on the way out' if for_exit else ''})"
               if band_atr else f" ({int(config.regime_fast_ma_days)}d average)")
        ),
        blocking=not side_ok,
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
    # One ceiling, applied to whichever end of the open is adverse for this direction.
    ceiling = float(config.gap_atr)
    if bearish:
        calm_open = gap_atr <= ceiling
        gap_label = "Open not a gap up"
        gap_limit = f"gap ≤ +{ceiling:.2f} ATR (down-gaps are allowed)"
    else:
        calm_open = gap_atr >= -ceiling
        gap_label = "Open not a gap down"
        gap_limit = f"gap ≥ -{ceiling:.2f} ATR (up-gaps are allowed)"
    checks.append(Check(
        label=gap_label,
        ok=calm_open,
        value=f"{gap_atr:+.2f} ATR ({(session_open / prior_close - 1.0) if prior_close else 0:+.2%})",
        limit=gap_limit,
        blocking=not calm_open and not for_exit,
    ))

    # ── VWAP: is today's average buyer under water or in front? ────────────────────────
    vwap = session_vwap(intraday_today)
    opening = opening_range(intraday_today, int(config.opening_range_minutes))
    readings.update({"vwap": vwap, "opening_range": opening})
    # Above VWAP, or below it but recovering off the opening-range low -- the spec's "above, or
    # recovering toward". A symbol pinned under VWAP *and* under its opening low is not pulling
    # back within an uptrend, it is falling.
    if bearish:
        # The reflection: a symbol pinned *above* VWAP and above its opening high is not selling
        # off within a downtrend, it is rallying.
        on_side = bool(vwap > 0 and price < vwap)
        fading = bool(vwap > 0 and 0 < opening["high"] and price < opening["high"] and price >= vwap)
        vwap_ok = on_side or fading
        vwap_note = (" — below" if on_side else (" — fading off the opening high" if fading
                                                 else " — above, and above the opening high"))
        vwap_limit = "below VWAP, or fading toward it"
    else:
        on_side = price > vwap > 0
        fading = bool(vwap > 0 and price > opening["low"] > 0 and price <= vwap)
        vwap_ok = on_side or fading
        vwap_note = (" — above" if on_side else (" — recovering off the opening low" if fading
                                                 else " — below, and below the opening low"))
        vwap_limit = "above VWAP, or recovering toward it"
    checks.append(Check(
        label="Under VWAP" if bearish else "Holding VWAP",
        ok=vwap_ok,
        value=(f"${price:,.2f} vs VWAP ${vwap:,.2f}" + vwap_note if vwap > 0
               else "no intraday volume yet"),
        limit=vwap_limit,
        blocking=not vwap_ok and not for_exit,
    ))


    # ── directional volume: is today's tape buyer- or seller-heavy so far? ─────────────
    # A reading, not a gate, for the same reason the slope is: it is genuine information about
    # today's tape and has not been measured to reject anything. Promote it to a blocking check
    # only once it has earned that the way ``entry_reach``/``exit_reach`` did -- against
    # forward returns, not intuition.
    volume_split = directional_volume(intraday_today)
    readings["volume_imbalance"] = volume_split["imbalance"]
    checks.append(Check(
        label="Directional volume",
        ok=True,
        value=(
            f"{volume_split['imbalance']:+.0%} imbalance "
            f"(buy {volume_split['buy_volume']:,.0f} / sell {volume_split['sell_volume']:,.0f})"
            f" — buy/sell split by each bar's own open-to-close"
            if (volume_split["buy_volume"] + volume_split["sell_volume"]) > 0
            else "no directional volume yet"
        ),
        # A reading, so no ``limit`` -- see the trend check above.
        gate=False,
    ))

    eligible = all(not check.blocking for check in checks)
    return eligible, readings, checks


#: The long-side name this module carried while it was the only side. Kept so nothing that only
#: ever wanted the bull gates has to learn about directions.
def bull_regime(
    daily_bars: pd.DataFrame,
    intraday_today: pd.DataFrame,
    *,
    price: float,
    config: Any,
    for_exit: bool = False,
) -> tuple[bool, dict[str, Any], list[Check]]:
    return regime(
        daily_bars, intraday_today, price=price, config=config,
        direction=CALL, for_exit=for_exit,
    )
