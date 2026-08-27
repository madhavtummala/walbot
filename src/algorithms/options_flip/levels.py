"""The entry level, the target, and -- the part that matters -- how often each is actually reached.

``E = P - k_entry × ATR`` and ``T = P + k_target × ATR``, where the multiples come from what
comparable past sessions did rather than from intuition. A limit order controls price and does
not guarantee a fill, so the two probabilities below are not decoration: a strategy that models
its entry as certain is measuring a trade it often never had.

**Why this replaces a fixed offset.** A bid always the same distance below the open is a filter
that admits days which fell and excludes days which rose -- measured on 185 sessions per symbol,
the days a fixed 0.40% bid *missed* were worth +1.73% on IBIT and +0.67% on GLD against −0.34%
and −0.42% for the days it filled, both at p < 0.001. The fix is not a different constant. It is
to condition the level on the day in front of you, and to refuse the setup when the conditional
probability of being filled in time is poor.

**Buckets are deliberately shallow.** The reference design buckets on nine features; three levels
on each is nineteen thousand cells against the ~180 sessions a symbol has here, so every cell
would hold nothing and every quantile would be one observation wearing a confidence interval.
This buckets on the two features that survived out-of-sample testing -- how much of the session
remains, and how far price already sits from the open -- and falls back to the unconditional
distribution when a bucket is thin. The honest limitation is stated on the check: a conditional
quantile from fifteen samples is reported with its sample size so a reader can discount it.

**What the data says about each side.** Forward-run is weakly predictable out of sample (R² about
+0.14 pooled, +0.19 on IBIT); forward-dip is not (R² about 0.00, negative on IBIT). So the target
is priced from a conditional quantile and the *entry* leans on the touch probability rather than
on a precise depth -- the model is used where it has skill and distrusted where it does not.

**The horizons are deliberately asymmetric.** A decent pullback is asked for *today*, because the
entry is abandoned unfilled at the close; a larger move is asked for over the whole hold, because
that is how long the position has to find it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

#: Below this many comparable sessions a bucket is anecdote and the unconditional sample is used.
MIN_BUCKET = 20


def _day_frames(intraday_history: pd.DataFrame, lookback: int = 0) -> list[pd.DataFrame]:
    """The most recent ``lookback`` past sessions, each sorted. Zero means all of them."""
    if intraday_history is None or intraday_history.empty:
        return []
    frames = [frame.sort_values("ts") for _day, frame in intraday_history.groupby("day")]
    return frames[-lookback:] if lookback and lookback > 0 else frames


def excursion_samples(
    intraday_history: pd.DataFrame, *, minute: int, atr: float, lookback: int = 0,
    run_horizon: int = 1,
) -> pd.DataFrame:
    """For each past session: the further dip and further run from ``minute``, in ATR units.

    **The two are measured over different horizons, and that asymmetry is the design.** The entry
    has to fill *today* or it is abandoned at the close, so the dip is a single-session measure.
    The target has ``max_hold_sessions`` to be reached, so the run is measured over that many.

    Pricing both over one session is a bug this module carried until it was measured: on IBIT
    from 10:00, the median further run is 0.256 ATR over one session and 0.596 over three -- a
    factor of 2.3, and 3.3 on GLD. Charging three sessions of theta against a one-session target
    is what made the modelled profit negative on a setup whose direction was right.

    In ATR units so one set of multiples works across symbols and volatility regimes.
    """
    rows = []
    frames = _day_frames(intraday_history, lookback)
    span = max(int(run_horizon), 1)
    for index, frame in enumerate(frames):
        upto = frame[frame["minute"] <= minute]
        after = frame[frame["minute"] > minute]
        if upto.empty or after.empty:
            continue
        price = float(upto["close"].astype(float).iloc[-1])
        session_open = float(frame["open"].astype(float).iloc[0])
        if price <= 0 or atr <= 0:
            continue
        # The run window runs on into the following sessions, which is where a multi-day hold
        # actually earns. A window that runs off the end of the history is skipped rather than
        # truncated: a short window would understate the run and bias the target downward.
        forward = frames[index:index + span]
        if len(forward) < span:
            continue
        highs = [float(after["high"].astype(float).max())]
        highs += [float(f["high"].astype(float).max()) for f in forward[1:]]
        rows.append({
            "dip": (price - float(after["low"].astype(float).min())) / atr,
            "run": (max(highs) - price) / atr,
            "pos_vs_open": (price / session_open - 1.0) if session_open > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def conditional_levels(
    intraday_history: pd.DataFrame,
    *,
    minute: int,
    price: float,
    session_open: float,
    atr: float,
    config: Any,
) -> dict[str, Any]:
    """``E``, ``T`` and the probabilities of reaching them, from comparable sessions.

    Returns absolute prices, not offsets, plus the sample the quantiles came from so the deck can
    report how much evidence is behind them.
    """
    blank = {
        "entry": 0.0, "target": 0.0, "p_touch": 0.0, "p_target": 0.0,
        "sample": 0, "conditional": False, "k_entry": 0.0, "k_target": 0.0,
    }
    if atr <= 0 or price <= 0:
        return blank
    samples = excursion_samples(
        intraday_history, minute=minute, atr=atr,
        lookback=int(getattr(config, "level_lookback_days", 0) or 0),
        run_horizon=int(getattr(config, "max_hold_sessions", 1) or 1),
    )
    if samples.empty:
        return blank

    # Condition on where price sits against its open -- the one day-shape feature that held up
    # out of sample. A tolerance rather than a bucket edge, so a day never falls between cells.
    position = (price / session_open - 1.0) if session_open > 0 else 0.0
    near = samples[(samples["pos_vs_open"] - position).abs() <= float(config.bucket_tolerance)]
    # A hard floor, not one that scales with the pool.
    #
    # It used to scale -- ``min(20, max(n // 2, 5))`` -- so a 20-session lookback could produce a
    # six-session "conditional" bucket, and a quantile of six observations is an order statistic.
    # Live on 2026-08-27 that put IBIT's target at 6.09 ATR, asking $45.14 -> $52.15: a 15.5%
    # move in four sessions, which is the +26.8% rally sitting inside the lookback being read
    # back out as a forecast. The median 4-session run is 0.81 ATR at every lookback tested; only
    # the *tail* moves, from 2.12 ATR at 80 sessions to 5.29 at 20. A small bucket lands in it.
    floor = MIN_BUCKET
    conditional = len(near) >= floor
    pool = near if conditional else samples

    # The entry sits at a dip depth a *majority* of comparable days reach, because an entry that
    # never fills is the failure mode this design exists to price.
    k_entry = float(np.quantile(pool["dip"].values, 1.0 - float(config.entry_reach)))
    # Measured, not assumed equal to the knob. They agree by construction here, and the check
    # costs nothing -- a placement that silently stopped matching its own probability is exactly
    # the failure this number exists to catch.
    p_touch = float((pool["dip"].values >= k_entry).mean())

    # The target is taken over the *dipped* subset, not the whole pool, and that is not a detail.
    #
    # Taken over the whole pool it contradicted itself: a target at the 60th percentile of every
    # session is by construction reached by about 40% of them, so a floor demanding 45% could
    # almost never be met. Measured over 1,507 candidate sessions the pair rejected 4,990 of
    # ~5,000 opportunities and armed two trades -- a gate that fires twice in six months is not
    # selective, it is broken.
    #
    # Conditioning on the dip makes ``target_reach`` mean exactly what it says: the share of
    # comparable days that pulled back this far *and then* ran far enough to pay. That is the
    # number the trade actually depends on, and it makes a separate minimum redundant -- the
    # target is placed at the reach the caller asked for, so it cannot disagree with itself.
    dipped = pool[pool["dip"].values >= k_entry]
    runs = dipped["run"].values if len(dipped) >= floor else pool["run"].values
    k_target = float(np.quantile(runs, 1.0 - float(config.target_reach)))
    p_target = float((runs >= k_target).mean()) if len(runs) else 0.0

    entry = price - k_entry * atr
    target = entry + k_target * atr

    return {
        "entry": entry, "target": target,
        "p_touch": p_touch, "p_target": p_target,
        "sample": int(len(pool)), "conditional": bool(conditional),
        "k_entry": k_entry, "k_target": k_target,
    }
