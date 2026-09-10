"""The entry level, the target, and how often each is actually reached.

``E = P - k_entry × ATR`` and ``T = P + k_target × ATR``, with the multiples conditioned on what
comparable past sessions did (bucketed on session-fraction-remaining and position-vs-open, the
two features that held up out of sample) rather than a fixed offset. A limit order does not
guarantee a fill, so the touch/target probabilities matter as much as the levels themselves.

Horizons are asymmetric: the dip is measured over one session (an unfilled entry is abandoned at
the close), the run over the whole hold (that's how long the position has to find it).
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

    The dip is a single-session measure (an unfilled entry is abandoned at the close); the run is
    measured over ``run_horizon`` (``max_hold_sessions``) sessions, since that's how long a hold
    has to find it. ATR units so one set of multiples works across symbols and vol regimes.
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
        # Skip rather than truncate a window that runs off the end of history -- a short window
        # understates the run and biases the target downward.
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
    # out of sample. A tolerance, not a bucket edge, so a day never falls between cells.
    position = (price / session_open - 1.0) if session_open > 0 else 0.0
    near = samples[(samples["pos_vs_open"] - position).abs() <= float(config.bucket_tolerance)]
    # A hard floor rather than one that scales with the pool -- a small scaled bucket let a
    # tail move sitting inside the lookback get read back out as a forecast (see MIN_BUCKET).
    floor = MIN_BUCKET
    conditional = len(near) >= floor
    pool = near if conditional else samples

    # The entry sits at a dip depth a majority of comparable days reach, since an unfilled entry
    # is the failure mode this design exists to price.
    k_entry = float(np.quantile(pool["dip"].values, 1.0 - float(config.entry_reach)))
    p_touch = float((pool["dip"].values >= k_entry).mean())

    # The target is taken over the *dipped* subset, not the whole pool: conditioning on the dip
    # makes ``exit_reach`` mean what it says (days that pulled back this far and then ran far
    # enough), rather than a target defined against the whole pool contradicting its own floor.
    dipped = pool[pool["dip"].values >= k_entry]
    runs = dipped["run"].values if len(dipped) >= floor else pool["run"].values
    k_target = float(np.quantile(runs, 1.0 - float(config.exit_reach)))
    p_target = float((runs >= k_target).mean()) if len(runs) else 0.0

    entry = price - k_entry * atr
    target = entry + k_target * atr

    return {
        "entry": entry, "target": target,
        "p_touch": p_touch, "p_target": p_target,
        "sample": int(len(pool)), "conditional": bool(conditional),
        "k_entry": k_entry, "k_target": k_target,
    }
