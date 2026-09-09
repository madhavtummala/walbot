"""Predicting the chosen option contract's band from its own price history.

This replaces the old assumption behind :func:`~.excursion.option_price_for` -- that a contract's
own series is too thin to predict and the underlying must be translated instead. The underlying
is still translated as the *cold-start* fallback, because a freshly chosen contract genuinely has
no history of its own yet. But once a selected contract has accumulated enough sessions, its own
premium low and run are the thing being traded, and they are predicted here, in premium dollars,
from the same excursion-quantile machinery the underlying band uses.

The reuse is deliberate and cheap. ``conditional_levels`` and ``excursion_samples`` in
:mod:`.levels` are agnostic to what instrument their bars describe -- they only need
``ts``/``minute``/``day`` columns and an ATR in the instrument's own price units. The option's
own 5m history is normalized to that shape, its own daily ATR is computed in premium, and the
resulting ``entry``/``target`` are already in the units the resting order uses.

**Why the option's own history now wins when it exists.** The option's low and run are what the
exit actually collects: an option that retraces its own premium low before running is where the
fill and the target both happen, and nothing about the underlying's path captures the contract's
theta drift, its widening spread, or the way a deep-ITM call barely moves. Translating the
underlying through a static delta ignores all three. Predicting the premium directly is the
generalisation the user asked for; the underlying translation is retained only as the fallback
for the no-history case.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...core.interfaces import MARKET_TZ
from .indicators import average_true_range
from .levels import conditional_levels


def prepare_option_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw option OHLCV to the ``ts``/``minute``/``day`` shape the band machinery needs.

    ``fetch_option_price_history`` returns bars keyed on a UTC ``timestamp``; ``excursion_samples``
    reads ``minute``, ``day`` and a market-time ``ts`` column. Rows without a parsable time are
    dropped so a sparse option print does not become a ``NaT`` key.
    """
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "minute", "day"])
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    frame["minute"] = frame["ts"].dt.hour * 60 + frame["ts"].dt.minute
    frame["day"] = frame["ts"].dt.date
    return frame[["ts", "open", "high", "low", "close", "minute", "day"]].dropna(subset=["ts"])


def option_daily_atr(option_bars: pd.DataFrame, window: int = 14) -> float:
    """The contract's own daily true range in premium, Wilder-averaged over ``window`` sessions.

    Aggregates the 5m bars into per-session OHLC candles and feeds them to the same ATR used on
    the underlying, so a premium's gap (a fresh contract that lists and immediately reprices) is
    part of the move a position is exposed to rather than being ignored.
    """
    if option_bars is None or option_bars.empty:
        return 0.0
    frame = option_bars.copy()
    frame["day"] = pd.to_datetime(frame["ts"]).dt.date
    daily = frame.groupby("day").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
    ).reset_index()
    return average_true_range(daily, window=int(window))


def option_band(
    option_bars: pd.DataFrame,
    *,
    minute: int,
    option_mark: float,
    session_open: float,
    config: Any,
    max_hold: int = 1,
    atr: float | None = None,
) -> dict[str, Any]:
    """The entry and target in *premium*, from the option contract's own price history.

    Mirrors :func:`~.levels.conditional_levels` but run on the option's own bars and its own ATR.
    ``option_mark`` takes the place of the underlying's ``price``, and the returned ``entry`` /
    ``target`` are premium dollars. When the sample is too thin to be trusted (below
    ``MIN_BUCKET`` comparable sessions) the quantiles still fall back to the unconditional
    distribution, just as the underlying band does -- there is no separate cold-start here, that
    is the caller's decision when the whole sample is empty.
    """
    levels = conditional_levels(
        option_bars, minute=minute, price=option_mark, session_open=session_open,
        atr=atr if atr else 0.0, config=config,
    )
    return {**levels, "source": "option", "atr": atr}


def choose_band(
    option_bars: pd.DataFrame,
    *,
    minute: int,
    option_mark: float,
    session_open: float,
    config: Any,
    max_hold: int,
    underlying_translation: dict[str, float] | None,
    contract: Any,
    underlying_now: float,
) -> dict[str, Any]:
    """Return the best band -- the option's own when sane, else the underlying translation.

    ``underlying_translation`` carries the underlying-derived ``entry``/``target`` (premium,
    via ``option_price_for``) from the caller so this module stays pure of the underlying fetch.
    ``contract`` supplies the ``delta`` and ``mark`` for the fallback path.

    The option's own band wins only when it has a non-empty sample *and* its output passes a
    sanity gate. The option's price history for a single contract is thin and noisy -- a handful
    of sessions can put ``k_entry`` below zero or ask a target several times the premium -- so
    absurd output is discarded in favour of the translation, which is the honest answer when the
    contract cannot yet speak for itself.
    """
    if option_bars is not None and not option_bars.empty:
        atr = option_daily_atr(option_bars, int(getattr(config, "atr_days", 14)))
        band = option_band(
            option_bars, minute=minute, option_mark=option_mark,
            session_open=session_open, config=config, max_hold=max_hold, atr=atr,
        )
        if _sane(band, option_mark):
            return band

    def _fallback() -> dict[str, Any]:
        return {
            **underlying_translation,
            "source": "underlying",
            "atr": None,
            "entry_option": getattr(contract, "midpoint", option_mark) if contract is not None else option_mark,
            "target_option": underlying_translation.get("target", 0.0),
        }

    if underlying_translation and underlying_translation.get("entry", 0.0) > 0 and contract is not None:
        return _fallback()
    return {"entry": 0.0, "target": 0.0, "p_touch": 0.0, "p_target": 0.0,
            "sample": 0, "conditional": False, "k_entry": 0.0, "k_target": 0.0,
            "source": "none", "atr": None}


def _sane(band: dict[str, Any], option_mark: float) -> bool:
    """Whether the option band's own prediction is usable rather than an artifact of a thin sample.

    Guards the three ways a small sample lies: an entry at or below the session's own low-bound
    (``k_entry`` at or below zero means "dip to here is a majority event", which is trivially
    meaningless after only a handful of sessions), an entry that fails to clear its own cost, and
    a target so far above the premium that it can only be the tail of the distribution read back
    out as a forecast.
    """
    return bool(
        band.get("sample", 0) > 0
        and option_mark > 0
        and band.get("entry", 0.0) > 0.0
        and band.get("target", 0.0) > band.get("entry", 0.0)
        and band.get("k_entry", 0.0) >= 0.0
        and band.get("target", 0.0) <= option_mark * 3.0
    )
