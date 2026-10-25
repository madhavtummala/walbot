"""Predict the chosen option contract's band from its own price history, falling back to the
underlying translation when the contract has no history of its own yet (a fresh pick) or its own
prediction is an artifact of a thin sample. Reuses :mod:`.levels`'s excursion machinery, which is
agnostic to what instrument its bars describe.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...core.interfaces import MARKET_TZ
from .indicators import average_true_range
from .levels import conditional_levels


def prepare_option_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw option OHLCV (UTC ``timestamp``) to the ``ts``/``minute``/``day`` shape
    ``excursion_samples`` needs. Rows without a parsable time are dropped."""
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "minute", "day"])
    frame = bars.copy()
    frame["ts"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    frame["minute"] = frame["ts"].dt.hour * 60 + frame["ts"].dt.minute
    frame["day"] = frame["ts"].dt.date
    return frame[["ts", "open", "high", "low", "close", "minute", "day"]].dropna(subset=["ts"])


def option_daily_atr(option_bars: pd.DataFrame, window: int = 14) -> float:
    """The contract's own daily true range in premium, Wilder-averaged over ``window`` sessions."""
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

    Mirrors :func:`~.levels.conditional_levels` run on the option's own bars and ATR.
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
    """Return the best band -- the option's own when it has a sample and passes :func:`_sane`,
    else ``underlying_translation`` (the underlying-derived entry/target the caller already
    computed via ``option_price_for``)."""
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
    """Whether the option band's own prediction is usable, or a thin-sample artifact."""
    return bool(
        band.get("sample", 0) > 0
        and option_mark > 0
        and band.get("entry", 0.0) > 0.0
        and band.get("target", 0.0) > band.get("entry", 0.0)
        and band.get("k_entry", 0.0) >= 0.0
        and band.get("target", 0.0) <= option_mark * 3.0
    )
