"""Bar resolutions: what each provider can serve, and how many bars a window needs.

A lookback is stated in *market* minutes, not wall-clock ones, and a provider serves whatever
grid it has. Both conversions live here so a caller never has to know either.
"""

from __future__ import annotations

import logging
import math

from ..core.config import Config
from ..data.bars import TRADING_MINUTES_PER_DAY
from ..data.duckdb_store import DAILY_INTERVAL_MINUTES

logger = logging.getLogger(__name__)

#: Minute frequencies Schwab's pricehistory endpoint accepts. Anything else is a 400.
SCHWAB_MINUTE_FREQUENCIES = frozenset({1, 5, 10, 15, 30})

#: Bar resolutions each provider can actually serve, finest first. yfinance's sub-15m data is
#: capped at about a week of history, so 15 minutes stays its practical floor; Schwab has no
#: such cap, which is what makes a finer default worth taking.
PROVIDER_BAR_MINUTES: dict[str, tuple[int, ...]] = {
    "schwab": tuple(sorted(SCHWAB_MINUTE_FREQUENCIES)),
    "alpaca": (1, 5, 15, 30, 60),
    "finnhub": (1, 5, 15, 30, 60),
    "yfinance": (15, 30, 60),
}

def default_bar_minutes(config: Config) -> int:
    """The configured preferred resolution for fine-grained bars."""
    return max(int(getattr(config, "market_data_bar_minutes", 5) or 5), 1)


def resolve_bar_minutes(provider: str, wanted: int) -> int:
    """The finest grid ``provider`` can serve at or below ``wanted``.

    Coarser rather than an error: horizons are stated in minutes now, so a provider that
    cannot hit the requested grid still answers the question, just with less resolution.
    """
    # A daily request is not on the intraday grid. ``PROVIDER_BAR_MINUTES`` lists the minute
    # frequencies a vendor serves, so resolving 1440 against Alpaca's ``(1, 5, 15, 30, 60)``
    # picked 60 and quietly turned every daily fetch into an hourly one.
    if int(wanted) >= DAILY_INTERVAL_MINUTES:
        return DAILY_INTERVAL_MINUTES
    supported = PROVIDER_BAR_MINUTES.get(provider.lower())
    if not supported:
        return max(int(wanted), 1)
    eligible = [minutes for minutes in supported if minutes <= int(wanted)]
    return max(eligible) if eligible else min(supported)


def bars_for_minutes(lookback_minutes: int, bar_minutes: int) -> int:
    """How many bars of ``bar_minutes`` a window of ``lookback_minutes`` of market time spans.

    Lookbacks count minutes the market was open, so a 4800-minute window is about twelve
    sessions -- and asking a provider for 4800/5 = 960 five-minute bars would fetch nearly
    four times what it needs. One session is 390 minutes however finely it is sliced.
    """
    if lookback_minutes <= 0 or bar_minutes <= 0:
        return 0
    sessions = lookback_minutes / TRADING_MINUTES_PER_DAY
    per_session = max(TRADING_MINUTES_PER_DAY // bar_minutes, 1)
    return max(int(math.ceil(sessions * per_session)) + 1, 1)

