"""Cash-distribution sources.

One module per provider, each implementing :class:`~src.core.interfaces.DividendProvider`.
Nothing here writes to ``market_bars``: the price cache records what the market printed, and
distributions are booked separately as the cash events they are.
"""

from __future__ import annotations

from .alpaca import AlpacaDividendProvider
from .yfinance import YFinanceDividendProvider

#: Preference order. Alpaca first: it is authenticated, it is the same vendor as the trading
#: API, and it carries ``payable_date`` and the ``special`` flag that yfinance does not.
#: yfinance needs no credentials, so it stays as the fallback when Alpaca is unconfigured or
#: rate-limited. The two were measured to agree on all 778 events across the tradable
#: universe, so falling back changes availability rather than answers.
DIVIDEND_PROVIDERS = {
    "alpaca": AlpacaDividendProvider,
    "yfinance": YFinanceDividendProvider,
}

DEFAULT_DIVIDEND_PROVIDER_ORDER = ["alpaca", "yfinance"]

__all__ = [
    "DEFAULT_DIVIDEND_PROVIDER_ORDER",
    "DIVIDEND_PROVIDERS",
    "AlpacaDividendProvider",
    "YFinanceDividendProvider",
]
