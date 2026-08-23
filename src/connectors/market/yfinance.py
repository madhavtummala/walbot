"""yfinance market data.

No quotes: yfinance has no quote endpoint worth the name, so ``fetch_price`` reports nothing and
the dispatcher falls through to the next provider for every symbol.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd

from ...data.bars import calendar_days_for
from ...data.duckdb_store import DAILY_INTERVAL_MINUTES
from ..base import MarketDataProvider
from ..frames import _empty_bars
from ..sources import ProviderUnavailable

logger = logging.getLogger(__name__)

#: yfinance caps intraday history at 60 days regardless of what is asked for.
MAX_INTRADAY_DAYS = 59


class YFinance(MarketDataProvider):
    name = "yfinance"

    def fetch_price(self, symbols: list[str], **extra: Any) -> dict[str, dict[str, Any]]:
        return {}

    def fetch_bars(
        self,
        symbols: list[str],
        *,
        interval_minutes: int,
        lookback_bars: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        **extra: Any,
    ) -> dict[str, pd.DataFrame]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ProviderUnavailable("yfinance is not installed") from exc

        daily = interval_minutes >= DAILY_INTERVAL_MINUTES
        kwargs: dict[str, Any] = {
            "interval": "1d" if daily else f"{interval_minutes}m",
            "auto_adjust": False,
            "progress": False,
            "threads": False,
        }
        if start_date is not None or end_date is not None:
            kwargs.update({"start": start_date, "end": end_date})
        else:
            kwargs["period"] = self._period(interval_minutes, lookback_bars)

        # One symbol at a time: a batched download returns a MultiIndex frame that has to be
        # unstacked per symbol anyway, and a single bad ticker fails the whole batch.
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                frames[symbol] = yf.download(symbol, **kwargs)
            except Exception as exc:  # noqa: BLE001 - one bad symbol must not fail the rest
                logger.warning("Skipping yfinance bars for %s after provider error: %s", symbol, exc)
                frames[symbol] = _empty_bars()
        return frames

    @staticmethod
    def _period(interval_minutes: int, lookback_bars: int) -> str:
        """How far back to ask for, in yfinance's own ``Nd`` vocabulary."""
        if interval_minutes >= DAILY_INTERVAL_MINUTES:
            # Double the bar count so weekends and holidays still leave enough sessions.
            return f"{max(int(lookback_bars * 2), 30)}d"
        days = calendar_days_for(lookback_bars * interval_minutes)
        return f"{min(days, MAX_INTRADAY_DAYS)}d"
