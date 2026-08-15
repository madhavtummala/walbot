"""Cash dividends from yfinance.

The fallback source: no credentials, no rate limit worth planning around. It reports only an
ex-date and an amount, so ``payable_date`` is filled with the ex-date -- the payment is right
and only the settlement lag is missing.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, List

from src.core.interfaces import CashDividend, DividendProvider

logger = logging.getLogger(__name__)


class YFinanceDividendProvider(DividendProvider):
    name = "yfinance"

    def __init__(self, config: Any = None) -> None:
        self._config = config

    def fetch_dividends(
        self, symbols: List[str], start: date, end: date
    ) -> List[CashDividend]:
        import pandas as pd
        import yfinance as yf

        wanted = sorted({str(symbol).upper() for symbol in symbols if symbol})
        out: List[CashDividend] = []
        for symbol in wanted:
            try:
                series = yf.Ticker(symbol).dividends
            except Exception as error:  # one bad symbol must not lose the rest of the batch
                logger.warning("yfinance dividends failed for %s: %s", symbol, error)
                continue
            if series is None or len(series) == 0:
                continue
            for stamp, amount in series.items():
                ex_date = pd.Timestamp(stamp).date()
                value = float(amount or 0.0)
                if value <= 0 or ex_date < start or ex_date > end:
                    continue
                out.append(
                    CashDividend(
                        symbol=symbol,
                        ex_date=ex_date,
                        amount=value,
                        payable_date=ex_date,
                        record_date=None,
                        special=False,
                        source=self.name,
                    )
                )
        logger.info("yfinance returned %s cash dividends for %s symbols", len(out), len(wanted))
        return out
