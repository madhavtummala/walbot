"""Cash dividends from Alpaca's corporate-actions API."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, List

from src.core.interfaces import CashDividend, DividendProvider

logger = logging.getLogger(__name__)

#: Alpaca rejects very long symbol lists on one request, so batches stay modest.
_SYMBOL_BATCH = 20


class AlpacaDividendProvider(DividendProvider):
    """Distribution history from ``/v1beta1/corporate-actions``.

    Preferred over the price-derived alternatives because it reports the payment as the event
    it was -- an amount against an ex-date -- rather than something inferred from a price gap
    that also contains that day's market move.
    """

    name = "alpaca"

    def __init__(self, config: Any) -> None:
        self._config = config

    def _client(self):
        from alpaca.data.historical.corporate_actions import CorporateActionsClient

        key = getattr(self._config, "alpaca_api_key", "")
        secret = getattr(self._config, "alpaca_api_secret", "")
        if not key or not secret:
            raise RuntimeError("Alpaca credentials are not configured")
        return CorporateActionsClient(api_key=key, secret_key=secret)

    def fetch_dividends(
        self, symbols: List[str], start: date, end: date
    ) -> List[CashDividend]:
        from alpaca.data.requests import CorporateActionsRequest

        wanted = sorted({str(symbol).upper() for symbol in symbols if symbol})
        if not wanted:
            return []
        client = self._client()

        out: List[CashDividend] = []
        for index in range(0, len(wanted), _SYMBOL_BATCH):
            batch = wanted[index : index + _SYMBOL_BATCH]
            response = client.get_corporate_actions(
                CorporateActionsRequest(
                    symbols=batch,
                    types=["cash_dividend"],
                    start=start,
                    end=end,
                    limit=1000,
                )
            )
            data = response.data if hasattr(response, "data") else response
            for item in (data or {}).get("cash_dividends", []) or []:
                amount = float(getattr(item, "rate", 0.0) or 0.0)
                ex_date = getattr(item, "ex_date", None)
                if amount <= 0 or ex_date is None:
                    continue
                out.append(
                    CashDividend(
                        symbol=str(item.symbol).upper(),
                        ex_date=ex_date,
                        amount=amount,
                        # Falls back to the ex-date so a ledger always has a date to credit
                        # on; only the settlement lag is lost, never the payment.
                        payable_date=getattr(item, "payable_date", None) or ex_date,
                        record_date=getattr(item, "record_date", None),
                        special=bool(getattr(item, "special", False)),
                        source=self.name,
                    )
                )
        logger.info("Alpaca returned %s cash dividends for %s symbols", len(out), len(wanted))
        return out
