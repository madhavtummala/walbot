"""Schwab option chains, and the pre-market read that decides whether to use them.

Two capabilities that sit deliberately *outside* :class:`~src.connectors.base.MarketDataProvider`.
That ABC is equity price-and-bars, and both of these break its assumptions in ways that would
damage callers who never asked for them:

* A chain is not a symbol's price series. It is a few hundred instruments discovered at request
  time, which is the opposite of the "declare your symbols up front" contract the provider
  interface and the bar cache are built on.
* Pre-market data cannot enter the bar cache at all. ``market_bars`` is keyed by
  ``(provider, symbol, interval_minutes, timestamp)`` with no session dimension, so extended-hours
  bars would merge into the regular-hours series for the same symbol and interval and silently
  corrupt it. Every horizon conversion downstream also assumes ``TRADING_MINUTES_PER_DAY = 390``
  (see ``src/data/bars.py``), which an 04:00-20:00 session is not.

So both go straight to the API and neither is cached. A chain is only useful for the seconds in
which its quotes are current, and the pre-market summary is read once per symbol per session.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import pandas as pd

from ...common.config_utils import json_number
from ...core.interfaces import MARKET_TZ
from ...core.options import CALL, PUT, OptionContract
from ..http import _bearer_auth_header, _request_json
from ..sources import EOD_MARKET_CATEGORY, MARKET_CATEGORY, ProviderUnavailable, _schwab_token
from .schwab import PRICE_HISTORY_URL

logger = logging.getLogger(__name__)

CHAINS_URL = "https://api.schwabapi.com/marketdata/v1/chains"

#: The regular session opens at 09:30 market time; anything earlier in the day is pre-market.
#: Schwab serves extended hours from 04:00, and the useful signal is concentrated in the last
#: hour or so, but the whole window is read so the volume figure means something.
REGULAR_OPEN = time(9, 30)
PREMARKET_OPEN = time(4, 0)


def fetch_option_chain(
    config: Any,
    underlying: str,
    *,
    option_type: str = "",
    min_dte: int = 0,
    max_dte: int = 120,
    as_of: date | None = None,
) -> list[OptionContract]:
    """Quoted contracts for ``underlying``, filtered to a DTE window.

    The DTE window is applied in the request rather than after it. A full chain on a liquid ETF
    is thousands of rows across every weekly expiry; asking for the fortnight the strategy can
    actually trade is the difference between a small response and a slow one.

    Greeks come back on the same rows as the quotes, which is what makes a delta band usable
    without a second call or a pricing model of our own.
    """
    token = _schwab_token(config, MARKET_CATEGORY)
    if not token:
        raise ProviderUnavailable("Schwab access token is not configured")

    today = as_of or datetime.now(MARKET_TZ).date()
    contract_type = {CALL: "CALL", PUT: "PUT"}.get(option_type, "ALL")
    payload = _request_json(
        "schwab", MARKET_CATEGORY, CHAINS_URL,
        {
            "symbol": str(underlying).upper(),
            "contractType": contract_type,
            "fromDate": (today + timedelta(days=max(min_dte, 0))).isoformat(),
            "toDate": (today + timedelta(days=max(max_dte, min_dte))).isoformat(),
            # Schwab computes the greeks only when asked; without this delta comes back as NaN.
            "includeUnderlyingQuote": "false",
            "strategy": "SINGLE",
        },
        headers=_bearer_auth_header(token),
    ) or {}

    contracts: list[OptionContract] = []
    for map_key, kind in (("callExpDateMap", CALL), ("putExpDateMap", PUT)):
        contracts.extend(_contracts_from_map(payload.get(map_key) or {}, underlying, kind))
    contracts.sort(key=lambda contract: (contract.expiry, contract.strike))
    return contracts


def _contracts_from_map(exp_map: dict[str, Any], underlying: str, option_type: str) -> list[OptionContract]:
    """Flatten Schwab's ``{"2025-01-17:14": {"150.0": [row]}}`` nesting into contracts.

    The expiry key carries a ``:N`` suffix giving Schwab's own days-to-expiry. It is dropped and
    the date re-derived, because that count is relative to when the response was built and this
    contract may be compared against a run timestamp that is not quite the same moment.
    """
    contracts: list[OptionContract] = []
    for expiry_key, strikes in (exp_map or {}).items():
        expiry = pd.to_datetime(str(expiry_key).split(":")[0], errors="coerce")
        if pd.isna(expiry):
            continue
        for rows in (strikes or {}).values():
            for row in rows or []:
                contract = _contract_from_row(row, underlying, option_type, expiry.date())
                if contract is not None:
                    contracts.append(contract)
    return contracts


def _contract_from_row(
    row: dict[str, Any], underlying: str, option_type: str, expiry: date
) -> OptionContract | None:
    """One chain row as an :class:`OptionContract`, or ``None`` if it is not usable.

    Schwab publishes its own OSI string in ``symbol``; it is preferred over rebuilding one, so a
    contract we send back on an order leg is byte-identical to the one the venue named.
    """
    osi = str(row.get("symbol") or "").strip().upper()
    strike = json_number(row.get("strikePrice")) or 0.0
    if not osi or strike <= 0:
        return None
    bid = json_number(row.get("bid")) or 0.0
    ask = json_number(row.get("ask")) or 0.0
    mark = json_number(row.get("mark")) or 0.0
    delta = _greek(row.get("delta"), limit=1.0)
    return OptionContract(
        osi_symbol=osi,
        underlying=str(underlying).upper(),
        option_type=option_type,
        strike=float(strike),
        expiry=expiry,
        bid=float(bid),
        ask=float(ask),
        mark=float(mark),
        delta=delta,
        open_interest=int(json_number(row.get("openInterest")) or 0),
        volume=int(json_number(row.get("totalVolume")) or 0),
        implied_volatility=_greek(row.get("volatility"), limit=100.0),
    )


#: Schwab's "not computed" sentinel. It appears on every greek and on implied volatility when the
#: chain is served outside the hours it calculates them, and it is a *number* -- so it flows
#: through arithmetic untouched unless it is caught here. Left alone it made every contract look
#: like delta -999, which put nothing inside any delta band and reported as "no strike in band":
#: a gate pointing at the wrong setting entirely.
GREEK_UNAVAILABLE = -999.0


def _greek(value: Any, *, limit: float) -> float:
    """A greek, or ``0.0`` when the provider did not compute one.

    Bounds-checked rather than compared against the sentinel exactly, so any out-of-range marker
    is caught the same way -- a delta outside [-1, 1] is not a delta whatever its value.
    """
    number = json_number(value)
    if number is None or abs(float(number)) > limit:
        return 0.0
    return float(number)


def fetch_premarket_summary(
    config: Any, symbol: str, *, as_of: datetime | None = None
) -> dict[str, Any]:
    """This session's pre-market activity for ``symbol``.

    Returns ``{last, prior_close, change_pct, volume, bars, session_date}``. ``bars`` is the count
    of extended-hours minutes observed, and it matters as much as the price: a 0.8% gap on two
    prints is noise, and the caller needs to be able to tell that from a real one.

    Uses one-minute bars over today only, with ``needExtendedHoursData`` on -- the single place in
    this codebase where that flag is set, and the response is deliberately never written to the
    bar cache. ``prior_close`` comes from Schwab's own ``previousClose`` rather than from the last
    regular-hours bar we happen to hold, so the gap is measured against the same close the
    exchange used.
    """
    token = _schwab_token(config, EOD_MARKET_CATEGORY)
    if not token:
        raise ProviderUnavailable("Schwab access token is not configured")

    moment = (as_of or datetime.now(timezone.utc)).astimezone(MARKET_TZ)
    session_date = moment.date()
    start = datetime.combine(session_date, PREMARKET_OPEN, tzinfo=MARKET_TZ)
    payload = _request_json(
        "schwab", EOD_MARKET_CATEGORY, PRICE_HISTORY_URL,
        {
            "symbol": str(symbol).upper(),
            "frequencyType": "minute",
            "frequency": 1,
            "startDate": int(start.timestamp() * 1000),
            "endDate": int(moment.timestamp() * 1000),
            "needExtendedHoursData": "true",
            "needPreviousClose": "true",
        },
        headers=_bearer_auth_header(token),
    ) or {}

    prior_close = float(json_number(payload.get("previousClose")) or 0.0)
    open_cut = datetime.combine(session_date, REGULAR_OPEN, tzinfo=MARKET_TZ)
    last, volume, bars = 0.0, 0.0, 0
    for candle in payload.get("candles") or []:
        stamp = candle.get("datetime")
        if stamp is None:
            continue
        when = datetime.fromtimestamp(int(stamp) / 1000, tz=timezone.utc).astimezone(MARKET_TZ)
        # Strictly before the bell. A run at 09:35 gets today's pre-market and none of today's
        # regular session, which is what makes the answer identical whenever it is asked.
        if when >= open_cut:
            continue
        last = float(candle.get("close") or last)
        volume += float(candle.get("volume") or 0.0)
        bars += 1

    change_pct = (last / prior_close - 1.0) if last > 0 and prior_close > 0 else 0.0
    return {
        "last": last,
        "prior_close": prior_close,
        "change_pct": change_pct,
        "volume": volume,
        "bars": bars,
        "session_date": session_date.isoformat(),
    }
