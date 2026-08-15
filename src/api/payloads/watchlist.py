"""The dashboard watchlist -- a view preference, not a trading input.

Split out of the single ``api_payloads`` module, which had grown to 1253 lines covering nine
unrelated domains. The public names are unchanged and still importable from ``api_payloads``.
"""


from __future__ import annotations

from ...brokerages.alpaca_client import create_data_client

import logging
from typing import Any
from ...core.config import (

    get_config,
)
from ...core.market_context import load_latest_prices
from ...data.duckdb_store import DAILY_INTERVAL_MINUTES, read_bars
from ...data.state_store import load_state, save_state

logger = logging.getLogger(__name__)


#: Watchlist symbols live in the state store, not a config file: it is a per-user view
#: preference rather than something that changes how an algorithm trades.
WATCHLIST_STATE_KEY = "watchlist"

DEFAULT_WATCHLIST = ["SPY", "QQQ", "GLD", "TLT"]


def _watchlist_symbols() -> list[str]:
    stored = load_state(WATCHLIST_STATE_KEY, None)
    if not isinstance(stored, list):
        return list(DEFAULT_WATCHLIST)
    return [str(symbol).strip().upper()[:10] for symbol in stored if str(symbol).strip()]


def watchlist_payload() -> dict[str, Any]:
    """Watchlist tickers with their latest price and move since the previous close."""
    symbols = _watchlist_symbols()
    rows: list[dict[str, Any]] = [{"symbol": symbol, "price": None, "change": None} for symbol in symbols]
    if not symbols:
        return {"symbols": symbols, "rows": rows}

    config = get_config()
    prices: dict[str, float] = {}
    try:
        prices = load_latest_prices(symbols, config, create_data_client(config))
    except Exception as error:  # noqa: BLE001 - a quote outage must not blank the sidebar
        logger.warning("Could not price watchlist: %s", error)

    for row in rows:
        price = float(prices.get(row["symbol"], 0.0) or 0.0)
        row["price"] = price or None
        # Previous close comes from the cached daily bars, so the sidebar costs no extra call.
        try:
            bars = read_bars(row["symbol"], interval_minutes=DAILY_INTERVAL_MINUTES, limit=2)
            if price and not bars.empty and len(bars) >= 1:
                previous = float(bars["close"].iloc[-2] if len(bars) >= 2 else bars["close"].iloc[-1])
                if previous:
                    row["change"] = (price - previous) / previous
        except Exception:  # noqa: BLE001 - a missing bar just means no percentage
            continue
    return {"symbols": symbols, "rows": rows}


def save_watchlist_payload(symbols: Any) -> dict[str, Any]:
    if not isinstance(symbols, list):
        raise ValueError("Watchlist must be a list of symbols.")
    cleaned: list[str] = []
    for symbol in symbols:
        ticker = str(symbol or "").strip().upper()[:10]
        if ticker and ticker not in cleaned:
            cleaned.append(ticker)
    save_state(WATCHLIST_STATE_KEY, cleaned[:30])
    return watchlist_payload()
