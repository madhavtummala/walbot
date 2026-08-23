"""The tradable universe and recommendations for changing it.

Split out of the single ``api_payloads`` module, which had grown to 1253 lines covering nine
unrelated domains. The public names are unchanged and still importable from ``api_payloads``.
"""


from __future__ import annotations

from ...brokerages.alpaca.client import create_data_client
from ...data.universe import load_tradable_names

import logging
from typing import Any

import pandas as pd


from ...data import fetch_daily_bars
from ...core.config import (
    get_config,
    save_universe_symbols,
)
from ...data.universe_selector import candidate_specs_by_symbol, preferred_symbols, recommend_universe_rows

logger = logging.getLogger(__name__)
from .system import _display_path





def universe_payload() -> dict[str, Any]:
    config = get_config()
    tradable_names = load_tradable_names(config.tradables_csv)
    tradables = set(tradable_names) if tradable_names else set(config.symbols)
    specs = candidate_specs_by_symbol(tradables)

    rows: list[dict[str, Any]] = []
    for symbol in config.symbols:
        spec = specs.get(symbol)
        rows.append(
            {
                "symbol": symbol,
                "name": tradable_names.get(symbol, symbol),
                "bucket": spec.bucket if spec else "",
                "tradable": symbol in tradables,
                "enabled": symbol in config.symbols,
            }
        )

    return {"rows": rows, "count": len(rows)}


def recommend_universe_payload(body: dict[str, Any] | None = None) -> dict[str, Any]:
    body = body or {}
    config = get_config()
    max_symbols = min(max(int(body.get("max_symbols") or 12), 3), 24)
    force_refresh = bool(body.get("refresh", True))
    tradable_names = load_tradable_names(config.tradables_csv)
    candidates = preferred_symbols(set(tradable_names))
    if not candidates:
        raise RuntimeError("No preferred universe candidates were present in the tradables CSV.")

    data_client = create_data_client(config)
    bars_by_symbol = fetch_daily_bars(
        candidates,
        lookback_days=320,
        ma_days=252,
        extra_buffer_days=60,
        data_client=data_client,
        force_refresh=force_refresh,
    )
    recommendation = recommend_universe_rows(
        tradable_names=tradable_names,
        bars_by_symbol=bars_by_symbol,
        max_symbols=max_symbols,
    )
    return {
        **recommendation,
        "max_symbols": max_symbols,
        "data_feed": config.alpaca_data_feed,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "current": universe_payload()["rows"],
    }


def apply_universe_payload(body: dict[str, Any]) -> dict[str, Any]:
    config = get_config()
    tradable_names = load_tradable_names(config.tradables_csv)
    specs = candidate_specs_by_symbol(set(tradable_names))
    validate_against_master = bool(tradable_names)
    raw_rows = body.get("rows") or []
    raw_symbols = body.get("symbols") or []
    if isinstance(raw_symbols, str):
        raw_symbols = [symbol.strip().upper() for symbol in raw_symbols.split(",") if symbol.strip()]

    proposed_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    if raw_rows:
        for row in raw_rows:
            symbol = str(row.get("symbol", row.get("Ticker", ""))).strip().upper()
            if not symbol or symbol in seen:
                continue
            if validate_against_master and symbol not in tradable_names:
                raise ValueError(f"{symbol} is not present in the tradables CSV.")
            spec = specs.get(symbol)
            proposed_rows.append(
                {
                    "symbol": symbol,
                    "name": tradable_names.get(symbol) or str(row.get("name") or symbol),
                    "bucket": str(row.get("bucket") or (spec.bucket if spec else "")).strip(),
                }
            )
            seen.add(symbol)
    else:
        for symbol in [str(item).strip().upper() for item in raw_symbols]:
            if not symbol or symbol in seen:
                continue
            if validate_against_master and symbol not in tradable_names:
                raise ValueError(f"{symbol} is not present in the tradables CSV.")
            spec = specs.get(symbol)
            proposed_rows.append(
                {
                    "symbol": symbol,
                    "name": tradable_names.get(symbol) or symbol,
                    "bucket": spec.bucket if spec else "",
                }
            )
            seen.add(symbol)

    if len(proposed_rows) < 3:
        raise ValueError("Universe must include at least 3 tradable symbols.")
    if len(proposed_rows) > 24:
        raise ValueError("Universe must include 24 symbols or fewer.")

    config_path = save_universe_symbols([row["symbol"] for row in proposed_rows])
    return {
        "saved": True,
        "path": _display_path(config_path),
        "universe": universe_payload(),
    }
