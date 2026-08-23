"""The tradable universe, read from the configured CSV.

Lives here rather than behind an API payload because it is a *data* question. Bursty DCA used
to answer it by importing ``api.api_payloads.universe_payload`` -- an algorithm reaching up
into the web layer, for a symbol set, so that filtering a plan pulled in FastAPI's payload
builders and everything they import.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYMBOL_COLUMNS = ("ticker", "symbol")


def resolve_project_path(path: str) -> Path:
    csv_path = Path(path)
    if csv_path.is_absolute():
        return csv_path
    return PROJECT_ROOT / csv_path


def load_tradable_names(path: str) -> dict[str, str]:
    """Load a symbol -> name lookup from a tradables CSV."""
    csv_path = resolve_project_path(path)
    if not csv_path.exists():
        logger.warning("Tradables CSV does not exist: %s", csv_path)
        return {}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}

        field_lookup = {field.lower(): field for field in reader.fieldnames}
        symbol_field = next((field_lookup[column] for column in SYMBOL_COLUMNS if column in field_lookup), None)
        name_field = field_lookup.get("name")
        if symbol_field is None:
            raise ValueError(f"{csv_path} must contain a Ticker or Symbol column.")

        names: dict[str, str] = {}
        for row in reader:
            symbol = str(row.get(symbol_field, "")).strip().upper()
            if not symbol or symbol in names:
                continue
            names[symbol] = str(row.get(name_field, "")).strip() if name_field else ""
        return names


def tradable_symbols(config: Any) -> set[str]:
    """Every symbol the account may trade, upper-cased.

    The configured symbol list intersected with the tradables CSV, falling back to the symbol
    list alone when no CSV is configured -- the same rule ``universe_payload`` renders, stated
    once here so the deck and the algorithms cannot disagree about what is tradable.

    ``config`` is required rather than fetched. ``core.config.schema`` imports this module for
    :func:`load_tradable_names`, so reaching back for ``get_config()`` -- even deferred inside
    the function -- closes a cycle between configuration and the data it configures.
    """
    configured = {str(symbol).strip().upper() for symbol in (getattr(config, "symbols", None) or [])}
    names = load_tradable_names(getattr(config, "tradables_csv", "") or "")
    return (configured & set(names)) if names else configured
