from __future__ import annotations

import csv
import logging
from pathlib import Path

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
