from __future__ import annotations

import json
from datetime import timezone
from typing import Any

import pandas as pd

from .duckdb_store import DUCKDB_STATE_PATH, _connect

STATE_DUCKDB_PATH = DUCKDB_STATE_PATH


def load_state(key: str, default: Any, db_path: str = STATE_DUCKDB_PATH) -> Any:
    with _connect(db_path) as connection:
        row = connection.execute("SELECT value FROM app_state WHERE key = ?", [key]).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return default

    return default


def save_state(key: str, value: Any, db_path: str = STATE_DUCKDB_PATH) -> Any:
    encoded = json.dumps(value, sort_keys=True)
    updated_at = pd.Timestamp.now(tz=timezone.utc).to_pydatetime()
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO app_state (key, value, updated_at)
            VALUES (?, ?, ?)
            """,
            [key, encoded, updated_at],
        )
    return value


def delete_state(key: str, db_path: str = STATE_DUCKDB_PATH) -> None:
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM app_state WHERE key = ?", [key])
