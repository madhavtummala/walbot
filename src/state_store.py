from __future__ import annotations

import json
import os
import sqlite3
from datetime import timezone
from typing import Any

import pandas as pd

from .universe import resolve_project_path

STATE_DB_PATH = os.getenv("STATE_DB_PATH", "data/trading_bot.sqlite")


def _connect(db_path: str = STATE_DB_PATH) -> sqlite3.Connection:
    resolved = resolve_project_path(db_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(resolved)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return connection


def load_state(key: str, default: Any, db_path: str = STATE_DB_PATH) -> Any:
    with _connect(db_path) as connection:
        row = connection.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return default

    return default


def save_state(key: str, value: Any, db_path: str = STATE_DB_PATH) -> Any:
    encoded = json.dumps(value, sort_keys=True)
    updated_at = pd.Timestamp.now(tz=timezone.utc).isoformat()
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO app_state (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, encoded, updated_at),
        )
    return value


def delete_state(key: str, db_path: str = STATE_DB_PATH) -> None:
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM app_state WHERE key = ?", (key,))
