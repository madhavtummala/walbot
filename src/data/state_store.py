from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import timezone
from typing import Any, Iterator

import pandas as pd

from .duckdb_store import DUCKDB_STATE_PATH, _connect

STATE_DUCKDB_PATH = DUCKDB_STATE_PATH

#: When set, reads and writes go to this dict instead of DuckDB. A backtest replays algorithms
#: that carry state between runs -- DCA's accrued budget, Rally Rotation's eligibility history --
#: and without this it would read and then overwrite the live account's state. A ContextVar
#: rather than a parameter so no algorithm has to know it is being replayed.
_EPHEMERAL_STATE: ContextVar[dict[str, Any] | None] = ContextVar("ephemeral_state", default=None)


def algorithm_state_key(algorithm_id: str, account_id: str) -> str:
    """Where one binding's algorithm state lives.

    Keyed on the pair, because a binding is an algorithm *and* an account: the runtime drives
    several at once, and two accounts running the same algorithm are two separate books. A key
    without the account would let fills in one draw down the other's accrued budget and share
    its cooldowns. Built here rather than by each algorithm, which is what left the codebase
    with two hand-rolled key formats for one concept.
    """
    return f"algorithm_state:{algorithm_id}:{account_id or 'default'}"


@contextmanager
def ephemeral_state(initial: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Redirect all state reads and writes to a throwaway dict for the duration of the block."""
    store: dict[str, Any] = dict(initial or {})
    token = _EPHEMERAL_STATE.set(store)
    try:
        yield store
    finally:
        _EPHEMERAL_STATE.reset(token)


def load_state(key: str, default: Any, db_path: str | None = None) -> Any:
    store = _EPHEMERAL_STATE.get()
    if store is not None:
        return store.get(key, default)
    with _connect(db_path) as connection:
        row = connection.execute("SELECT value FROM app_state WHERE key = ?", [key]).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return default

    return default


def save_state(key: str, value: Any, db_path: str | None = None) -> Any:
    store = _EPHEMERAL_STATE.get()
    if store is not None:
        store[key] = value
        return value
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


def delete_state(key: str, db_path: str | None = None) -> None:
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM app_state WHERE key = ?", [key])
