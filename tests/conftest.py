from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_state_database(tmp_path_factory):
    """Point the DuckDB state file at a throwaway path for the whole test session.

    Several tests drive the real submit path (``pipeline.place_orders``), which now records an
    audit row per order. Without this they would write test orders -- and any other state a run
    persists -- straight into the developer's ``data/walbot.duckdb``.
    """
    previous = os.environ.get("STATE_DUCKDB_PATH")
    os.environ["STATE_DUCKDB_PATH"] = str(tmp_path_factory.mktemp("state") / "walbot.duckdb")

    # The path is read at import time into module-level constants, so rebind the live ones.
    from src.data import duckdb_store, state_store

    path = os.environ["STATE_DUCKDB_PATH"]
    originals = {
        (duckdb_store, "DUCKDB_STATE_PATH"): duckdb_store.DUCKDB_STATE_PATH,
        (state_store, "STATE_DUCKDB_PATH"): state_store.STATE_DUCKDB_PATH,
    }
    for (module, name) in originals:
        setattr(module, name, path)
    try:
        yield path
    finally:
        for (module, name), value in originals.items():
            setattr(module, name, value)
        if previous is None:
            os.environ.pop("STATE_DUCKDB_PATH", None)
        else:
            os.environ["STATE_DUCKDB_PATH"] = previous
