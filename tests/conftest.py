from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_config_file(tmp_path_factory):
    """Point every config section at a throwaway copy of the real ``config/walbot.yaml``.

    Tests read the same content they always did, but anything that *writes* -- ``save_controls``
    and the account editors both rewrite whole documents -- lands on the copy instead of the
    developer's live trading config. A test that redirects only some of the section env vars
    used to write through to the real file for the rest, which is how a suite run stripped the
    deployment keys off every algorithm.

    ``TRADING_CONFIG_FILE`` alone is enough: every section falls back to it unless a test names
    that section's own env var, which stays possible.
    """
    from src.core.config.paths import config_file_path

    source = config_file_path()
    destination = tmp_path_factory.mktemp("config") / "walbot.yaml"
    if source.exists():
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    previous = os.environ.get("TRADING_CONFIG_FILE")
    os.environ["TRADING_CONFIG_FILE"] = str(destination)
    try:
        yield destination
    finally:
        if previous is None:
            os.environ.pop("TRADING_CONFIG_FILE", None)
        else:
            os.environ["TRADING_CONFIG_FILE"] = previous


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
