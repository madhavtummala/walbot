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


@pytest.fixture(autouse=True)
def _no_background_recomputes():
    """Keep a plain read from starting real work on a background thread.

    Reading a lazy payload now schedules a recompute -- an algorithm run, or a replay of months
    of bars -- and a test that merely asks what is cached would otherwise launch one. That gives
    a suite which quietly backtests in the background, races the state store it is asserting on,
    and is slow for a reason nothing in the test mentions.

    Spawning is disabled instead, and the caches are cleared either side so one test's snapshot
    is never another's starting state. A test that wants the value computed opts in with
    ``run_lazy_inline``.
    """
    from src.api.payloads import algorithms, backtest

    fields = (algorithms.SIGNALS, backtest.BACKTESTS)
    originals = [field._spawn for field in fields]
    for field in fields:
        field.reset()
        field._spawn = lambda work: None
    try:
        yield
    finally:
        for field, original in zip(fields, originals):
            field._spawn = original
            field.reset()


@pytest.fixture
def run_lazy_inline():
    """Opt one lazy field into computing on the calling thread, for a test that wants the value."""
    def enable(field):
        field._spawn = lambda work: work()
    return enable
