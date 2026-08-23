from __future__ import annotations

import threading
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.data import bars as bars_module
from src.data import cache_warmup, duckdb_store, provider_cache


def _bars(dates: list[str], start_price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates, utc=True),
            "open": [start_price + index for index in range(len(dates))],
            "high": [start_price + index + 1 for index in range(len(dates))],
            "low": [start_price + index - 1 for index in range(len(dates))],
            "close": [start_price + index + 0.5 for index in range(len(dates))],
            "volume": [1_000_000 + index for index in range(len(dates))],
        }
    )


def test_clear_market_bars_is_targeted(tmp_path) -> None:
    """One store, two resolutions: clearing dailies must leave the minute bars alone."""
    db_path = str(tmp_path / "market.duckdb")
    bars = _bars(["2026-01-02"])
    duckdb_store.write_market_bars("yfinance", "SPY", 1440, bars, db_path=db_path)
    duckdb_store.write_market_bars("yfinance", "SPY", 5, bars, db_path=db_path)

    deleted = duckdb_store.clear_market_bars(
        provider="yfinance",
        symbols=["SPY"],
        interval_minutes=1440,
        db_path=db_path,
    )

    assert deleted == 1
    summary = duckdb_store.market_bars_summary(provider="yfinance", symbols=["SPY"], db_path=db_path)
    assert [(row["interval_minutes"], row["timeframe"], row["rows"]) for row in summary] == [(5, "5m", 1)]


def test_clear_cached_payloads_can_match_key_prefix(tmp_path) -> None:
    db_path = str(tmp_path / "state.duckdb")
    provider_cache.save_cached_payload("intraday_market_data", "yfinance", "SPY:15:78", [{"close": 1}], 60, db_path=db_path)
    provider_cache.save_cached_payload("intraday_market_data", "yfinance", "QQQ:15:78", [{"close": 2}], 60, db_path=db_path)

    deleted = provider_cache.clear_cached_payloads(
        category="intraday_market_data",
        provider="yfinance",
        cache_key_prefixes=["SPY:15:"],
        db_path=db_path,
    )

    assert deleted == 1
    assert provider_cache.load_cached_payload("intraday_market_data", "yfinance", "SPY:15:78", db_path=db_path) is None
    assert provider_cache.load_cached_payload("intraday_market_data", "yfinance", "QQQ:15:78", db_path=db_path) == [{"close": 2}]


def test_warm_market_data_cache_forces_yfinance_fetches(monkeypatch) -> None:
    calls = []
    bars = _bars(["2026-01-02"])

    def fake_eod(symbols, _config, *, lookback_bars, force_refresh, provider, start_date, end_date):
        calls.append(("eod", symbols, lookback_bars, force_refresh, provider, start_date, end_date))
        return {symbol: bars for symbol in symbols}

    def fake_history(symbols, _config, *, lookback_minutes, bar_minutes, force_refresh, provider, start_date, end_date):
        calls.append(("intraday", symbols, lookback_minutes, bar_minutes, force_refresh, provider, start_date, end_date))
        return {symbol: bars for symbol in symbols}

    class MockConfig:
        eod_market_data_provider_order = ["yfinance"]
        intraday_market_data_provider_order = ["yfinance"]

    monkeypatch.setattr(cache_warmup, "get_config", lambda: MockConfig())
    monkeypatch.setattr(cache_warmup, "_clear_market_cache", lambda *_args, **_kwargs: {"eod_duckdb_rows": 0})
    monkeypatch.setattr(cache_warmup, "market_bars_summary", lambda **_kwargs: [{"symbol": "SPY", "rows": 1}])
    monkeypatch.setattr("src.connectors.fetch_eod_market_bars", fake_eod)
    monkeypatch.setattr("src.connectors.fetch_market_history", fake_history)

    result = cache_warmup.warm_market_data_cache(["SPY"], clear=True)

    assert result["fetched"]["eod_rows"] == {"SPY": 1}
    assert result["fetched"]["intraday_rows"] == {"SPY": 1}
    assert calls == [
        ("eod", ["SPY"], 98, True, "yfinance", None, None),
        ("intraday", ["SPY"], 1170, 5, True, "yfinance", None, None),
    ]


def test_warm_market_data_cache_selects_algorithm_intraday_date_range(monkeypatch) -> None:
    calls = []
    bars = _bars(["2026-06-03T15:00:00Z"])

    monkeypatch.setattr(cache_warmup, "_algorithm_symbols", lambda algorithm_id=None: ["QQQM", "BIL"])
    class MockConfig:
        eod_market_data_provider_order = ["yfinance"]
        intraday_market_data_provider_order = ["yfinance"]

    monkeypatch.setattr(cache_warmup, "get_config", lambda: MockConfig())
    monkeypatch.setattr(cache_warmup, "market_bars_summary", lambda **_kwargs: [])
    monkeypatch.setattr(
        "src.connectors.fetch_market_history",
        lambda symbols, _config, **kwargs: calls.append((symbols, kwargs)) or {symbol: bars for symbol in symbols},
    )
    monkeypatch.setattr(
        "src.connectors.fetch_eod_market_bars",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("EOD should not be fetched")),
    )

    result = cache_warmup.warm_market_data_cache(
        algorithm_id="rally_rotation",
        start_date="2026-06-03",
        end_date="2026-06-03",
        warm_eod=False,
        warm_intraday=True,
    )

    symbols, kwargs = calls[0]
    assert symbols == ["BIL", "QQQM"]
    assert kwargs["lookback_minutes"] == 390
    assert kwargs["start_date"].isoformat() == "2026-06-03T05:00:00+00:00"
    assert kwargs["end_date"].isoformat() == "2026-06-04T05:00:00+00:00"
    assert result["cleared"] == {}
    assert result["warmed"] == {"daily": False, "intraday": True}
    # yfinance cannot serve the 5-minute default, so the rows actually written are 15m ones.
    assert result["bar_minutes"] == 15


def test_warm_market_data_cache_rejects_reversed_date_range() -> None:
    try:
        cache_warmup._parse_date_range("2026-06-04", "2026-06-03")
    except ValueError as exc:
        assert str(exc) == "start date must be on or before end date"
    else:
        raise AssertionError("Expected reversed date range to fail")


def test_eod_date_range_uses_provider_timestamp_convention() -> None:
    yfinance_start, yfinance_end = cache_warmup._parse_date_range(
        "2026-06-03",
        "2026-06-03",
        timezone_name=cache_warmup._eod_timezone("yfinance"),
    )
    start, end = cache_warmup._parse_date_range(
        "2026-06-03",
        "2026-06-03",
        timezone_name=cache_warmup._eod_timezone("alpaca"),
    )

    assert yfinance_start.isoformat() == "2026-06-03T00:00:00+00:00"
    assert yfinance_end.isoformat() == "2026-06-04T00:00:00+00:00"
    assert start.isoformat() == "2026-06-03T04:00:00+00:00"
    assert end.isoformat() == "2026-06-04T04:00:00+00:00"


def test_read_history_blends_fine_bars_with_the_daily_tail(tmp_path) -> None:
    """A window longer than the fine bars reach is completed from the coarser tier.

    This is what lets a horizon be stated in minutes at all: twelve sessions is longer than a
    fresh intraday cache holds, and without the blend every symbol would score flat there.
    """
    db_path = str(tmp_path / "market.duckdb")
    daily = _bars([f"2026-05-{day:02d}T20:00:00Z" for day in range(4, 15)])
    fine = _bars(
        [stamp.isoformat() for stamp in pd.date_range("2026-05-14T14:30:00Z", periods=12, freq="5min")],
        start_price=200.0,
    )
    duckdb_store.write_market_bars("schwab", "SPY", 1440, daily, db_path=db_path)
    duckdb_store.write_market_bars("schwab", "SPY", 5, fine, db_path=db_path)

    # Ten sessions of market time, which the twelve 5-minute bars cannot begin to cover.
    blended = bars_module.read_history(
        "SPY",
        lookback_minutes=10 * 390,
        end=datetime(2026, 5, 14, 16, tzinfo=timezone.utc),
        db_path=db_path,
    )

    assert blended["timestamp"].is_monotonic_increasing
    resolutions = set(blended["interval_minutes"])
    assert resolutions == {5, 1440}, "both tiers contribute"
    # The fine bars own the recent end; the dailies only fill what they do not reach.
    assert blended["interval_minutes"].iloc[-1] == 5
    assert blended["interval_minutes"].iloc[0] == 1440


def test_read_history_prefers_the_finest_tier_that_covers_the_window(tmp_path) -> None:
    """When fine bars already span the window, no coarser tier is mixed in."""
    db_path = str(tmp_path / "market.duckdb")
    fine = _bars(
        [stamp.isoformat() for stamp in pd.date_range("2026-05-14T13:30:00Z", periods=40, freq="5min")],
        start_price=200.0,
    )
    duckdb_store.write_market_bars("schwab", "SPY", 1440, _bars(["2026-05-13T20:00:00Z"]), db_path=db_path)
    duckdb_store.write_market_bars("schwab", "SPY", 5, fine, db_path=db_path)

    blended = bars_module.read_history(
        "SPY",
        lookback_minutes=195,
        end=datetime(2026, 5, 14, 17, tzinfo=timezone.utc),
        db_path=db_path,
    )

    assert set(blended["interval_minutes"]) == {5}


def test_the_store_migrates_category_rows_onto_interval_minutes(tmp_path) -> None:
    """EOD and intraday were two categories of the same thing; the accumulated rows survive.

    The backtester's depth *is* this cache, so a migration that dropped rows would quietly
    shorten every replay window rather than failing loudly.
    """
    import duckdb

    db_path = str(tmp_path / "legacy.duckdb")
    with duckdb.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE market_bars (
                category VARCHAR NOT NULL, provider VARCHAR NOT NULL, symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL, timestamp TIMESTAMPTZ NOT NULL,
                open DOUBLE NOT NULL, high DOUBLE NOT NULL, low DOUBLE NOT NULL,
                close DOUBLE NOT NULL, volume DOUBLE NOT NULL, adjusted_close DOUBLE,
                raw_json VARCHAR,
                PRIMARY KEY (category, provider, symbol, timeframe, timestamp)
            )
            """
        )
        connection.executemany(
            "INSERT INTO market_bars VALUES (?, ?, ?, ?, ?, 100, 101, 99, 100.5, 1000, 100.5, NULL)",
            [
                ("eod_market_data", "schwab", "SPY", "1d", datetime(2026, 5, 4, 20, tzinfo=timezone.utc)),
                ("eod_market_data", "schwab", "SPY", "1d", datetime(2026, 5, 5, 20, tzinfo=timezone.utc)),
                ("intraday_market_data", "schwab", "SPY", "15m", datetime(2026, 5, 5, 14, 30, tzinfo=timezone.utc)),
            ],
        )

    summary = duckdb_store.market_bars_summary(db_path=db_path)

    assert sum(row["rows"] for row in summary) == 3, "no row is lost in the migration"
    assert {(row["interval_minutes"], row["rows"]) for row in summary} == {(1440, 2), (15, 1)}
    # And the migrated rows are readable through the new path, not just countable.
    assert len(duckdb_store.read_bars("SPY", interval_minutes=15, db_path=db_path)) == 1


def test_provider_bars_are_stamped_when_the_close_happened() -> None:
    """Providers stamp a bar at its start; the close is the price at its end.

    Left alone that is lookahead: a daily bar stamped at midnight carries that session's
    *closing* price, so asking "what was the price at 10:00" hands back something that will
    not be known for another six hours. Stamping happens where a payload becomes bars, so a
    fetcher's return value and the rows it stores cannot disagree by an interval.
    """
    from src.connectors.cache import _provider_bars

    intraday = _provider_bars(_bars(["2026-06-04T14:30:00Z"]), 5)
    assert pd.Timestamp(intraday["timestamp"].iloc[0]) == pd.Timestamp("2026-06-04T14:35:00Z")

    # Daily bars land on the 16:00 exchange-local session close, whatever midnight the
    # provider chose -- which is what lets two providers' rows for one session line up.
    session_close = pd.Timestamp("2026-06-04 16:00", tz="America/New_York").tz_convert("UTC")
    for stamp in ("2026-06-04T04:00:00Z", "2026-06-04T05:00:00Z", "2026-06-04T00:00:00Z"):
        daily = _provider_bars(_bars([stamp]), duckdb_store.DAILY_INTERVAL_MINUTES)
        assert pd.Timestamp(daily["timestamp"].iloc[0]) == session_close, stamp


def test_the_store_records_the_stamps_it_is_given(tmp_path) -> None:
    """Stamping is the connector's job, so writing must not shift anything a second time."""
    db_path = str(tmp_path / "stamps.duckdb")
    stamped = _bars(["2026-06-04T14:35:00Z"])
    duckdb_store.write_market_bars("schwab", "SPY", 5, stamped, db_path=db_path)

    stored = duckdb_store.read_bars("SPY", interval_minutes=5, db_path=db_path)
    assert pd.Timestamp(stored["timestamp"].iloc[0]) == pd.Timestamp("2026-06-04T14:35:00Z")


def test_a_bar_that_has_not_closed_yet_is_not_cached(tmp_path) -> None:
    """A bar fetched mid-interval is a partial aggregate, and nothing ever revisits it.

    Timestamps name the bar's end, so the test is simply whether it lies in the future.
    """
    db_path = str(tmp_path / "partial.duckdb")
    now = pd.Timestamp.now(tz="UTC").floor("min")
    closed = _bars([(now - pd.Timedelta(minutes=5)).isoformat()])
    still_forming = _bars([(now + pd.Timedelta(minutes=3)).isoformat()])

    assert duckdb_store.write_market_bars("schwab", "SPY", 5, closed, db_path=db_path) == 1
    assert duckdb_store.write_market_bars("schwab", "SPY", 5, still_forming, db_path=db_path) == 0

    stored = duckdb_store.read_bars("SPY", interval_minutes=5, db_path=db_path)
    assert len(stored) == 1
    assert pd.Timestamp(stored["timestamp"].iloc[0]) <= now


def test_prune_keeps_only_the_feed_grids_and_symbols_still_in_use(tmp_path, monkeypatch) -> None:
    """The cache accumulated three providers and a superseded grid; pruning is deliberate."""
    from src.data import cache_prune

    db_path = str(tmp_path / "prune.duckdb")
    bars = _bars(["2026-05-04T14:00:00Z"])
    duckdb_store.write_market_bars("schwab", "SPY", 5, bars, db_path=db_path)          # keep
    duckdb_store.write_market_bars("schwab", "SPY", 1440, bars, db_path=db_path)       # keep
    duckdb_store.write_market_bars("schwab", "SPY", 15, bars, db_path=db_path)         # superseded grid
    duckdb_store.write_market_bars("alpaca", "SPY", 1440, bars, db_path=db_path)       # retired provider
    duckdb_store.write_market_bars("schwab", "DROPPED", 5, bars, db_path=db_path)      # left the universe

    class _Config:
        symbols = ["SPY"]

    monkeypatch.setattr(cache_prune, "get_config", lambda: _Config())

    dry = cache_prune.prune(db_path=db_path, apply=False)
    assert dry["applied"] is False
    assert dry["rows"] == {"kept": 2, "other_intervals": 1, "other_providers": 1, "outside_universe": 1}
    # A dry run must not have touched anything.
    assert len(duckdb_store.market_bars_summary(db_path=db_path)) == 5

    applied = cache_prune.prune(db_path=db_path, apply=True)
    assert applied["deleted"] == 3 and applied["rows_after"] == 2
    remaining = {(r["provider"], r["interval_minutes"], r["symbol"])
                 for r in duckdb_store.market_bars_summary(db_path=db_path)}
    assert remaining == {("schwab", 5, "SPY"), ("schwab", 1440, "SPY")}


def test_warming_defaults_to_the_tradable_universe_not_what_is_held(monkeypatch) -> None:
    """A symbol added to the universe must get cached before any algorithm trades it.

    Defaulting to the union of algorithm universes inverted that: nothing was warmed until
    something already held it, so a newly added symbol had no history on its first run.
    """
    class _Config:
        symbols = ["AAA", "BBB", "VTEB"]

    monkeypatch.setattr(cache_warmup, "get_config", lambda: _Config())
    monkeypatch.setattr(cache_warmup, "_algorithm_symbols", lambda algorithm_id=None: ["AAA"])

    assert cache_warmup._wanted_symbols(None) == ["AAA", "BBB", "VTEB"]
    # Naming an algorithm still narrows to just what that algorithm needs.
    assert cache_warmup._wanted_symbols(None, algorithm_id="rally_rotation") == ["AAA"]
    # And an explicit list always wins.
    assert cache_warmup._wanted_symbols(["zzz"]) == ["ZZZ"]


def test_audit_flags_a_tier_disagreement_at_the_session_close(tmp_path, monkeypatch) -> None:
    """A 5m bar ending at the close and that session's daily bar describe the same instant.

    Reads blend the tiers, so a systematic gap between them is a step discontinuity in the
    middle of a price series, not a rounding curiosity.
    """
    from src.data import cache_audit

    db_path = str(tmp_path / "audit.duckdb")
    close_ts = pd.Timestamp("2026-06-04 15:55", tz="America/New_York").tz_convert("UTC")

    def bar(price: float) -> pd.DataFrame:
        frame = _bars([close_ts.isoformat()])
        frame["close"] = [price]
        return frame

    # AGREE: both tiers report the same closing price.
    duckdb_store.write_market_bars("schwab", "AGREE", 5, bar(100.0), db_path=db_path)
    duckdb_store.write_market_bars("schwab", "AGREE", 1440, bar(100.0), db_path=db_path)
    # DIVERGE: the daily tier is back-adjusted, the intraday tier is raw.
    duckdb_store.write_market_bars("schwab", "DIVERGE", 5, bar(100.0), db_path=db_path)
    duckdb_store.write_market_bars("schwab", "DIVERGE", 1440, bar(98.0), db_path=db_path)
    # LONELY: intraday only, so there is nothing to compare against.
    duckdb_store.write_market_bars("schwab", "LONELY", 5, bar(100.0), db_path=db_path)

    class _Config:
        symbols = ["AGREE", "DIVERGE", "LONELY"]

    monkeypatch.setattr(cache_audit, "get_config", lambda: _Config())
    report = cache_audit.audit(db_path=db_path)

    by_symbol = {row["symbol"]: row for row in report["rows"]}
    assert by_symbol["AGREE"]["match_rate"] == 1.0
    assert by_symbol["DIVERGE"]["match_rate"] == 0.0
    assert by_symbol["DIVERGE"]["median_rel_gap"] == pytest.approx(2 / 98, abs=1e-4)
    assert report["symbols_without_overlap"] == ["LONELY"]


def test_read_only_pool_reads_cheaply(tmp_path) -> None:
    """A backtest's read pool must never lock other readers out.

    DuckDB read-only connections coexist with other processes' read-only connections, which
    is the whole reason the replay gets one.
    """
    db_path = str(tmp_path / "ro.duckdb")
    duckdb_store.write_market_bars("yfinance", "SPY", 1440, _bars(["2026-01-02"]), db_path=db_path)

    with duckdb_store.pooled_connections(db_path, read_only=True):
        with duckdb_store._connect(db_path) as connection:
            assert duckdb_store.connection_is_read_only(connection) is True
        assert len(duckdb_store.read_bars("SPY", interval_minutes=1440, db_path=db_path)) == 1


def test_read_only_pool_upgrades_rather_than_failing_a_write(tmp_path) -> None:
    """A read-only batch describes that job, not the whole process.

    The scheduler still writes its run state while a backtest reads, and DuckDB fixes the mode
    per file per process -- so the write upgrades the shared handle instead of raising
    "Can't open a connection to same database file with a different configuration".
    """
    db_path = str(tmp_path / "upgrade.duckdb")
    duckdb_store.write_market_bars("yfinance", "SPY", 1440, _bars(["2026-01-02"]), db_path=db_path)

    with duckdb_store.pooled_connections(db_path, read_only=True):
        assert duckdb_store._HANDLES[str(duckdb_store.resolve_project_path(db_path))].read_only is True

        written = duckdb_store.write_market_bars("yfinance", "SPY", 1440, _bars(["2026-01-05"]), db_path=db_path)

        assert written == 1
        assert duckdb_store._HANDLES[str(duckdb_store.resolve_project_path(db_path))].read_only is False
        assert len(duckdb_store.read_bars("SPY", interval_minutes=1440, db_path=db_path)) == 2


def test_other_threads_keep_reading_through_a_read_only_pool(tmp_path) -> None:
    """The crash this pool caused: another thread's plain read died mid-backtest.

    Pooling was thread-local while DuckDB's mode is per process, so any other thread opening
    the same file -- read-write by default, even to read a row -- was refused outright.
    """
    db_path = str(tmp_path / "threads.duckdb")
    duckdb_store.write_market_bars("yfinance", "SPY", 1440, _bars(["2026-01-02"]), db_path=db_path)
    outcome: dict[str, object] = {}

    def read_from_another_thread() -> None:
        try:
            outcome["rows"] = len(duckdb_store.read_bars("SPY", interval_minutes=1440, db_path=db_path))
        except Exception as exc:  # noqa: BLE001 - the failure under test
            outcome["error"] = exc

    with duckdb_store.pooled_connections(db_path, read_only=True):
        thread = threading.Thread(target=read_from_another_thread)
        thread.start()
        thread.join(timeout=10)

    assert outcome.get("error") is None
    assert outcome["rows"] == 1


def test_write_pool_still_writes(tmp_path) -> None:
    """The default pool keeps its read-write mode: schema work and cache writes need it."""
    db_path = str(tmp_path / "rw.duckdb")

    with duckdb_store.pooled_connections(db_path):
        connection = duckdb_store._connect(db_path)
        assert duckdb_store.connection_is_read_only(connection) is False
        written = duckdb_store.write_market_bars("yfinance", "SPY", 1440, _bars(["2026-01-02"]), db_path=db_path)
        assert written == 1
        assert len(duckdb_store.read_bars("SPY", interval_minutes=1440, db_path=db_path)) == 1


def test_read_only_pool_creates_a_missing_file_as_read_write(tmp_path) -> None:
    """There is nothing to read in a file that does not exist yet.

    A read-only connection cannot create one, so the fallback opens read-write to build the
    schema and then lets the batch go back to reading.
    """
    db_path = str(tmp_path / "fresh.duckdb")

    with duckdb_store.pooled_connections(db_path, read_only=True):
        assert duckdb_store.read_bars("SPY", db_path=db_path).empty
        # A schema-ensure write on the read-only connection is skipped, not fatal.
        assert duckdb_store.read_bars("SPY", interval_minutes=5, db_path=db_path).empty


def test_read_only_pool_closes_its_connection_on_exit(tmp_path) -> None:
    """The cached handle must not leak across batch jobs or test runs."""
    db_path = str(tmp_path / "ro-exit.duckdb")
    duckdb_store.write_market_bars("yfinance", "SPY", 1440, _bars(["2026-01-02"]), db_path=db_path)

    with duckdb_store.pooled_connections(db_path, read_only=True):
        duckdb_store.read_bars("SPY", interval_minutes=1440, db_path=db_path)
        assert len(duckdb_store._HANDLES) == 1

    assert len(duckdb_store._HANDLES) == 0


def test_a_locked_database_is_waited_for_rather_than_failed_on(tmp_path, monkeypatch) -> None:
    """Another process holding the file is usually a short write, not a permanent state.

    DuckDB has no busy-timeout of its own -- ``connect()`` fails the instant the lock is
    incompatible -- so the waiting is done here. Without it, any overlap between the dashboard
    and a batch job surfaced as an error page.
    """
    db_path = str(tmp_path / "locked.duckdb")
    attempts = {"n": 0}

    def flaky_connect(path, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("Conflicting lock is held in /proc/other")
        return duckdb_store._duckdb().connect(path, **kwargs)

    monkeypatch.setattr(duckdb_store, "LOCK_RETRY_SECONDS", 0.01)
    connection = duckdb_store._open_with_retry(lambda: flaky_connect(db_path), db_path)

    assert attempts["n"] == 3, "it gave up instead of waiting for the lock"
    connection.close()


def test_waiting_for_a_lock_is_bounded(tmp_path, monkeypatch) -> None:
    """A request thread that waits is a server thread held, so the wait has to end."""
    monkeypatch.setattr(duckdb_store, "LOCK_RETRY_SECONDS", 0.01)
    monkeypatch.setattr(duckdb_store, "LOCK_WAIT_SECONDS", 0.05)

    def always_locked():
        raise RuntimeError("Conflicting lock is held")

    with pytest.raises(RuntimeError, match="Conflicting lock"):
        duckdb_store._open_with_retry(always_locked, "never.duckdb")


def test_an_unrelated_error_is_not_retried(tmp_path, monkeypatch) -> None:
    """Only a lock conflict is worth waiting on; anything else must surface immediately."""
    monkeypatch.setattr(duckdb_store, "LOCK_RETRY_SECONDS", 10.0)  # would hang if retried
    attempts = {"n": 0}

    def broken():
        attempts["n"] += 1
        raise RuntimeError("no such table: market_bars")

    with pytest.raises(RuntimeError, match="no such table"):
        duckdb_store._open_with_retry(broken, "x.duckdb")
    assert attempts["n"] == 1
