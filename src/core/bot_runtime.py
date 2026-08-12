from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Any
from zoneinfo import ZoneInfo

from src.core.config import DEFAULT_STRATEGY_ID, get_config
from src.core import pipeline
from ..api.controls import load_controls
from ..algorithms.registry import get_algorithm_class
from ..core.interfaces import Schedule, schedule_minutes
from ..execution.live_runner import run_once
from ..common.logging_utils import log_position_changes

logger = logging.getLogger(__name__)
MARKET_TZ = ZoneInfo("America/Chicago")
MARKET_OPEN_MINUTE = (8 * 60) + 30
MARKET_CLOSE_MINUTE = 15 * 60

@dataclass
class RuntimeState:
    enabled: bool = False
    running: bool = False
    account_id: str = ""
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_error: str = ""
    last_run_date: str = ""
    last_run_key: str = ""


class _RuntimeLoop:
    def __init__(self, name: str, enabled_fn, run_fn, interval_fn, run_key_fn=None, account_id_fn=None) -> None:
        self.name = name
        self._enabled_fn = enabled_fn
        self._run_fn = run_fn
        self._interval_fn = interval_fn
        self._run_key_fn = run_key_fn
        self._account_id_fn = account_id_fn or (lambda controls: str(controls.get("trading_account_id") or ""))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = RuntimeState()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name=f"dashboard-{self.name}-runtime", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def snapshot(self) -> dict[str, Any]:
        controls = load_controls()
        with self._lock:
            state = self._state.__dict__.copy()
        state["enabled"] = self._enabled_fn(controls)
        state["account_id"] = str(self._account_id_fn(controls) or state.get("account_id") or "")
        return state

    def _loop(self) -> None:
        while not self._stop.is_set():
            controls = load_controls()
            account_id = str(self._account_id_fn(controls) or "")
            enabled = self._enabled_fn(controls)
            with self._lock:
                self._state.enabled = enabled
                self._state.account_id = account_id
            if enabled:
                run_key = self._next_run_key()
                if run_key is not None:
                    self._run_guarded(account_id, run_key)
            self._wake.wait(self._check_seconds())
            self._wake.clear()

    def _check_seconds(self) -> int:
        try:
            return max(int(self._interval_fn()), 15)
        except Exception:
            return 300

    def _next_run_key(self) -> str | None:
        if self._run_key_fn is None:
            return ""
        run_key = self._run_key_fn()
        if not run_key:
            return None
        with self._lock:
            if self._state.last_run_key == run_key:
                return None
        return str(run_key)

    def _run_guarded(self, account_id: str, run_key: str) -> None:
        with self._lock:
            if self._state.running:
                return
            self._state.running = True
            self._state.last_started_at = datetime.now(timezone.utc).isoformat()
            self._state.last_error = ""
            if run_key:
                self._state.last_run_key = run_key
        
        logger.info("Starting %s run (key=%s)", self.name, run_key or "manual")
        try:
            self._run_fn(account_id or None)
            with self._lock:
                self._state.last_run_date = date.today().isoformat()
            logger.info("Completed %s run successfully", self.name)
        except Exception as exc:  # pragma: no cover - surfaced via status payload.
            logger.exception("%s runtime failed", self.name)
            with self._lock:
                self._state.last_error = str(exc)
        finally:
            with self._lock:
                self._state.running = False
                self._state.last_finished_at = datetime.now(timezone.utc).isoformat()


def _algorithm_enabled(controls: dict[str, Any]) -> bool:
    # No sentinel strategy to screen out any more: every id names a real algorithm, so the
    # power toggle is the whole answer. Options still has a "none" because it can be off
    # while equities trade.
    return bool(controls.get("algorithm_enabled"))


def _options_enabled(controls: dict[str, Any]) -> bool:
    config = get_config()
    return (
        bool(controls.get("options_trading_enabled"))
        and str(controls.get("options_strategy") or "none") != "none"
        and not config.kill_switch
    )


def _run_algorithm(account_id: str | None) -> None:
    run_once(account_id=account_id)


def _run_options(account_id: str | None) -> None:
    from .algorithms.options.swing import run_options_once

    run_options_once(account_id=account_id)


def _active_schedule(controls: dict[str, Any]) -> Schedule:
    """The selected algorithm's declared cadence.

    Cadence is a property of the strategy, not of the deployment, so it is read off the class
    rather than from config. An unknown strategy falls back to the base default, which only
    matters for how often the idle loop wakes to re-read controls.
    """
    strategy = str(controls.get("active_strategy") or DEFAULT_STRATEGY_ID)
    try:
        return get_algorithm_class(strategy).schedule
    except (KeyError, ValueError, TypeError):
        return Schedule()


def _is_regular_market_hours(now: datetime, schedule: Schedule) -> bool:
    local_now = now.astimezone(MARKET_TZ) if now.tzinfo else now.replace(tzinfo=MARKET_TZ)
    if local_now.weekday() not in schedule.weekdays:
        return False
    minute_of_day = (local_now.hour * 60) + local_now.minute
    start_minute = max(schedule_minutes(schedule.start_time, MARKET_OPEN_MINUTE), MARKET_OPEN_MINUTE)
    end_minute = min(schedule_minutes(schedule.end_time, MARKET_CLOSE_MINUTE), MARKET_CLOSE_MINUTE)
    return start_minute <= minute_of_day < end_minute


def _algorithm_jitter_offset_minutes(bucket: datetime, refresh_minutes: int, jitter_minutes: int) -> int:
    jitter_minutes = max(int(jitter_minutes or 0), 0)
    if jitter_minutes <= 0:
        return 0
    seed = f"{bucket.date().isoformat()}:{bucket.hour:02d}:{bucket.minute:02d}:{refresh_minutes}".encode("utf-8")
    return int.from_bytes(sha256(seed).digest()[:4], "big") % (jitter_minutes + 1)


def _algorithm_bucket_key(now: datetime, schedule: Schedule) -> str | None:
    if not _is_regular_market_hours(now, schedule):
        return None
    refresh_minutes = max(int(schedule.refresh_minutes or 30), 1)
    local_now = now.astimezone(MARKET_TZ) if now.tzinfo else now.replace(tzinfo=MARKET_TZ)
    minute_of_day = (local_now.hour * 60) + local_now.minute
    start_minute = max(schedule_minutes(schedule.start_time, MARKET_OPEN_MINUTE), MARKET_OPEN_MINUTE)
    bucket_minute = start_minute + (((minute_of_day - start_minute) // refresh_minutes) * refresh_minutes)
    bucket_hour, bucket_minute = divmod(bucket_minute, 60)
    bucket = local_now.replace(hour=bucket_hour, minute=bucket_minute, second=0, microsecond=0)
    jitter_offset = _algorithm_jitter_offset_minutes(bucket, refresh_minutes, schedule.jitter_minutes)
    scheduled = bucket + timedelta(minutes=jitter_offset)
    if local_now < scheduled:
        return None
    return f"algorithm:{bucket.isoformat(timespec='minutes')}"


def _algorithm_run_key() -> str | None:
    return _algorithm_bucket_key(datetime.now(MARKET_TZ), _active_schedule(load_controls()))


def _options_run_key() -> str | None:
    """Options keep the base cadence: the swing algorithm is not in the equities registry, so
    it has no ``Schedule`` of its own to read."""
    return _algorithm_bucket_key(datetime.now(MARKET_TZ), Schedule())


def _algorithm_check_seconds() -> int:
    return _active_schedule(load_controls()).check_seconds


def _options_check_seconds() -> int:
    return Schedule().check_seconds


def _options_account_id(controls: dict[str, Any]) -> str:
    return str(controls.get("options_trading_account_id") or controls.get("trading_account_id") or "")


class BotRuntime:
    """Two loops, not three.

    DCA used to get its own loop with its own cron, because it was not an algorithm. Now that
    it is one, it runs on the equities loop like everything else and its cadence comes from
    ``DCAAlgorithm.schedule``. Keeping the third loop would have let two schedulers drive the
    same accrual state at once.
    """

    def __init__(self) -> None:
        self.algorithm = _RuntimeLoop(
            "algorithm",
            _algorithm_enabled,
            _run_algorithm,
            _algorithm_check_seconds,
            _algorithm_run_key,
        )
        self.options = _RuntimeLoop(
            "options",
            _options_enabled,
            _run_options,
            _options_check_seconds,
            _options_run_key,
            _options_account_id,
        )

    def start(self) -> None:
        self.algorithm.start()
        self.options.start()

    def stop(self) -> None:
        self.algorithm.stop()
        self.options.stop()

    def snapshot(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm.snapshot(),
            "options": self.options.snapshot(),
        }


bot_runtime = BotRuntime()
