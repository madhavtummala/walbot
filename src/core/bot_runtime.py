from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Any
from zoneinfo import ZoneInfo

from src.core.config import DEFAULT_STRATEGY_ID
from ..api.controls import (
    ORIGIN_SCHEDULE,
    binding_refusal,
    find_binding,
    frequency_minutes,
    load_controls,
    normalize_frequency,
)
from ..algorithms.registry import get_algorithm_class
from ..core.interfaces import Schedule, schedule_minutes
from ..execution.live_runner import run_once

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
            self._run_fn(account_id or None, run_key)
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


def _binding_frequency(binding_id: str) -> str:
    binding = find_binding(load_controls(), binding_id)
    return normalize_frequency((binding or {}).get("frequency"))


def _binding_enabled(binding_id: str):
    """Enabled check scoped to one binding, re-read from controls on every tick.

    Delegates to ``binding_refusal`` so the scheduler and the MCP tools apply one rule about
    which origin owns a binding, rather than each carrying its own opinion.
    """

    def enabled(controls: dict[str, Any]) -> bool:
        return not binding_refusal(find_binding(controls, binding_id), ORIGIN_SCHEDULE)

    return enabled


def _binding_account_id(binding_id: str):
    def account_id(controls: dict[str, Any]) -> str:
        binding = find_binding(controls, binding_id)
        return str((binding or {}).get("account_id") or controls.get("trading_account_id") or "")

    return account_id


def _binding_strategy(binding_id: str, controls: dict[str, Any] | None = None) -> str:
    controls = controls if controls is not None else load_controls()
    binding = find_binding(controls, binding_id)
    return str((binding or {}).get("strategy") or DEFAULT_STRATEGY_ID)


def _binding_schedule(binding_id: str) -> Schedule:
    try:
        return get_algorithm_class(_binding_strategy(binding_id)).schedule
    except (KeyError, ValueError, TypeError):
        return Schedule()


def _frequency_minutes(frequency: str) -> int:
    """Minutes between fires. Only reached for scheduled bindings, so ``mcp`` cannot appear."""
    return frequency_minutes(frequency) or 60


def _binding_run_fn(binding_id: str):
    def run(account_id: str | None, run_key: str = "") -> None:
        # Resolved per run, so switching a binding's strategy takes effect on the next tick.
        run_once(account_id=account_id, strategy=_binding_strategy(binding_id))

    return run


def _binding_run_key(binding_id: str):
    def run_key() -> str | None:
        frequency = _binding_frequency(binding_id)
        # No cadence means nothing for the clock to bucket -- an agent decides when this runs.
        if frequency_minutes(frequency) is None:
            return None
        local_now = datetime.now(MARKET_TZ)
        # The binding owns the cadence, the algorithm still owns the session window. Without
        # this a 1hr binding fires at 03:00 on a Sunday: run_once bails at the market-closed
        # check, but only after a full data fetch, and it still stamps "last run" with a time
        # the market was shut.
        if not _is_regular_market_hours(local_now, _binding_schedule(binding_id)):
            return None
        minutes = _frequency_minutes(frequency)
        if minutes >= 24 * 60:
            bucket = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            bucket_minute = ((local_now.hour * 60) + local_now.minute) // minutes * minutes
            bucket_hour, minute = divmod(bucket_minute, 60)
            bucket = local_now.replace(hour=bucket_hour, minute=minute, second=0, microsecond=0)
        return f"{binding_id}:{bucket.isoformat(timespec='minutes')}"

    return run_key


def _binding_check_seconds(binding_id: str):
    def check_seconds() -> int:
        minutes = frequency_minutes(_binding_frequency(binding_id))
        # An agent-driven binding still wakes, but only to re-read controls in case its
        # frequency changed; there is no cadence of its own to follow.
        return 300 if minutes is None else max(minutes * 60, 60)

    return check_seconds


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


class BotRuntime:
    """One scheduler loop per algorithm binding.

    Bindings are user-editable at runtime, so the loop set is reconciled against controls
    rather than fixed at construction: adding a binding in the dashboard starts a loop for it,
    removing one stops that loop and leaves the others alone.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._algorithm_loops: dict[str, _RuntimeLoop] = {}
        self._started = False

    def _make_loop(self, binding_id: str) -> _RuntimeLoop:
        return _RuntimeLoop(
            f"algorithm-{binding_id}",
            _binding_enabled(binding_id),
            _binding_run_fn(binding_id),
            _binding_check_seconds(binding_id),
            _binding_run_key(binding_id),
            _binding_account_id(binding_id),
        )

    def reconcile(self) -> None:
        try:
            bindings = load_controls().get("bindings") or []
        except Exception:  # noqa: BLE001 - a bad config must not kill the runtime
            logger.exception("Could not read bindings; leaving runtime loops as they are")
            return
        wanted = {str(binding.get("id")) for binding in bindings}
        with self._lock:
            for binding_id in list(self._algorithm_loops):
                if binding_id not in wanted:
                    self._algorithm_loops.pop(binding_id).stop()
            for binding_id in wanted:
                if binding_id not in self._algorithm_loops:
                    loop = self._make_loop(binding_id)
                    self._algorithm_loops[binding_id] = loop
                    if self._started:
                        loop.start()

    def start(self) -> None:
        self._started = True
        self.reconcile()
        with self._lock:
            loops = list(self._algorithm_loops.values())
        for loop in loops:
            loop.start()

    def stop(self) -> None:
        self._started = False
        with self._lock:
            loops = list(self._algorithm_loops.values())
        for loop in loops:
            loop.stop()

    @property
    def algorithm(self) -> _RuntimeLoop:
        """The first binding's loop, for callers that predate multiple bindings."""
        self.reconcile()
        with self._lock:
            loops = list(self._algorithm_loops.values())
        return loops[0] if loops else self._make_loop("b1")

    def snapshot(self) -> dict[str, Any]:
        self.reconcile()
        with self._lock:
            loops = dict(self._algorithm_loops)
        bindings = {binding_id: loop.snapshot() for binding_id, loop in loops.items()}
        first = next(iter(bindings.values()), None)
        return {
            "bindings": bindings,
            "algorithm": first if first is not None else RuntimeState().__dict__,
        }


bot_runtime = BotRuntime()
