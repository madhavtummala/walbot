from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from src.core.config import DEFAULT_STRATEGY_ID
from ..api.controls import (
    ORIGIN_SCHEDULE,
    algorithm_default_cron,
    binding_refusal,
    find_binding,
    load_controls,
    normalize_cron,
)
from ..core.cron import CronError, cron_fire_key, parse_cron
from ..core.interfaces import MARKET_TZ
from ..execution.live_runner import run_once

logger = logging.getLogger(__name__)

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


def _binding_cron(binding_id: str) -> str:
    """This binding's schedule as saved, re-read every tick so an edit takes effect at once."""
    controls = load_controls()
    binding = find_binding(controls, binding_id)
    return normalize_cron((binding or {}).get("cron"), _binding_strategy(binding_id, controls))


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


def _binding_run_fn(binding_id: str):
    def run(account_id: str | None, run_key: str = "") -> None:
        # Resolved per run, so switching a binding's strategy takes effect on the next tick.
        run_once(account_id=account_id, strategy=_binding_strategy(binding_id))

    return run


def _binding_run_key(binding_id: str):
    """The fire time this tick belongs to, used once and then remembered as ``last_run_key``.

    Keyed on the *scheduled* minute rather than the current one, so every poll inside a fire's
    grace window yields the same key and the loop runs it exactly once.
    """

    def run_key() -> str | None:
        cron = _binding_cron(binding_id)
        # An empty cron is not a cadence the clock failed to parse -- it is a binding that
        # says an agent decides when it runs.
        if not cron:
            return None
        try:
            spec = parse_cron(cron)
        except CronError:
            # Saving is validated, so reaching here means a hand-edited config. Refusing to
            # fire is the safe reading: a schedule nobody can parse must not be guessed at.
            logger.warning("Binding %s has an unusable schedule %r; not running it", binding_id, cron)
            return None
        fired_at = cron_fire_key(spec, datetime.now(MARKET_TZ))
        if fired_at is None:
            return None
        return f"{binding_id}:{fired_at}"

    return run_key


def _binding_check_seconds(binding_id: str) -> Any:
    def check_seconds() -> int:
        # Wake far more often than any schedule fires. A cron names a minute, and a loop that
        # slept until roughly the next one would drift past it; polling cheaply and testing the
        # expression is what makes an 11:00 run land at 11:00. GRACE_MINUTES is the slack that
        # keeps a slow tick from missing its own fire time entirely.
        return 300

    return check_seconds


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
