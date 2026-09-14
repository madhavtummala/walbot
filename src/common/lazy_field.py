"""A value too slow to compute while someone is waiting for a page.

Some figures cost a year of broker transactions to answer. Computing them on every page load
makes reloading to check a price re-read a year of history; computing them never leaves the page
showing a blank and telling the user to press a button.

This does neither. A read is always instant -- it answers with whatever was last computed -- and
quietly starts a recompute in the background when what it has is missing or stale. Nobody waits,
and the value fills itself in. A caller who does want to wait says so once, with ``force``, which
is what a manual Refresh is: the user deciding this is worth the delay.

    ANALYTICS = LazyField("account analytics", compute_analytics)

    ANALYTICS.get(account_id)                # a page load: instant, self-filling
    ANALYTICS.get(account_id, force=True)    # Refresh: synchronous, always fresh
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

#: How long a computed value is served before a read starts refreshing it in the background.
#: Generous on purpose: the staleness is visible (every snapshot carries ``computed_at``) and a
#: user who wants certainty has a button, so there is nothing to buy by recomputing sooner.
DEFAULT_TTL_SECONDS = 900.0


def _spawn_in_thread(work: Callable[[], None]) -> None:
    """Run in a daemon thread, so a pending recompute never holds up a shutdown."""
    threading.Thread(target=work, daemon=True).start()


class LazyField:
    """One named value, computed per key, in the background unless forced.

    ``compute`` takes a key and returns the value. It may raise: a failed recompute leaves the
    previous value in place rather than replacing it with nothing, because a figure from twenty
    minutes ago is worth more than a blank.
    """

    def __init__(
        self,
        name: str,
        compute: Callable[[str], Any],
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        spawn: Callable[[Callable[[], None]], None] = _spawn_in_thread,
    ):
        self.name = name
        self._compute = compute
        self._ttl = float(ttl_seconds)
        # Injectable so a test can run the "background" work inline and assert on the result
        # instead of sleeping until a thread happens to finish.
        self._spawn = spawn
        self._values: Dict[str, Dict[str, Any]] = {}
        self._in_flight: set[str] = set()
        self._lock = threading.Lock()

    def get(self, key: str, *, force: bool = False) -> Dict[str, Any]:
        """A snapshot: ``{value, computed_at, state, error}``.

        ``state`` is ``ready`` when the value is real, ``computing`` while the first one is
        being worked out, and ``error`` when the last attempt failed with nothing cached to fall
        back on. ``computed_at`` always travels with the value, because a cached figure that
        does not say how old it is invites being read as live.
        """
        if force:
            return self._compute_now(key)

        with self._lock:
            cached = self._values.get(key)
            stale = not cached or (time.monotonic() - cached["_at"]) > self._ttl
            start = stale and key not in self._in_flight
            if start:
                self._in_flight.add(key)
        if start:
            # Fire and forget. The caller gets today's answer now and a better one next time.
            self._spawn(lambda: self._refresh(key))
        if cached:
            return self._snapshot(cached)
        return {"value": None, "computed_at": "", "state": "computing", "error": ""}

    def peek(self, key: str) -> Dict[str, Any]:
        """What is cached, without starting anything. For callers that must not cause work."""
        with self._lock:
            cached = self._values.get(key)
        return self._snapshot(cached) if cached else {
            "value": None, "computed_at": "", "state": "computing", "error": ""
        }

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)

    # -- internals ------------------------------------------------------------------------

    def _compute_now(self, key: str) -> Dict[str, Any]:
        try:
            value = self._compute(key)
        except Exception as error:  # noqa: BLE001 - one slow figure is not the whole page
            logger.warning("Could not compute %s for %s: %s", self.name, key, error)
            with self._lock:
                cached = self._values.get(key)
            # The previous value survives a failed refresh, and says what went wrong beside it.
            if cached:
                return {**self._snapshot(cached), "error": str(error)}
            return {"value": None, "computed_at": "", "state": "error", "error": str(error)}
        entry = {
            "value": value,
            "_at": time.monotonic(),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._values[key] = entry
        return self._snapshot(entry)

    def _refresh(self, key: str) -> None:
        try:
            self._compute_now(key)
        finally:
            # Released whatever happened, or one failure would wedge this key forever.
            with self._lock:
                self._in_flight.discard(key)

    @staticmethod
    def _snapshot(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "value": entry["value"],
            "computed_at": entry["computed_at"],
            "state": "ready",
            "error": "",
        }
