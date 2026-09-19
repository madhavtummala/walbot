"""A value too slow to compute while someone is waiting for a page.

Some figures cost a year of broker transactions to answer. Computing them on every page load
makes reloading to check a price re-read a year of history; computing them never leaves the page
showing a blank and telling the user to press a button.

This does neither. A read is always instant -- it answers with whatever was last computed -- and
quietly starts a recompute in the background when what it has is missing or stale. Nobody waits,
and the value fills itself in.

Three modes, and the difference between the last two is what the reader sees while the work runs:

    ANALYTICS = LazyField("account analytics", compute_analytics)

    ANALYTICS.get(account_id)                  # a visit: last value now, recompute behind it
    ANALYTICS.get(account_id, force=True)      # wait for a fresh one, and answer with it
    ANALYTICS.recompute(account_id)            # drop it, recompute in the background

``recompute`` is what an explicit reload asks for. Forcing would hold the request open for the
length of the work -- tolerable for a broker read, not for a backtest that replays months of
bars -- so instead the value is dropped, the work is started, and the snapshot comes back saying
``computing``. The page has nothing to show and draws its skeleton, which is precisely the
"clear it and load it again" the reader asked for, without a request that hangs for minutes.

Every snapshot carries ``refreshing``, which says whether work is in flight for that key right
now. That is what lets a page serve a stale value *and* know to look again shortly, rather than
displaying it forever because the read that triggered the recompute has already returned.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

#: How long a computed value is served before a read starts refreshing it in the background.
#: Generous on purpose: the staleness is visible (every snapshot carries ``computed_at``), and
#: since a visit now starts a refresh by itself, a shorter window would mostly buy duplicate
#: work rather than fresher numbers. Instances that move faster pass their own.
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
            # Read under the same lock that decided to start, so the flag cannot say "idle" for
            # work this very call just queued.
            refreshing = start or key in self._in_flight
        if start:
            # Fire and forget. The caller gets today's answer now and a better one next time.
            self._spawn(lambda: self._refresh(key))
        if cached:
            return self._snapshot(cached, refreshing=refreshing)
        return {
            "value": None, "computed_at": "", "state": "computing",
            "error": "", "refreshing": refreshing,
        }

    def recompute(self, key: str) -> Dict[str, Any]:
        """Drop the cached value and compute a new one in the background.

        What an explicit reload asks for, and the reason ``force`` is not the answer to it:
        forcing holds the request open for the length of the work, which for a backtest is
        minutes. Here the value goes, the work starts, and the snapshot says ``computing`` -- so
        the page paints its skeleton immediately and fills in when the result lands.

        Deduped through the same ``_in_flight`` set as the lazy path, so hitting reload twice
        starts one recompute rather than two. ``force`` deliberately does not dedupe: a caller
        that asked to wait for a fresh value must not be handed one computed before it asked.
        """
        with self._lock:
            self._values.pop(key, None)
            start = key not in self._in_flight
            if start:
                self._in_flight.add(key)
        if start:
            self._spawn(lambda: self._refresh(key))
        return {
            "value": None, "computed_at": "", "state": "computing", "error": "", "refreshing": True,
        }

    def refreshing(self, key: str) -> bool:
        """Whether a compute for this key is running right now."""
        with self._lock:
            return key in self._in_flight

    def peek(self, key: str) -> Dict[str, Any]:
        """What is cached, without starting anything. For callers that must not cause work."""
        with self._lock:
            cached = self._values.get(key)
            refreshing = key in self._in_flight
        return self._snapshot(cached, refreshing=refreshing) if cached else {
            "value": None, "computed_at": "", "state": "computing",
            "error": "", "refreshing": refreshing,
        }

    def reset(self) -> None:
        """Forget every value and every in-flight marker.

        For test isolation. These instances are module-level, so without it one test's cached
        value -- or a marker left by a recompute that was never allowed to run -- is still there
        for the next one, and a suite that passes in order fails when run alone.
        """
        with self._lock:
            self._values.clear()
            self._in_flight.clear()

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
            return {
                "value": None, "computed_at": "", "state": "error",
                "error": str(error), "refreshing": False,
            }
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
    def _snapshot(entry: Dict[str, Any], *, refreshing: bool = False) -> Dict[str, Any]:
        return {
            "value": entry["value"],
            "computed_at": entry["computed_at"],
            "state": "ready",
            "error": "",
            # Whether a newer value is on its way. A reader serving this snapshot uses it to
            # decide whether to look again shortly, which is the only way a background result
            # reaches a page that has already been painted.
            "refreshing": refreshing,
        }
