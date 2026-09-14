"""The lazy-field primitive: instant reads, background recomputes, forced refresh."""

from __future__ import annotations

import time

from src.common.lazy_field import LazyField


def _inline(compute, **kwargs) -> LazyField:
    """A field whose "background" work runs inline, so a test never sleeps on a thread."""
    return LazyField("test", compute, spawn=lambda work: work(), **kwargs)


def test_the_first_read_answers_immediately_and_starts_the_work() -> None:
    calls: list[str] = []
    field = _inline(lambda key: calls.append(key) or f"value-{key}")

    first = field.get("a")

    assert first["value"] is None
    assert first["state"] == "computing"
    assert calls == ["a"], "the read started the computation without waiting for it"
    assert field.get("a")["value"] == "value-a"


def test_a_fresh_value_is_reused_rather_than_recomputed() -> None:
    calls: list[str] = []
    field = _inline(lambda key: calls.append(key) or "value")

    field.get("a")
    for _ in range(5):
        field.get("a")

    assert calls == ["a"]


def test_a_stale_value_is_served_while_a_new_one_is_worked_out() -> None:
    """The reader is never made to wait for the refresh their read triggered."""
    calls: list[int] = []

    def compute(_key):
        calls.append(len(calls))
        return f"generation-{len(calls)}"

    field = _inline(compute, ttl_seconds=0.0)
    field.get("a")
    assert field.peek("a")["value"] == "generation-1"

    # Stale now, so this read hands back the old value and computes the next one behind it.
    served = field.get("a")

    assert served["value"] == "generation-1", "answered from what it had, not from the new work"
    assert field.peek("a")["value"] == "generation-2"


def test_force_waits_and_returns_the_new_value() -> None:
    calls: list[int] = []
    field = _inline(lambda _key: calls.append(1) or f"generation-{len(calls)}")

    field.get("a")

    assert field.get("a", force=True)["value"] == "generation-2"


def test_one_computation_is_started_however_many_readers_arrive() -> None:
    """Every panel on a freshly-opened page reads this; only one crawl should result."""
    pending: list = []
    calls: list[str] = []
    field = LazyField("test", lambda key: calls.append(key) or "v", spawn=pending.append)

    for _ in range(4):
        field.get("a")

    assert len(pending) == 1
    pending[0]()
    assert calls == ["a"]


def test_a_failure_with_nothing_cached_reports_itself() -> None:
    def compute(_key):
        raise RuntimeError("broker is down")

    field = _inline(compute)
    field.get("a")

    snapshot = field.get("a", force=True)

    assert snapshot["value"] is None
    assert snapshot["state"] == "error"
    assert "broker is down" in snapshot["error"]


def test_a_failed_refresh_leaves_the_previous_value_standing() -> None:
    """A figure from twenty minutes ago is worth more than a blank."""
    attempts = {"n": 0}

    def compute(_key):
        attempts["n"] += 1
        if attempts["n"] > 1:
            raise RuntimeError("broker is down")
        return "good"

    field = _inline(compute)
    field.get("a")

    snapshot = field.get("a", force=True)

    assert snapshot["value"] == "good"
    assert "broker is down" in snapshot["error"]


def test_a_failed_computation_does_not_wedge_the_key() -> None:
    """The in-flight marker has to be released whatever happened, or this key never retries."""
    attempts = {"n": 0}

    def compute(_key):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("transient")
        return "good"

    field = _inline(compute)
    field.get("a")  # fails
    field.get("a")  # must be allowed to try again

    assert field.peek("a")["value"] == "good"


def test_keys_do_not_share_a_value() -> None:
    field = _inline(lambda key: f"value-{key}")
    field.get("a")
    field.get("b")

    assert field.peek("a")["value"] == "value-a"
    assert field.peek("b")["value"] == "value-b"


def test_peek_never_starts_work() -> None:
    calls: list[str] = []
    field = _inline(lambda key: calls.append(key) or "v")

    assert field.peek("a")["state"] == "computing"
    assert calls == []


def test_the_real_spawner_computes_off_the_calling_thread() -> None:
    """The default really is a thread -- the inline spawn used everywhere else is a test seam."""
    done = []
    field = LazyField("test", lambda _key: done.append(1) or "v")

    field.get("a")

    deadline = time.monotonic() + 5.0
    while not done and time.monotonic() < deadline:
        time.sleep(0.01)
    assert done, "the background computation never ran"
    assert field.peek("a")["value"] == "v"
