from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.core import plan_cache
from src.core.interfaces import AlgorithmPlan, Intent


@pytest.fixture(autouse=True)
def _empty():
    plan_cache.clear()
    yield
    plan_cache.clear()


def _plan() -> AlgorithmPlan:
    return AlgorithmPlan(strategy="rally_rotation", intents=[Intent(symbol="AAA", value=0.5)])


def test_a_stashed_plan_comes_back_identical() -> None:
    """Identity, not equality: the object that executes is the one the algorithm produced, so
    nothing in between can have re-derived it from a serialised copy."""
    plan = _plan()
    token = plan_cache.stash(plan, binding_id="b1", account_id="paper")

    pending = plan_cache.claim(token)

    assert pending.plan is plan
    assert pending.binding_id == "b1"
    assert pending.account_id == "paper"


def test_claiming_consumes_the_token() -> None:
    token = plan_cache.stash(_plan(), binding_id="b1", account_id="paper")
    plan_cache.claim(token)

    with pytest.raises(plan_cache.PlanUnavailable, match="already been used"):
        plan_cache.claim(token)


def test_an_unknown_token_is_refused() -> None:
    with pytest.raises(plan_cache.PlanUnavailable, match="get_algorithm_plan"):
        plan_cache.claim("nonsense")


def test_an_empty_token_is_refused_rather_than_matching_something() -> None:
    plan_cache.stash(_plan(), binding_id="b1", account_id="paper")

    for empty in ("", None):
        with pytest.raises(plan_cache.PlanUnavailable):
            plan_cache.claim(empty)


def test_a_plan_expires(monkeypatch) -> None:
    """Prices ride on the plan and are committed rather than re-read, so age is staleness."""
    token = plan_cache.stash(_plan(), binding_id="b1", account_id="paper", ttl_seconds=60)
    monkeypatch.setattr(plan_cache, "_now", lambda: datetime.now(timezone.utc) + timedelta(seconds=61))

    with pytest.raises(plan_cache.PlanUnavailable):
        plan_cache.claim(token)


def test_a_plan_inside_its_ttl_survives(monkeypatch) -> None:
    token = plan_cache.stash(_plan(), binding_id="b1", account_id="paper", ttl_seconds=60)
    monkeypatch.setattr(plan_cache, "_now", lambda: datetime.now(timezone.utc) + timedelta(seconds=30))

    assert plan_cache.claim(token).binding_id == "b1"


def test_expired_plans_do_not_accumulate(monkeypatch) -> None:
    """The cache is never swept on a timer, so eviction has to ride on the calls it does get."""
    for _ in range(3):
        plan_cache.stash(_plan(), binding_id="b1", account_id="paper", ttl_seconds=1)
    assert len(plan_cache._pending) == 3

    monkeypatch.setattr(plan_cache, "_now", lambda: datetime.now(timezone.utc) + timedelta(seconds=30))
    plan_cache.stash(_plan(), binding_id="b1", account_id="paper")

    assert len(plan_cache._pending) == 1


def test_tokens_are_unguessable_and_distinct() -> None:
    tokens = {plan_cache.stash(_plan(), binding_id="b1", account_id="paper") for _ in range(50)}

    assert len(tokens) == 50
    assert all(len(token) >= 20 for token in tokens)
