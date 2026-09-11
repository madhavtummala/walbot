from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src import mcp_server
from src.api import controls as controls_module
from src.core import plan_cache
from src.core.config import Config
from src.core.interfaces import AlgorithmPlan, DesiredOrder, Intent, OrderRequest


@pytest.fixture(autouse=True)
def _empty_plan_cache():
    """Tokens are process-global, so one test's leftovers are another's phantom plan."""
    plan_cache.clear()
    yield
    plan_cache.clear()


class DummyMCP:
    def __init__(self) -> None:
        self.tools = []

    def tool(self):
        def decorator(func):
            self.tools.append(func)
            setattr(self, func.__name__, func)
            return func

        return decorator


def _binding(**overrides) -> dict:
    """A binding an agent is allowed to drive: switched on, parked on ``mcp``."""
    binding = {"id": "b1", "strategy": "rally_rotation", "account_id": "paper", "enabled": True, "cron": ""}
    binding.update(overrides)
    return binding


def _build(monkeypatch, bindings: list[dict] | None = None) -> DummyMCP:
    fake_server = DummyMCP()
    monkeypatch.setattr(mcp_server, "_server", lambda *args, **kwargs: fake_server)
    controls = {"bindings": [_binding()] if bindings is None else bindings}
    # Patched in both namespaces: the tools read controls directly, and
    # ``resolve_binding_for_origin`` reads them through its own module.
    monkeypatch.setattr(mcp_server, "load_controls", lambda *a, **k: controls)
    monkeypatch.setattr(controls_module, "load_controls", lambda *a, **k: controls)
    mcp_server.create_mcp_server()
    return fake_server


def test_create_mcp_server_exposes_expected_tools(monkeypatch) -> None:
    fake_server = _build(monkeypatch)

    assert [tool.__name__ for tool in fake_server.tools] == [
        "list_bindings",
        "get_algorithm_plan",
        "list_accounts",
        "get_account",
        "get_account_orders",
        "place_orders",
    ]


def _plan(**overrides) -> AlgorithmPlan:
    fields = {
        "strategy": "rally_rotation",
        "mode": "target",
        "intents": [Intent(symbol="AAA", kind="weight", value=0.5)],
        "latest_prices": {"AAA": 100.0},
        "signals": {"AAA": {"score": 1.0}},
        "state": {"accrued": 12.0},
    }
    fields.update(overrides)
    return AlgorithmPlan(**fields)


def _token(plan: AlgorithmPlan | None = None, *, binding_id: str = "b1", account_id: str = "default") -> str:
    """Stash a plan the way get_algorithm_plan would, and return its token.

    ``account_id`` matches ``Config()``'s own default, so a test that does not care about the
    account is not tripped by the guard that a plan must execute against the book it was sized
    against.
    """
    return plan_cache.stash(plan if plan is not None else _plan(), binding_id=binding_id, account_id=account_id)


def _placing(monkeypatch, *, positions: dict | None = None, outcome: dict | None = None) -> dict:
    """Stub out everything downstream of the gates and capture the plan that reaches execute."""
    captured: dict = {}

    class _Book:
        def get_positions(self):
            return dict(positions or {})

    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=False))
    monkeypatch.setattr(mcp_server, "resolve_brokerage", lambda config: _Book())
    monkeypatch.setattr(mcp_server, "record_orders", lambda *a, **k: None)

    def fake_execute(plan, config, brokerage, **kwargs):
        captured["plan"] = plan
        return outcome if outcome is not None else {"status": "submitted", "order_results": []}

    monkeypatch.setattr(mcp_server, "execute_algorithm", fake_execute)
    return captured


def _shares(plan: AlgorithmPlan) -> dict[str, float]:
    return {intent.symbol: intent.value for intent in plan.intents if intent.kind == "shares"}


def test_the_plan_payload_carries_what_review_needs() -> None:
    """The agent reads this to form a judgement. It is no longer read back, so the only
    requirement is that nothing a reviewer needs is missing."""
    payload = mcp_server._plan_payload(_plan())

    assert payload["strategy"] == "rally_rotation"
    assert payload["intents"] == [{"symbol": "AAA", "kind": "weight", "value": 0.5}]
    assert payload["latest_prices"] == {"AAA": 100.0}
    assert payload["state"] == {"accrued": 12.0}


def test_desired_orders_reach_the_agent_for_review() -> None:
    """An order-book algorithm's proposal lives entirely in ``desired_orders`` -- ``intents``
    is empty for this shape -- so omitting it showed the agent an empty plan."""
    plan = AlgorithmPlan(
        strategy="options_flip",
        desired_orders=[
            DesiredOrder(
                key="GLD:target",
                request=OrderRequest(
                    symbol="GLD   260918C00380000", action="sell", quantity=1,
                    order_type="limit", limit_price=33.02, time_in_force="gtc",
                    asset_type="option", extra={"position_intent": "sell_to_close"},
                ),
                replace_tolerance=0.02,
            ),
        ],
    )
    payload = mcp_server._plan_payload(plan)

    assert payload["intents"] == []
    assert payload["desired_orders"][0]["key"] == "GLD:target"
    assert payload["desired_orders"][0]["request"]["limit_price"] == 33.02
    assert payload["desired_orders"][0]["request"]["time_in_force"] == "gtc"
    assert payload["desired_orders"][0]["replace_tolerance"] == 0.02


def test_signals_pass_through_whole_for_a_non_rally_shape() -> None:
    """Was a fixed whitelist of Rally Rotation's own fields (score/reason/...), which silently
    emptied an Options Flip signal -- its row carries ``checks``/``estimate``/``contract``
    instead, none of which that list named."""
    plan = AlgorithmPlan(
        strategy="options_flip",
        signals={"GLD": {"state": "held", "headline": "Holding GLD 380 call", "checks": [
            {"label": "Holding VWAP", "ok": True, "value": "above", "blocking": False},
        ], "estimate": {"direction": "call", "contract": "GLD   260918C00380000"}}},
    )
    payload = mcp_server._plan_payload(plan)
    assert payload["signals"]["GLD"]["headline"] == "Holding GLD 380 call"
    assert payload["signals"]["GLD"]["checks"][0]["label"] == "Holding VWAP"
    assert payload["signals"]["GLD"]["estimate"]["contract"] == "GLD   260918C00380000"


def test_get_algorithm_plan_hands_back_a_token_that_places_that_same_plan(monkeypatch) -> None:
    """The two-call seam: the agent reviews a payload and submits a token, and the plan that
    executes is the one this side kept -- never one rebuilt from what came back."""
    fake_server = _build(monkeypatch)
    captured = _placing(monkeypatch)
    monkeypatch.setattr(mcp_server, "run_algorithm", lambda algorithm, config: _plan())

    proposed = fake_server.get_algorithm_plan("rally_rotation")
    assert proposed["status"] == "ok"
    assert proposed["plan_token"]

    assert fake_server.place_orders(proposed["plan_token"])["status"] == "submitted"
    assert captured["plan"].target_weights == {"AAA": 0.5}
    # The opaque ledger survives because it never left: ``execute`` commits the state on the
    # plan it is given, and a round trip through the agent was where it used to be droppable.
    assert captured["plan"].state == {"accrued": 12.0}


def test_a_plan_token_is_single_use(monkeypatch) -> None:
    """A retried tool call must not submit the same batch of orders a second time."""
    fake_server = _build(monkeypatch)
    _placing(monkeypatch)
    token = _token()

    assert fake_server.place_orders(token)["status"] == "submitted"

    replayed = fake_server.place_orders(token)
    assert replayed["status"] == "refused"
    assert "already been used" in replayed["reason"]


def test_an_unknown_plan_token_is_refused(monkeypatch) -> None:
    fake_server = _build(monkeypatch)
    _placing(monkeypatch)

    result = fake_server.place_orders("not-a-real-token")

    assert result["status"] == "refused"
    assert "get_algorithm_plan" in result["reason"]


def test_an_expired_plan_is_refused_rather_than_submitted_stale(monkeypatch) -> None:
    """A plan commits the prices it was built with, so an old one submits stale limit prices."""
    fake_server = _build(monkeypatch)
    _placing(monkeypatch)
    token = plan_cache.stash(_plan(), binding_id="b1", account_id="paper", ttl_seconds=1)
    # Reach past the clock rather than sleeping: the expiry is a timestamp comparison.
    monkeypatch.setattr(plan_cache, "_now", lambda: datetime.now(timezone.utc) + timedelta(seconds=30))

    result = fake_server.place_orders(token)

    assert result["status"] == "refused"
    assert "get_algorithm_plan" in result["reason"]


def test_a_binding_switched_off_after_planning_refuses_the_submission(monkeypatch) -> None:
    """The gap between reviewing and submitting is exactly where a kill decision lands."""
    controls = {"bindings": [_binding()]}
    fake_server = _build(monkeypatch, controls["bindings"])
    _placing(monkeypatch)
    token = _token()

    # Switched off while the agent was reading the news.
    off = {"bindings": [_binding(enabled=False)]}
    monkeypatch.setattr(mcp_server, "load_controls", lambda *a, **k: off)
    monkeypatch.setattr(controls_module, "load_controls", lambda *a, **k: off)

    result = fake_server.place_orders(token)

    assert result["status"] == "refused"
    assert "switched off" in result["reason"]


def test_a_plan_is_refused_if_its_binding_now_points_at_another_account(monkeypatch) -> None:
    """Quantities are sized against one book's holdings and equity; repointed, they describe an
    account these orders would no longer reach."""
    fake_server = _build(monkeypatch, [_binding(account_id="schwab2")])
    _placing(monkeypatch)
    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=False, account_id="schwab2"))

    result = fake_server.place_orders(_token(account_id="paper"))

    assert result["status"] == "refused"
    assert "sized against paper" in result["reason"]


def test_skip_holds_a_position_instead_of_selling_it(monkeypatch) -> None:
    """The trap this exists to avoid: under ``target`` mode the intent list *is* the portfolio,
    so dropping a row targets zero -- a veto expressed as a deletion would sell the position the
    agent meant to leave alone."""
    fake_server = _build(monkeypatch)
    captured = _placing(monkeypatch, positions={"AAA": 30.0})

    result = fake_server.place_orders(_token(), [{"op": "skip", "symbol": "AAA"}])

    assert result["status"] == "submitted"
    # Pinned to what is held: target equals current, so the sizer plans no trade at all.
    assert _shares(captured["plan"]) == {"AAA": 30.0}
    assert result["applied_edits"] == [{"op": "skip", "symbol": "AAA", "effect": "held at 30 shares"}]


def test_skip_under_incremental_mode_drops_the_increment(monkeypatch) -> None:
    """Incremental intents are deltas and anything unlisted is genuinely left alone, so here the
    right spelling of "leave it" really is removal."""
    fake_server = _build(monkeypatch)
    captured = _placing(monkeypatch, positions={"AAA": 30.0})
    token = _token(_plan(mode="incremental"))

    result = fake_server.place_orders(token, [{"op": "skip", "symbol": "AAA"}])

    assert result["status"] == "submitted"
    assert captured["plan"].intents == []


def test_exit_closes_a_position_the_plan_wanted_to_keep(monkeypatch) -> None:
    fake_server = _build(monkeypatch)
    captured = _placing(monkeypatch, positions={"AAA": 30.0})

    result = fake_server.place_orders(_token(), [{"op": "exit", "symbol": "AAA"}])

    assert _shares(captured["plan"]) == {"AAA": 0.0}
    assert result["applied_edits"] == [{"op": "exit", "symbol": "AAA", "effect": "closed to zero"}]


def test_exit_under_incremental_mode_sells_what_is_held(monkeypatch) -> None:
    """An incremental target is ``current + value``, so reaching zero means a negative delta."""
    fake_server = _build(monkeypatch)
    captured = _placing(monkeypatch, positions={"AAA": 30.0})

    fake_server.place_orders(_token(_plan(mode="incremental")), [{"op": "exit", "symbol": "AAA"}])

    assert _shares(captured["plan"]) == {"AAA": -30.0}


def test_an_edit_cannot_change_a_size(monkeypatch) -> None:
    """The fence from SOUL.md: the agent judges, it does not size. There is no edit op that
    takes an amount, so a payload trying to smuggle one is refused on the op name."""
    fake_server = _build(monkeypatch)
    _placing(monkeypatch, positions={"AAA": 30.0})

    result = fake_server.place_orders(_token(), [{"op": "set_weight", "symbol": "AAA", "value": 0.35}])

    assert result["status"] == "refused"
    assert "Unknown edit op" in result["reason"]


def test_edits_are_refused_for_an_order_book_plan(monkeypatch) -> None:
    """Reconciliation cancels whatever is not in ``desired_orders``, so removing a leg there
    cancels a resting stop rather than declining an action. No safe partial edit exists."""
    fake_server = _build(monkeypatch, [_binding(strategy="options_flip")])
    _placing(monkeypatch)
    plan = AlgorithmPlan(
        strategy="options_flip",
        desired_orders=[
            DesiredOrder(key="GLD:stop", request=OrderRequest(symbol="GLD", action="sell", quantity=1)),
        ],
    )

    result = fake_server.place_orders(_token(plan), [{"op": "skip", "symbol": "GLD"}])

    assert result["status"] == "refused"
    assert "cancels a resting order" in result["reason"]


def test_an_edit_naming_an_unrelated_symbol_is_refused(monkeypatch) -> None:
    """Almost always a mistyped ticker, and a silently ignored veto is the kind that gets
    noticed after the order fills."""
    fake_server = _build(monkeypatch)
    _placing(monkeypatch, positions={"AAA": 30.0})

    result = fake_server.place_orders(_token(), [{"op": "skip", "symbol": "ZZZ"}])

    assert result["status"] == "refused"
    assert "neither proposed by this plan nor held" in result["reason"]


def test_conflicting_edits_for_one_symbol_are_refused(monkeypatch) -> None:
    fake_server = _build(monkeypatch)
    _placing(monkeypatch, positions={"AAA": 30.0})

    result = fake_server.place_orders(
        _token(), [{"op": "skip", "symbol": "AAA"}, {"op": "exit", "symbol": "AAA"}]
    )

    assert result["status"] == "refused"
    assert "Conflicting edits" in result["reason"]


def test_placing_without_edits_reports_an_empty_edit_list(monkeypatch) -> None:
    """So a wrap can tell "the agent vetoed nothing" from "the field is missing"."""
    fake_server = _build(monkeypatch)
    _placing(monkeypatch)

    assert fake_server.place_orders(_token())["applied_edits"] == []


def test_place_orders_refuses_a_binding_the_scheduler_drives(monkeypatch) -> None:
    """The invariant: one origin per enabled binding, never both.

    A binding with a cron is the scheduler's. Letting an agent submit for it too is
    two live origins on one algorithm, which is what this gate exists to prevent.
    """
    fake_server = _build(monkeypatch, [_binding(cron="30 9 * * 1-5")])

    result = fake_server.place_orders(_token())

    assert result["status"] == "refused"
    assert "scheduler places its orders" in result["reason"].replace(", so the ", " ")


def test_place_orders_refuses_a_switched_off_binding(monkeypatch) -> None:
    fake_server = _build(monkeypatch, [_binding(enabled=False)])

    result = fake_server.place_orders(_token())

    assert result["status"] == "refused"
    assert "switched off" in result["reason"]


def test_place_orders_refuses_an_algorithm_with_no_binding(monkeypatch) -> None:
    fake_server = _build(monkeypatch, [_binding(strategy="dca")])

    # No binding id on the stash, so resolution falls back to the plan's own strategy.
    result = fake_server.place_orders(_token(binding_id=""))

    assert result["status"] == "refused"
    assert "No binding is configured" in result["reason"]


def test_place_orders_refuses_to_guess_between_two_eligible_bindings(monkeypatch) -> None:
    """Two bindings can share a strategy on different accounts, and guessing the binding is
    guessing the account -- the difference between a paper order and a real one."""
    fake_server = _build(monkeypatch, [_binding(id="b1"), _binding(id="b2", account_id="schwab")])

    result = fake_server.place_orders(_token(binding_id=""))

    assert result["status"] == "refused"
    assert "binding_id" in result["reason"]

    # Naming one resolves it.
    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=True))
    named = fake_server.place_orders(_token(binding_id="b2"))
    assert named["status"] == "skipped"  # got past the gate, stopped by the kill switch


def test_place_orders_uses_the_bindings_account_not_the_default(monkeypatch) -> None:
    """The bug this replaced: get_config() with no account_id resolves the *default* account,
    so an algorithm bound to a live account could have had its orders sent to a paper one."""
    fake_server = _build(monkeypatch, [_binding(account_id="schwab2")])
    seen: dict = {}

    def fake_get_config(**kwargs):
        seen.update(kwargs)
        return Config(kill_switch=True)

    monkeypatch.setattr(mcp_server, "get_config", fake_get_config)
    fake_server.place_orders(_token())

    assert seen["account_id"] == "schwab2"


def test_get_algorithm_plan_runs_for_a_scheduled_binding_but_says_it_cannot_trade(monkeypatch) -> None:
    """Computing a proposal is a read, like a backtest, so it is not gated -- but the agent is
    told plainly that acting on it will be refused."""
    fake_server = _build(monkeypatch, [_binding(cron="30 9 * * 1-5")])
    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=True))

    result = fake_server.get_algorithm_plan("rally_rotation")

    assert result["can_place_orders"] is False
    assert result["status"] == "error"  # stopped by the kill switch, not by the binding


def test_list_bindings_reports_what_the_agent_may_drive(monkeypatch) -> None:
    fake_server = _build(
        monkeypatch,
        [_binding(id="b1"), _binding(id="b2", strategy="rally_rotation", cron="30 9 * * 1-5"), _binding(id="b3", enabled=False)],
    )

    rows = {row["binding_id"]: row for row in fake_server.list_bindings()["bindings"]}

    assert rows["b1"]["can_place_orders"] is True
    assert rows["b1"]["driven_by"] == "mcp"
    assert rows["b2"]["can_place_orders"] is False
    assert rows["b2"]["driven_by"] == "schedule"
    assert rows["b3"]["can_place_orders"] is False


def test_one_binding_is_driven_by_exactly_one_origin() -> None:
    """The scheduler and the MCP tools ask the same function, so the two can never both say yes."""
    for cron in ("*/15 9-15 * * 1-5", "0 11 * * 1-5", "30 9 * * 1", ""):
        binding = _binding(cron=cron)
        schedule_ok = not controls_module.binding_refusal(binding, controls_module.ORIGIN_SCHEDULE)
        mcp_ok = not controls_module.binding_refusal(binding, controls_module.ORIGIN_MCP)
        assert schedule_ok != mcp_ok, cron

    off = _binding(enabled=False)
    assert controls_module.binding_refusal(off, controls_module.ORIGIN_SCHEDULE)
    assert controls_module.binding_refusal(off, controls_module.ORIGIN_MCP)
