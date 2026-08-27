from __future__ import annotations

from datetime import datetime, timezone

from src import mcp_server
from src.api import controls as controls_module
from src.core.config import Config


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
        "get_current_positions",
        "place_orders",
    ]


def _plan_payload(**overrides) -> dict:
    payload = {
        "strategy": "rally_rotation",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "mode": "target",
        "intents": [{"symbol": "AAA", "kind": "weight", "value": 0.5}],
        "latest_prices": {"AAA": 100.0},
        "signals": {"AAA": {"score": 1.0}},
        "state": {"accrued": 12.0},
    }
    payload.update(overrides)
    return payload


def test_plan_payload_round_trips_through_the_agent() -> None:
    """Whatever the agent hands back is what gets committed, state included."""
    restored = mcp_server._plan_from_payload(_plan_payload())

    assert restored.strategy == "rally_rotation"
    assert restored.target_weights == {"AAA": 0.5}
    assert restored.latest_prices == {"AAA": 100.0}
    # Opaque to the agent, but it has to survive the round trip untouched: ``execute`` commits
    # the state on the plan it is given, so a payload that dropped it would reset the ledger.
    assert restored.state == {"accrued": 12.0}


def test_an_agents_edited_intents_are_the_ones_that_get_placed(monkeypatch) -> None:
    """The whole point of the two-call seam: review, edit, then submit what was reviewed."""
    fake_server = _build(monkeypatch)
    placed: dict = {}

    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=False))
    monkeypatch.setattr(mcp_server, "resolve_brokerage", lambda config: object())
    monkeypatch.setattr(mcp_server, "record_orders", lambda *a, **k: None)

    def fake_execute(plan, config, brokerage, **kwargs):
        placed["weights"] = plan.target_weights
        return {"status": "submitted", "order_results": []}

    monkeypatch.setattr(mcp_server, "execute_algorithm", fake_execute)
    edited = _plan_payload(intents=[{"symbol": "AAA", "kind": "weight", "value": 0.25}])

    assert fake_server.place_orders(edited)["status"] == "submitted"
    assert placed["weights"] == {"AAA": 0.25}


def test_place_orders_refuses_a_binding_the_scheduler_drives(monkeypatch) -> None:
    """The invariant: one origin per enabled binding, never both.

    A binding with a cron is the scheduler's. Letting an agent submit for it too is
    two live origins on one algorithm, which is what this gate exists to prevent.
    """
    fake_server = _build(monkeypatch, [_binding(cron="30 9 * * 1-5")])

    result = fake_server.place_orders(_plan_payload())

    assert result["status"] == "refused"
    assert "scheduler places its orders" in result["reason"].replace(", so the ", " ")


def test_place_orders_refuses_a_switched_off_binding(monkeypatch) -> None:
    fake_server = _build(monkeypatch, [_binding(enabled=False)])

    result = fake_server.place_orders(_plan_payload())

    assert result["status"] == "refused"
    assert "switched off" in result["reason"]


def test_place_orders_refuses_an_algorithm_with_no_binding(monkeypatch) -> None:
    fake_server = _build(monkeypatch, [_binding(strategy="dca")])

    result = fake_server.place_orders(_plan_payload())

    assert result["status"] == "refused"
    assert "No binding is configured" in result["reason"]


def test_place_orders_refuses_to_guess_between_two_eligible_bindings(monkeypatch) -> None:
    """Two bindings can share a strategy on different accounts, and guessing the binding is
    guessing the account -- the difference between a paper order and a real one."""
    fake_server = _build(monkeypatch, [_binding(id="b1"), _binding(id="b2", account_id="schwab")])

    result = fake_server.place_orders(_plan_payload())

    assert result["status"] == "refused"
    assert "binding_id" in result["reason"]

    # Naming one resolves it.
    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=True))
    named = fake_server.place_orders(_plan_payload(), "b2")
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
    fake_server.place_orders(_plan_payload())

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
