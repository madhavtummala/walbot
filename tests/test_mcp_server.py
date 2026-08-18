from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
    binding = {"id": "b1", "strategy": "rally_rotation", "account_id": "paper", "enabled": True, "frequency": "mcp"}
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
        "get_algorithm_result",
        "get_current_positions",
        "place_orders",
    ]


def _result_payload(**overrides) -> dict:
    payload = {
        "strategy": "rally_rotation",
        "as_of": datetime.now(timezone.utc).isoformat(),
        "target_weights": {"AAA": 0.5},
        "latest_prices": {"AAA": 100.0},
        "signals": {"AAA": {"score": 1.0}},
    }
    payload.update(overrides)
    return payload


def test_place_orders_rejects_weights_over_one(monkeypatch) -> None:
    fake_server = _build(monkeypatch)

    result = fake_server.place_orders(_result_payload(), {"AAA": 0.7, "BBB": 0.5})

    assert result["status"] == "error"
    assert "exceeds 1.0" in result["reason"]


def test_place_orders_rejects_empty_and_negative_weights(monkeypatch) -> None:
    fake_server = _build(monkeypatch)

    assert fake_server.place_orders(_result_payload(), {})["status"] == "error"
    assert "negative" in fake_server.place_orders(_result_payload(), {"AAA": -0.2})["reason"]


def test_place_orders_refuses_a_stale_result(monkeypatch) -> None:
    fake_server = _build(monkeypatch)
    stale = _result_payload(as_of=(datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())

    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=False))
    monkeypatch.setattr(mcp_server, "resolve_brokerage", lambda config: object())

    result = fake_server.place_orders(stale)

    assert result["status"] == "error"
    assert "too stale" in result["reason"]


def test_result_payload_round_trips_through_the_agent() -> None:
    payload = _result_payload()

    restored = mcp_server._result_from_payload(payload)

    assert restored.strategy == "rally_rotation"
    assert restored.target_weights == {"AAA": 0.5}
    assert restored.latest_prices == {"AAA": 100.0}


def test_place_orders_refuses_a_binding_the_scheduler_drives(monkeypatch) -> None:
    """The invariant: one origin per enabled binding, never both.

    A binding on a clock frequency is the scheduler's. Letting an agent submit for it too is
    two live origins on one algorithm, which is what this gate exists to prevent.
    """
    fake_server = _build(monkeypatch, [_binding(frequency="1hr")])

    result = fake_server.place_orders(_result_payload())

    assert result["status"] == "refused"
    assert "scheduler places its orders" in result["reason"].replace(", so the ", " ")


def test_place_orders_refuses_a_switched_off_binding(monkeypatch) -> None:
    fake_server = _build(monkeypatch, [_binding(enabled=False)])

    result = fake_server.place_orders(_result_payload())

    assert result["status"] == "refused"
    assert "switched off" in result["reason"]


def test_place_orders_refuses_an_algorithm_with_no_binding(monkeypatch) -> None:
    fake_server = _build(monkeypatch, [_binding(strategy="dca")])

    result = fake_server.place_orders(_result_payload())

    assert result["status"] == "refused"
    assert "No binding is configured" in result["reason"]


def test_place_orders_refuses_to_guess_between_two_eligible_bindings(monkeypatch) -> None:
    """Two bindings can share a strategy on different accounts, and guessing the binding is
    guessing the account -- the difference between a paper order and a real one."""
    fake_server = _build(monkeypatch, [_binding(id="b1"), _binding(id="b2", account_id="schwab")])

    result = fake_server.place_orders(_result_payload())

    assert result["status"] == "refused"
    assert "binding_id" in result["reason"]

    # Naming one resolves it.
    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=True))
    named = fake_server.place_orders(_result_payload(), None, "b2")
    assert named["status"] == "skipped"  # got past the gate, stopped by the kill switch


def test_place_orders_uses_the_bindings_account_not_the_default(monkeypatch) -> None:
    """The bug this replaced: get_config() with no account_id resolves the *default* account,
    so an algorithm bound to a live account could have had its orders sent to a paper one."""
    fake_server = _build(monkeypatch, [_binding(account_id="schwab_individual")])
    seen: dict = {}

    def fake_get_config(**kwargs):
        seen.update(kwargs)
        return Config(kill_switch=True)

    monkeypatch.setattr(mcp_server, "get_config", fake_get_config)
    fake_server.place_orders(_result_payload())

    assert seen["account_id"] == "schwab_individual"


def test_get_algorithm_result_runs_for_a_scheduled_binding_but_says_it_cannot_trade(monkeypatch) -> None:
    """Computing a proposal is a read, like a backtest, so it is not gated -- but the agent is
    told plainly that acting on it will be refused."""
    fake_server = _build(monkeypatch, [_binding(frequency="1hr")])
    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=True))

    result = fake_server.get_algorithm_result("rally_rotation")

    assert result["can_place_orders"] is False
    assert result["status"] == "error"  # stopped by the kill switch, not by the binding


def test_list_bindings_reports_what_the_agent_may_drive(monkeypatch) -> None:
    fake_server = _build(
        monkeypatch,
        [_binding(id="b1"), _binding(id="b2", strategy="rally_rotation", frequency="1hr"), _binding(id="b3", enabled=False)],
    )

    rows = {row["binding_id"]: row for row in fake_server.list_bindings()["bindings"]}

    assert rows["b1"]["can_place_orders"] is True
    assert rows["b1"]["driven_by"] == "mcp"
    assert rows["b2"]["can_place_orders"] is False
    assert rows["b2"]["driven_by"] == "schedule"
    assert rows["b3"]["can_place_orders"] is False


def test_one_binding_is_driven_by_exactly_one_origin() -> None:
    """The scheduler and the MCP tools ask the same function, so the two can never both say yes."""
    for frequency in ("15m", "30m", "1hr", "2hr", "1d", "mcp"):
        binding = _binding(frequency=frequency)
        schedule_ok = not controls_module.binding_refusal(binding, controls_module.ORIGIN_SCHEDULE)
        mcp_ok = not controls_module.binding_refusal(binding, controls_module.ORIGIN_MCP)
        assert schedule_ok != mcp_ok, frequency

    off = _binding(enabled=False)
    assert controls_module.binding_refusal(off, controls_module.ORIGIN_SCHEDULE)
    assert controls_module.binding_refusal(off, controls_module.ORIGIN_MCP)


def test_validate_target_weights_normalises_symbols() -> None:
    cleaned, error = mcp_server._validate_target_weights({" bbc ": "0.4", "gld": 0.2})

    assert error is None
    assert cleaned == {"BBC": 0.4, "GLD": 0.2}
