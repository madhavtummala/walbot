from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src import mcp_server
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


def _build(monkeypatch) -> DummyMCP:
    fake_server = DummyMCP()
    monkeypatch.setattr(mcp_server, "_server", lambda *args, **kwargs: fake_server)
    mcp_server.create_mcp_server()
    return fake_server


def test_create_mcp_server_exposes_expected_tools(monkeypatch) -> None:
    fake_server = _build(monkeypatch)

    assert [tool.__name__ for tool in fake_server.tools] == [
        "get_algorithm_result",
        "get_current_positions",
        "place_orders",
    ]


def _result_payload(**overrides) -> dict:
    payload = {
        "strategy": "fast_momentum",
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

    assert restored.strategy == "fast_momentum"
    assert restored.target_weights == {"AAA": 0.5}
    assert restored.latest_prices == {"AAA": 100.0}


def test_validate_target_weights_normalises_symbols() -> None:
    cleaned, error = mcp_server._validate_target_weights({" bbc ": "0.4", "gld": 0.2})

    assert error is None
    assert cleaned == {"BBC": 0.4, "GLD": 0.2}
