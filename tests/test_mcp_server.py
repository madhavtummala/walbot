from __future__ import annotations

from src import mcp_server


class DummyMCP:
    def __init__(self) -> None:
        self.tools = []

    def tool(self):
        def decorator(func):
            self.tools.append(func)
            setattr(self, func.__name__, func)
            return func

        return decorator


def test_create_mcp_server_exposes_expected_tools(monkeypatch) -> None:
    fake_server = DummyMCP()
    monkeypatch.setattr(mcp_server, "_server", lambda *args, **kwargs: fake_server)
    monkeypatch.setattr(mcp_server, "strategy_signals_payload", lambda strategy="momentum_social": {"strategy": strategy, "ok": True, "leaders": []})

    mcp = mcp_server.create_mcp_server()

    assert [tool.__name__ for tool in fake_server.tools] == [
        "get_live_signals",
        "get_portfolio_preview",
        "get_planned_orders",
        "place_orders",
    ]
    assert mcp.get_live_signals("fast_momentum") == {"strategy": "fast_momentum", "ok": True, "leaders": []}
