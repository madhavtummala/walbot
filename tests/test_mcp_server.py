from __future__ import annotations

from src import mcp_server


def test_request_trade_approval_payload_uses_notification_approval(monkeypatch) -> None:
    calls = []

    def fake_request_trade_approval(planned_orders, *, approval_id, timeout_seconds, poll_seconds):
        calls.append((planned_orders, approval_id, timeout_seconds, poll_seconds))
        return True

    monkeypatch.setattr(mcp_server, "request_trade_approval_via_notifications", fake_request_trade_approval)

    result = mcp_server.request_trade_approval_payload(
        [{"symbol": "AAA", "action": "buy", "quantity": 2}],
        approval_id="abc123",
        timeout_seconds=30,
        poll_seconds=2,
    )

    assert result == {"approved": True, "approval_id": "abc123", "requested": True}
    assert calls == [([{"symbol": "AAA", "action": "buy", "quantity": 2}], "abc123", 30, 2)]


def test_request_trade_approval_payload_skips_empty_plans(monkeypatch) -> None:
    monkeypatch.setattr(
        mcp_server,
        "request_trade_approval_via_notifications",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not request approval")),
    )

    assert mcp_server.request_trade_approval_payload([], approval_id="abc123") == {
        "approved": False,
        "approval_id": "abc123",
        "requested": False,
        "reason": "no_planned_orders",
    }
