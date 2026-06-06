from __future__ import annotations

import argparse
import os
import uuid
from typing import Any

from src.api.api_payloads import (
    backtest_payload,
    controls_payload,
    dca_payload,
    status_payload,
    strategy_signals_payload,
    universe_payload,
)
from src.notifications.base import NotificationMessage
from src.notifications.service import (
    format_portfolio_change_message,
    request_trade_approval as request_trade_approval_via_notifications,
    send_notification,
)


def _server(name: str, host: str, port: int):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised in container builds with mcp installed.
        raise RuntimeError("The MCP runtime is not installed. Install the 'mcp' package to use --mcp mode.") from exc

    try:
        return FastMCP(name, host=host, port=port)
    except TypeError:
        return FastMCP(name)


def request_trade_approval_payload(
    planned_orders: list[dict[str, Any]],
    *,
    approval_id: str = "",
    timeout_seconds: int = 300,
    poll_seconds: int = 5,
) -> dict[str, Any]:
    if not isinstance(planned_orders, list) or not planned_orders:
        return {"approved": False, "approval_id": approval_id, "requested": False, "reason": "no_planned_orders"}
    approval_id = approval_id.strip() or uuid.uuid4().hex[:10]
    approved = request_trade_approval_via_notifications(
        planned_orders,
        approval_id=approval_id,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
    )
    return {"approved": bool(approved), "approval_id": approval_id, "requested": True}


def create_mcp_server(host: str = "0.0.0.0", port: int = 8001):
    mcp = _server("walbot", host, port)

    @mcp.tool()
    def get_status() -> dict[str, Any]:
        """Return dashboard status, bot state, risk config, and redacted runtime config."""
        return status_payload()

    @mcp.tool()
    def get_controls() -> dict[str, Any]:
        """Return dashboard trading controls."""
        return controls_payload()

    @mcp.tool()
    def get_universe() -> dict[str, Any]:
        """Return the configured trading universe."""
        return universe_payload()

    @mcp.tool()
    def get_dca_plan() -> dict[str, Any]:
        """Return the DCA plan and allocation preview."""
        return dca_payload()

    @mcp.tool()
    def generate_strategy_signals(strategy: str = "momentum_social") -> dict[str, Any]:
        """Generate current strategy signal rows for review."""
        return strategy_signals_payload(strategy=strategy)

    @mcp.tool()
    def run_backtest(strategy: str = "momentum_social", period: str = "", refresh: bool = False) -> dict[str, Any]:
        """Run or load a strategy backtest payload."""
        return backtest_payload({"strategy": strategy, "period": period, "refresh": refresh} if period else {"strategy": strategy, "refresh": refresh})

    @mcp.tool()
    def format_portfolio_notification(order_results: list[dict[str, Any]]) -> dict[str, Any]:
        """Format submitted order results into a portfolio-change notification."""
        text = format_portfolio_change_message(order_results)
        return {"text": text or "", "has_changes": bool(text)}

    @mcp.tool()
    def send_text_notification(text: str, subject: str = "") -> dict[str, Any]:
        """Send a text notification through configured notification providers."""
        sent = send_notification(NotificationMessage(text=text, subject=subject))
        return {"sent": sent}

    @mcp.tool()
    def request_trade_approval(
        planned_orders: list[dict[str, Any]],
        approval_id: str = "",
        timeout_seconds: int = 300,
        poll_seconds: int = 5,
    ) -> dict[str, Any]:
        """Request Telegram approve/deny for planned orders and return the user's decision."""
        return request_trade_approval_payload(
            planned_orders,
            approval_id=approval_id,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
        )

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Walbot MCP tools.")
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", default=int(os.getenv("MCP_PORT", "8001")), type=int)
    parser.add_argument("--transport", default=os.getenv("MCP_TRANSPORT", "sse"))
    args = parser.parse_args()

    mcp = create_mcp_server(host=args.host, port=args.port)
    try:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    except TypeError:
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
