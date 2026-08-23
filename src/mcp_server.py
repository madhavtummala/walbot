from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

import uvicorn

from src.api.controls import (
    ORIGIN_MCP,
    binding_driver,
    binding_refusal,
    find_binding,
    load_controls,
    resolve_binding_for_origin,
)
from src.core.config import get_config
from src.core.interfaces import MODE_TARGET, AlgorithmPlan, Intent
from src.core.pipeline import UnknownBrokerageError, read_snapshot, resolve_brokerage
from src.core.runner import execute_algorithm, run_algorithm
from src.data.order_journal import record_orders

from mcp.server.fastmcp.server import Settings as FastMCPSettings

# The mcp library's Settings model annotates ``lifespan`` with a forward reference that
# resolves only after the whole module is loaded; constructing it before then makes
# pydantic-settings warn that the field definition is incomplete. Rebuild once against the
# library's own namespace so server construction stays quiet.
FastMCPSettings.model_rebuild()

logger = logging.getLogger(__name__)

DEFAULT_ALGORITHM = "rally_rotation"


def _plan_payload(plan: AlgorithmPlan) -> dict[str, Any]:
    """Serialise a plan for the agent, keeping only what review and execution need.

    ``state`` rides along opaquely. The agent has no business reading an accrued budget, but
    it has to hand it back untouched: the plan it returns is the plan that gets committed.
    """
    return {
        "strategy": plan.strategy,
        "as_of": plan.as_of.isoformat(),
        "mode": plan.mode,
        "intents": [
            {"symbol": intent.symbol, "kind": intent.kind, "value": round(intent.value, 6)}
            for intent in plan.intents
        ],
        "latest_prices": {symbol: round(price, 4) for symbol, price in plan.latest_prices.items()},
        "signals": {
            symbol: {
                "score": row.get("score"),
                "reason": row.get("reason"),
                "signal": row.get("signal"),
                "score_components": row.get("score_components"),
                "realized_volatility": row.get("realized_volatility"),
            }
            for symbol, row in plan.signals.items()
        },
        "allocation_mode": plan.metadata.get("allocation_mode"),
        "state": plan.state,
    }


def _plan_from_payload(payload: dict[str, Any]) -> AlgorithmPlan:
    """Rebuild a plan from the payload the agent was given, edits included."""
    return AlgorithmPlan(
        strategy=str(payload.get("strategy") or DEFAULT_ALGORITHM),
        intents=[
            Intent(symbol=str(row["symbol"]).upper(), kind=str(row.get("kind") or "weight"), value=float(row["value"]))
            for row in (payload.get("intents") or [])
        ],
        signals=payload.get("signals") or {},
        latest_prices={str(k).upper(): float(v) for k, v in (payload.get("latest_prices") or {}).items()},
        metadata={"allocation_mode": payload.get("allocation_mode")},
        mode=str(payload.get("mode") or MODE_TARGET),
        as_of=datetime.fromisoformat(payload["as_of"]) if payload.get("as_of") else datetime.now(timezone.utc),
        state=payload.get("state") or {},
    )


def _server(name: str, host: str, port: int):
    """Build the server object, across the 1.x/2.x rename.

    mcp 2.0 dropped ``mcp.server.fastmcp`` and renamed ``FastMCP`` to ``MCPServer`` under
    ``mcp.server.mcpserver``. The tool-registration and ``run`` surfaces this module uses are
    unchanged, so both are supported rather than pinning the project to the old package.
    """
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2.0
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as MCPServer  # mcp < 2.0
        except ImportError as exc:  # pragma: no cover - exercised in container builds with mcp installed.
            raise RuntimeError(
                "The MCP runtime is not installed. Install the 'mcp' package to use --mcp mode."
            ) from exc

    # 2.x moved host/port off the constructor and onto run(); 1.x accepted them in both places.
    try:
        return MCPServer(name, host=host, port=port)
    except TypeError:
        return MCPServer(name)


def create_mcp_server(host: str = "0.0.0.0", port: int = 8001):
    mcp = _server("walbot", host, port)

    @mcp.tool()
    def list_bindings() -> dict[str, Any]:
        """List the configured algorithm bindings and say which ones this agent may trade.

        A binding pairs an algorithm with an account and decides what drives it: a cron
        expression in market time, or an empty ``cron`` for "an agent drives this one". Only
        bindings reported here with ``can_place_orders: true`` will accept place_orders; the
        rest are switched off or are the scheduler's to run.
        """
        controls = load_controls()
        return {
            "status": "ok",
            "bindings": [
                {
                    "binding_id": binding["id"],
                    "algorithm": binding["strategy"],
                    "account_id": binding["account_id"],
                    "enabled": bool(binding["enabled"]),
                    "cron": binding["cron"],
                    "driven_by": binding_driver(binding),
                    "can_place_orders": not binding_refusal(binding, ORIGIN_MCP),
                    "reason": binding_refusal(binding, ORIGIN_MCP),
                }
                for binding in (controls.get("bindings") or [])
            ],
        }

    @mcp.tool()
    def get_algorithm_plan(algorithm: str = DEFAULT_ALGORITHM, binding_id: str = "") -> dict[str, Any]:
        """Run the algorithm against market data and return the plan it proposes.

        Returns the intents it wants to act on, the score and reason behind each symbol, and
        the prices the plan was built from. Nothing is submitted and nothing is remembered.
        Pass the whole payload back to place_orders -- it carries the prices and the state that
        step needs.

        Deliberately runs whatever it is asked to, switched on or not: computing a plan is the
        same read-only act as a backtest, and "what would this do right now" is worth answering
        for an algorithm the scheduler owns. Whether the plan may be *acted* on is reported as
        ``can_place_orders`` rather than decided here.
        """
        binding, refusal = resolve_binding_for_origin(ORIGIN_MCP, binding_id=binding_id, strategy=algorithm)
        # The account matters even for a read: a proposal is sized against the holdings and
        # equity of a specific account, so reporting one binding's plan against another's book
        # would be wrong in exactly the way that is hard to notice.
        config = get_config(account_id=(binding or {}).get("account_id") or None, strategy_id=algorithm)
        # Carried on every return, including the failures: an agent that is told only "kill
        # switch is enabled" cannot tell whether the binding would have accepted an order.
        context = {
            "binding_id": (binding or {}).get("id", ""),
            "account_id": config.account_id,
            "can_place_orders": binding is not None,
            "reason": refusal,
        }
        if config.kill_switch:
            return {"strategy": algorithm, "status": "error", **context, "reason": "Kill switch is enabled"}
        return {"status": "ok", **context, **_plan_payload(run_algorithm(algorithm, config))}

    @mcp.tool()
    def get_current_positions(binding_id: str = "") -> dict[str, Any]:
        """Return the live holdings reported by the brokerage, with equity and cash.

        Reports the default account unless ``binding_id`` names one, which matters as soon as
        more than one account is configured -- see list_bindings.
        """
        binding = find_binding(load_controls(), binding_id) if binding_id else None
        if binding_id and binding is None:
            return {"status": "error", "reason": f"No binding with id {binding_id!r}"}
        config = get_config(account_id=(binding or {}).get("account_id") or None)
        try:
            brokerage = resolve_brokerage(config)
        except UnknownBrokerageError as exc:
            return {"status": "error", "reason": str(exc)}

        account_state = brokerage.get_account_state()
        snapshot = read_snapshot(config, brokerage)
        return {
            "status": "ok",
            "account_id": config.account_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "equity": snapshot.equity,
            "cash": float(account_state.get("cash", 0.0)),
            "buying_power": float(account_state.get("buying_power", 0.0)),
            "positions": [
                {"symbol": symbol, "shares": shares}
                for symbol, shares in sorted(snapshot.positions.items())
            ],
        }

    @mcp.tool()
    def place_orders(algorithm_plan: dict[str, Any], binding_id: str = "") -> dict[str, Any]:
        """Submit orders for a reviewed plan, and return the resulting weight changes.

        ``algorithm_plan`` is the payload get_algorithm_plan returned. Pass it back unchanged to
        submit the proposal as-is, or edit its ``intents`` first -- that list is the complete
        intended action, so a held symbol dropped from a weight plan is sold to zero. Everything
        else in the payload must come back untouched: it carries the prices that size the shares
        and the state the algorithm will commit. Submits immediately.

        Only accepts bindings this agent drives -- switched on, with an empty ``cron``. Call
        list_bindings to see which those are, and name ``binding_id`` when one algorithm is
        bound to more than one account.

        Orders are fitted to the account's available funds before submission, so ``status``
        distinguishes three different things and only one of them is worth reacting to:

        - ``submitted`` -- every leg went out at the size asked for.
        - ``submitted_reduced`` -- the batch was deliberately trimmed to what the account can
          pay for. This is a success. Do NOT resubmit: the legs in ``funding.reduced`` were
          shrunk on purpose, and re-sending them asks for money that is not there.
        - ``unfunded`` / ``partial`` / ``rejected`` -- something did not reach the market.
          ``unfunded`` lists legs no amount of shrinking could fund, ``rejected`` lists legs
          the broker itself refused; each row carries the reason.

        ``funding`` explains how the batch was paid for -- buying power, the reserve held
        back, sale proceeds, and any cash-equivalent holdings liquidated to cover a shortfall.
        """
        plan = _plan_from_payload(algorithm_plan)
        # Resolved from configuration, never from the payload: ``algorithm_plan`` is whatever
        # the agent sent back, so the binding it claims cannot be the thing that authorises it.
        binding, refusal = resolve_binding_for_origin(ORIGIN_MCP, binding_id=binding_id, strategy=plan.strategy)
        if binding is None:
            return {"strategy": plan.strategy, "status": "refused", "reason": refusal}
        # The binding's account, not the default one. get_config() with no account_id resolves
        # whatever account is default, so an algorithm bound to a live account could have its
        # orders submitted to a paper one -- or the reverse. The scheduler has always passed the
        # binding's account through run_once; this path simply never did.
        config = get_config(account_id=binding["account_id"] or None, strategy_id=plan.strategy)

        if config.kill_switch:
            logger.warning("Kill switch is enabled. Skipping order placement for %s.", plan.strategy)
            return {"strategy": plan.strategy, "status": "skipped", "reason": "Kill switch is enabled"}

        try:
            brokerage = resolve_brokerage(config)
        except UnknownBrokerageError as exc:
            return {"strategy": plan.strategy, "status": "error", "reason": str(exc)}

        try:
            outcome = execute_algorithm(plan, config, brokerage)
        except ValueError as exc:
            return {"strategy": plan.strategy, "status": "error", "reason": str(exc)}
        # Journalled here as well as in the live runner: an agent-driven order is still
        # this algorithm's order, and the dashboard should not have a blind spot for it.
        record_orders(plan.strategy, config.account_id, outcome.get("order_results") or [])
        return outcome

    return mcp


def mcp_asgi_app(mcp, transport: str = "sse"):
    """The MCP server as a plain ASGI app, for serving it inside an existing process.

    ``mcp.run()`` builds this same app and then calls ``uvicorn.run`` on it, which owns the
    process: it creates the event loop and installs signal handlers. That is right for
    standalone use and wrong for embedding, so the app is taken directly instead.
    """
    builders = {
        "sse": "sse_app",
        "http": "streamable_http_app",
        "streamable-http": "streamable_http_app",
        "streamable_http": "streamable_http_app",
    }
    attribute = builders.get(transport)
    if attribute is None:
        raise ValueError(f"MCP transport {transport!r} cannot be served in-process; use 'sse' or 'streamable-http'.")
    # Named rather than called directly because the 1.x/2.x rename that ``_server`` works
    # around could move these too -- better a clear error than an AttributeError.
    builder = getattr(mcp, attribute, None)
    if builder is None:
        raise RuntimeError(f"The installed MCP runtime has no {attribute}(); cannot serve {transport!r} in-process.")
    return builder()


def serve_in_thread(*, host: str, port: int, transport: str = "sse") -> threading.Thread:
    """Serve the MCP tools on ``port`` from a thread of the *calling* process.

    Same process, deliberately. DuckDB permits one read-write process per database file and
    refuses every other opener -- including read-only ones -- so an MCP server in its own
    process contends with the dashboard for ``data/walbot.duckdb`` and one of them loses with
    "Conflicting lock is held". Sharing a process means sharing ``duckdb_store``'s connection
    pool, where DuckDB's own MVCC interleaves the readers and writers properly.

    A thread rather than a second event loop in the main thread because uvicorn installs
    process-wide signal handlers in ``Server.capture_signals``, and two servers doing that in
    one thread leaves the second one's handlers in place -- so a SIGTERM would stop only one of
    them and the process would hang on shutdown. Off the main thread uvicorn skips signal
    handling entirely, which is exactly what an embedded server wants.
    """
    mcp = create_mcp_server(host=host, port=port)
    server = uvicorn.Server(
        uvicorn.Config(
            mcp_asgi_app(mcp, transport),
            host=host,
            port=port,
            log_level=os.getenv("UVICORN_LOG_LEVEL", "warning"),
        )
    )

    def _serve() -> None:
        try:
            asyncio.run(server.serve())
        except Exception:  # noqa: BLE001 - a dead MCP server must not take the dashboard with it
            logger.exception("MCP tool server stopped")

    # Daemon so it cannot hold the process open: shutdown is the dashboard's to decide, and the
    # old subprocess was terminated rather than drained too.
    thread = threading.Thread(target=_serve, name="walbot-mcp", daemon=True)
    thread.start()
    return thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Walbot MCP tools.")
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", default=int(os.getenv("MCP_PORT", "8001")), type=int)
    parser.add_argument("--transport", default=os.getenv("MCP_TRANSPORT", "sse"))
    args = parser.parse_args()

    mcp = create_mcp_server(host=args.host, port=args.port)
    # stdio has nowhere to bind, and mcp 2.x rejects the kwargs on that transport.
    if args.transport == "stdio":
        mcp.run(transport=args.transport)
        return
    try:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    except TypeError:
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
