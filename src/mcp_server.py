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
    load_controls,
    resolve_binding_for_origin,
)
from src.core.config import get_config
from src.core.interfaces import MODE_TARGET, AlgorithmPlan, DesiredOrder, Intent, OrderRequest
from src.core.pipeline import (
    TradingRefused,
    UnknownBrokerageError,
    assert_account_tradable,
    resolve_brokerage,
)
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


def _order_request_payload(request: OrderRequest) -> dict[str, Any]:
    return {
        "symbol": request.symbol, "action": request.action, "quantity": request.quantity,
        "order_type": request.order_type, "limit_price": request.limit_price,
        "stop_price": request.stop_price, "client_order_id": request.client_order_id,
        "time_in_force": request.time_in_force, "asset_type": request.asset_type,
        "strategy": request.strategy,
        "children": [_order_request_payload(child) for child in request.children],
        "extra": dict(request.extra),
    }


def _order_request_from_payload(payload: dict[str, Any]) -> OrderRequest:
    return OrderRequest(
        symbol=str(payload["symbol"]), action=str(payload["action"]),
        quantity=float(payload["quantity"]),
        order_type=str(payload.get("order_type") or "market"),
        limit_price=payload.get("limit_price"), stop_price=payload.get("stop_price"),
        client_order_id=payload.get("client_order_id"),
        time_in_force=str(payload.get("time_in_force") or "day"),
        asset_type=str(payload.get("asset_type") or "equity"),
        strategy=str(payload.get("strategy") or "single"),
        children=tuple(_order_request_from_payload(child) for child in (payload.get("children") or [])),
        extra=dict(payload.get("extra") or {}),
    )


def _plan_payload(plan: AlgorithmPlan) -> dict[str, Any]:
    """Serialise a plan for the agent, keeping only what review and execution need.

    ``state`` rides along opaquely. The agent has no business reading an accrued budget, but
    it has to hand it back untouched: the plan it returns is the plan that gets committed.

    ``signals`` is passed through whole rather than through a fixed key list. It used to
    whitelist Rally Rotation's own shape (``score``/``reason``/``signal``/...), which silently
    dropped everything an order-book algorithm like Options Flip actually reports -- its signal
    carries ``checks``, ``estimate`` and ``contract`` instead, none of which that whitelist
    named, so an agent reviewing an Options Flip plan saw five empty fields and nothing real.
    Every algorithm's signal row is already plain JSON-able data (see ``options_flip.algorithm.
    _signal``, ``rally_rotation``'s own row builder), so there is nothing left to filter.

    ``desired_orders`` carries an order-book algorithm's actual proposal -- the resting
    buy/sell/stop legs Options Flip's ``plan()`` builds instead of ``intents``. Omitting it here
    was a real bug, not a simplification: ``place_orders`` rebuilds an ``AlgorithmPlan`` from
    whatever the agent sends back, and a plan rebuilt with no ``desired_orders`` reconciles
    against an empty wanted-set, which cancels every order the position currently has resting
    at the broker (a live stop and target included) and replaces them with nothing.
    """
    return {
        "strategy": plan.strategy,
        "as_of": plan.as_of.isoformat(),
        "mode": plan.mode,
        "intents": [
            {"symbol": intent.symbol, "kind": intent.kind, "value": round(intent.value, 6)}
            for intent in plan.intents
        ],
        "desired_orders": [
            {
                "key": order.key,
                "request": _order_request_payload(order.request),
                "replace_tolerance": order.replace_tolerance,
            }
            for order in plan.desired_orders
        ],
        "latest_prices": {symbol: round(price, 4) for symbol, price in plan.latest_prices.items()},
        "signals": dict(plan.signals),
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
        desired_orders=[
            DesiredOrder(
                key=str(row["key"]),
                request=_order_request_from_payload(row["request"]),
                replace_tolerance=float(row.get("replace_tolerance") or 0.0),
            )
            for row in (payload.get("desired_orders") or [])
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

        The proposal lives in one of two places depending on the algorithm's shape, and reading
        the wrong one for a given strategy will look like an empty plan:

        - Allocation strategies (Bursty DCA, Rally Rotation) propose a *portfolio*: read
          ``intents`` (what to hold) and ``mode`` (whether the list is the complete target or
          only the symbols to touch). ``signals`` explains the reasoning per symbol.
        - Options Flip proposes an *order book* instead: ``intents`` is always empty for it.
          Read ``desired_orders`` for what should be resting at the broker right now (entry bid,
          or a held position's target/stop), and ``signals[symbol]`` for the reasoning --
          ``checks`` (each gate, pass/fail and why), ``estimate`` (the band prediction and
          greeks-priced profit), and ``contract``/``state``/``headline``.

        Pass the whole payload back to place_orders unchanged, or edited -- it carries the
        prices and the state that step needs regardless of which proposal shape it holds.

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
    def list_accounts() -> dict[str, Any]:
        """Name every configured account. Start here, then ask the other two tools per account.

        Cheap and money-free: ids, labels, broker, and ``deployments`` (the algorithms bound to
        each). Reading a portfolio is this list plus one get_account call per row -- fanned out
        this way rather than as one all-accounts call so a slow or unreachable broker costs you
        that account and not the answer.
        """
        from src.api.payloads.accounts import accounts_payload

        payload = accounts_payload()
        return {
            "status": "ok",
            "default_account": payload["default"],
            "accounts": [
                {k: row[k] for k in ("id", "label", "broker", "deployments", "credentials_ready")}
                for row in payload["rows"]
            ],
        }

    @mcp.tool()
    def get_account(account_id: str = "") -> dict[str, Any]:
        """Holdings, cash and P/L for one account. Defaults to the default account.

        The account is the unit, not the algorithm: a broker reports one blended position per
        symbol, so two algorithms trading the same account cannot be told apart here.

        Carries ``equity``, ``cash``, ``day_pl`` (and percent), ``total_pl``, ``dividend_pl``
        and the holdings in ``rows``. ``day_pl: null`` means the broker did not report where the
        session started -- "unknown", not "flat". An unreachable broker fills ``error`` and
        leaves the figures null rather than reporting zeros.

        This is the same function the dashboard's account page calls, through the brokerage
        interface, so every broker answers it the same way.
        """
        from src.api.payloads.accounts import positions_payload

        return {"status": "ok", "updated_at": datetime.now(timezone.utc).isoformat(), **positions_payload(account_id)}

    @mcp.tool()
    def get_account_orders(account_id: str = "", limit: int = 40) -> dict[str, Any]:
        """One account's recent orders, in every state, most recent first.

        Filled, partially filled, replaced, cancelled, rejected and still-resting arrive in one
        list -- a live order simply appears with a resting status and an unfilled quantity, so
        there is no separate working-orders question to ask.

        Capped at ``limit`` rather than windowed by time, which matters for good-till-cancelled
        orders: a stop that has rested for two days is current exposure but was submitted long
        ago, and any 24-hour window would drop it.

        Same function the dashboard's account page calls. It reads the broker's own record where
        the broker keeps one, so manual trades show up too; where it does not, it falls back to
        the bot's order journal, and a row's ``status`` says which vocabulary you are reading.
        """
        from src.api.payloads.accounts import account_activity_payload

        return {"status": "ok", **account_activity_payload(account_id, limit=limit)}

    @mcp.tool()
    def place_orders(algorithm_plan: dict[str, Any], binding_id: str = "") -> dict[str, Any]:
        """Submit orders for a reviewed plan. The response shape depends on the plan's own shape.

        ``algorithm_plan`` is the payload get_algorithm_plan returned. Pass it back unchanged to
        submit the proposal as-is, or edit it first. Everything not deliberately edited must
        come back untouched: ``latest_prices`` and ``state`` are committed as given, not
        recomputed. Submits immediately.

        Only accepts bindings this agent drives -- switched on, with an empty ``cron``. Call
        list_bindings to see which those are, and name ``binding_id`` when one algorithm is
        bound to more than one account.

        **Allocation strategies (Bursty DCA, Rally Rotation) -- edit ``intents``.** That list is
        the complete intended action under ``mode: "target"``, so a held symbol dropped from it
        is sold to zero. Orders are fitted to the account's available funds before submission,
        so ``status`` distinguishes:

        - ``submitted`` -- every leg went out at the size asked for.
        - ``submitted_reduced`` -- the batch was deliberately trimmed to what the account can
          pay for. This is a success. Do NOT resubmit: the legs in ``funding.reduced`` were
          shrunk on purpose, and re-sending them asks for money that is not there.
        - ``unfunded`` / ``partial`` / ``rejected`` -- something did not reach the market.
          ``unfunded`` lists legs no amount of shrinking could fund, ``rejected`` lists legs
          the broker itself refused; each row carries the reason.

        ``funding`` explains how the batch was paid for -- buying power, the reserve held
        back, sale proceeds, and any cash-equivalent holdings liquidated to cover a shortfall.

        **Options Flip -- edit ``desired_orders``, never ``intents`` (it is always empty for
        this strategy).** Each entry is one resting order the algorithm wants at the broker
        right now, keyed by role (``SYMBOL:entry`` / ``:target`` / ``:stop``). Missing or
        dropping a key here is not "leave it as-is" -- reconciliation cancels whatever is not
        in this list, so an edited payload missing a held position's ``:stop`` cancels that
        stop at the broker. If you did not mean to touch a symbol, leave every one of its keys
        exactly as returned. The response carries no ``funding``/``diff``: read
        ``order_results`` (each entry ``submitted``/``replaced``/``cancelled``/``unchanged``/
        ``rejected``, with the order id and reason where relevant) and ``working_orders`` (what
        is now actually resting).
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

        try:
            # Refused before authenticating: there is no point reaching a venue we have already
            # decided not to trade. The same rule is re-asserted inside ``execute_algorithm``,
            # which is where it is actually enforced -- this is the early out, not the check.
            assert_account_tradable(config)
        except TradingRefused as refusal:
            logger.warning("Refused to execute %s: %s", plan.strategy, refusal.reason)
            return {"strategy": plan.strategy, "status": "skipped", "reason": refusal.reason}

        try:
            brokerage = resolve_brokerage(config)
        except UnknownBrokerageError as exc:
            return {"strategy": plan.strategy, "status": "error", "reason": str(exc)}

        try:
            # The kill switch and the paper-only restriction are asserted inside, so this path
            # no longer has to remember them -- and can no longer forget one, which is how a
            # paper-only algorithm became executable against a live account from here.
            outcome = execute_algorithm(plan, config, brokerage)
        except TradingRefused as refusal:
            logger.warning("Refused to execute %s: %s", plan.strategy, refusal.reason)
            return {"strategy": plan.strategy, "status": "skipped", "reason": refusal.reason}
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
