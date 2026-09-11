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
from src.core.interfaces import AlgorithmPlan, OrderRequest
from src.core.pipeline import (
    TradingRefused,
    UnknownBrokerageError,
    assert_account_tradable,
    resolve_brokerage,
)
from src.core.plan_cache import PlanUnavailable, claim, stash
from src.core.plan_edits import PlanEditRefused, apply_edits
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


def _plan_payload(plan: AlgorithmPlan) -> dict[str, Any]:
    """Serialise a plan for the agent, keeping only what review and execution need.

    Review only. Nothing here is read back: ``place_orders`` takes a token and uses the plan
    this was serialised from, so these fields inform the agent's judgement and cannot alter what
    executes. ``state`` rides along opaquely for the same reason it always did -- the agent has
    no business reading an accrued budget -- but it can no longer be edited on the way back.

    ``signals`` is passed through whole rather than through a fixed key list. It used to
    whitelist Rally Rotation's own shape (``score``/``reason``/``signal``/...), which silently
    dropped everything an order-book algorithm like Options Flip actually reports -- its signal
    carries ``checks``, ``estimate`` and ``contract`` instead, none of which that whitelist
    named, so an agent reviewing an Options Flip plan saw five empty fields and nothing real.
    Every algorithm's signal row is already plain JSON-able data (see ``options_flip.algorithm.
    _signal``, ``rally_rotation``'s own row builder), so there is nothing left to filter.

    ``desired_orders`` carries an order-book algorithm's actual proposal -- the resting
    buy/sell/stop legs Options Flip's ``plan()`` builds instead of ``intents``. It is here so
    the agent can see what would rest at the broker; reconciliation reads the held plan.
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

        Returns a ``plan_token``. Review the plan, then pass *only* that token to place_orders;
        the plan itself stays on this side and is never sent back. It expires in about a minute
        and a half, and is good for one submission -- an expired token is a normal outcome, not
        a fault: call this again and review the fresh plan.

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
        plan = run_algorithm(algorithm, config)
        # Held here rather than round-tripped through the agent, so the plan that executes is
        # the plan that was reviewed. The token is the only part of this payload place_orders
        # reads back; everything else is for the agent to form a judgement on.
        token = stash(plan, binding_id=context["binding_id"], account_id=config.account_id)
        return {"status": "ok", **context, "plan_token": token, **_plan_payload(plan)}

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
    def place_orders(plan_token: str, edits: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Submit a reviewed plan. The response shape depends on the plan's own shape.

        ``plan_token`` is the token get_algorithm_plan returned. The plan itself never travels
        back: it is held here, so what executes is exactly what you reviewed. Submits
        immediately. The token is single-use and expires in about ninety seconds -- if it has,
        call get_algorithm_plan again rather than treating it as a failure.

        **To decline a plan, do not call this tool.** That is the whole veto, and it needs no
        argument. Say in your report what you declined and why.

        ``edits`` is the narrower case: the plan is broadly right but one symbol is not, because
        of something the algorithm could not see. Each edit is ``{"op": ..., "symbol": ...}``
        and names an action, never an amount -- sizing stays the algorithm's:

        - ``skip`` -- leave the symbol exactly as it is, neither bought nor sold today.
        - ``exit`` -- close the position in that symbol.

        Editing is refused for an order-book plan (Options Flip): a leg removed there cancels a
        resting order rather than declining it, so that shape is submit-whole or decline-whole.
        An edit naming a symbol the plan neither proposes nor holds is refused too, rather than
        ignored, because that is nearly always a mistyped ticker.

        The binding is the one the plan was computed against, re-checked here -- a binding
        switched off between planning and submitting refuses, and so does the kill switch.

        **Allocation strategies (Bursty DCA, Rally Rotation).** Orders are fitted to the
        account's available funds before submission, so ``status`` distinguishes:

        - ``submitted`` -- every leg went out at the size asked for.
        - ``submitted_reduced`` -- the batch was deliberately trimmed to what the account can
          pay for. This is a success. Do NOT resubmit: the legs in ``funding.reduced`` were
          shrunk on purpose, and re-sending them asks for money that is not there.
        - ``unfunded`` / ``partial`` / ``rejected`` -- something did not reach the market.
          ``unfunded`` lists legs no amount of shrinking could fund, ``rejected`` lists legs
          the broker itself refused; each row carries the reason.

        ``funding`` explains how the batch was paid for -- buying power, the reserve held
        back, sale proceeds, and any cash-equivalent holdings liquidated to cover a shortfall.

        **Options Flip** proposes an order book instead. The response carries no
        ``funding``/``diff``: read ``order_results`` (each entry ``submitted``/``replaced``/
        ``cancelled``/``unchanged``/``rejected``, with the order id and reason where relevant)
        and ``working_orders`` (what is now actually resting).
        """
        try:
            pending = claim(plan_token)
        except PlanUnavailable as unavailable:
            return {"status": "refused", "reason": unavailable.reason}
        plan = pending.plan
        # Re-resolved rather than trusted from the stash: a binding can be switched off, or
        # handed to the scheduler, in the gap between proposing a plan and submitting it, and
        # the authorisation that matters is the one in force now.
        binding, refusal = resolve_binding_for_origin(
            ORIGIN_MCP, binding_id=pending.binding_id, strategy=plan.strategy
        )
        if binding is None:
            return {"strategy": plan.strategy, "status": "refused", "reason": refusal}
        # The binding's account, not the default one. get_config() with no account_id resolves
        # whatever account is default, so an algorithm bound to a live account could have its
        # orders submitted to a paper one -- or the reverse. The scheduler has always passed the
        # binding's account through run_once; this path simply never did.
        config = get_config(account_id=binding["account_id"] or None, strategy_id=plan.strategy)
        # A plan's quantities are sized against one specific book's holdings and equity. If the
        # binding has been repointed since it was proposed, those numbers describe an account
        # this order would no longer reach.
        if pending.account_id and pending.account_id != config.account_id:
            return {
                "strategy": plan.strategy,
                "status": "refused",
                "reason": (
                    f"That plan was sized against {pending.account_id}, but {binding['id']} now "
                    f"points at {config.account_id}. Call get_algorithm_plan again."
                ),
            }

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
            # Applied here, against the live book rather than the one the plan was built on:
            # ``skip`` means "leave this position where it is", and where it is now is the only
            # honest reading of that.
            plan, applied_edits = apply_edits(plan, edits, positions=brokerage.get_positions())
        except PlanEditRefused as refused:
            return {"strategy": plan.strategy, "status": "refused", "reason": refused.reason}

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
        # Reported back even when empty: an outcome that shows only its final orders makes "the
        # agent vetoed two symbols" read identically to "the algorithm proposed nothing", and
        # those want very different follow-ups from whoever reads the wrap.
        return {**outcome, "applied_edits": applied_edits}

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
