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
    deployment_driver,
    deployment_refusal,
    find_deployment,
    load_controls,
    resolve_deployment_for_origin,
)
from src.core.config import get_config
from src.core.interfaces import AlgorithmPlan, OrderRequest
from src.core.strategy_models import STRATEGY_LABELS
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


#: Money to the cent, ratios to four places -- a ratio is read as a percent with two decimals.
#: Applied only on the way out of a tool, never to the payload functions themselves, because
#: the dashboard formats its own display and would rather keep the full float.
_RATIO_FIELDS = ("day_pl_percent", "unrealized_plpc")

#: What an order does not have. A market order carries no limit and no stop, an unfilled one no
#: fill price, and an order nobody refused no reason; all four say nothing a reader did not
#: already know from ``order_type`` and ``status``. Dropped from the row rather than sent empty.
_ORDER_ABSENCES = ("limit_price", "stop_price", "filled_avg_price", "reason")


def _compact(value: Any, *, drop: tuple[str, ...] = ()) -> Any:
    """The same answer in fewer characters, for tools whose reader pays by the token.

    An agent reading a portfolio holds every one of these rows in its context at once, so a
    float printed to seventeen places costs it real room for no information -- ``unrealized_pl``
    came back as ``66.39599999999973`` where ``66.4`` is the whole of what anyone can act on.
    Rounding only; nothing is renamed, reordered, or reinterpreted, and a ``None`` stays
    ``None`` except for the keys in ``drop``, where absence is the same fact as empty. Those
    drop on any empty value, not just ``None``: a reason is missing as ``""`` and a price as
    ``null``, and both mean the row has nothing to say about it.
    """
    if isinstance(value, dict):
        return {
            key: _compact(item, drop=drop)
            for key, item in value.items()
            if not (key in drop and not item)
        }
    if isinstance(value, list):
        return [_compact(item, drop=drop) for item in value]
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    return round(value, 2)


def _compact_rows(rows: Any, *, ratios: tuple[str, ...] = (), drop: tuple[str, ...] = ()) -> Any:
    """``_compact`` for a list of rows, with the ratio columns kept at four places."""
    if not isinstance(rows, list):
        return rows
    compacted = []
    for row in rows:
        if not isinstance(row, dict):
            compacted.append(_compact(row, drop=drop))
            continue
        tidy = _compact(row, drop=drop)
        for field in ratios:
            if isinstance(row.get(field), float):
                tidy[field] = round(row[field], 4)
        compacted.append(tidy)
    return compacted

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
    def list_algorithms() -> dict[str, Any]:
        """Every algorithm this bot has, and for each one where it trades and who drives it.

        Start here: this is the only way to learn which algorithm ids exist, and the id is what
        every other algorithm tool takes. ``description`` is one line on what each one does --
        enough to say which algorithm a report is about without calling get_algorithm_plan,
        which is the only way to see what it proposes *today*.

        Deployment is a property of the algorithm, reported inline:

        - ``deployed: false`` -- it names no account, so it has nothing to trade against.
          ``get_algorithm_plan`` still works on it (against the default account) but
          ``place_orders`` will refuse.
        - ``account_id`` -- the one account it trades. An algorithm never has more than one;
          several algorithms may share an account, which costs attribution, not correctness.
        - ``cron`` -- a schedule in market time means the scheduler owns it, and an agent may
          not place its orders. Empty means an agent drives it. Exactly one of the two, always,
          reported as ``driven_by``.

        Only rows with ``can_place_orders: true`` will accept place_orders; every other row
        carries the ``reason`` it will not.
        """
        from src.algorithms.explainers import EXPLAINERS
        from src.algorithms.registry import ALGORITHMS

        controls = load_controls()
        rows = []
        for algorithm in ALGORITHMS.names():
            deployment = find_deployment(controls, algorithm)
            refusal = deployment_refusal(deployment, ORIGIN_MCP)
            rows.append(
                {
                    "algorithm": algorithm,
                    "name": STRATEGY_LABELS.get(algorithm, algorithm),
                    # The one-line form, not the explainer's paragraph: this is a list of every
                    # algorithm at once, and three paragraphs here would cost more than the
                    # whole rest of the payload.
                    "description": (EXPLAINERS.get(algorithm) or {}).get("headline", ""),
                    "deployed": deployment is not None,
                    "account_id": (deployment or {}).get("account_id", ""),
                    "enabled": bool((deployment or {}).get("enabled", False)),
                    "cron": (deployment or {}).get("cron", ""),
                    "driven_by": deployment_driver(deployment) if deployment else "",
                    "can_place_orders": not refusal,
                    "reason": refusal,
                }
            )
        return {"status": "ok", "algorithms": rows}

    @mcp.tool()
    def get_algorithm_plan(algorithm: str = DEFAULT_ALGORITHM) -> dict[str, Any]:
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
        the plan itself stays on this side and is never sent back. It expires in about five
        minutes, and is good for one submission -- an expired token is a normal outcome, not a
        fault: call this again and review the fresh plan.

        Runs whatever it is asked to, switched on or not; whether the plan may be *acted* on
        is reported as ``can_place_orders`` rather than decided here. Sized against the account
        the algorithm is deployed to. An algorithm deployed nowhere is still planned, against
        the default account, and reports ``can_place_orders: false``.
        """
        deployment = find_deployment(load_controls(), algorithm)
        refusal = deployment_refusal(deployment, ORIGIN_MCP)
        # The account comes from the deployment itself, not from the permission check: a
        # proposal is sized against the holdings and equity of a specific account, and a
        # switched-off algorithm still has an account to be sized against. Reading it off the
        # authorisation instead is how a refused deployment's plan came to be computed against
        # whichever account happened to be the default.
        config = get_config(account_id=(deployment or {}).get("account_id") or None, strategy_id=algorithm)
        # Carried on every return, including the failures: an agent that is told only "kill
        # switch is enabled" cannot tell whether the deployment would have accepted an order.
        context = {
            "account_id": config.account_id,
            "can_place_orders": not refusal,
            "reason": refusal,
        }
        if config.kill_switch:
            return {"strategy": algorithm, "status": "error", **context, "reason": "Kill switch is enabled"}
        plan = run_algorithm(algorithm, config)
        # Held here rather than round-tripped through the agent, so the plan that executes is
        # the plan that was reviewed. The token is the only part of this payload place_orders
        # reads back; everything else is for the agent to form a judgement on.
        token = stash(plan, algorithm=algorithm, account_id=config.account_id)
        return {"status": "ok", **context, "plan_token": token, **_plan_payload(plan)}

    @mcp.tool()
    def get_price(symbol: str, timestamp: str = "") -> dict[str, Any]:
        """One price for one symbol, from this bot's own market data.

        **Prefer this over a web search for any price.** A search engine answers with whatever
        a page said when it was indexed, which for a quote is routinely days old and is never
        marked as such. This reads the bar store the algorithms trade on.

        ``timestamp`` is ISO 8601 -- ``"2026-07-15"`` or ``"2026-07-15T14:30:00Z"`` -- and
        defaults to now. The answer is the nearest bar in time, from the five-minute series
        where the store has it and the daily series otherwise, so a stamp naming a time gets
        an intraday price for a recent date and a session close for an old one.

        **Read ``as_of``, always, and quote it rather than the timestamp you asked for.** It is
        what was actually struck, and the gap can be minutes on a recent intraday date or days
        on a weekend, a holiday, or a date before the symbol listed. ``price`` is that bar's
        close.

        A bare date is midnight, so for a date with intraday bars it answers near that
        session's *open* rather than its close -- roughly a day's range away from what "the
        closing price on the 15th" would mean. Name a time if you need one end of the session.
        Two prices read the same way are still a sound comparison: a return between two bare
        dates measures open to open.

        One symbol, one point. A 60-day return is two calls and a subtraction; a trend is a
        call per point you want.

        A symbol the store has never held answers ``status: "error"`` with a null ``price``,
        never a zero -- a zero is a number someone might do arithmetic with.
        """
        from src.api.payloads.accounts import _parse_stamp
        from src.data.duckdb_store import read_closest_bar

        ticker = str(symbol or "").strip().upper()[:12]
        if not ticker:
            return {"status": "error", "symbol": "", "price": None,
                    "error": "No symbol given. Pass one, e.g. 'SPY'."}

        raw = str(timestamp or "").strip()
        wanted_at = _parse_stamp(raw) if raw else datetime.now(timezone.utc)
        if wanted_at is None:
            return {"status": "error", "symbol": ticker, "price": None,
                    "error": f"Could not read {timestamp!r} as a date. Use ISO 8601, e.g. '2026-07-15'."}

        try:
            bar = read_closest_bar(ticker, wanted_at)
        except Exception as error:  # noqa: BLE001 - reported as itself, not as "no such symbol"
            logger.warning("Price lookup failed for %s: %s", ticker, error)
            return {"status": "error", "symbol": ticker, "price": None,
                    "error": f"Could not read prices for {ticker}: {error}"}
        if bar is None:
            return {"status": "error", "symbol": ticker, "price": None,
                    "error": f"Nothing could price {ticker}."}

        struck = bar["timestamp"]
        return {
            "status": "ok",
            "symbol": ticker,
            "price": round(bar["close"], 4),
            "as_of": struck.isoformat() if hasattr(struck, "isoformat") else str(struck),
            "error": "",
        }

    @mcp.tool()
    def list_accounts() -> dict[str, Any]:
        """Every account, each with a headline of what it is worth and what it did today.

        Start here. Ids, labels, broker and ``deployments`` (the algorithms bound to each),
        plus the figures that decide whether an account is worth a closer look: ``equity``,
        ``cash``, ``day_pl`` (and percent), ``total_pl``, ``realized_pl``, ``positions`` and
        ``orders_today``.

        ``total_pl`` is *open* P/L -- the gain on what is still held, since each position was
        opened -- and is the same figure get_account_positions reports under that name.
        ``day_pl`` is only today's move. The two are not interchangeable.

        **An account flat on the day with no orders needs no further call.** That is what this
        is for: read the headline, then spend get_account_positions and get_account_orders only
        on the accounts that actually did something.

Every figure is current, ``realized_pl`` included. The answer is small -- each field is
        one number -- but not instant: every row is a live broker read and realized P/L matches
        a year of fills, so expect seconds, not bytes. Read per account, so a slow or
        unreachable broker costs you that account's figures and not the listing; it reports
        ``error`` and nulls, never zeros, because "we could not ask" and "nothing happened" are
        different answers.
        """
        from src.api.payloads.accounts import account_headline, accounts_payload

        payload = accounts_payload()
        accounts = []
        for row in payload["rows"]:
            identity = {k: row[k] for k in ("id", "label", "broker", "deployments", "credentials_ready")}
            # Only for accounts that could answer. Asking a broker we have no keys for buys a
            # guaranteed error per row and tells the reader nothing they cannot see from
            # ``credentials_ready``.
            headline = account_headline(row["id"]) if row.get("credentials_ready") else {}
            accounts.append({**identity, **headline})
        return {
            "status": "ok",
            "default_account": payload["default"],
            # Rounded like every other money figure these tools hand over, so the same number
            # does not arrive to two different precisions from two different tools.
            "accounts": _compact_rows(accounts, ratios=_RATIO_FIELDS, drop=("error",)),
        }

    @mcp.tool()
    def get_account_positions(account_id: str = "") -> dict[str, Any]:
        """Holdings, cash and P/L for one account. Defaults to the default account.

        The account is the unit, not the algorithm: a broker reports one blended position per
        symbol, so two algorithms trading the same account cannot be told apart here.

        ``day_pl: null`` means the broker did not report where the session started --
        "unknown", not "flat". An unreachable broker fills ``error`` and leaves the figures
        null, never zero.

        Three P/L figures, never interchangeable. ``total_pl`` and a row's ``unrealized_pl``
        are *open* P/L, the whole gain since each position was opened; ``day_pl`` is only
        today's move. A position bought months ago and one bought this morning differ in the
        first and can agree in the second.

        ``realized_pl`` is profit already banked, **year to date**, as is ``dividend_pl``. The
        window is not a trailing year, so do not infer one from it. An account that closed a
        winning trade and went back to cash holds nothing, so its open and day figures are both
        zero while ``realized_pl`` carries the entire gain -- never read a zero ``total_pl`` as
        "this account has not made money". ``realized_unmatched`` above zero means some sells'
        opening buys predate the window and could not be priced: the total is real but partial.

        Money is rounded to the cent and ratios to four places -- ``day_pl_percent: 0.0013`` is
        a 0.13% day. A total and its parts can differ by a cent.
        """
        # Year to date because that is what a broker's own statement totals, making it the one
        # figure here a user can check us against. Fills are still read over a trailing year --
        # matching this year's sells needs last year's buys -- but no trailing-year total is
        # ever reported. The same function the dashboard's account page calls, through the
        # brokerage interface, so every broker answers it the same way.
        from src.api.payloads.accounts import account_analytics_payload, positions_payload

        # Forced, unlike a page load: an agent gets one shot at an answer and cannot come back
        # a second later to see whether a background recompute landed.
        positions = positions_payload(account_id)
        analytics = account_analytics_payload(account_id, refresh=True)
        # Merged by hand rather than by ``**``, which let the second dict's empty ``error`` erase
        # the first's real one -- an unreachable broker reported null balances and said nothing
        # about why. ``state`` and ``computed_at`` are dropped or renamed for the same reason:
        # beside an account's balances, a bare "state" reads as the account's.
        #
        # ``dividend_rows`` is dropped because it was never anything but weight here: up to forty
        # distributions, each with the broker's own description string, carried on every call to
        # summarise them into the one number -- ``dividend_pl`` -- that this tool documents and
        # any reader actually uses. The dashboard still renders the rows; it reads the analytics
        # payload directly and is unaffected.
        carried = {
            key: value for key, value in analytics.items()
            if key not in ("error", "state", "computed_at", "account_id", "dividend_rows")
        }
        errors = [text for text in (positions.get("error"), analytics.get("error")) if text]
        merged = {
            "status": "ok",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **positions,
            **carried,
            # Only the realized/dividend half is ever served from a computation; the balances
            # above are always read live, so one timestamp could not have covered both.
            "realized_computed_at": analytics.get("computed_at", ""),
            "error": "; ".join(errors),
        }
        # Rounded last, so nothing above has to think about presentation. Row nulls are kept:
        # this tool's whole claim about ``day_pl`` is that null means "unknown, not flat", and a
        # missing key could not say that. ``_compact_rows`` is handed the merged dict as a
        # single-row list so the account's own ``day_pl_percent`` is rounded as the ratio it is
        # -- at two places a 0.13% day reads as 0.0, which is the one number that must not round
        # to nothing.
        compacted = _compact_rows([merged], ratios=_RATIO_FIELDS)[0]
        compacted["rows"] = _compact_rows(merged["rows"], ratios=_RATIO_FIELDS)
        return compacted

    @mcp.tool()
    def get_account_orders(account_id: str = "", limit: int = 20) -> dict[str, Any]:
        """**Today's** orders for one account, in every state, most recent first.

        Filled, partially filled, replaced, cancelled, rejected and still-resting arrive in one
        list -- a live order simply appears with a resting status and an unfilled quantity, so
        there is no separate working-orders question to ask.

        A refused order carries ``reason``, the broker's own words for why. Only a refused one:
        the key is absent rather than empty when there is nothing to explain, as are
        ``limit_price`` and ``stop_price`` on a market order and ``filled_avg_price`` on an
        order that has not filled. Absent here means "does not apply", never "unknown".

        ``limit`` is per account and counts orders, not fills. Twenty covers any ordinary
        session; raise it for an account you know traded heavily, since the oldest of the day
        are the ones dropped.

        Today means the current *trading* day in market time, not the last 24 hours and not the
        UTC day: an order entered at 4pm ET is still today's.

        Note what this excludes. A good-till-cancelled stop placed last week is still live
        exposure but was entered before the window, so it will not appear here -- read holdings
        from get_account_positions rather than inferring them from this list, and do not
        conclude from an empty result that the account is flat.

        Reads the broker's own record, so a trade placed by hand in the broker's app shows up
        exactly like one this bot placed. An order the broker refused outright never reached it
        and has no id there, so those come from the bot's own journal and carry the reason it
        gave at submission -- the only place that explanation exists.
        """
        # The same function the dashboard's account page calls, but narrowed: the page caps by
        # count alone and shows orders from any date. The local paper book has no broker, so
        # its journal is the whole record rather than a supplement to one.
        from src.api.payloads.accounts import account_activity_payload, market_day_start

        payload = account_activity_payload(account_id, limit=limit, since=market_day_start())
        return {
            "status": "ok",
            **payload,
            "rows": _compact_rows(payload.get("rows", []), drop=_ORDER_ABSENCES),
        }

    @mcp.tool()
    def place_orders(plan_token: str, edits: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Submit a reviewed plan. The response shape depends on the plan's own shape.

        ``plan_token`` is the token get_algorithm_plan returned; the plan itself never travels
        back, so what executes is exactly what you reviewed. Submits immediately. Single-use,
        and expires in about five minutes -- an expired token is normal, not a failure: plan
        again and resubmit.

        **To decline a plan, do not call this tool.** That is the whole veto, and it needs no
        argument. Say in your report what you declined and why.

        ``edits`` is the narrower case: the plan is broadly right but one symbol is not, because
        of something the algorithm could not see. Each edit is ``{"op": ..., "symbol": ...}``
        and names an action, never an amount -- sizing stays the algorithm's:

        - ``skip`` -- leave the symbol exactly as it is, neither bought nor sold today.
        - ``exit`` -- close the position in that symbol.

        Editing is refused for an order-book plan (Options Flip): a leg removed there cancels a
        resting order rather than declining it, so that shape is submit-whole or decline-whole.
        An edit naming a symbol the plan neither proposes nor holds is refused rather than
        ignored -- that is nearly always a mistyped ticker.

        An algorithm switched off between planning and submitting refuses, as does the kill
        switch.

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
        # Re-resolved rather than trusted from the stash: a deployment can be switched off, or
        # handed to the scheduler, in the gap between proposing a plan and submitting it, and
        # the authorisation that matters is the one in force now.
        deployment, refusal = resolve_deployment_for_origin(ORIGIN_MCP, algorithm=pending.algorithm)
        if deployment is None:
            return {"strategy": plan.strategy, "status": "refused", "reason": refusal}
        # The deployment's account, not the default one. get_config() with no account_id
        # resolves whatever account is default, so an algorithm deployed to a live account
        # could have its orders submitted to a paper one -- or the reverse. The scheduler has
        # always passed the deployment's account through run_once; this path simply never did.
        config = get_config(account_id=deployment["account_id"] or None, strategy_id=plan.strategy)
        # A plan's quantities are sized against one specific book's holdings and equity. If the
        # algorithm has been repointed since it was proposed, those numbers describe an account
        # this order would no longer reach.
        if pending.account_id and pending.account_id != config.account_id:
            return {
                "strategy": plan.strategy,
                "status": "refused",
                "reason": (
                    f"That plan was sized against {pending.account_id}, but {pending.algorithm} "
                    f"now trades {config.account_id}. Call get_algorithm_plan again."
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
