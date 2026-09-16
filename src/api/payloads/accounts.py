"""Account state: balances, holdings, income and broker activity.

Split out of the single ``api_payloads`` module. Public names are unchanged and still
importable from ``api_payloads``.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from ...brokerages.alpaca.client import create_trading_client
from ...common.config_utils import json_number
from ...core.config import (
    UnknownAccountError,
    get_account_broker_type,
    get_config,
    load_accounts_config,
    config_transaction,
    save_accounts_config,
)
from ...common.lazy_field import LazyField
from ...core.interfaces import MARKET_TZ
from ...data.order_journal import load_order_journal
from ..controls import load_controls

logger = logging.getLogger(__name__)


#: How far back fills are read. A trailing year even though the figure reported is year to
#: date, and the gap is deliberate: a sell in January closes a buy from last autumn, and
#: without that buy in hand the match fails and the close is counted as unmatched instead of
#: priced. The window is the matching basis, not the reporting period -- see ``_realized_pl``.
REALIZED_ACTIVITY_DAYS = 364

#: How far back the account page totals income. Just under a year: comparable to a trailing
#: yield, small enough to stay one request, and inside Schwab's transactions window, which
#: rejects a range of exactly 365 days as "more than a year".
DIVIDEND_ACTIVITY_DAYS = 364

#: Fields a deployment target carries. Secrets are deliberately absent: accounts reference the
#: *names* of environment variables, so the dashboard can wire up a target without ever handling
#: an API key. Setting the secret stays a deploy-time action on the host.
ACCOUNT_FIELDS = ("label", "broker", "base_url", "data_feed", "api_key_env", "api_secret_env")

#: Sort floor for a row whose timestamp will not parse. Oldest rather than newest: an undateable
#: row is pushed to the bottom of the view rather than promoted above orders that are genuinely
#: recent, which is what a "now" default would do.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def positions_payload(account_id: str = "") -> dict[str, Any]:
    """Live holdings and P/L for one account.

    Per account rather than per algorithm: the broker reports a single blended position per
    symbol, so two algorithms trading the same account cannot be told apart here.

    Balances and holdings only -- one read of what the account is right now. Income and realized
    P/L have to crawl a year of transactions to answer, so they live in
    :func:`account_analytics_payload` and are computed on request rather than on every page
    load. Nothing here costs more than the position read itself.
    """
    try:
        config = get_config(account_id=account_id) if account_id else get_config()
    except UnknownAccountError as error:
        # Named but not configured. Reported as itself rather than served from another
        # account, which is the only honest answer and the one this used to get wrong.
        return {
            "account_id": account_id, "account_label": account_id, "equity": None,
            "cash": None, "day_pl": None, "day_pl_percent": None, "total_pl": None,
            "rows": [], "error": str(error),
        }
    payload: dict[str, Any] = {
        "account_id": config.account_id,
        "account_label": config.account_label,
        "equity": None,
        "cash": None,
        "day_pl": None,
        "day_pl_percent": None,
        "total_pl": None,
        "rows": [],
        "error": "",
    }
    # One path for every broker, through the registry. There used to be a hand-written Alpaca
    # branch here reading the SDK directly, which is how Day P/L and Open P/L came to be
    # computed two different ways and drift apart.
    broker = get_account_broker_type(config.account_id)
    return {**payload, **_brokerage_positions(config, broker)}


def _dividend_pl(brokerage: Any, config: Any) -> dict[str, Any]:
    """Income received, through the brokerage interface rather than per-broker branching.

    Kept apart from ``total_pl``: a distribution is cash that arrived, not a change in what
    the holdings are worth.

    Takes an already-resolved brokerage rather than resolving its own. Each resolution builds a
    fresh session, and a fresh Schwab session means another OAuth exchange and another account
    lookup -- four helpers resolving independently made one page load authenticate four times.
    """
    try:
        end = datetime.now(timezone.utc).date()
        rows = brokerage.get_dividend_activity(end - timedelta(days=DIVIDEND_ACTIVITY_DAYS), end)
    except Exception as error:  # noqa: BLE001 - income is a detail, not the whole page
        logger.warning("Could not read dividend activity for %s: %s", config.account_id, error)
        return {"dividend_pl": None, "dividend_rows": []}
    # Year to date, the same window realized P/L reports, so the two figures beside each other
    # cover the same span. The fetch stays a trailing year because ``dividend_rows`` is the
    # recent-activity list and wants more than this year's.
    year_start = date(end.year, 1, 1).isoformat()
    return {
        "dividend_pl": float(
            sum(float(row.get("amount") or 0.0) for row in rows
                if str(row.get("date") or "")[:10] >= year_start)
        ),
        "dividend_rows": rows[:40],
    }


def _realized_pl(brokerage: Any, config: Any) -> dict[str, Any]:
    """Profit already banked, matched from the broker's own fill record.

    Kept apart from ``total_pl``, which only ever measures positions still held: an account
    that opened a call, closed it at a profit and went back to cash has a true open P/L of
    zero, and the whole gain lives here. Reporting only the open figure made such an account
    look like it had never made a penny.
    """
    from ...brokerages.realized import realized_from_fills

    blank = {"realized_pl": None, "realized_closes": 0, "realized_unmatched": 0}
    try:
        end = datetime.now(timezone.utc).date()
        fills = brokerage.get_fills(end - timedelta(days=REALIZED_ACTIVITY_DAYS), end)
    except Exception as error:  # noqa: BLE001 - one figure is not the whole page
        logger.warning("Could not read fills for %s: %s", config.account_id, error)
        return blank
    # None means the venue cannot report fills at all, which is not the same as an account
    # that has not closed anything -- the first is unknown, the second is zero.
    if fills is None:
        return blank
    # Year to date and nothing else: it is the figure a broker's own statement shows, and so
    # the only one a reader can check us against. The trailing year is still fetched, because
    # matching this year's sells needs last year's buys, but it is never reported.
    ytd = realized_from_fills(fills, since=date(end.year, 1, 1).isoformat())
    return {
        "realized_pl": ytd["realized_pl"],
        "realized_closes": ytd["closes"],
        "realized_unmatched": ytd["unmatched"],
    }


def _compute_analytics(account_id: str) -> dict[str, Any]:
    """The slow half of an account page: income and realized P/L, a year of transactions each."""
    config = get_config(account_id=account_id) if account_id else get_config()
    from ...core.pipeline import resolve_brokerage

    # Resolved once and handed to both. Each resolution builds a fresh session, and for Schwab
    # a fresh session is another OAuth exchange and another account lookup -- helpers resolving
    # independently made one page load authenticate four times.
    brokerage = resolve_brokerage(config)
    return {**_dividend_pl(brokerage, config), **_realized_pl(brokerage, config)}


#: In memory rather than the state store: it caches something the broker can always be asked
#: again, so losing it on restart costs one recompute and never correctness.
ANALYTICS = LazyField("account analytics", _compute_analytics)


def account_analytics_payload(account_id: str = "", *, refresh: bool = False) -> dict[str, Any]:
    """Income and realized P/L for one account -- the figures that cost a year of transactions.

    Lazy, not manual. A plain read is instant and answers with whatever was last computed, while
    starting a recompute in the background if that is missing or stale -- so the value fills
    itself in without anyone waiting for it, and nothing has to tell the user to go press a
    button. ``refresh`` is the button: compute now, synchronously, and answer with the result.
    """
    try:
        config = get_config(account_id=account_id) if account_id else get_config()
    except UnknownAccountError as error:
        return {**_blank_analytics(account_id), "state": "error", "error": str(error)}

    snapshot = ANALYTICS.get(config.account_id, force=refresh)
    return {
        **_blank_analytics(config.account_id),
        **(snapshot["value"] or {}),
        "computed_at": snapshot["computed_at"],
        "state": snapshot["state"],
        "error": snapshot["error"],
    }


def _blank_analytics(account_id: str) -> dict[str, Any]:
    """Nulls, not zeros. "Not computed yet" and "this account earned nothing" are different."""
    return {
        "account_id": account_id,
        "computed_at": "",
        "state": "computing",
        "dividend_pl": None,
        "dividend_rows": [],
        # Banked, as opposed to ``total_pl``, which only measures what is still held.
        "realized_pl": None,
        "realized_closes": 0,
        "realized_unmatched": 0,
        "error": "",
    }


def account_headline(account_id: str) -> dict[str, Any]:
    """One account in one row: what it is worth, how it moved, and how busy it was today.

    The summary an agent reads *before* deciding which accounts are worth a closer look. An
    account that is flat, holds what it held yesterday and placed nothing needs no further
    call, and saying so in a dozen numbers is far cheaper than the two detail reads it saves.

    Best effort, and never raises. Each of the three sources is caught on its own, so a broker
    that is slow on positions still contributes its order count, and an account that is
    unreachable entirely reports nulls and an ``error`` rather than zeros -- "we could not ask"
    and "nothing happened" must not look alike in a summary, because only one of them is
    news.

    ``realized_pl`` is forced current, like every other figure here. It costs a year of
    transactions to match, which is the slowest thing in this row -- but it is seconds, not
    bytes, and an agent gets one shot at an answer. Reading the last computed value instead
    returned ``null`` on the first call after a restart, which is the call a scheduled brief
    actually makes: a figure that is absent exactly when it is wanted is not worth the time it
    saves.
    """
    row: dict[str, Any] = {
        "equity": None, "cash": None, "day_pl": None, "day_pl_percent": None,
        "total_pl": None, "realized_pl": None, "positions": None, "orders_today": None,
        "error": "",
    }
    errors: list[str] = []

    try:
        positions = positions_payload(account_id)
        row.update({
            key: positions.get(key)
            for key in ("equity", "cash", "day_pl", "day_pl_percent", "total_pl")
        })
        row["positions"] = len(positions.get("rows") or [])
        if positions.get("error"):
            errors.append(str(positions["error"]))
    except Exception as error:  # noqa: BLE001 - one account must not blank the listing
        logger.warning("Headline positions failed for %s: %s", account_id, error)
        errors.append(str(error))

    try:
        # Forced, like get_account_positions and for the same reason: an agent cannot come back
        # a second later to see whether a background recompute landed. Measured at about five
        # seconds across five accounts, which is time rather than payload.
        analytics = account_analytics_payload(account_id, refresh=True)
        row["realized_pl"] = analytics.get("realized_pl")
    except Exception as error:  # noqa: BLE001
        logger.warning("Headline analytics failed for %s: %s", account_id, error)

    try:
        # Counted rather than listed, and with a cap high enough that the count is a count and
        # not the cap: the detail tool's twenty would quietly report "20" for a busy session.
        activity = account_activity_payload(account_id, limit=200, since=market_day_start())
        row["orders_today"] = len(activity.get("rows") or [])
        if activity.get("error"):
            errors.append(str(activity["error"]))
    except Exception as error:  # noqa: BLE001
        logger.warning("Headline activity failed for %s: %s", account_id, error)
        errors.append(str(error))

    row["error"] = "; ".join(dict.fromkeys(errors))
    return row


def _account_items(raw: dict[str, Any]) -> dict[str, Any]:
    accounts = raw.get("accounts") if isinstance(raw.get("accounts"), dict) else {}
    items = accounts.get("items") if isinstance(accounts.get("items"), dict) else {}
    return items


def accounts_payload() -> dict[str, Any]:
    """Every deployment target, with whether its credentials are actually present."""
    raw = load_accounts_config()
    items = _account_items(raw)
    controls = load_controls()
    # Several algorithms may be deployed to one account, so this is a list per account. The
    # reverse never happens: an algorithm names at most one account.
    deployed: dict[str, list[str]] = {}
    for deployment in controls.get("deployments") or []:
        deployed.setdefault(str(deployment.get("account_id") or ""), []).append(
            str(deployment.get("algorithm") or "")
        )

    rows = []
    for account_id, section in items.items():
        section = section if isinstance(section, dict) else {}
        key_env = str(section.get("api_key_env") or "")
        secret_env = str(section.get("api_secret_env") or "")
        missing = [name for name in (key_env, secret_env) if name and not os.getenv(name)]
        rows.append(
            {
                "id": str(account_id),
                "label": str(section.get("label") or account_id),
                "broker": str(section.get("broker") or "alpaca"),
                "base_url": str(section.get("base_url") or ""),
                "data_feed": str(section.get("data_feed") or ""),
                "api_key_env": key_env,
                "api_secret_env": secret_env,
                "credentials_ready": not missing,
                "missing_env": missing,
                "deployments": sorted(deployed.get(str(account_id), [])),
            }
        )
    rows.sort(key=lambda row: row["id"])
    return {"default": str(raw.get("default") or ""), "rows": rows}


def save_account_payload(body: dict[str, Any]) -> dict[str, Any]:
    account_id = str(body.get("id") or "").strip()[:80]
    if not account_id:
        raise ValueError("A deployment target needs an id.")
    if not account_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Target id may only contain letters, numbers, dashes, and underscores.")

    with config_transaction():
        raw = load_accounts_config()
        accounts = raw.setdefault("accounts", {})
        if not isinstance(accounts, dict):
            accounts = {}
            raw["accounts"] = accounts
        items = accounts.setdefault("items", {})
        if not isinstance(items, dict):
            items = {}
            accounts["items"] = items

        section = items.get(account_id) if isinstance(items.get(account_id), dict) else {}
        for field_name in ACCOUNT_FIELDS:
            if field_name in body:
                section[field_name] = str(body.get(field_name) or "")
        section.setdefault("broker", "alpaca")
        items[account_id] = section
        if not raw.get("default"):
            raw["default"] = account_id
        save_accounts_config(raw)
    return accounts_payload()


def delete_account_payload(account_id: str) -> dict[str, Any]:
    account_id = str(account_id or "").strip()
    with config_transaction():
        raw = load_accounts_config()
        items = _account_items(raw)
        if account_id not in items:
            raise ValueError(f"No deployment target named {account_id}.")
        if len(items) <= 1:
            raise ValueError("Keep at least one deployment target.")

        # Inside the transaction: the deployments are read to decide whether this delete is
        # allowed, so a deployment added between the check and the write would be left
        # pointing at an account that no longer resolves.
        controls = load_controls()
        in_use = [
            d for d in (controls.get("deployments") or []) if str(d.get("account_id")) == account_id
        ]
        if in_use:
            # Deleting a target out from under a running deployment would leave it pointed at an
            # account that no longer resolves, which fails at order time rather than here.
            names = ", ".join(str(d.get("algorithm")) for d in in_use)
            raise ValueError(f"{account_id} still runs {names}. Undeploy first.")

        items.pop(account_id)
        if str(raw.get("default") or "") == account_id:
            raw["default"] = next(iter(items), "")
        save_accounts_config(raw)
    return accounts_payload()


def _account_rows(brokerage: Any, *, label: str, account_id: str) -> dict[str, Any]:
    """Holdings, cash and both P/L figures, from any brokerage through the shared interface.

    One implementation for every broker. There were two near-identical copies -- one for the
    local paper book and one for everything else -- and adding the day's move to the second
    silently left the first showing a blank, which is precisely the failure a second copy
    exists to cause.
    """
    try:
        state = brokerage.get_account_state()
        rows = brokerage.get_position_details()
    except Exception as error:  # noqa: BLE001 - an unreachable broker must not blank the page
        logger.warning("Could not read %s positions for %s: %s", label, account_id, error)
        return {"error": str(error)}

    equity = float(state.get("equity") or 0.0)
    # Only when the broker reports where the session started. Absent stays absent: a missing
    # opening value is "unknown", and defaulting it to zero would render the whole balance as
    # today's gain.
    opening = state.get("last_equity")
    day_pl = (equity - float(opening)) if opening else None

    return {
        "equity": equity,
        "cash": float(state.get("cash") or 0.0),
        "day_pl": day_pl,
        "day_pl_percent": (day_pl / float(opening)) if day_pl is not None and float(opening) else None,
        # Open P/L: the broker's own since-opened figure per position, not the day's move.
        # Zero when nothing is held: no open position is a real answer, not an unknown one.
        # A failed read never reaches here -- it returns an error above.
        "total_pl": sum(float(row["unrealized_pl"]) for row in rows),
        "rows": rows,
    }


def _brokerage_positions(config: Any, broker: str) -> dict[str, Any]:
    """Holdings and cash for any non-Alpaca brokerage, through the shared interface."""
    from ...core.pipeline import resolve_brokerage

    try:
        brokerage = resolve_brokerage(config)
    except Exception as error:  # noqa: BLE001 - an unknown broker must not blank the page
        logger.warning("Could not resolve %s for %s: %s", broker, config.account_id, error)
        return {"error": str(error)}
    return _account_rows(brokerage, label=broker, account_id=config.account_id)


def _paper_activity(config: Any, limit: int, since: datetime | None = None) -> dict[str, Any]:
    """Order history for the local paper book, from the bot's own journal.

    The paper brokerage fills immediately and keeps no order log, so the journal written at
    submission is the whole record -- which is complete here, since nothing but this bot can
    trade a local book.
    """
    rows = []
    for entry in load_order_journal(account_id=config.account_id, limit=limit):
        rows.append(
            {
                "symbol": entry.get("symbol", ""),
                "side": entry.get("side", ""),
                "status": entry.get("status", ""),
                "qty": entry.get("quantity"),
                "filled_qty": entry.get("quantity") if entry.get("status") == "submitted" else 0.0,
                "filled_avg_price": entry.get("price") or None,
                "order_type": entry.get("order_type", ""),
                "limit_price": entry.get("limit_price") or None,
                "stop_price": entry.get("stop_price") or None,
                "submitted_at": entry.get("submitted_at", ""),
                # The journal has recorded this since it was written and nothing read it, so a
                # refused paper order arrived here saying only "rejected". It is the one field
                # a reader of a rejection actually wants.
                "reason": entry.get("reason", ""),
            }
        )
    return {"rows": _since(rows, since)}


def market_day_start() -> datetime:
    """Midnight of the current *trading* day, in the market's own timezone.

    Not UTC midnight: a 4pm ET fill lands on the following UTC day for half the year, so a
    UTC-midnight cutoff would file this afternoon's trades under tomorrow and answer "nothing
    traded today" to someone looking at a position opened an hour ago.
    """
    return datetime.now(MARKET_TZ).replace(hour=0, minute=0, second=0, microsecond=0)


def _parse_stamp(value: Any) -> datetime | None:
    """One order timestamp as an aware datetime, or ``None`` when it cannot be read.

    Every broker spells this differently and none of them are quite ``fromisoformat``: Schwab
    sends ``+0000`` without the colon, Alpaca a ``Z``, and the bot's own journal a naive local
    stamp. A stamp that parses to nothing is treated as undateable by the caller rather than
    as old, so a filter can never silently delete an order over a formatting quirk.
    """
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    # ``+0000`` -> ``+00:00``. ISO 8601 allows the colonless form; fromisoformat did not until
    # 3.11, and being explicit here keeps the parse independent of the interpreter version.
    if len(text) >= 5 and text[-5] in "+-" and text[-4:].isdigit():
        text = f"{text[:-2]}:{text[-2:]}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    # A naive stamp is the journal's, written in local time by the host that placed the order.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=datetime.now().astimezone().tzinfo)


def _since(rows: list[dict[str, Any]], since: datetime | None) -> list[dict[str, Any]]:
    """Drop orders entered before ``since``. No cutoff keeps everything.

    An unparseable timestamp is kept. The alternative -- dropping it -- would hide a real order
    because of a format this code did not anticipate, which is the worse of the two failures.
    """
    if since is None:
        return rows
    return [row for row in rows if (_parse_stamp(row.get("submitted_at")) or since) >= since]


def _brokerage_activity(
    config: Any, broker: str, limit: int, since: datetime | None = None
) -> dict[str, Any]:
    """Recent orders from any non-Alpaca brokerage, in the shape the account page expects.

    Every state, not just the resting ones: ``get_orders("")`` applies no status filter, so a
    filled, cancelled or rejected order arrives alongside a working one. That is the whole
    point of this view -- a filled order is the thing you came to look for.

    Bracket legs arrive flattened by the brokerage, so a stop that is still a child of its
    entry is listed as its own row rather than hidden inside one.
    """
    from ...core.pipeline import resolve_brokerage

    try:
        brokerage = resolve_brokerage(config)
        orders = brokerage.get_orders("")
    except Exception as error:  # noqa: BLE001 - a broker outage must not blank the page
        logger.warning("Could not load %s activity for %s: %s", broker, config.account_id, error)
        return {"error": str(error)}

    rows = []
    for order in orders or []:
        filled_price = json_number(order.get("filled_avg_price"))
        rows.append(
            {
                "symbol": str(order.get("symbol") or ""),
                "side": str(order.get("action") or ""),
                "status": str(order.get("status") or ""),
                "qty": json_number(order.get("quantity")),
                "filled_qty": json_number(order.get("filled_quantity")),
                # Zero means "no fill yet" in the brokerage's vocabulary, and the page reads a
                # null as "--". Passing the zero through would print a $0.00 fill.
                "filled_avg_price": filled_price or None,
                "order_type": str(order.get("order_type") or ""),
                "limit_price": json_number(order.get("limit_price")) or None,
                "stop_price": json_number(order.get("stop_price")) or None,
                "submitted_at": str(order.get("entered_time") or ""),
                # Empty unless the broker refused it and said why. A venue that reports no
                # reason leaves this empty rather than filling in a stand-in phrase, so a
                # missing explanation is never mistaken for the broker's own words.
                "reason": str(order.get("reason") or ""),
            }
        )
    rows.sort(key=lambda row: row["submitted_at"], reverse=True)
    # Trimmed by date first, so ``limit`` caps what survives the window rather than deciding
    # which orders the window gets to consider.
    return {"rows": _since(rows, since)[:limit]}


def _refused_before_the_broker(config: Any, limit: int, since: datetime | None) -> list[dict[str, Any]]:
    """Orders this bot tried to place and the broker refused outright, from the bot's journal.

    Invisible to every other path here, and the most important rows in the view. When
    ``submit_order`` raises -- "account not eligible to trade uncovered option contracts", a
    halt, no buying power -- the broker never creates an order, so it has no id and appears in
    no order feed. Reading the broker alone therefore reports a clean session for an account
    whose every order was thrown out, which is the one failure a reader must never be handed
    quietly.

    Only entries with no ``order_id`` are taken, which is exactly the set that never reached
    the broker: anything the broker did create is in its own feed already, with its own live
    status, and is the better record of the two. So these can be concatenated without a
    dedup pass -- the two sources are disjoint by construction.
    """
    rows = []
    for entry in load_order_journal(account_id=config.account_id, limit=limit):
        if str(entry.get("status") or "") != "rejected" or entry.get("order_id"):
            continue
        rows.append(
            {
                "symbol": entry.get("symbol", ""),
                "side": entry.get("side", ""),
                "status": "rejected",
                "qty": entry.get("quantity"),
                "filled_qty": 0.0,
                "filled_avg_price": None,
                "order_type": entry.get("order_type", ""),
                "limit_price": entry.get("limit_price") or None,
                "stop_price": entry.get("stop_price") or None,
                "submitted_at": entry.get("submitted_at", ""),
                # The whole point of the row. The broker's own words, caught at submission,
                # which is the only moment they exist -- nothing persists them but this.
                "reason": entry.get("reason", ""),
            }
        )
    return _since(rows, since)


def account_activity_payload(
    account_id: str = "", limit: int = 40, since: datetime | None = None
) -> dict[str, Any]:
    """Recent broker orders for one account.

    ``since`` bounds how far back the view reaches, and is the caller's choice rather than this
    function's: the dashboard asks for a week, the MCP tool for the trading day. Passing
    ``None`` returns everything the broker offered, bounded only by ``limit``.

    Read straight from the brokerage rather than a local mirror: the broker is the only source
    that knows about fills, partial fills, and cancels after submission -- and the only one that
    knows about a trade placed by hand in the broker's own app, which no journal here can see.
    The local paper book is the exception: there is no broker, so the bot's journal is the record.
    """
    try:
        config = get_config(account_id=account_id) if account_id else get_config()
    except UnknownAccountError as error:
        return {"account_id": account_id, "rows": [], "error": str(error)}
    payload: dict[str, Any] = {"account_id": config.account_id, "rows": [], "error": ""}
    broker = get_account_broker_type(config.account_id)
    if broker == "paper":
        return {**payload, **_paper_activity(config, limit, since)}
    if broker != "alpaca":
        # The broker's own feed, through the shared interface. The bot's journal was standing in
        # here, which made this view "orders this bot placed" rather than the account's activity:
        # anything traded by hand in the broker's own app was invisible, and an agent reading the
        # MCP tool would conclude the account had never bought what it plainly holds.
        activity = _brokerage_activity(config, broker, limit, since)
        return {**payload, **activity,
                "rows": _newest_first(
                    list(activity.get("rows") or []) + _refused_before_the_broker(config, limit, since)
                )[:limit]}
    try:
        client = create_trading_client(config)
        try:
            orders = client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit))
        except Exception:  # noqa: BLE001 - older SDKs reject the filter; fall back to open orders
            orders = client.get_orders()
        rows = []
        for order in orders:
            submitted = getattr(order, "submitted_at", None) or getattr(order, "created_at", None)
            rows.append(
                {
                    "symbol": str(getattr(order, "symbol", "")),
                    "side": _enum_value(getattr(order, "side", "")),
                    "status": _enum_value(getattr(order, "status", "")),
                    "qty": json_number(getattr(order, "qty", None)),
                    "filled_qty": json_number(getattr(order, "filled_qty", None)),
                    "filled_avg_price": json_number(getattr(order, "filled_avg_price", None)),
                    # What the order is *asking*, which is the only price a resting order has:
                    # it has no fill yet, so ``filled_avg_price`` is null and the row could say
                    # nothing about price at all. A market order genuinely has none -- it takes
                    # whatever the book offers -- and says so rather than inventing one.
                    "order_type": _enum_value(getattr(order, "order_type", "")) or _enum_value(getattr(order, "type", "")),
                    "limit_price": json_number(getattr(order, "limit_price", None)),
                    "stop_price": json_number(getattr(order, "stop_price", None)),
                    "submitted_at": submitted.isoformat() if hasattr(submitted, "isoformat") else str(submitted or ""),
                    # Alpaca's order model carries no rejection reason -- there is no such field
                    # on it, so a rejected order arrives saying only "rejected". Read with
                    # ``getattr`` anyway in case a later SDK grows one, and left empty rather
                    # than filled with a guess: the bot's own journal recorded the reason the
                    # broker gave at submission, and that is where it would have to come from.
                    "reason": str(getattr(order, "reject_reason", "") or ""),
                }
            )
        payload["rows"] = _newest_first(
            _since(rows, since) + _refused_before_the_broker(config, limit, since)
        )[:limit]
    except Exception as error:  # noqa: BLE001 - a broker outage must not blank the page
        logger.warning("Could not load activity for %s: %s", config.account_id, error)
        payload["error"] = str(error)
    return payload


def _newest_first(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Most recent first, across rows that came from different sources.

    Sorted on the parsed stamp rather than the string: the broker spells its timestamps one way
    and the journal another, and comparing those as text interleaves them by punctuation.
    """
    return sorted(rows, key=lambda row: (_parse_stamp(row.get("submitted_at")) or _EPOCH), reverse=True)


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")
