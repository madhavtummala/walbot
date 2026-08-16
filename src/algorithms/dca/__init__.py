from __future__ import annotations

from copy import deepcopy
from typing import Any

from ...common.config_utils import as_float
from ...core.config import load_dca_config, save_dca_config

DCA_PLAN_SECTION = "dca_plan"

#: Per-account plans. The monthly budgets *are* DCA's config, and config is per account the
#: same way accrual state is (``dca_accrual:{strategy}:{account}``) -- two DCA bindings on
#: different accounts must not share one budget. ``dca_plan`` above stays readable as the
#: template for any account that has never been edited.
DCA_PLANS_SECTION = "dca_plans"

#: Hard ceiling on a per-symbol budget. Every plan amount means **dollars per month, per
#: symbol** -- the same meaning in DCA and Bursty DCA, so switching between them never
#: silently changes the spend rate.
DCA_MAX_ITEM_AMOUNT = 5_000.0

#: Which variant runs the plan. Both accrue identically; Bursty adds a signal to the predicate.
DCA_ALGORITHMS = ("dca", "bursty_dca")

#: The plan carries **what to buy and how much**, and nothing else. Cadence lives on the
#: algorithm class as a ``Schedule``, and whether it runs at all is ``algorithm_enabled``
#: plus ``active_strategy`` in controls -- the same switch every other algorithm uses. A
#: separate ``enabled`` here used to let DCA run on its own loop alongside another algorithm,
#: which meant two loops could drive the same accrual state at different cadences.
DEFAULT_DCA_PLAN: dict[str, Any] = {
    "max_item_amount": DCA_MAX_ITEM_AMOUNT,
    "buy": {
        "amount": 100.0,
        "items": [
            {"symbol": "SPY", "amount": 25.0},
            {"symbol": "QQQ", "amount": 20.0},
            {"symbol": "GLD", "amount": 15.0},
            {"symbol": "TLT", "amount": 10.0},
        ],
    },
    "sell": {
        "amount": 0.0,
        "items": [],
    },
}

BUCKETS = ("buy", "sell")


def _universe_lookup(universe_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["symbol"]).upper(): row for row in universe_rows if row.get("symbol")}


def sanitize_dca_plan(plan: dict[str, Any] | None, universe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize a DCA plan and keep only symbols present in the configured universe."""
    raw_plan = deepcopy(DEFAULT_DCA_PLAN)
    if plan:
        ignored_bucket_keys = set(BUCKETS)
        raw_plan.update({key: value for key, value in plan.items() if key not in ignored_bucket_keys})
        for bucket in BUCKETS:
            if isinstance(plan.get(bucket), dict):
                raw_plan[bucket].update(plan[bucket])

    universe = _universe_lookup(universe_rows)
    sanitized = {
        "max_item_amount": min(
            max(as_float(raw_plan.get("max_item_amount"), default=DCA_MAX_ITEM_AMOUNT), 1.0),
            DCA_MAX_ITEM_AMOUNT,
        ),
    }

    for bucket in BUCKETS:
        bucket_plan = raw_plan.get(bucket, {})
        raw_items = bucket_plan.get("items", [])
        assigned_symbols: set[str] = set()
        items: list[dict[str, Any]] = []
        for item in raw_items:
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol or symbol not in universe or symbol in assigned_symbols:
                continue
            assigned_symbols.add(symbol)
            amount = min(
                as_float(item.get("amount"), default=0.0),
                sanitized["max_item_amount"],
                DCA_MAX_ITEM_AMOUNT,
            )
            sanitized_item = {
                "symbol": symbol,
                "amount": amount,
            }
            items.append(sanitized_item)

        sanitized[bucket] = {
            "amount": sum(item["amount"] for item in items),
            "items": items,
        }

    return sanitized


def _raw_plan_from_config(raw_config: dict[str, Any], account_id: str = "") -> dict[str, Any]:
    bot_section = raw_config.get("dca_bot")
    if not isinstance(bot_section, dict):
        return DEFAULT_DCA_PLAN
    plans = bot_section.get(DCA_PLANS_SECTION)
    if account_id and isinstance(plans, dict) and isinstance(plans.get(account_id), dict):
        return plans[account_id]
    # No plan of its own yet: fall back to the single pre-account plan, which keeps an existing
    # config working and seeds a new account with something sensible rather than an empty board.
    section = bot_section.get(DCA_PLAN_SECTION)
    if isinstance(section, dict):
        return section
    return DEFAULT_DCA_PLAN


def load_dca_plan(
    universe_rows: list[dict[str, Any]], path: str | None = None, account_id: str = ""
) -> dict[str, Any]:
    raw_config = load_dca_config(path)
    return sanitize_dca_plan(_raw_plan_from_config(raw_config, account_id), universe_rows)


def save_dca_plan(
    plan: dict[str, Any],
    universe_rows: list[dict[str, Any]],
    path: str | None = None,
    account_id: str = "",
) -> dict[str, Any]:
    sanitized = sanitize_dca_plan(plan, universe_rows)
    raw_config = load_dca_config(path)
    dca_bot = raw_config.setdefault("dca_bot", {})
    if not isinstance(dca_bot, dict):
        dca_bot = {}
        raw_config["dca_bot"] = dca_bot
    if account_id:
        plans = dca_bot.setdefault(DCA_PLANS_SECTION, {})
        if not isinstance(plans, dict):
            plans = {}
            dca_bot[DCA_PLANS_SECTION] = plans
        plans[account_id] = sanitized
    else:
        # No account in play (a bare CLI edit): keep writing the shared template.
        dca_bot[DCA_PLAN_SECTION] = sanitized
    save_dca_config(raw_config, path)
    return sanitized


def unknown_plan_symbols(universe_rows: list[dict[str, Any]] | None = None, path: str | None = None) -> list[str]:
    """Plan symbols that are not in the tradable universe.

    ``sanitize_dca_plan`` drops them, so without this a typo in a bucket is invisible: the row
    simply never appears and the money is silently never spent.
    """
    if universe_rows is None:
        from ...api.api_payloads import universe_payload

        universe_rows = universe_payload()["rows"]
    universe = _universe_lookup(universe_rows)
    raw_plan = _raw_plan_from_config(load_dca_config(path))
    unknown: list[str] = []
    for bucket in BUCKETS:
        for item in (raw_plan.get(bucket) or {}).get("items", []) or []:
            symbol = str(item.get("symbol", "")).strip().upper()
            if symbol and symbol not in universe and symbol not in unknown:
                unknown.append(symbol)
    return unknown


def allocation_preview(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Create planned DCA order rows from exact per-symbol dollar amounts.

    Always rendered. The plan describes what would be bought; whether anything is actually
    bought is the algorithm bot's switch, and a preview that vanished when the bot was off
    could not be reviewed before turning it on.
    """
    rows: list[dict[str, Any]] = []

    for bucket in BUCKETS:
        bucket_plan = plan.get(bucket, {})
        items = bucket_plan.get("items", [])
        bucket_total = sum(as_float(item.get("amount")) for item in items)
        if bucket_total <= 0 or not items:
            continue

        for index, item in enumerate(items):
            notional = as_float(item.get("amount"))
            if notional <= 0:
                continue
            weight = notional / bucket_total if bucket_total else 0.0
            rows.append(
                {
                    "bucket": bucket,
                    "action": bucket,
                    "symbol": item["symbol"],
                    "name": "",
                    "theme": "",
                    "rank": index + 1,
                    "weight": weight,
                    "notional": notional,
                }
            )
    return rows

