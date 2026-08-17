from __future__ import annotations

from copy import deepcopy
from typing import Any

from ...common.config_utils import as_float

#: Where a plan lives inside its algorithm's ordinary config section.
#:
#: DCA is a normal algorithm that happens to have a custom editor on the Tune screen, and
#: nothing more. Its plan is therefore ordinary algorithm tuning -- ``algorithms.<id>.plan``,
#: read and written through ``/api/algorithm-config`` like every other algorithm's knobs.
#:
#: It used to live in a ``dca_bot`` section of its own, keyed by account (``dca_plans``) with
#: a shared template (``dca_plan``) behind it. That bought a per-account dimension no other
#: algorithm has, at the cost of two real problems: ``dca`` and ``bursty_dca`` were forced to
#: share one budget, because the key had no room for the algorithm; and the editor and the
#: views could disagree about which account's plan they were looking at. Keying by algorithm
#: fixes both, and the price -- two accounts running ``dca`` now share a plan -- is the same
#: price every other algorithm already pays for its tuning.
PLAN_KEY = "plan"

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
    """Normalize a DCA plan and keep only symbols present in the configured universe.

    An absent or empty plan sanitizes to empty buckets. This used to start from
    ``DEFAULT_DCA_PLAN``, which meant clearing every bubble off the board silently restored a
    built-in basket the next time the plan was read -- the board showed nothing and the
    algorithm would have bought SPY, QQQ, GLD and TLT.
    """
    raw_plan: dict[str, Any] = {bucket: {"items": []} for bucket in BUCKETS}
    if plan:
        for bucket in BUCKETS:
            if isinstance(plan.get(bucket), dict):
                raw_plan[bucket].update(plan[bucket])

    universe = _universe_lookup(universe_rows)
    sanitized: dict[str, Any] = {}

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
            amount = min(as_float(item.get("amount"), default=0.0), DCA_MAX_ITEM_AMOUNT)
            items.append({"symbol": symbol, "amount": amount})

        sanitized[bucket] = {
            "amount": sum(item["amount"] for item in items),
            "items": items,
        }

    return sanitized


def raw_plan_from_config(config: Any, algorithm_id: str) -> dict[str, Any]:
    """The plan as written in ``algorithms.<algorithm_id>.plan``, unsanitized.

    Unsanitized because two callers need different things from it: ``sanitize_dca_plan`` for
    what can actually be traded, and ``unknown_plan_symbols`` for what was asked for and
    silently dropped. Sanitizing first would make the second question unanswerable.

    No plan configured means an empty plan -- buy nothing -- rather than ``DEFAULT_DCA_PLAN``.
    The dashboard renders whatever is in the config section, so a built-in fallback here would
    have the algorithm quietly trading a basket the board showed as empty. The default is a
    *seed* for a new config, not a silent standing order.
    """
    section = (getattr(config, "algorithm_configs", {}) or {}).get(algorithm_id) or {}
    plan = section.get(PLAN_KEY)
    return plan if isinstance(plan, dict) else {}


def unknown_plan_symbols(raw_plan: dict[str, Any], universe_rows: list[dict[str, Any]]) -> list[str]:
    """Plan symbols that are not in the tradable universe.

    ``sanitize_dca_plan`` drops them, so without this a typo in a bucket is invisible: the row
    simply never appears and the money is silently never spent.
    """
    universe = _universe_lookup(universe_rows)
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

