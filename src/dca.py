from __future__ import annotations

from copy import deepcopy
from typing import Any

from .config import load_dca_config, save_dca_config

DCA_PLAN_SECTION = "dca_plan"
DCA_MAX_ITEM_AMOUNT = 50.0

DEFAULT_DCA_PLAN: dict[str, Any] = {
    "enabled": False,
    "frequency": "weekly",
    "schedule_pattern": "0 12 * * 1-5",
    "next_run_date": "",
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

FREQUENCIES = {"daily", "weekly", "biweekly", "monthly"}
BUCKETS = ("buy", "sell")


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(parsed, 0.0)


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
        "enabled": _as_bool(raw_plan.get("enabled")),
        "frequency": raw_plan.get("frequency") if raw_plan.get("frequency") in FREQUENCIES else "weekly",
        "schedule_pattern": str(raw_plan.get("schedule_pattern") or DEFAULT_DCA_PLAN["schedule_pattern"])[:80],
        "next_run_date": str(raw_plan.get("next_run_date") or ""),
        "max_item_amount": min(
            max(_as_float(raw_plan.get("max_item_amount"), default=DCA_MAX_ITEM_AMOUNT), 1.0),
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
                _as_float(item.get("amount"), default=0.0),
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


def _raw_plan_from_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    bot_section = raw_config.get("dca_bot")
    if isinstance(bot_section, dict):
        section = bot_section.get(DCA_PLAN_SECTION)
        if isinstance(section, dict):
            return section
    return DEFAULT_DCA_PLAN


def load_dca_plan(universe_rows: list[dict[str, Any]], path: str | None = None) -> dict[str, Any]:
    raw_config = load_dca_config(path)
    return sanitize_dca_plan(_raw_plan_from_config(raw_config), universe_rows)


def save_dca_plan(plan: dict[str, Any], universe_rows: list[dict[str, Any]], path: str | None = None) -> dict[str, Any]:
    sanitized = sanitize_dca_plan(plan, universe_rows)
    raw_config = load_dca_config(path)
    dca_bot = raw_config.setdefault("dca_bot", {})
    if not isinstance(dca_bot, dict):
        dca_bot = {}
        raw_config["dca_bot"] = dca_bot
    dca_bot[DCA_PLAN_SECTION] = sanitized
    save_dca_config(raw_config, path)
    return sanitized


def allocation_preview(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Create planned DCA order rows from exact per-symbol dollar amounts."""
    rows: list[dict[str, Any]] = []
    if not plan.get("enabled"):
        return rows

    for bucket in BUCKETS:
        bucket_plan = plan.get(bucket, {})
        items = bucket_plan.get("items", [])
        bucket_total = sum(_as_float(item.get("amount")) for item in items)
        if bucket_total <= 0 or not items:
            continue

        for index, item in enumerate(items):
            notional = _as_float(item.get("amount"))
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
                    "frequency": plan.get("frequency", "weekly"),
                    "next_run_date": plan.get("next_run_date", ""),
                }
            )
    return rows
