from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.algorithms.registry import canonical_algorithm_id
from src.core.config import (
    DEFAULT_STRATEGY_ID,
    load_algorithm_bot_config,
    load_options_bot_config,
    save_algorithm_bot_config,
    save_options_bot_config,
)

#: A binding is one algorithm pointed at one account, with its own on/off switch. The dashboard
#: renders one panel per binding, and the runtime gives each its own scheduler loop, so several
#: strategies can be tried in parallel against different accounts from a single deployment.
DEFAULT_BINDING: dict[str, Any] = {
    "id": "",
    "strategy": DEFAULT_STRATEGY_ID,
    "account_id": "",
    "enabled": False,
    "frequency": "1hr",
}

VALID_FREQUENCIES = {"15m", "30m", "1hr", "2hr", "1d", "mcp"}


def _normalize_frequency(value: Any) -> str:
    candidate = str(value or "1hr").strip().lower()
    if candidate in {"15m", "30m", "1hr", "2hr", "1d", "mcp"}:
        return candidate
    return "1hr"

DEFAULT_CONTROLS: dict[str, Any] = {
    "trading_account_id": "",
    "bindings": [],
    "options": {
        "enabled": False,
        "strategy": "none",
        "account_id": "",
    },
}


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _binding_id(existing: set[str], preferred: str = "") -> str:
    """Stable per-binding key, used for runtime loops and dashboard panels alike."""
    candidate = str(preferred or "").strip()[:60]
    if candidate and candidate not in existing:
        return candidate
    index = 1
    while f"b{index}" in existing:
        index += 1
    return f"b{index}"


def sanitize_binding(raw: dict[str, Any] | None, existing_ids: set[str], fallback_account: str = "") -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    saved_strategy = str(raw.get("strategy") or DEFAULT_STRATEGY_ID)
    strategy = canonical_algorithm_id(saved_strategy)[:80]
    enabled = _as_bool(raw.get("enabled"), default=False)
    # "none" used to force the bot idle regardless of the enabled flag, so a config saved that
    # way can hold enabled: true while the dashboard showed off. Migrating it to DCA without
    # clearing the flag would start live trading on upgrade -- land in the off state instead.
    if saved_strategy.strip().lower() == "none":
        enabled = False
    return {
        "id": _binding_id(existing_ids, str(raw.get("id") or "")),
        "strategy": strategy,
        "account_id": str(raw.get("account_id") or fallback_account or "")[:80],
        "enabled": enabled,
        "frequency": _normalize_frequency(raw.get("frequency")),
    }


def _bindings_from_raw(controls: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the binding list, falling back to the single-binding shape that predates it."""
    raw_bindings = controls.get("bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        legacy = controls.get("equities") if isinstance(controls.get("equities"), dict) else {}
        legacy_strategy = controls.get("active_strategy", legacy.get("strategy"))
        legacy_enabled = controls.get("algorithm_enabled", legacy.get("enabled"))
        if legacy_strategy is None and legacy_enabled is None:
            return []
        raw_bindings = [
            {
                "strategy": legacy_strategy if legacy_strategy is not None else DEFAULT_STRATEGY_ID,
                "enabled": legacy_enabled,
                "account_id": controls.get("trading_account_id", ""),
            }
        ]

    fallback_account = str(controls.get("trading_account_id") or "")
    sanitized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw_bindings:
        binding = sanitize_binding(entry, seen, fallback_account)
        seen.add(binding["id"])
        sanitized.append(binding)
    return sanitized


def sanitize_controls(controls: dict[str, Any] | None) -> dict[str, Any]:
    raw = deepcopy(DEFAULT_CONTROLS)
    if controls:
        raw.update(
            {key: value for key, value in controls.items() if key not in {"options", "options_trading", "bindings"}}
        )
        if isinstance(controls.get("options"), dict):
            raw["options"].update(controls["options"])
        if "options_strategy" in controls:
            raw["options"]["strategy"] = controls.get("options_strategy")
        if "options_trading_enabled" in controls:
            raw["options"]["enabled"] = controls.get("options_trading_enabled")
        if "options_trading_account_id" in controls:
            raw["options"]["account_id"] = controls.get("options_trading_account_id")
        raw["bindings"] = controls.get("bindings", raw["bindings"])

    bindings = _bindings_from_raw({**raw, "bindings": raw.get("bindings")})
    if not bindings:
        bindings = [sanitize_binding(None, set(), str(raw.get("trading_account_id") or ""))]

    options_strategy = str(raw["options"].get("strategy") or DEFAULT_CONTROLS["options"]["strategy"])[:80]
    options_enabled = _as_bool(raw["options"].get("enabled"), default=False) and options_strategy != "none"
    trading_account_id = str(raw.get("trading_account_id") or bindings[0]["account_id"] or "")[:80]
    options_account_id = str(raw["options"].get("account_id") or trading_account_id or "")[:80]

    # The first binding is mirrored onto the pre-binding keys so anything still reading a single
    # strategy -- saved backtests, the live runner's default, older callers -- keeps working.
    primary = bindings[0]
    return {
        "trading_account_id": trading_account_id,
        "bindings": bindings,
        "options": {
            "enabled": options_enabled,
            "strategy": options_strategy,
            "account_id": options_account_id,
        },
        "options_trading": {
            "enabled": options_enabled,
            "strategy": options_strategy,
            "account_id": options_account_id,
        },
        "equities": {
            "enabled": primary["enabled"],
            "strategy": primary["strategy"],
        },
        "algorithm": {
            "enabled": primary["enabled"],
            "strategy": primary["strategy"],
        },
        "algorithm_enabled": primary["enabled"],
        "active_strategy": primary["strategy"],
        "options_trading_enabled": options_enabled,
        "options_strategy": options_strategy,
        "options_trading_account_id": options_account_id,
    }


def find_binding(controls: dict[str, Any], binding_id: str) -> dict[str, Any] | None:
    for binding in controls.get("bindings") or []:
        if str(binding.get("id")) == str(binding_id):
            return binding
    return None


def _raw_controls_from_bot_configs(path: str | None = None) -> dict[str, Any]:
    algorithm_config = load_algorithm_bot_config(path)
    options_config = load_options_bot_config(path)
    raw: dict[str, Any] = {}
    algorithm_bot = algorithm_config.get("algorithm_bot") if isinstance(algorithm_config.get("algorithm_bot"), dict) else {}
    options_bot = options_config.get("options_bot") if isinstance(options_config.get("options_bot"), dict) else {}
    if algorithm_bot:
        raw["trading_account_id"] = algorithm_bot.get("trading_account_id", "")
        if isinstance(algorithm_bot.get("bindings"), list) and algorithm_bot["bindings"]:
            raw["bindings"] = algorithm_bot["bindings"]
        else:
            raw["equities"] = {
                "enabled": algorithm_bot.get("enabled", False),
                "strategy": algorithm_bot.get("strategy", DEFAULT_STRATEGY_ID),
            }
    if options_bot:
        raw["options"] = {
            "enabled": options_bot.get("enabled", False),
            "strategy": options_bot.get("strategy", "none"),
            "account_id": options_bot.get("account_id", raw.get("trading_account_id", "")),
        }
    return raw


def load_controls(path: str | None = None) -> dict[str, Any]:
    bot_controls = _raw_controls_from_bot_configs(path)
    if bot_controls:
        return sanitize_controls(bot_controls)
    return sanitize_controls(DEFAULT_CONTROLS)


def save_controls(controls: dict[str, Any], path: str | None = None) -> dict[str, Any]:
    sanitized = sanitize_controls(controls)
    algorithm_config = load_algorithm_bot_config(path)
    algorithm_bot = algorithm_config.setdefault("algorithm_bot", {})
    if not isinstance(algorithm_bot, dict):
        algorithm_bot = {}
        algorithm_config["algorithm_bot"] = algorithm_bot
    algorithm_bot.update(
        {
            "trading_account_id": sanitized["trading_account_id"],
            "bindings": sanitized["bindings"],
            # Kept in step with the first binding so a rollback still reads a sane single bot.
            "enabled": sanitized["equities"]["enabled"],
            "strategy": sanitized["equities"]["strategy"],
        }
    )
    save_algorithm_bot_config(algorithm_config, path)

    options_config = load_options_bot_config(path)
    options_bot = options_config.setdefault("options_bot", {})
    if not isinstance(options_bot, dict):
        options_bot = {}
        options_config["options_bot"] = options_bot
    options_bot.update(
        {
            "enabled": sanitized["options"]["enabled"],
            "strategy": sanitized["options"]["strategy"],
            "account_id": sanitized["options"]["account_id"],
        }
    )
    save_options_bot_config(options_config, path)
    return sanitized
