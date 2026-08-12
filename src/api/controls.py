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

DEFAULT_CONTROLS: dict[str, Any] = {
    "trading_account_id": "",
    "equities": {
        "enabled": False,
        "strategy": DEFAULT_STRATEGY_ID,
    },
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


def sanitize_controls(controls: dict[str, Any] | None) -> dict[str, Any]:
    raw = deepcopy(DEFAULT_CONTROLS)
    if controls:
        raw.update({key: value for key, value in controls.items() if key not in {"algorithm", "options_trading", "equities", "options"}})
        if isinstance(controls.get("equities"), dict):
            raw["equities"].update(controls["equities"])
        if isinstance(controls.get("options"), dict):
            raw["options"].update(controls["options"])
        if "active_strategy" in controls:
            raw["equities"]["strategy"] = controls.get("active_strategy")
        if "algorithm_enabled" in controls:
            raw["equities"]["enabled"] = controls.get("algorithm_enabled")
        if "options_strategy" in controls:
            raw["options"]["strategy"] = controls.get("options_strategy")
        if "options_trading_enabled" in controls:
            raw["options"]["enabled"] = controls.get("options_trading_enabled")
        if "options_trading_account_id" in controls:
            raw["options"]["account_id"] = controls.get("options_trading_account_id")

    # Every equity strategy id now names a real algorithm -- a saved "none" resolves to DCA --
    # so the power toggle alone decides whether the bot trades.
    saved_strategy = str(raw["equities"].get("strategy") or DEFAULT_CONTROLS["equities"]["strategy"])
    algorithm_strategy = canonical_algorithm_id(saved_strategy)[:80]
    algorithm_enabled = _as_bool(raw["equities"].get("enabled"), default=False)
    # "none" used to force the bot idle regardless of the enabled flag, so a config saved that
    # way can hold enabled: true while the dashboard showed off. Migrating it to DCA without
    # clearing the flag would start live trading on upgrade -- land in the off state instead
    # and let the power toggle be pressed deliberately.
    if saved_strategy.strip().lower() == "none":
        algorithm_enabled = False
    options_strategy = str(raw["options"].get("strategy") or DEFAULT_CONTROLS["options"]["strategy"])[:80]
    options_enabled = _as_bool(raw["options"].get("enabled"), default=False) and options_strategy != "none"
    options_account_id = str(raw["options"].get("account_id") or raw.get("trading_account_id") or "")[:80]
    return {
        "trading_account_id": str(raw.get("trading_account_id") or "")[:80],
        "equities": {
            "enabled": algorithm_enabled,
            "strategy": algorithm_strategy,
        },
        "options": {
            "enabled": options_enabled,
            "strategy": options_strategy,
            "account_id": options_account_id,
        },
        "algorithm": {
            "enabled": algorithm_enabled,
            "strategy": algorithm_strategy,
        },
        "options_trading": {
            "enabled": options_enabled,
            "strategy": options_strategy,
            "account_id": options_account_id,
        },
        "algorithm_enabled": algorithm_enabled,
        "active_strategy": algorithm_strategy,
        "options_trading_enabled": options_enabled,
        "options_strategy": options_strategy,
        "options_trading_account_id": options_account_id,
    }


def _raw_controls_from_bot_configs(path: str | None = None) -> dict[str, Any]:
    algorithm_config = load_algorithm_bot_config(path)
    options_config = load_options_bot_config(path)
    raw: dict[str, Any] = {}
    algorithm_bot = algorithm_config.get("algorithm_bot") if isinstance(algorithm_config.get("algorithm_bot"), dict) else {}
    options_bot = options_config.get("options_bot") if isinstance(options_config.get("options_bot"), dict) else {}
    if algorithm_bot:
        raw["trading_account_id"] = algorithm_bot.get("trading_account_id", "")
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
