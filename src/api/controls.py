from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from src.algorithms.registry import canonical_algorithm_id
from src.common.config_utils import as_bool
from src.core.config import (
    DEFAULT_STRATEGY_ID,
    load_algorithm_bot_config,
    load_options_bot_config,
    save_algorithm_bot_config,
    save_options_bot_config,
)

logger = logging.getLogger(__name__)

#: Every binding cadence, and how many minutes it means. ``mcp`` maps to ``None`` because it
#: names no cadence at all -- it is the absence of one, delegated to an agent.
#:
#: One mapping rather than a list of names beside a separate lookup of their lengths: the set
#: was previously written out four times (twice here, twice in ``core.bot_runtime``, each with
#: its own default), so adding a cadence to one of them meant the dashboard accepted a value
#: the scheduler silently rewrote to 1hr and then timed as 60 minutes.
BINDING_FREQUENCIES: dict[str, int | None] = {
    "15m": 15,
    "30m": 30,
    "1hr": 60,
    "2hr": 120,
    "1d": 24 * 60,
    "mcp": None,
}

DEFAULT_FREQUENCY = "1hr"
VALID_FREQUENCIES = set(BINDING_FREQUENCIES)


def normalize_frequency(value: Any) -> str:
    candidate = str(value or DEFAULT_FREQUENCY).strip().lower()
    return candidate if candidate in BINDING_FREQUENCIES else DEFAULT_FREQUENCY


def frequency_minutes(frequency: Any) -> int | None:
    """How often a cadence fires, or ``None`` for ``mcp``, which never fires on a clock."""
    return BINDING_FREQUENCIES[normalize_frequency(frequency)]


#: A binding is one algorithm pointed at one account, with its own on/off switch. The dashboard
#: renders one panel per binding, and the runtime gives each its own scheduler loop, so several
#: strategies can be tried in parallel against different accounts from a single deployment.
DEFAULT_BINDING: dict[str, Any] = {
    "id": "",
    "strategy": DEFAULT_STRATEGY_ID,
    "account_id": "",
    "enabled": False,
    "frequency": DEFAULT_FREQUENCY,
}

DEFAULT_CONTROLS: dict[str, Any] = {
    "trading_account_id": "",
    "bindings": [],
    "options": {
        "enabled": False,
        "strategy": "none",
        "account_id": "",
    },
}


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
    enabled = as_bool(raw.get("enabled"), default=False)
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
        "frequency": normalize_frequency(raw.get("frequency")),
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
    options_enabled = as_bool(raw["options"].get("enabled"), default=False) and options_strategy != "none"
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


#: The two origins that can drive a binding into *placing orders*. Backtest and live-signal
#: reads are deliberately not origins: they compute a proposal and submit nothing, so they run
#: whatever they are asked to, switched on or not.
ORIGIN_SCHEDULE = "schedule"
ORIGIN_MCP = "mcp"


def binding_driver(binding: dict[str, Any] | None) -> str:
    """Which origin is allowed to place this binding's orders.

    Exactly one, always: ``frequency: "mcp"`` does not *add* an agent driver, it hands the
    clock's job to one. A binding is driven by the scheduler or by an agent, never both.
    """
    return ORIGIN_SCHEDULE if frequency_minutes((binding or {}).get("frequency")) is not None else ORIGIN_MCP


def binding_refusal(binding: dict[str, Any] | None, origin: str) -> str:
    """Why ``origin`` may not place orders for ``binding``, or ``""`` if it may.

    The single implementation of the rule, because it previously had one and a half: the
    scheduler enforced it in ``bot_runtime._binding_enabled`` and the MCP tools enforced
    nothing at all, so an agent could trade a binding that was switched off, or one the
    scheduler was driving at the same time. A rule that only one of two callers applies is the
    same failure mode as an algorithm implemented twice -- see docs/refactor-consolidation-plan.md.
    """
    if not binding:
        return "No binding is configured for it"
    if not binding.get("enabled"):
        return f"Binding {binding.get('id')} is switched off"
    driver = binding_driver(binding)
    if driver != origin:
        if origin == ORIGIN_MCP:
            return (
                f"Binding {binding.get('id')} runs on the {binding.get('frequency')} schedule, so the scheduler "
                "places its orders. Set its frequency to 'mcp' to drive it from an agent instead."
            )
        return f"Binding {binding.get('id')} is driven by an agent over MCP, not by the schedule"
    return ""


def bindings_for_strategy(controls: dict[str, Any], strategy: str) -> list[dict[str, Any]]:
    """Every binding running ``strategy``. More than one is legal -- ids are unique, strategies are not."""
    wanted = canonical_algorithm_id(str(strategy or ""))
    return [b for b in (controls.get("bindings") or []) if str(b.get("strategy")) == wanted]


def account_for_strategy(strategy: str, controls: dict[str, Any] | None = None) -> str:
    """The account a read-only view of ``strategy`` should be computed against.

    Signal views and backtests are not origins -- they place nothing, so they run whatever they
    are asked to and cannot refuse the way ``resolve_binding_for_origin`` does. They still have
    to answer *for some account*, because some algorithms are configured per account: a DCA
    plan is per account, so computing the view against the default account rendered one plan
    while the dashboard's own editor wrote another, and neither view ever showed an edit.
    Reading the binding is what makes the two agree.

    An enabled binding wins over a switched-off one, since that is the deployment the view is
    describing. ``""`` means no binding names this strategy, and the caller falls back to the
    default account -- which is the right answer for an algorithm whose config is not per
    account, and the only available one for a strategy that is not deployed anywhere.
    """
    controls = controls if controls is not None else load_controls()
    candidates = bindings_for_strategy(controls, strategy)
    if not candidates:
        return ""
    preferred = next((binding for binding in candidates if binding.get("enabled")), candidates[0])
    return str(preferred.get("account_id") or "")


def resolve_binding_for_origin(
    origin: str,
    *,
    binding_id: str = "",
    strategy: str = "",
    controls: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """The binding ``origin`` may act through, or ``(None, reason)``.

    Addressed by ``binding_id`` when the caller knows it. Falling back to ``strategy`` is for
    callers that only know an algorithm name -- the MCP tools -- and it deliberately refuses
    when the answer is ambiguous rather than picking: two bindings may share a strategy on
    different accounts, and guessing which one to trade is guessing which account to trade.
    """
    controls = controls if controls is not None else load_controls()
    if binding_id:
        binding = find_binding(controls, binding_id)
        if binding is None:
            return None, f"No binding with id {binding_id!r}"
        return (binding, "") if not (reason := binding_refusal(binding, origin)) else (None, reason)

    candidates = bindings_for_strategy(controls, strategy)
    if not candidates:
        return None, f"No binding is configured for {strategy!r}"
    eligible = [b for b in candidates if not binding_refusal(b, origin)]
    if len(eligible) == 1:
        return eligible[0], ""
    if not eligible:
        # One candidate has one honest reason; several have several, so say them all rather
        # than reporting whichever happened to be first.
        return None, "; ".join(f"{b['id']}: {binding_refusal(b, origin)}" for b in candidates)
    return None, (
        f"{strategy!r} has {len(eligible)} bindings an agent may drive "
        f"({', '.join(b['id'] for b in eligible)}). Name one with binding_id."
    )


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
