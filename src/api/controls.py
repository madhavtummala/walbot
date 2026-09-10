from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from src.algorithms.registry import canonical_algorithm_id
from src.common.config_utils import as_bool
from src.core.config import (
    DEFAULT_STRATEGY_ID,
    config_transaction,
    load_algorithm_bot_config,
    save_algorithm_bot_config,
)

logger = logging.getLogger(__name__)

#: Empty cron: no clock drives this binding (the job the legacy ``mcp`` frequency used to do).
AGENT_DRIVEN_CRON = ""

#: What a pre-cron binding's ``frequency`` becomes when migrated.
_LEGACY_AGENT_FREQUENCY = "mcp"


def algorithm_default_cron(strategy: Any) -> str:
    """The cron a binding gets before anyone has chosen one, read off the algorithm class."""
    from src.algorithms.registry import get_algorithm_class

    try:
        return str(get_algorithm_class(canonical_algorithm_id(strategy)).cron)
    except (KeyError, ValueError, TypeError, AttributeError):
        from src.algorithms.base import BaseAlgorithm

        return str(BaseAlgorithm.cron)


def normalize_cron(value: Any, strategy: Any = None) -> str:
    """A cron this system will actually run, or ``""`` for an agent-driven binding.

    An unparseable expression falls back to the algorithm's default rather than to ``""``,
    since that would silently reclassify a scheduled binding as agent-driven and stop it trading.
    """
    from src.core.cron import CronError, parse_cron

    text = " ".join(str(value or "").split())
    if not text:
        return AGENT_DRIVEN_CRON
    if text.lower() == _LEGACY_AGENT_FREQUENCY:
        return AGENT_DRIVEN_CRON
    try:
        return parse_cron(text).expression
    except CronError:
        fallback = algorithm_default_cron(strategy)
        logger.warning("Unusable cron %r; falling back to the algorithm default %r", text, fallback)
        return fallback


def cron_is_scheduled(cron: Any) -> bool:
    """Whether a clock drives this binding at all."""
    return bool(str(cron or "").strip())


#: One algorithm pointed at one account, with its own on/off switch and scheduler loop.
DEFAULT_BINDING: dict[str, Any] = {
    "id": "",
    "strategy": DEFAULT_STRATEGY_ID,
    "account_id": "",
    "enabled": False,
    "cron": AGENT_DRIVEN_CRON,
}

DEFAULT_CONTROLS: dict[str, Any] = {
    "trading_account_id": "",
    "bindings": [],
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
    # A legacy "none" strategy forced the bot idle regardless of "enabled"; migrating it must
    # not flip a previously-idle binding into live trading.
    if saved_strategy.strip().lower() == "none":
        enabled = False
    return {
        "id": _binding_id(existing_ids, str(raw.get("id") or "")),
        "strategy": strategy,
        "account_id": str(raw.get("account_id") or fallback_account or "")[:80],
        "enabled": enabled,
        "cron": _binding_cron(raw, strategy),
    }


def _binding_cron(raw: dict[str, Any], strategy: str) -> str:
    """This binding's schedule, migrating a pre-cron ``frequency`` when that is all there is."""
    # An explicit ``cron`` is honoured whatever it says, empty included. An *absent* key means
    # this binding has never chosen, and must not land agent-driven by default -- a deployment
    # created from the dashboard would then sit switched on and never run.
    if "cron" in raw:
        return normalize_cron(raw.get("cron"), strategy)
    legacy = str(raw.get("frequency") or "").strip().lower()
    if legacy == _LEGACY_AGENT_FREQUENCY:
        return AGENT_DRIVEN_CRON
    return algorithm_default_cron(strategy)


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
            {key: value for key, value in controls.items() if key not in {"bindings"}}
        )
        raw["bindings"] = controls.get("bindings", raw["bindings"])

    bindings = _bindings_from_raw({**raw, "bindings": raw.get("bindings")})
    if not bindings:
        bindings = [sanitize_binding(None, set(), str(raw.get("trading_account_id") or ""))]

    trading_account_id = str(raw.get("trading_account_id") or bindings[0]["account_id"] or "")[:80]

    # Mirrored onto the pre-binding keys for callers still reading a single strategy.
    primary = bindings[0]
    return {
        "trading_account_id": trading_account_id,
        "bindings": bindings,
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
    }


def find_binding(controls: dict[str, Any], binding_id: str) -> dict[str, Any] | None:
    for binding in controls.get("bindings") or []:
        if str(binding.get("id")) == str(binding_id):
            return binding
    return None


#: The two origins that can drive a binding into *placing orders*. Backtest and live-signal
#: reads are not origins: they compute a proposal and submit nothing.
ORIGIN_SCHEDULE = "schedule"
ORIGIN_MCP = "mcp"


def binding_driver(binding: dict[str, Any] | None) -> str:
    """Which origin is allowed to place this binding's orders. Exactly one, always."""
    return ORIGIN_SCHEDULE if cron_is_scheduled((binding or {}).get("cron")) else ORIGIN_MCP


def binding_refusal(binding: dict[str, Any] | None, origin: str) -> str:
    """Why ``origin`` may not place orders for ``binding``, or ``""`` if it may."""
    if not binding:
        return "No binding is configured for it"
    if not binding.get("enabled"):
        return f"Binding {binding.get('id')} is switched off"
    driver = binding_driver(binding)
    if driver != origin:
        if origin == ORIGIN_MCP:
            return (
                f"Binding {binding.get('id')} runs on the schedule '{binding.get('cron')}', so the scheduler "
                "places its orders. Clear its schedule to drive it from an agent instead."
            )
        return f"Binding {binding.get('id')} is driven by an agent over MCP, not by the schedule"
    return ""


def bindings_for_strategy(controls: dict[str, Any], strategy: str) -> list[dict[str, Any]]:
    """Every binding running ``strategy``. More than one is legal -- ids are unique, strategies are not."""
    wanted = canonical_algorithm_id(str(strategy or ""))
    return [b for b in (controls.get("bindings") or []) if str(b.get("strategy")) == wanted]


def account_for_strategy(strategy: str, controls: dict[str, Any] | None = None) -> str:
    """The account a read-only view of ``strategy`` should be computed against.

    An enabled binding wins over a switched-off one. ``""`` means no binding names this
    strategy, and the caller falls back to the default account.
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
    callers that only know an algorithm name and deliberately refuses when ambiguous: two
    bindings may share a strategy on different accounts.
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
        return None, "; ".join(f"{b['id']}: {binding_refusal(b, origin)}" for b in candidates)
    return None, (
        f"{strategy!r} has {len(eligible)} bindings an agent may drive "
        f"({', '.join(b['id'] for b in eligible)}). Name one with binding_id."
    )


def _raw_controls_from_bot_configs(path: str | None = None) -> dict[str, Any]:
    algorithm_config = load_algorithm_bot_config(path)
    raw: dict[str, Any] = {}
    algorithm_bot = algorithm_config.get("algorithm_bot") if isinstance(algorithm_config.get("algorithm_bot"), dict) else {}
    if algorithm_bot:
        raw["trading_account_id"] = algorithm_bot.get("trading_account_id", "")
        if isinstance(algorithm_bot.get("bindings"), list) and algorithm_bot["bindings"]:
            raw["bindings"] = algorithm_bot["bindings"]
        else:
            raw["equities"] = {
                "enabled": algorithm_bot.get("enabled", False),
                "strategy": algorithm_bot.get("strategy", DEFAULT_STRATEGY_ID),
            }
    return raw


def load_controls(path: str | None = None) -> dict[str, Any]:
    bot_controls = _raw_controls_from_bot_configs(path)
    if bot_controls:
        return sanitize_controls(bot_controls)
    return sanitize_controls(DEFAULT_CONTROLS)


def save_controls(controls: dict[str, Any], path: str | None = None) -> dict[str, Any]:
    sanitized = sanitize_controls(controls)
    # The load below and the save at the end are one operation: this rewrites the whole
    # document from what it just read, so a concurrent save landing in between would be
    # overwritten wholesale. See :func:`config_transaction`.
    with config_transaction():
        algorithm_config = load_algorithm_bot_config(path)
        algorithm_bot = algorithm_config.setdefault("algorithm_bot", {})
        if not isinstance(algorithm_bot, dict):
            algorithm_bot = {}
            algorithm_config["algorithm_bot"] = algorithm_bot
        algorithm_bot.update(
            {
                "trading_account_id": sanitized["trading_account_id"],
                "bindings": sanitized["bindings"],
                # Kept in step with the first binding for rollback compatibility.
                "enabled": sanitized["equities"]["enabled"],
                "strategy": sanitized["equities"]["strategy"],
            }
        )
        save_algorithm_bot_config(algorithm_config, path)
    return sanitized
