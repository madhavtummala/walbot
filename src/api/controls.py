from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from src.algorithms.registry import canonical_algorithm_id
from src.common.config_utils import as_bool
from src.core.config import (
    config_transaction,
    load_algorithm_bot_config,
    load_algorithms_config,
    save_algorithm_bot_config,
    save_algorithms_config,
)

logger = logging.getLogger(__name__)

#: Empty cron: no clock drives this deployment, so an agent does.
AGENT_DRIVEN_CRON = ""

#: The keys a deployment owns on an algorithm's config section. Everything else under that
#: section is the algorithm's own tuning and is passed through untouched.
DEPLOYMENT_KEYS = ("account_id", "enabled", "cron")


def algorithm_default_cron(strategy: Any) -> str:
    """The cron a deployment gets before anyone has chosen one, read off the algorithm class."""
    from src.algorithms.registry import get_algorithm_class

    try:
        return str(get_algorithm_class(canonical_algorithm_id(strategy)).cron)
    except (KeyError, ValueError, TypeError, AttributeError):
        from src.algorithms.base import BaseAlgorithm

        return str(BaseAlgorithm.cron)


def normalize_cron(value: Any, strategy: Any = None) -> str:
    """A cron this system will actually run, or ``""`` for an agent-driven deployment.

    An unparseable expression falls back to the algorithm's default rather than to ``""``,
    since that would silently reclassify a scheduled deployment as agent-driven and stop it
    trading.
    """
    from src.core.cron import CronError, parse_cron

    text = " ".join(str(value or "").split())
    if not text:
        return AGENT_DRIVEN_CRON
    try:
        return parse_cron(text).expression
    except CronError:
        fallback = algorithm_default_cron(strategy)
        logger.warning("Unusable cron %r; falling back to the algorithm default %r", text, fallback)
        return fallback


def cron_is_scheduled(cron: Any) -> bool:
    """Whether a clock drives this deployment at all."""
    return bool(str(cron or "").strip())


DEFAULT_CONTROLS: dict[str, Any] = {
    "deployments": [],
}


def sanitize_deployment(algorithm: str, raw: dict[str, Any] | None) -> dict[str, Any] | None:
    """One algorithm's deployment, or ``None`` if it names no account.

    An algorithm with tuning but no ``account_id`` is simply not deployed: it has nowhere to
    trade, so it gets no scheduler loop and refuses orders. That is the whole "undeployed"
    state -- there is no separate flag for it.
    """
    raw = raw if isinstance(raw, dict) else {}
    account_id = str(raw.get("account_id") or "")[:80]
    if not account_id:
        return None
    strategy = canonical_algorithm_id(str(algorithm))[:80]
    return {
        "algorithm": strategy,
        "account_id": account_id,
        "enabled": as_bool(raw.get("enabled"), default=False),
        # An *absent* cron means this deployment has never chosen one, and must not land
        # agent-driven by default -- a deployment created from the dashboard would then sit
        # switched on and never run. An explicit cron is honoured whatever it says, empty
        # included, because empty is how you hand a deployment to an agent.
        "cron": (
            normalize_cron(raw.get("cron"), strategy)
            if "cron" in raw
            else algorithm_default_cron(strategy)
        ),
    }


def sanitize_controls(controls: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a controls document. One deployment per algorithm, at most one account each."""
    raw = deepcopy(DEFAULT_CONTROLS)
    if controls:
        raw.update({key: value for key, value in controls.items() if key != "deployments"})
        raw["deployments"] = controls.get("deployments") or []

    deployments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw["deployments"] if isinstance(raw["deployments"], list) else []:
        entry = entry if isinstance(entry, dict) else {}
        algorithm = str(entry.get("algorithm") or entry.get("strategy") or "")
        if not algorithm:
            continue
        deployment = sanitize_deployment(algorithm, entry)
        if deployment is None or deployment["algorithm"] in seen:
            # One algorithm, one account. A duplicate is a malformed document rather than a
            # second deployment, and dropping it is safer than letting two loops race the
            # same algorithm state.
            if deployment is not None:
                logger.warning("Ignoring a second deployment of %r; an algorithm runs on one account", algorithm)
            continue
        seen.add(deployment["algorithm"])
        deployments.append(deployment)

    deployments.sort(key=lambda row: row["algorithm"])
    return {"deployments": deployments}


def find_deployment(controls: dict[str, Any], algorithm: str) -> dict[str, Any] | None:
    """This algorithm's deployment, or ``None`` if it is not deployed."""
    wanted = canonical_algorithm_id(str(algorithm or ""))
    for deployment in controls.get("deployments") or []:
        if str(deployment.get("algorithm")) == wanted:
            return deployment
    return None


#: The two origins that can drive a deployment into *placing orders*. Backtest and live-signal
#: reads are not origins: they compute a proposal and submit nothing.
ORIGIN_SCHEDULE = "schedule"
ORIGIN_MCP = "mcp"


def deployment_driver(deployment: dict[str, Any] | None) -> str:
    """Which origin is allowed to place this deployment's orders. Exactly one, always."""
    return ORIGIN_SCHEDULE if cron_is_scheduled((deployment or {}).get("cron")) else ORIGIN_MCP


def deployment_refusal(deployment: dict[str, Any] | None, origin: str) -> str:
    """Why ``origin`` may not place orders for ``deployment``, or ``""`` if it may."""
    if not deployment:
        return "It is not deployed to an account"
    algorithm = deployment.get("algorithm")
    if not deployment.get("enabled"):
        return f"{algorithm} is switched off"
    driver = deployment_driver(deployment)
    if driver != origin:
        if origin == ORIGIN_MCP:
            return (
                f"{algorithm} runs on the schedule '{deployment.get('cron')}', so the scheduler "
                "places its orders. Clear its schedule to drive it from an agent instead."
            )
        return f"{algorithm} is driven by an agent over MCP, not by the schedule"
    return ""


def account_for_strategy(strategy: str, controls: dict[str, Any] | None = None) -> str:
    """The account a read-only view of ``strategy`` should be computed against.

    ``""`` means the algorithm is not deployed, and the caller falls back to the default
    account.
    """
    controls = controls if controls is not None else load_controls()
    deployment = find_deployment(controls, strategy)
    return str((deployment or {}).get("account_id") or "")


def primary_algorithm(controls: dict[str, Any] | None = None) -> str:
    """One algorithm to stand in where a caller has no particular one in mind.

    For the handful of views that predate deployments and still want a single strategy -- a
    bare ``run_once()`` from a shell, the status page's config lookup. The first deployment in
    id order, or the default algorithm when nothing is deployed at all.
    """
    from src.core.config import DEFAULT_STRATEGY_ID

    controls = controls if controls is not None else load_controls()
    deployments = controls.get("deployments") or []
    return str(deployments[0]["algorithm"]) if deployments else DEFAULT_STRATEGY_ID


def resolve_deployment_for_origin(
    origin: str,
    *,
    algorithm: str = "",
    controls: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """The deployment ``origin`` may act through, or ``(None, reason)``.

    An algorithm addresses exactly one deployment, so there is nothing to disambiguate: it is
    deployed and drivable by this origin, or it is not.
    """
    controls = controls if controls is not None else load_controls()
    deployment = find_deployment(controls, algorithm)
    if deployment is None:
        return None, f"{algorithm!r} is not deployed to an account"
    reason = deployment_refusal(deployment, origin)
    return (None, reason) if reason else (deployment, "")


def _controls_from_algorithms_config(path: str | None = None) -> dict[str, Any]:
    """Read every algorithm section's deployment keys into a controls document."""
    from src.core.config.coercion import _algorithm_sections

    sections = _algorithm_sections(load_algorithms_config(path))
    algorithm_bot = load_algorithm_bot_config(path).get("algorithm_bot")
    return {
        "deployments": [
            {"algorithm": algorithm, **{key: section[key] for key in DEPLOYMENT_KEYS if key in section}}
            for algorithm, section in sections.items()
        ],
    }


def load_controls(path: str | None = None) -> dict[str, Any]:
    return sanitize_controls(_controls_from_algorithms_config(path))


def save_controls(controls: dict[str, Any], path: str | None = None) -> dict[str, Any]:
    """Write the deployment keys back onto their algorithm sections, tuning untouched."""
    sanitized = sanitize_controls(controls)
    deployed = {row["algorithm"]: row for row in sanitized["deployments"]}
    # The load below and the save at the end are one operation: this rewrites the whole
    # document from what it just read, so a concurrent save landing in between would be
    # overwritten wholesale. See :func:`config_transaction`.
    with config_transaction():
        document = load_algorithms_config(path)
        sections = document.get("algorithms") if isinstance(document.get("algorithms"), dict) else document
        for algorithm, section in sections.items():
            if not isinstance(section, dict) or algorithm in {"algorithm_bot", "runtime"}:
                continue
            deployment = deployed.pop(str(algorithm), None)
            if deployment is None:
                # Undeploying is the removal of these keys, not a falsy value written over
                # them: a section left with ``account_id: ""`` would read back as deployed
                # nowhere, which is the same thing said less clearly.
                for key in DEPLOYMENT_KEYS:
                    section.pop(key, None)
                continue
            section.update({key: deployment[key] for key in DEPLOYMENT_KEYS})
        # A deployment naming an algorithm with no tuning section yet: create one holding just
        # its deployment keys, so the dashboard can deploy an algorithm nobody has tuned.
        for algorithm, deployment in deployed.items():
            sections[algorithm] = {key: deployment[key] for key in DEPLOYMENT_KEYS}
        save_algorithms_config(document, path)

        # Re-read rather than reuse ``document``: by default every section resolves to the one
        # unified walbot.yaml, so this has to pick up the algorithms written just above --
        # and when the sections are split across files it is a separate document anyway.
        bot_document = load_algorithm_bot_config(path)
        save_algorithm_bot_config(bot_document, path)
    return sanitized
