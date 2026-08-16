"""Turning loosely-typed YAML and environment values into the types ``Config`` declares.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from ...common.config_utils import as_bool, direct_or_env



def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _normalize_keyed_items(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(key): item if isinstance(item, dict) else {} for key, item in value.items()}
    if isinstance(value, list):
        items: dict[str, dict[str, Any]] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or item.get("name") or item.get("provider") or "").strip()
            if not item_id:
                continue
            items[item_id] = {key: val for key, val in item.items() if key not in {"id", "name"}}
        return items
    return {}


def _normalize_data_sources(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    sources = raw.get("data_sources", raw.get("connectors", raw))
    if not isinstance(sources, dict):
        return {}
    normalized: dict[str, Any] = {}
    for category in ("market_data", "intraday_market_data", "eod_market_data", "interday_market_data", "news_sentiment", "sentiment_data", "dividends"):
        section = sources.get(category, {})
        if not isinstance(section, dict):
            normalized[category] = {"provider_order": [], "providers": {}}
            continue
        providers = _normalize_keyed_items(section.get("providers", []))
        normalized[category] = {
            **section,
            "provider_order": list(providers),
            "providers": providers,
        }
    if not normalized["eod_market_data"]["providers"] and normalized["market_data"]["providers"]:
        normalized["eod_market_data"] = dict(normalized["market_data"])
    if not normalized["market_data"]["providers"] and normalized["eod_market_data"]["providers"]:
        normalized["market_data"] = dict(normalized["eod_market_data"])
    if not normalized["sentiment_data"]["providers"] and normalized["news_sentiment"]["providers"]:
        normalized["sentiment_data"] = dict(normalized["news_sentiment"])
    if not normalized["news_sentiment"]["providers"] and normalized["sentiment_data"]["providers"]:
        normalized["news_sentiment"] = dict(normalized["sentiment_data"])
    return normalized


def _provider_credential(
    data_sources: dict[str, Any],
    category: str,
    provider: str,
    key: str,
    env_key: str,
    fallback_env: str = "",
) -> str:
    providers = _section(_section(data_sources, category), "providers")
    return direct_or_env(_section(providers, provider), key, env_key, fallback_env)


def _provider_secret(data_sources: dict[str, Any], category: str, provider: str, fallback_env: str = "") -> str:
    return _provider_credential(data_sources, category, provider, "api_key", "api_key_env", fallback_env)




def _config_value(section: dict[str, Any], key: str, env_name: str, default: Any) -> Any:
    if env_name in os.environ:
        return os.getenv(env_name)
    return section.get(key, default)


def reader(section: dict[str, Any]) -> Callable[..., Any]:
    """A ``read(key, default)`` bound to one config section.

    The cast follows the default's type and the environment variable defaults to the key in
    upper case, which is the convention every one of these fields already followed. Spelling
    all three out per field meant writing the default twice -- once to look up, once to fall
    back on -- and the two could drift apart without anything failing.

    Pass ``env=`` for the handful whose environment name is not simply the key: the options
    knobs are stored as ``swing_dte_min`` but read ``OPTIONS_SWING_DTE_MIN``.
    """

    def read(key: str, default: Any, *, env: str | None = None) -> Any:
        value = _config_value(section, key, env or key.upper(), default)
        if isinstance(default, bool):  # before int: bool is a subclass of it
            return _str_to_bool(str(value), default)
        if isinstance(default, int):
            return _as_int(value, default)
        if isinstance(default, float):
            return _as_float(value, default)
        if isinstance(default, list):
            return _as_list(value, default)
        # An explicitly null YAML key means "not set", so it takes the default rather than
        # stringifying to "None" -- which is what several of these fields used to do.
        return default if value is None else str(value)

    return read


def _str_to_bool(value: str | None, default: bool) -> bool:
    return as_bool(value, default)


def _parse_symbols(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    return symbols or list(default)


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        parsed = [item.strip() for item in value.split(",") if item.strip()]
        return parsed or list(default)
    if isinstance(value, list):
        parsed = [str(item).strip() for item in value if str(item).strip()]
        return parsed or list(default)
    return list(default)


def _algorithm_sections(raw_algorithms_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    if raw_algorithms_config:
        external = _section(raw_algorithms_config, "algorithms") or raw_algorithms_config
        sections.update(
            {
                str(key): value
                for key, value in external.items()
                if isinstance(value, dict) and key not in {"algorithm_bot", "runtime"}
            }
        )
    return sections
