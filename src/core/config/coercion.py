"""Turning loosely-typed YAML and environment values into the types ``Config`` declares.
"""

from __future__ import annotations

import os
from typing import Any



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
    for category in ("market_data", "intraday_market_data", "eod_market_data", "interday_market_data", "news_sentiment", "sentiment_data"):
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


def _direct_or_env(section: dict[str, Any], key: str, env_key: str, fallback_env: str = "") -> str:
    if section.get(key):
        return _env_ref(section.get(key), "")
    env_name = str(section.get(env_key) or "").strip()
    if env_name:
        return os.getenv(env_name, "")
    return os.getenv(fallback_env, "") if fallback_env else ""


def _provider_credential(
    data_sources: dict[str, Any],
    category: str,
    provider: str,
    key: str,
    env_key: str,
    fallback_env: str = "",
) -> str:
    section = _section(data_sources, category)
    providers = _section(section, "providers")
    provider_config = _section(providers, provider)
    if provider_config.get(key):
        return _env_ref(provider_config.get(key), "")
    env_name = str(provider_config.get(env_key) or "").strip()
    if env_name:
        return os.getenv(env_name, "")
    return os.getenv(fallback_env, "") if fallback_env else ""


def _provider_secret(data_sources: dict[str, Any], category: str, provider: str, fallback_env: str = "") -> str:
    return _provider_credential(data_sources, category, provider, "api_key", "api_key_env", fallback_env)


def _env_ref(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], default)
    return str(value)


def _config_value(section: dict[str, Any], key: str, env_name: str, default: Any) -> Any:
    if env_name in os.environ:
        return os.getenv(env_name)
    return section.get(key, default)


def _str_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


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
