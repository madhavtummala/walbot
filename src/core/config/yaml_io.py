"""Reading and writing the YAML files, including the fallback parser used when PyYAML is absent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import threading

try:
    import yaml
except ImportError:  # pragma: no cover - exercised when PyYAML is not installed.
    yaml = None

from .paths import LEGACY_CONFIG_FILES, config_file_path, accounts_file_path, algorithm_bot_file_path, algorithms_file_path, connectors_file_path, dca_config_file_path, options_bot_file_path, universe_file_path



def _load_yaml_file(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    content = config_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(content) if yaml is not None else _parse_simple_yaml(content)
    loaded = loaded or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must contain a mapping at {config_path}")
    return loaded


def load_accounts_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(accounts_file_path(path))


def load_connectors_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(connectors_file_path(path))


def load_algorithms_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(algorithms_file_path(path))


def load_algorithm_bot_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(algorithm_bot_file_path(path))


def load_options_bot_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(options_bot_file_path(path))


def load_dca_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(dca_config_file_path(path))


def load_universe_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(universe_file_path(path))


#: Serialises writes to the config document. Every saver is a read-modify-write of the whole
#: file now that the sections share one, so two requests saving different sections would
#: otherwise race and the loser's section would silently revert. FastAPI runs sync endpoints
#: in a threadpool, so that is reachable, not theoretical.
_CONFIG_WRITE_LOCK = threading.Lock()


def _save_yaml_config(config: dict[str, Any], config_path: Path) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        content = yaml.safe_dump(config, sort_keys=False)
    else:
        content = _dump_simple_yaml(config)
    with _CONFIG_WRITE_LOCK:
        # Written through a temporary file in the same directory: a crash or a full disk
        # leaves the previous config intact rather than a half-written one that fails to load
        # and takes the account and binding definitions with it.
        temporary = config_path.with_name(f".{config_path.name}.tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, config_path)
    return config_path


def save_dca_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, dca_config_file_path(path))


def save_algorithm_bot_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, algorithm_bot_file_path(path))


def save_accounts_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, accounts_file_path(path))


def save_algorithms_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, algorithms_file_path(path))


def save_options_bot_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, options_bot_file_path(path))


def save_universe_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, universe_file_path(path))


def save_universe_symbols(symbols: list[str], path: str | None = None) -> Path:
    raw_config = load_universe_config(path)
    universe = raw_config.setdefault("tradable_universe", {})
    if not isinstance(universe, dict):
        universe = {}
        raw_config["tradable_universe"] = universe
    universe["symbols"] = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    return save_universe_config(raw_config, path)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "''", '""'}:
        return ""
    if value == "[]":
        return []
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_simple_yaml(content: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[dict[str, Any]] = [{"indent": -1, "container": root, "parent": None, "key": None}]
    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        is_list_item = stripped.startswith("-")
        while stack and (indent < stack[-1]["indent"] or (indent == stack[-1]["indent"] and not is_list_item)):
            stack.pop()
        if stripped.startswith("- "):
            entry = stack[-1]
            container = entry["container"]
            if isinstance(container, dict) and not container and entry["parent"] is not None:
                container = []
                entry["parent"][entry["key"]] = container
                entry["container"] = container
            if isinstance(container, list):
                rest = stripped[2:].strip()
                if ":" in rest:
                    key, value = rest.split(":", 1)
                    item: dict[str, Any] = {}
                    if value.strip():
                        item[key.strip()] = _parse_scalar(value)
                    else:
                        child: dict[str, Any] = {}
                        item[key.strip()] = child
                        stack.append({"indent": indent + 1, "container": child, "parent": item, "key": key.strip()})
                    container.append(item)
                    stack.append({"indent": indent, "container": item, "parent": container, "key": None})
                else:
                    container.append(_parse_scalar(rest))
            continue
        if stripped == "-":
            entry = stack[-1]
            container = entry["container"]
            if isinstance(container, dict) and not container and entry["parent"] is not None:
                container = []
                entry["parent"][entry["key"]] = container
                entry["container"] = container
            if isinstance(container, list):
                item = {}
                container.append(item)
                stack.append({"indent": indent, "container": item, "parent": container, "key": None})
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        parent = stack[-1]["container"]
        if not isinstance(parent, dict):
            continue
        if value.strip():
            parent[key.strip()] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append({"indent": indent, "container": child, "parent": parent, "key": key.strip()})
    return root


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value == "":
        return '""'
    text = str(value)
    if any(char in text for char in [":", "#", "{", "}", "[", "]"]) or text.strip() != text:
        return f'"{text}"'
    return text


def _dump_simple_yaml(value: Any, indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if item == []:
                lines.append(f"{prefix}{key}: []")
                continue
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_dump_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
    elif isinstance(value, list):
        if not value:
            lines.append(f"{prefix}[]")
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_dump_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
    return "\n".join(line for line in lines if line != "") + ("\n" if indent == 0 else "")

def migrate_legacy_config(directory: Path | None = None) -> Path | None:
    """Fold pre-unification config files into the single document, once.

    Returns the written path, or None when there is nothing to do. An existing deployment
    keeps its tuning this way -- its DCA plan and per-account plans are in those files, and
    seeding fresh defaults over them would quietly reset months of accrual.
    """
    target = config_file_path()
    if target.exists():
        return None
    base = directory or target.parent
    merged: dict[str, Any] = {}
    for legacy in LEGACY_CONFIG_FILES:
        source = base / Path(legacy).name
        if source.is_file():
            merged.update(_load_yaml_file(source))
    if not merged:
        return None
    return _save_yaml_config(merged, target)
