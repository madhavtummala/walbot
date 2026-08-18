"""Where each config file lives, and the one-time migration off the legacy layout.
"""

from __future__ import annotations

import os
from pathlib import Path

from .defaults import ACCOUNTS_FILE, ALGORITHMS_FILE, ALGORITHM_BOT_FILE, CONFIG_FILE, CONNECTORS_FILE, DCA_BOT_FILE, UNIVERSE_FILE



def _project_root() -> Path:
    # src/core/config/paths.py -> src/core/config -> src/core -> src -> repo root. One deeper
    # than the single ``config.py`` this was split out of, which silently resolved the root to
    # ``src/`` and left every config file "missing" -- the accounts file included, so an
    # account id fell back to the unnamed sentinel instead of the configured default.
    return Path(__file__).resolve().parents[3]


def _resolve_path(path: str | None, default: str = ALGORITHM_BOT_FILE) -> Path:
    if not path:
        return _project_root() / default
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _project_root() / candidate


def config_file_path(path: str | None = None) -> Path:
    return _resolve_path(path or os.getenv("TRADING_CONFIG_FILE", CONFIG_FILE), CONFIG_FILE)


def _section_path(path: str | None, env_name: str, legacy_default: str) -> Path:
    """Where one section lives: the unified file, unless it is pointed somewhere explicitly.

    Sections share a document because their top-level keys never collided, so every loader
    still receives the shape it always did. An explicit path or the section's own env var
    still wins -- tests bind individual sections to temp files, and the migration reads the
    old ones.
    """
    if path or os.getenv(env_name):
        return _resolve_path(path or os.getenv(env_name), legacy_default)
    return config_file_path()


def accounts_file_path(path: str | None = None) -> Path:
    return _section_path(path, "TRADING_ACCOUNTS_FILE", ACCOUNTS_FILE)


def connectors_file_path(path: str | None = None) -> Path:
    return _section_path(path, "TRADING_CONNECTORS_FILE", CONNECTORS_FILE)


def _sibling_config_path(path: str | None, env_name: str, default: str) -> Path:
    return _section_path(path, env_name, default)


def algorithms_file_path(path: str | None = None) -> Path:
    return _sibling_config_path(path, "TRADING_ALGORITHMS_FILE", ALGORITHMS_FILE)


def algorithm_bot_file_path(path: str | None = None) -> Path:
    return _sibling_config_path(path, "TRADING_ALGORITHM_BOT_FILE", ALGORITHM_BOT_FILE)


def dca_config_file_path(path: str | None = None) -> Path:
    return _sibling_config_path(path, "TRADING_DCA_BOT_FILE", DCA_BOT_FILE)


def universe_file_path(path: str | None = None) -> Path:
    return _sibling_config_path(path, "TRADING_UNIVERSE_FILE", UNIVERSE_FILE)


#: The seven files this configuration used to live in, in merge order.
LEGACY_CONFIG_FILES = (
    ACCOUNTS_FILE,
    CONNECTORS_FILE,
    UNIVERSE_FILE,
    ALGORITHM_BOT_FILE,
    ALGORITHMS_FILE,
    DCA_BOT_FILE,
)
