"""Type coercion, parsing, and configuration helper utilities.

Consolidates scalar conversions, symbol list parsing, and legacy bar-to-minute
knob translations so algorithms, brokerages, and API payloads share a single set
of robust, single-purpose functions.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
from typing import Any, Mapping, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

LEGACY_BAR_MINUTES = 15
_warned_legacy_keys: set[str] = set()


def as_float(value: Any, default: float = 0.0) -> float:
    """Coerce value to a float, returning default if None, invalid, or not finite.

    Infinite as well as NaN, which it used to let through: ``math.isnan`` alone passes ``inf``
    on to whatever asked for a number, where it survives arithmetic silently and only surfaces
    as an unserialisable weight or a NaN much later. The private ``_finite`` copies this
    replaces all used ``isfinite``, so the shared helper was the lenient one.
    """
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def as_int(value: Any, default: int = 0) -> int:
    """Coerce value to an integer, returning default if None or invalid."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


#: The one truth table. There were five, and they disagreed: ``y`` was true only in the config
#: reader, an unrecognised string was ``True`` here (via ``bool(value)``) and ``False`` in the
#: dashboard's controls, and the Telegram provider tested the *false* set and defaulted to on --
#: so a typo'd value enabled notifications there and disabled the same setting everywhere else.
TRUE_WORDS = frozenset({"true", "1", "yes", "y", "on"})
FALSE_WORDS = frozenset({"false", "0", "no", "n", "off"})


def as_bool(value: Any, default: bool = False) -> bool:
    """Coerce value to boolean, falling back to ``default`` for anything unrecognised.

    Unrecognised means unrecognised: ``as_bool("maybe")`` used to reach ``bool(value)`` and
    return ``True``, which turns a typo in a config file into an enabled feature.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in TRUE_WORDS:
            return True
        if normalized in FALSE_WORDS:
            return False
        return default
    return bool(value)


def parse_symbols(value: Any, default: Sequence[str] = ()) -> list[str]:
    """Parse a comma-separated string or a list into upper-cased, stripped symbols.

    An explicitly empty *list* means an empty universe and is honoured. Only an absent value
    or a blank string means "unset", and takes the default. The two used to be conflated, so
    writing ``defensive_universe: []`` to turn a sleeve off silently restored the five-name
    default instead -- a setting that looks obeyed and is not.
    """
    if value is None:
        return list(default)
    if isinstance(value, str):
        parsed = [item.strip().upper() for item in value.split(",") if item.strip()]
        return parsed or list(default)
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip().upper() for item in value if str(item).strip()]
    return list(default)


def json_number(value: Any) -> float | None:
    """Coerce value to a JSON-safe float, mapping non-finite values to None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def env_ref(value: Any, default: str = "") -> str:
    """Resolve a ``${ENV_NAME}`` indirection, or pass a literal through.

    Lets a config file name an environment variable in place of a secret, so the yaml stays
    committable.
    """
    if value is None:
        return default
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], default)
    return str(value)


def direct_or_env(section: dict[str, Any], key: str, env_key: str, fallback_env: str = "") -> str:
    """A credential, from the config directly, from the env var it names, or from a default var.

    One resolution order for every secret in the project. It was previously written out three
    times -- ``config.coercion`` twice, once for accounts and once for providers, and the
    Telegram provider byte-for-byte again -- so where a secret could be spelled depended on
    which of the three read it.
    """
    if section.get(key):
        return env_ref(section.get(key), "")
    env_name = str(section.get(env_key) or "").strip()
    if env_name:
        return os.getenv(env_name, "")
    return os.getenv(fallback_env, "") if fallback_env else ""


def as_text(value: Any, default: str = "") -> str:
    """Coerce value to a stripped string, falling back when it is empty or absent."""
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def as_symbol(value: Any, default: str = "") -> str:
    """Coerce value to an upper-cased ticker."""
    return as_text(value, default).upper()


def tuning_section(config: Any, *algorithm_ids: str) -> dict[str, Any]:
    """The saved tuning for an algorithm, from the first id that has any.

    Several ids because an algorithm can be renamed while the config on disk keeps the old key.
    Reading them in order is what lets a rename not silently discard someone's tuning.
    """
    sections = getattr(config, "algorithm_configs", None)
    if not isinstance(sections, dict):
        return {}
    for algorithm_id in algorithm_ids:
        section = sections.get(algorithm_id)
        if isinstance(section, dict) and section:
            return section
    return {}


def account_sizing_fallbacks(config: Any) -> dict[str, Any]:
    """Tuning defaults that come from the account rather than from the algorithm.

    An algorithm that does not name its own per-trade minimum or drift threshold inherits the
    account's, so switching accounts changes it -- which is why these two cannot simply be
    dataclass defaults.
    """
    return {
        "per_trade_value_min": getattr(config, "min_trade_dollars", None),
        "rebalance_threshold": getattr(config, "rebalance_threshold", None),
    }


def as_symbol_map(value: Any, default: Mapping[str, str] | None = None) -> dict[str, str]:
    """Read a symbol -> label mapping, upper-casing the keys.

    Without this a ``dict`` field falls through to :func:`as_text` and arrives as the string
    repr of itself, which reads as "the setting does nothing" rather than as a mistake -- the
    exact failure the type-driven dispatch below exists to prevent.
    """
    if not isinstance(value, Mapping):
        return dict(default or {})
    return {str(key).strip().upper(): str(label).strip() for key, label in value.items() if str(key).strip()}


def _coercer_for(field: dataclasses.Field) -> Any:
    """How one dataclass field reads a raw config value, decided by its declared type.

    Type-driven rather than spelled out per field, because spelling it out is what made these
    loaders 80 lines each and let a new knob be added to the dataclass and forgotten in the
    parser -- where it reads as "the setting does nothing" rather than as a mistake.
    """
    declared = field.metadata.get("coerce")
    if declared == "symbol":
        return as_symbol
    if declared == "symbols":
        return parse_symbols

    annotation = field.type if isinstance(field.type, str) else getattr(field.type, "__name__", "")
    if "dict" in annotation:
        return as_symbol_map
    if "list" in annotation:
        return parse_symbols
    if "bool" in annotation:
        return as_bool
    if "float" in annotation:
        return as_float
    if "int" in annotation:
        return as_int
    return as_text


def load_tuning(
    cls: type[T],
    raw: Mapping[str, Any] | None,
    *,
    fallbacks: Mapping[str, Any] | None = None,
) -> T:
    """Build a frozen tuning dataclass from a saved config section.

    Every field is read by name and coerced by its declared type; a field the section does not
    mention keeps the dataclass default. Integer fields whose name ends ``_minutes`` go through
    :func:`minutes_knob`, so a config still written in bar counts is converted rather than
    ignored -- the field may declare a non-standard legacy name with
    ``metadata={"legacy_key": ...}``.

    ``fallbacks`` supplies a different default for named fields, for the handful whose default
    comes from the account config rather than from the dataclass (``per_trade_value_min`` falls
    back to ``min_trade_dollars``, for instance).
    """
    section = raw if isinstance(raw, dict) else {}
    fallbacks = fallbacks or {}
    defaults = cls()
    values: dict[str, Any] = {}

    for field in dataclasses.fields(cls):
        if not field.init:
            continue
        default = fallbacks.get(field.name)
        if default is None:
            default = getattr(defaults, field.name)
        coerce = _coercer_for(field)
        if coerce is as_int and field.name.endswith("_minutes"):
            values[field.name] = minutes_knob(
                section, field.name, int(default), legacy_key=field.metadata.get("legacy_key")
            )
            continue
        values[field.name] = coerce(section.get(field.name), default)

    return cls(**values)


def minutes_knob(raw: dict[str, Any], key: str, default: int, *, legacy_key: str | None = None) -> int:
    """Read a minute-valued knob, converting a config still written in bar counts.

    Horizons used to be counted in bars on an assumed 15-minute grid. Dashboard tuning
    lives in config files that may still have legacy bar keys, which are converted to minutes.
    """
    if not isinstance(raw, dict):
        return default
    if key in raw:
        try:
            return int(raw[key])
        except (TypeError, ValueError):
            return default
    legacy_key = legacy_key or key.replace("_minutes", "_bars")
    if legacy_key not in raw:
        return default
    try:
        bars = int(raw[legacy_key])
    except (TypeError, ValueError):
        return default
    try:
        grid = int(raw.get("intraday_bar_minutes", LEGACY_BAR_MINUTES)) or LEGACY_BAR_MINUTES
    except (TypeError, ValueError):
        grid = LEGACY_BAR_MINUTES
    minutes = bars * grid
    if legacy_key not in _warned_legacy_keys:
        _warned_legacy_keys.add(legacy_key)
        logger.info(
            "Config knob %s is counted in bars; reading it as %s=%s (%s bars x %sm). "
            "Save the algorithm from the dashboard to store it in minutes.",
            legacy_key, key, minutes, bars, grid,
        )
    return minutes
