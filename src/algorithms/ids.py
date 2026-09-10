"""Algorithm identity: the canonical ids and the retired names that still resolve to them.

Deliberately imports nothing, so both the registry and the config loader can read it without
creating an import cycle.
"""

from __future__ import annotations

DEFAULT_STRATEGY_ID = "bursty_dca"

#: Retired ids that still appear in saved controls, tuning sections, and cached backtests.
ALGORITHM_ALIASES = {
    "none": "bursty_dca",
    "intraday_pick": "options_flip",
}

#: Reverse of ``ALGORITHM_ALIASES``, for reading tuning saved under a retired id. A list per
#: id, since more than one retired name can map to the same current one.
LEGACY_ALGORITHM_IDS: dict[str, list[str]] = {}
for _old, _new in ALGORITHM_ALIASES.items():
    LEGACY_ALGORITHM_IDS.setdefault(_new, []).append(_old)


def canonical_algorithm_id(algorithm_id: str) -> str:
    normalized = str(algorithm_id or "").strip().lower()
    return ALGORITHM_ALIASES.get(normalized, normalized)
