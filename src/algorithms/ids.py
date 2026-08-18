"""Algorithm identity: the canonical ids and the retired names that still resolve to them.

Deliberately imports nothing. Identity is not behaviour -- and keeping it dependency-free is
what lets both the registry and the config loader read it.

That mattered structurally: ``get_config`` needs to canonicalise a strategy id before reading
tuning saved under it, and it used to reach into ``algorithms.registry`` to do so. Since the
registry imports the algorithm base, which reads market context, which loads config, that one
import closed three separate cycles. The loader now depends on this module instead, and this
module depends on nothing.
"""

from __future__ import annotations

DEFAULT_STRATEGY_ID = "bursty_dca"

#: Retired ids that still appear in saved controls, tuning sections, and cached backtests.
#: ``none`` is here because the deck used to carry a "None" card that meant "no algorithm".
#: ``dca`` is retired in favour of ``bursty_dca``, which can be tuned to behave identically.
ALGORITHM_ALIASES = {
    "dca_bot": "bursty_dca",
    "none": "bursty_dca",
    "dca": "bursty_dca",
}


#: Reverse of :data:`ALGORITHM_ALIASES`, for reading tuning saved under a retired id.
#:
#: A list per id, not a single value: a plain reverse dict would keep only the newest alias
#: and silently miss tuning still filed under an older one -- which is the key actually on disk.
LEGACY_ALGORITHM_IDS: dict[str, list[str]] = {}
for _old, _new in ALGORITHM_ALIASES.items():
    LEGACY_ALGORITHM_IDS.setdefault(_new, []).append(_old)


def canonical_algorithm_id(algorithm_id: str) -> str:
    normalized = str(algorithm_id or "").strip().lower()
    return ALGORITHM_ALIASES.get(normalized, normalized)
