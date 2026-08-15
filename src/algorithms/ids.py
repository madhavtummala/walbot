"""Algorithm identity: the canonical ids and the retired names that still resolve to them.

Deliberately imports nothing. Identity is not behaviour -- knowing that ``invest_spy`` means
``spy_rotation`` requires no algorithm class, no config, and no registry -- and keeping it
dependency-free is what lets both the registry and the config loader read it.

That mattered structurally: ``get_config`` needs to canonicalise a strategy id before reading
tuning saved under it, and it used to reach into ``algorithms.registry`` to do so. Since the
registry imports the algorithm base, which reads market context, which loads config, that one
import closed three separate cycles. The loader now depends on this module instead, and this
module depends on nothing.
"""

from __future__ import annotations

DEFAULT_STRATEGY_ID = "fast_momentum"

#: Retired ids that still appear in saved controls, tuning sections, and cached backtests.
#: ``invest_spy`` is aliased rather than renamed on disk because the id is the key under
#: ``algorithm_configs``: renaming it outright would drop saved tuning back to defaults with
#: no error anywhere.
#: ``none`` is here because the deck used to carry a "None" card that meant "no algorithm, DCA
#: keeps running underneath". DCA is a selectable algorithm now, so that card is gone and the
#: id it saved resolves to the thing it always actually did. Whether the bot trades at all is
#: ``algorithm_enabled``, not a sentinel strategy id.
ALGORITHM_ALIASES = {
    "invest_spy": "spy_rotation",
    "regime_rotation": "spy_rotation",
    "dca_bot": "dca",
    "none": "dca",
}


#: Reverse of :data:`ALGORITHM_ALIASES`, for reading tuning saved under a retired id.
#:
#: A list per id, not a single value: ``spy_rotation`` has been renamed twice (``invest_spy``
#: then ``regime_rotation``), and a plain reverse dict would keep only the newest of the two
#: and silently miss tuning still filed under the oldest -- which is the key actually on disk.
LEGACY_ALGORITHM_IDS: dict[str, list[str]] = {}
for _old, _new in ALGORITHM_ALIASES.items():
    LEGACY_ALGORITHM_IDS.setdefault(_new, []).append(_old)


def canonical_algorithm_id(algorithm_id: str) -> str:
    normalized = str(algorithm_id or "").strip().lower()
    return ALGORITHM_ALIASES.get(normalized, normalized)
