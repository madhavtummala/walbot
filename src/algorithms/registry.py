"""Which algorithms exist, and how to reach one without loading the others.

Every algorithm lives at ``<package>.algorithm:<Class>`` -- the same path for all of them, so
this file states which algorithms exist rather than where each one hid its class.
"""

from __future__ import annotations

from typing import Type

from ..common.registry import Registry
from .base import BaseAlgorithm
# Identity lives in ``ids``, which imports nothing, so the config loader can canonicalise a
# strategy id without importing this module and closing a cycle back through it.
from .ids import ALGORITHM_ALIASES, LEGACY_ALGORITHM_IDS, canonical_algorithm_id  # noqa: F401

ALGORITHMS: Registry[BaseAlgorithm] = Registry(
    "algorithm",
    BaseAlgorithm,
    {
        "bursty_dca": "src.algorithms.bursty_dca.algorithm:BurstyDCAAlgorithm",
        "rally_rotation": "src.algorithms.rally_rotation.algorithm:RallyRotationAlgorithm",
        "options_flip": "src.algorithms.options_flip.algorithm:OptionsFlipAlgorithm",
    },
)

for _retired, _current in ALGORITHM_ALIASES.items():
    ALGORITHMS.alias(_retired, _current)


def register_algorithm(algorithm_id: str, cls: Type[BaseAlgorithm]) -> None:
    """Add an algorithm at runtime. Used by tests to register a stub subject."""
    ALGORITHMS.register(algorithm_id, cls)


def get_algorithm_class(algorithm_id: str) -> Type[BaseAlgorithm]:
    """The class registered under ``algorithm_id``, importing its module on first use."""
    return ALGORITHMS.get(algorithm_id)
