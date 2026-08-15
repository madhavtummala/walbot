from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Type

from .base import BaseAlgorithm
# Identity lives in ``ids``, which imports nothing, so the config loader can canonicalise a
# strategy id without importing this module and closing a cycle back through it.
from .ids import ALGORITHM_ALIASES, LEGACY_ALGORITHM_IDS, canonical_algorithm_id  # noqa: F401

ALGORITHM_MODULES = {
    "fast_momentum": "src.algorithms.fast_momentum",
    # Not to be confused with the "dual_momentum" *scoring model* in core/strategy_models.py,
    # which is a daily-bar row builder used by the options swing algorithm. This id is the
    # algorithm; that string is a signal-row style.
    "dual_momentum": "src.algorithms.dual_momentum",
    "spy_rotation": "src.algorithms.invest_spy",
    "dca": "src.algorithms.dca",
    "bursty_dca": "src.algorithms.dca",
    "options_swing": "src.algorithms.options.swing",
}

ALGORITHM_REGISTRY: dict[str, str | Type[BaseAlgorithm]] = {
    "dca": "src.algorithms.dca.bot:DCAAlgorithm",
    "bursty_dca": "src.algorithms.dca.bursty:BurstyDCAAlgorithm",
    "fast_momentum": "src.algorithms.fast_momentum:FastMomentumAlgorithm",
    "dual_momentum": "src.algorithms.dual_momentum:DualMomentumAlgorithm",
    "spy_rotation": "src.algorithms.invest_spy:InvestSpyAlgorithm",
}

def get_algorithm_module(algorithm_id: str) -> ModuleType:
    normalized = canonical_algorithm_id(algorithm_id)
    module_path = ALGORITHM_MODULES.get(normalized)
    if not module_path:
        raise KeyError(f"Unknown algorithm: {algorithm_id}")
    return import_module(module_path)


def register_algorithm(algorithm_id: str, cls: Type[BaseAlgorithm]) -> None:
    normalized = str(algorithm_id or "").strip().lower()
    if not normalized:
        raise ValueError("algorithm_id is required")
    ALGORITHM_REGISTRY[normalized] = cls


def _load_algorithm_class(path: str) -> Type[BaseAlgorithm]:
    module_path, _, class_name = path.partition(":")
    if not module_path or not class_name:
        raise ValueError(f"Invalid algorithm class path: {path}")
    module = import_module(module_path)
    cls = getattr(module, class_name)
    if not isinstance(cls, type) or not issubclass(cls, BaseAlgorithm):
        raise TypeError(f"{path} is not an AlgorithmPlugin class")
    return cls


def get_algorithm_class(algorithm_id: str) -> Type[BaseAlgorithm]:
    normalized = canonical_algorithm_id(algorithm_id)
    entry = ALGORITHM_REGISTRY.get(normalized)
    if entry is None:
        raise KeyError(f"Unknown algorithm: {algorithm_id}")
    if isinstance(entry, str):
        cls = _load_algorithm_class(entry)
        ALGORITHM_REGISTRY[normalized] = cls
        return cls
    cls = entry
    return cls
