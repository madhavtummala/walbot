from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Type

from .base import BaseAlgorithm

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
