from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Type

from .base import BaseAlgorithm

ALGORITHM_MODULES = {
    "fast_momentum": "src.algorithms.fast_momentum",
    "invest_spy": "src.algorithms.invest_spy",
    "dca": "src.algorithms.dca",
    "options_swing": "src.algorithms.options.swing",
}

ALGORITHM_REGISTRY: dict[str, str | Type[BaseAlgorithm]] = {
    "momentum_social": "src.algorithms.generic:MomentumSocialAlgorithm",
    "trend_following": "src.algorithms.generic:TrendFollowingAlgorithm",
    "mean_reversion": "src.algorithms.generic:MeanReversionAlgorithm",
    "breakout": "src.algorithms.generic:BreakoutAlgorithm",
    "risk_parity": "src.algorithms.generic:RiskParityAlgorithm",
    "dual_momentum": "src.algorithms.generic:DualMomentumAlgorithm",
    "dca": "src.algorithms.dca.bot:DCABot",
    "dca_bot": "src.algorithms.dca.bot:DCABot",
    "fast_momentum": "src.algorithms.fast_momentum:FastMomentumAlgorithm",
    "invest_spy": "src.algorithms.invest_spy:InvestSpyAlgorithm",
}


def get_algorithm_module(algorithm_id: str) -> ModuleType:
    normalized = str(algorithm_id or "").strip().lower()
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
    normalized = str(algorithm_id or "").strip().lower()
    entry = ALGORITHM_REGISTRY.get(normalized)
    if entry is None:
        raise KeyError(f"Unknown algorithm: {algorithm_id}")
    if isinstance(entry, str):
        cls = _load_algorithm_class(entry)
        ALGORITHM_REGISTRY[normalized] = cls
        return cls
    cls = entry
    return cls
