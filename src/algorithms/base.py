from __future__ import annotations

import logging
from typing import Dict, Any, List
from ..core.interfaces import AlgorithmDecision, AlgorithmPlugin, AlgorithmContext, AlgorithmRequirements

logger = logging.getLogger(__name__)

class BaseAlgorithm(AlgorithmPlugin):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        class_id = getattr(self.__class__, "algorithm_id", "")
        self.algorithm_id = class_id or self.__class__.__name__.lower()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseAlgorithm:
        return cls(config)

    def requirements(self, config: Any, current_positions: Dict[str, int]) -> AlgorithmRequirements:
        return AlgorithmRequirements()

    def generate_signals(self, context: AlgorithmContext) -> List[Dict[str, Any]]:
        # Default implementation: no signals
        return []

    def decide(self, context: AlgorithmContext) -> AlgorithmDecision:
        return AlgorithmDecision()

    def log_signal(self, symbol: str, signal_type: str, details: Dict[str, Any]):
        logger.info(f"[{self.algorithm_id}] {symbol} SIGNAL: {signal_type} | {details}")
