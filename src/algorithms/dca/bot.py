from __future__ import annotations

from ..base import BaseAlgorithm
from ...core.interfaces import AlgorithmContext
from typing import List, Dict, Any

class DCABot(BaseAlgorithm):
    algorithm_id = "dca"
    """
    Dollar Cost Averaging Algorithm.
    Follows the plug-and-play interface but ignores signals.
    """
    
    def generate_signals(self, context: AlgorithmContext) -> List[Dict[str, Any]]:
        signals = []
        for symbol in self.config.get("symbols", []):
            # Simplified DCA logic: just buy a fixed amount
            amount = self.config.get("amount_per_trade", 100)
            signals.append({
                "symbol": symbol,
                "action": "BUY",
                "type": "MARKET",
                "amount": amount,
                "reason": "DCA Interval reached"
            })
        return signals
