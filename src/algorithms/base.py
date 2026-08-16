from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from ..common.config_utils import json_number
from ..core.interfaces import (
    AlgorithmDecision,
    AlgorithmPlugin,
    AlgorithmContext,
    AlgorithmRequirements,
    Intent,
    PortfolioSnapshot,
    SignalView,
    intents_from_weights,
    weights_from_intents,
)

logger = logging.getLogger(__name__)



def signal_view_from_decision(decision: AlgorithmDecision, latest_prices: Dict[str, float] | None = None) -> SignalView:
    """Build a generic dashboard view from an algorithm decision's signals and target weights."""
    latest_prices = latest_prices or {}
    leaders: List[Dict[str, Any]] = []
    for symbol, values in decision.signals.items():
        signal = int(values.get("signal", 0) or 0)
        side = "LONG" if signal > 0 else "SHORT" if signal < 0 else "FLAT"
        row = {key: json_number(val) if isinstance(val, (int, float)) else val for key, val in values.items()}
        row.update(
            {
                "symbol": symbol,
                "signal": side,
                "side": side,
                "target_weight": json_number(decision.target_weights.get(symbol, 0.0)),
                "close": row.get("close", json_number(latest_prices.get(symbol))),
                "reason": values.get("reason", ""),
            }
        )
        leaders.append(row)
    leaders.sort(key=lambda item: (item["signal"] != "LONG", item["signal"] == "FLAT", -abs(float(item.get("score") or 0.0))))
    return SignalView(leaders=leaders, summary=_summary_from(decision, leaders))


def _summary_from(decision: AlgorithmDecision, leaders: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Dashboard summary rows.

    Allocation-style algorithms describe themselves through ``metadata``; anything without it
    falls back to plain long/short counts.
    """
    allocation_mode = decision.metadata.get("allocation_mode")
    if not allocation_mode:
        long_count = sum(1 for row in leaders if row["signal"] == "LONG")
        short_count = sum(1 for row in leaders if row["signal"] == "SHORT")
        return [
            {"label": "Long", "value": str(long_count)},
            {"label": "Short", "value": str(short_count)},
            {"label": "Universe", "value": str(len(leaders))},
        ]

    exposure = sum(float(weight) for weight in decision.target_weights.values() if weight > 0)
    market_sentiment = decision.metadata.get("market_sentiment")
    return [
        {"label": "Allocation", "value": str(allocation_mode)},
        {"label": "Exposure", "value": f"{exposure:.0%}"},
        {
            "label": "Sentiment",
            "value": "No recent records" if not market_sentiment else f"{float(market_sentiment):+.2f}",
        },
        {"label": "Universe", "value": str(len(leaders))},
    ]

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

    def config_fingerprint(self, config: Any) -> Dict[str, Any]:
        """Everything this algorithm's behaviour depends on, for cache invalidation.

        A cached backtest is only valid while the inputs that produced it are unchanged, and
        "the inputs" differ per algorithm. This used to be handled by threading DCA's plan
        through three backtest signatures as a special case -- an argument that never reached
        the algorithm and existed solely to change a hash. An algorithm that reads extra
        configuration declares it here instead, and the backtest stays generic.

        The value only has to be stable and JSON-encodable; it is hashed, never read.
        """
        selected = (
            config.algorithm_configs.get(self.algorithm_id, {})
            if isinstance(getattr(config, "algorithm_configs", None), dict)
            else {}
        )
        return {"tuning": selected, "symbols": list(getattr(config, "symbols", []) or [])}

    def generate_signals(self, context: AlgorithmContext) -> List[Dict[str, Any]]:
        # Default implementation: no signals
        return []

    def decide(self, context: AlgorithmContext) -> AlgorithmDecision:
        return AlgorithmDecision()

    def refine(
        self,
        intents: List[Intent],
        signals: Dict[str, Dict[str, Any]],
        snapshot: PortfolioSnapshot,
        latest_prices: Dict[str, float],
        config: Any,
        as_of: datetime,
    ) -> List[Intent]:
        """Route weight intents through :meth:`refine_weights` and pass anything else through.

        Allocation algorithms think in weight dicts, so they override ``refine_weights`` and
        never see the intent wrapper. Algorithms that emit notional intents (DCA) override
        ``refine`` itself.
        """
        weights = weights_from_intents(intents)
        others = [intent for intent in intents if intent.kind != "weight"]
        refined = self.refine_weights(weights, signals, snapshot, latest_prices, config, as_of)
        return intents_from_weights(refined) + others

    def refine_weights(
        self,
        target_weights: Dict[str, float],
        signals: Dict[str, Dict[str, Any]],
        snapshot: PortfolioSnapshot,
        latest_prices: Dict[str, float],
        config: Any,
        as_of: datetime,
    ) -> Dict[str, float]:
        """Weight-mode step 2: adjust proposed weights for what is already held."""
        return dict(target_weights)

    def signal_view(self, config: Any, *, data_client: Any = None) -> SignalView:
        """Default dashboard view: run step 1 and map its proposal to rows.

        Uses ``analyze`` rather than ``decide`` so the dashboard never needs an account, and
        so it cannot hit the guards that divide by equity.
        """
        from ..core.market_context import build_algorithm_context

        context = build_algorithm_context(
            config, self.requirements(config, {}), data_client=data_client
        )
        return signal_view_from_decision(self.analyze(context), context.latest_prices)

