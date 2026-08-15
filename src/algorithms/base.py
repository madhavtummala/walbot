from __future__ import annotations

import logging
import math
from typing import Dict, Any, List
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


def json_number(value: Any) -> float | None:
    """Coerce ``value`` to a JSON-safe float, mapping non-finite values to ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


#: The grid every ``*_bars`` knob in this project was written against, before horizons moved
#: to wall-clock minutes. Used only to convert configs saved under the old names.
LEGACY_BAR_MINUTES = 15

_warned_legacy_keys: set[str] = set()


def minutes_knob(raw: Dict[str, Any], key: str, default: int, *, legacy_key: str | None = None) -> int:
    """Read a minute-valued knob, converting a config still written in bar counts.

    Horizons used to be counted in bars on an assumed 15-minute grid, so a saved
    ``micro_momentum_lookback_bars: 78`` means 1170 minutes and nothing else. Dashboard tuning
    lives in ``config/walbot.yaml``, which nobody is going to hand-migrate, so the old key is
    honoured and converted rather than silently ignored -- being ignored would drop the knob
    back to its default and quietly change how the strategy trades.
    """
    if not isinstance(raw, dict):
        return default
    if key in raw:
        try:
            return int(raw[key])
        except (TypeError, ValueError):
            return default
    legacy_key = legacy_key or key.replace("_minutes", "_bars")
    if legacy_key not in raw:
        return default
    try:
        bars = int(raw[legacy_key])
    except (TypeError, ValueError):
        return default
    try:
        grid = int(raw.get("intraday_bar_minutes", LEGACY_BAR_MINUTES)) or LEGACY_BAR_MINUTES
    except (TypeError, ValueError):
        grid = LEGACY_BAR_MINUTES
    minutes = bars * grid
    if legacy_key not in _warned_legacy_keys:
        _warned_legacy_keys.add(legacy_key)
        logger.info(
            "Config knob %s is counted in bars; reading it as %s=%s (%s bars x %sm). "
            "Save the algorithm from the dashboard to store it in minutes.",
            legacy_key, key, minutes, bars, grid,
        )
    return minutes


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
    ) -> List[Intent]:
        """Route weight intents through :meth:`refine_weights` and pass anything else through.

        Allocation algorithms think in weight dicts, so they override ``refine_weights`` and
        never see the intent wrapper. Algorithms that emit notional intents (DCA) override
        ``refine`` itself.
        """
        weights = weights_from_intents(intents)
        others = [intent for intent in intents if intent.kind != "weight"]
        refined = self.refine_weights(weights, signals, snapshot, latest_prices, config)
        return intents_from_weights(refined) + others

    def refine_weights(
        self,
        target_weights: Dict[str, float],
        signals: Dict[str, Dict[str, Any]],
        snapshot: PortfolioSnapshot,
        latest_prices: Dict[str, float],
        config: Any,
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

    def log_signal(self, symbol: str, signal_type: str, details: Dict[str, Any]):
        logger.info(f"[{self.algorithm_id}] {symbol} SIGNAL: {signal_type} | {details}")
