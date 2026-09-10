from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List

from ..common.config_utils import json_number, load_tuning, tuning_section
from ..core.interfaces import (
    ACTION_BLOCKED,
    ACTION_ENTER,
    ACTION_EXIT,
    ACTION_HOLD,
    ACTION_IDLE,
    AlgorithmContext,
    AlgorithmPlan,
    AlgorithmRequirements,
    Check,
    SignalRow,
    SignalView,
)
from ..core.pipeline import place_orders
from ..data.state_store import algorithm_state_key, save_state
from .ids import LEGACY_ALGORITHM_IDS

logger = logging.getLogger(__name__)


#: Reading order for the deck: the decisions that changed something first, then the holdings,
#: then the names that wanted in and were stopped, then the rest. A reader scanning the top of
#: the list should be seeing what this run actually did.
ACTION_ORDER = {ACTION_EXIT: 0, ACTION_ENTER: 1, ACTION_HOLD: 2, ACTION_BLOCKED: 3, ACTION_IDLE: 4}


def sort_rows(rows: List[SignalRow]) -> List[SignalRow]:
    """Order by what the run did, then by conviction (signed score) within each group."""
    return sorted(
        rows,
        key=lambda row: (ACTION_ORDER.get(row.action, 9), -_score_of(row), row.symbol),
    )


def _score_of(row: SignalRow) -> float:
    for metric in row.metrics:
        if metric.get("label") == "Score":
            try:
                return float(str(metric.get("value", "")).replace(",", ""))
            except ValueError:
                return 0.0
    return 0.0


def signal_view_from_plan(plan: AlgorithmPlan) -> SignalView:
    """The default rendering, for an algorithm that thinks in weights and scores.

    An algorithm whose decisions are not weights builds its own rows instead, into the same
    :class:`SignalRow` shape, so the deck never learns which algorithm it is drawing.
    """
    weights = plan.target_weights
    rows = [
        SignalRow(
            symbol=symbol,
            action=str(values.get("action") or _implied_action(values, weights.get(symbol, 0.0))),
            headline=str(values.get("reason") or ""),
            metrics=[
                {"label": "Score", "value": f"{float(values.get('score', 0.0)):.2f}"},
                {"label": "Weight", "value": f"{weights.get(symbol, 0.0):.1%}"},
                {"label": "Close", "value": _money(values.get("close") or plan.latest_prices.get(symbol))},
            ],
            checks=[Check(**check) for check in values.get("checks") or []],
        )
        for symbol, values in plan.signals.items()
    ]
    return SignalView(rows=sort_rows(rows), summary=_summary_from(plan, rows))


def _implied_action(values: Dict[str, Any], weight: float) -> str:
    """Fallback for an algorithm that has not classified its own rows yet."""
    if weight > 0:
        return ACTION_HOLD
    return ACTION_BLOCKED if values.get("eligible") else ACTION_IDLE


def _money(value: Any) -> str:
    number = json_number(value) if isinstance(value, (int, float)) else None
    return f"${number:,.2f}" if number else "--"


def _summary_from(plan: AlgorithmPlan, rows: List[SignalRow]) -> List[Dict[str, str]]:
    """The header strip: what the run did, in counts, above the rows that justify it."""
    counts = Counter(row.action for row in rows)
    summary = [
        {"label": "Allocation", "value": str(plan.metadata.get("allocation_mode") or "--")},
        {"label": "Exposure", "value": f"{sum(w for w in plan.target_weights.values() if w > 0):.0%}"},
        {"label": "Held", "value": str(counts[ACTION_HOLD] + counts[ACTION_ENTER])},
    ]
    # Only worth a slot when the run actually did one of these.
    for label, action in (("Entering", ACTION_ENTER), ("Exiting", ACTION_EXIT), ("Blocked", ACTION_BLOCKED)):
        if counts[action]:
            summary.append({"label": label, "value": str(counts[action])})
    # Only when the algorithm actually reads sentiment, not unconditionally.
    if "market_sentiment" in plan.metadata:
        market_sentiment = plan.metadata["market_sentiment"]
        summary.append({
            "label": "Sentiment",
            "value": "No recent records" if not market_sentiment else f"{float(market_sentiment):+.2f}",
        })
    summary.append({"label": "Universe", "value": str(len(rows))})
    return summary


class BaseAlgorithm:
    """The whole algorithm contract: declare what you need, plan, execute, render.

    ``plan`` is pure -- everything it reads arrives on :class:`AlgorithmContext`, so the live
    runner and the backtester drive the identical call. ``execute`` is the only half that
    touches the world: it places orders and persists whatever the run should remember.

    Every algorithm is a package with the same three modules: ``config.py`` (the frozen tuning
    dataclass, named via ``tuning_class``), ``algorithm.py`` (the subclass and its market
    logic), ``signals.py`` (the per-symbol rows and the rendering in :meth:`signal_view`).
    """

    algorithm_id: str = ""

    #: The frozen dataclass holding this algorithm's own knobs, one per algorithm, living in
    #: its ``config.py``. Named rather than loaded, because loading is framework work: see
    #: :meth:`tuning`. Every algorithm has one; there is no such thing as an untuned strategy.
    tuning_class: type | None = None

    #: Default cadence (cron, market time) -- a binding's own cron overrides this.
    cron: str = "*/30 9-15 * * 1-5"

    #: Whether the replay can meaningfully simulate this algorithm. The backtester fills at the
    #: mark and keeps no resting orders, so an algorithm whose decisions *are* resting orders
    #: (a limit, an exchange-side stop) has nothing to be simulated against.
    backtestable: bool = True
    not_backtestable_reason: str = ""

    #: Name of a purpose-built Tune-screen editor, or None for the generic parameter form.
    tune_editor: str | None = None

    #: The buckets a ``budgets`` board splits its symbols across, and what each amount means.
    #: Declared by the algorithm because only it knows: DCA divides a monthly budget into buy
    #: and sell, Options Flip divides a per-position dollar cap into call and put. The dashboard
    #: used to hardcode ``["buy", "sell"]``, which made the board unusable for any algorithm
    #: whose buckets were named anything else.
    tune_buckets: tuple[str, ...] = ("buy", "sell")
    #: One line under the board saying what a bubble's number is, in the algorithm's own terms.
    tune_budget_hint: str = "Dollars per month, per symbol"
    #: How a bubble's number should be read and edited: ``currency`` formats it as dollars,
    #: ``count`` as a plain integer. The board is one component, so the unit is a declaration
    #: rather than two implementations of the same board.
    tune_unit: str = "currency"
    #: Largest amount one bubble may hold, and the increment a scroll moves it by.
    tune_max_amount: float = 2000.0
    tune_step: float = 25.0

    #: Order-sizing floors. ``None`` takes the account's own setting (right for a portfolio
    #: algorithm); ``0.0`` switches the floor off (right for one that states increments, e.g.
    #: DCA, where a drift threshold would suppress the exact trade the plan asked for).
    min_trade_dollars: float | None = None
    rebalance_threshold: float | None = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        class_id = getattr(self.__class__, "algorithm_id", "")
        self.algorithm_id = class_id or self.__class__.__name__.lower()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseAlgorithm:
        return cls(config)

    def _ids_with_legacy(self) -> tuple[str, ...]:
        """This algorithm's id plus any retired ids that still resolve to it."""
        return (self.algorithm_id, *LEGACY_ALGORITHM_IDS.get(self.algorithm_id, []))

    def tuning(self, config: Any) -> Any:
        """This algorithm's knobs, read from its own config section and coerced by type."""
        if self.tuning_class is None:
            raise NotImplementedError(f"{type(self).__name__} declares no tuning_class")
        return load_tuning(self.tuning_class, tuning_section(config, *self._ids_with_legacy()))

    def requirements(self, config: Any, current_positions: Dict[str, int]) -> AlgorithmRequirements:
        """What data this algorithm needs, so the caller can load it once and hand it over."""
        return AlgorithmRequirements()

    def config_fingerprint(self, config: Any) -> Dict[str, Any]:
        """Everything this algorithm's behaviour depends on, for cache invalidation.

        Only has to be stable and JSON-encodable; it's hashed, never read. An algorithm that
        reads extra configuration beyond its tuning section overrides this to include it.
        """
        return {
            "tuning": tuning_section(config, *self._ids_with_legacy()),
            "symbols": list(getattr(config, "symbols", []) or []),
        }

    def plan(self, context: AlgorithmContext) -> AlgorithmPlan:
        """What to trade, as a pure function of ``context``.

        May not fetch, read a clock, or write anything -- state it wants kept goes on the
        returned plan and is committed by ``execute``, only if orders actually go out.
        """
        return AlgorithmPlan()

    def execute(
        self,
        plan: AlgorithmPlan,
        config: Any,
        brokerage: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Place ``plan``'s orders, then persist what the run should remember."""
        sizing = self.sizing(config)
        outcome = place_orders(
            plan.intents,
            config,
            brokerage,
            latest_prices=plan.latest_prices,
            signals=plan.signals,
            mode=plan.mode,
            min_trade_dollars=sizing["min_trade_dollars"],
            rebalance_threshold=sizing["rebalance_threshold"],
            **kwargs,
        )
        state = self.state_after(plan, outcome)
        if state is not None:
            save_state(algorithm_state_key(self.algorithm_id, getattr(config, "account_id", "")), state)
        return {"strategy": plan.strategy, **outcome}

    def state_after(self, plan: AlgorithmPlan, outcome: Dict[str, Any]) -> Dict[str, Any] | None:
        """What to remember once the orders have been sent. ``None`` writes nothing.

        Defaults to whatever ``plan`` proposed. An algorithm whose memory depends on what
        actually *filled* (DCA drawing down an accrued budget) overrides this and reads
        ``outcome`` instead, so a run that proposes but never submits keeps its budget.
        """
        return plan.state or None

    def sizing(self, config: Any) -> Dict[str, float]:
        """Order-sizing floors, from the class attributes above or the account config.

        No cash buffer here deliberately: exposure caps are a strategy decision the algorithm
        already applies to its own weights, while holding cash back to fund a batch is an
        account decision applied once in ``pipeline.place_orders``.
        """
        def floor(declared: float | None, account_key: str) -> float:
            if declared is not None:
                return float(declared)
            return float(getattr(config, account_key, 0.0) or 0.0)

        return {
            "min_trade_dollars": floor(self.min_trade_dollars, "min_trade_dollars"),
            "rebalance_threshold": floor(self.rebalance_threshold, "rebalance_threshold"),
        }

    def signal_view(self, plan: AlgorithmPlan) -> SignalView:
        """Render ``plan`` for the dashboard -- a rendering, never a second run.

        The default reads ``signals`` and the weight intents every plan has, which covers any
        algorithm that thinks in weights and scores. One that does not (DCA's budgets/accrual)
        overrides this and builds its own rows from the same plan.
        """
        return signal_view_from_plan(plan)
