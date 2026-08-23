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
    """Order by what the run did, then by conviction within each group.

    Conviction is the score, signed. Sorting on ``-abs(score)`` used to seat the *worst* name in
    the universe next to the best -- a score of -3.0 ahead of +1.0 -- which for a cross-sectional
    ranker throws away the entire signal.
    """
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

    Reads only what every plan has. An algorithm whose decisions are not weights (DCA's budgets,
    an option contract) builds its own rows in its ``signals.py`` instead -- but into the same
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
    # Only worth a slot when the run actually did one of these. A permanent "Entering 0 /
    # Exiting 0" strip is noise on the ~90% of sessions where a momentum book does nothing.
    for label, action in (("Entering", ACTION_ENTER), ("Exiting", ACTION_EXIT), ("Blocked", ACTION_BLOCKED)):
        if counts[action]:
            summary.append({"label": label, "value": str(counts[action])})
    # Only when the algorithm actually reads sentiment. It used to be unconditional, so an
    # algorithm with both sentiment weights at zero -- which is every one of them by default,
    # and there is no sentiment provider configured -- still rendered a "No recent records"
    # row, which reads as a broken feed rather than as a switched-off input.
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

    Two methods carry the strategy, and they differ in what they are allowed to touch.
    ``plan`` is pure -- everything it reads arrives on the :class:`AlgorithmContext`, so the
    live runner and the backtester drive the identical call and cannot drift apart. ``execute``
    is the half that touches the world: it places the orders and persists whatever the run
    should remember.

    Nothing here reaches for a brokerage or a state store during ``plan``. That is what makes
    looking at an algorithm free of consequence: the dashboard and a reviewing agent run the
    same code the scheduler does, and neither can move money or shift the next run's memory.

    **Every algorithm is a package with the same three modules**, so that knowing one is
    knowing all of them:

    ``config.py``
        The frozen tuning dataclass, and nothing else. Named through ``tuning_class``; the
        loading is done here, by :meth:`tuning`.
    ``algorithm.py``
        The subclass and the market logic only it uses -- ``requirements``, ``plan``, and
        ``state_after`` when its memory depends on what filled.
    ``signals.py``
        The per-symbol rows ``plan`` publishes and the :meth:`signal_view` that renders them.
        Display is separated because it is the one part with no bearing on what gets traded,
        and mixing it in is how an algorithm grows indicator maths no decision ever reads.

    ``__init__.py`` stays a docstring. Further private modules are for an ``algorithm.py``
    that would otherwise be unreadable, not a default.
    """

    algorithm_id: str = ""

    #: The frozen dataclass holding this algorithm's own knobs, one per algorithm, living in
    #: its ``config.py``. Named rather than loaded, because loading is framework work: see
    #: :meth:`tuning`. Every algorithm has one; there is no such thing as an untuned strategy.
    tuning_class: type | None = None

    #: Default cadence, as a cron expression in market time. See :mod:`src.core.cron`.
    #:
    #: A *default*, not a rule: a binding carries its own cron and overrides this. The class
    #: still states one because a new deployment needs a sensible cadence before anyone has
    #: chosen one, and because what cadence an algorithm can usefully run at is a property of
    #: the strategy -- running Bursty DCA every fifteen minutes only burns API calls.
    cron: str = "*/30 9-15 * * 1-5"

    #: Name of a purpose-built editor the Tune screen should render for this algorithm's
    #: configuration, or None for the generic parameter form. An algorithm whose real
    #: configuration is not a list of scalars declares it here, so the dashboard asks the
    #: backend which editor to use instead of keeping its own list of special algorithms.
    tune_editor: str | None = None

    #: Order-sizing floors. ``None`` takes the account's own setting, which is right for an
    #: algorithm that states a portfolio; ``0.0`` switches the floor off, which is right for one
    #: that states increments -- a drift threshold applied to a DCA buy suppresses exactly the
    #: trade the plan asked for. An algorithm whose floors come from its own tuning overrides
    #: :meth:`sizing` instead.
    min_trade_dollars: float | None = None
    rebalance_threshold: float | None = None

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        class_id = getattr(self.__class__, "algorithm_id", "")
        self.algorithm_id = class_id or self.__class__.__name__.lower()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> BaseAlgorithm:
        return cls(config)

    def tuning(self, config: Any) -> Any:
        """This algorithm's knobs, read from its own config section and coerced by type.

        The one way an algorithm reads its configuration. Each used to write its own loader,
        which is how one of the three ended up bypassing :func:`load_tuning` entirely and
        silently losing legacy-key and ``_minutes`` handling -- a bug that only exists because
        the contract left the job open. Declare ``tuning_class`` and this is done for you,
        retired ids included.
        """
        if self.tuning_class is None:
            raise NotImplementedError(f"{type(self).__name__} declares no tuning_class")
        ids = (self.algorithm_id, *LEGACY_ALGORITHM_IDS.get(self.algorithm_id, []))
        return load_tuning(self.tuning_class, tuning_section(config, *ids))

    def requirements(self, config: Any, current_positions: Dict[str, int]) -> AlgorithmRequirements:
        """What data this algorithm needs, so the caller can load it once and hand it over."""
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
        ids = (self.algorithm_id, *LEGACY_ALGORITHM_IDS.get(self.algorithm_id, []))
        return {
            "tuning": tuning_section(config, *ids),
            "symbols": list(getattr(config, "symbols", []) or []),
        }

    def plan(self, context: AlgorithmContext) -> AlgorithmPlan:
        """What to trade, as a pure function of ``context``.

        Reads bars, holdings, equity and the memory of previous runs -- all of it handed over
        as data. May not fetch, may not read a clock (``context.timestamp`` is the moment this
        run describes), and may not write anything: state it wants kept goes on the returned
        plan and is committed by ``execute``, only if orders actually go out.
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

        Defaults to whatever ``plan`` proposed, which is right for an algorithm whose memory
        is a function of its own decision. An algorithm whose memory depends on what actually
        *filled* -- DCA drawing down an accrued budget -- overrides this and reads ``outcome``,
        so a run that proposes but never submits keeps its budget rather than spending it on
        nothing.
        """
        return plan.state or None

    def sizing(self, config: Any) -> Dict[str, float]:
        """Order-sizing floors, from the class attributes above or the account config.

        Deliberately no cash buffer. How much of the book to deploy is a strategy decision,
        already stated by whatever gross-exposure cap the algorithm applies to its own weights;
        holding cash back to *fund* a batch is an account decision, applied once against
        buying power in ``pipeline.place_orders``. Fusing the two is what made every
        exposure-capped algorithm override this to zero to avoid under-investing twice.
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
        """Render ``plan`` for the dashboard.

        A rendering, never a second run: the caller hands over the same plan the scheduler
        would act on, so the deck shows what would actually trade rather than a parallel
        approximation of it. The default reads the two fields every plan has -- ``signals``
        for the rows and the weight intents for the exposure -- which is enough for any
        algorithm that thinks in weights and scores. One that does not (DCA has budgets and
        accrual, not ranks) overrides this and builds its own rows from the same plan.
        """
        return signal_view_from_plan(plan)
