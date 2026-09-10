"""Running an algorithm: the two calls every driver makes, and the line between them.

``run_algorithm`` is read-only -- it loads what the algorithm declared it needs, hands it over
as a context, and returns the plan. The dashboard, a reviewing agent and the scheduler all make
this identical call, so looking at what an algorithm would do costs nothing and changes nothing.
``execute_algorithm`` is the half that acts: it places the orders and lets the algorithm record
what came back.
"""

from __future__ import annotations

from dataclasses import replace
import logging
from typing import Any

from src.algorithms.registry import get_algorithm_class
from src.core.interfaces import AlgorithmPlan, Brokerage
from src.core.market_context import build_algorithm_context
from src.core.pipeline import assert_tradable, read_snapshot, resolve_brokerage

logger = logging.getLogger(__name__)


def run_algorithm(
    strategy: str,
    config,
    *,
    data_client: Any = None,
    brokerage: Brokerage | None = None,
    algorithm: Any = None,
) -> AlgorithmPlan:
    """Run ``strategy`` and return the plan it proposes.

    Reads holdings and equity from the brokerage so the algorithm can see them, but places
    nothing and persists nothing: running an algorithm to look at what it would do must never
    change what it does next. ``execute_algorithm`` is the half that writes.
    """
    algorithm = algorithm or get_algorithm_class(strategy).from_config(config)
    requirements = algorithm.requirements(config, {})
    book = brokerage or resolve_brokerage(config)
    snapshot = read_snapshot(config, book)
    context = build_algorithm_context(
        config,
        requirements,
        algorithm_id=algorithm.algorithm_id,
        positions=snapshot.positions,
        cost_basis=_cost_basis(book) if requirements.needs_cost_basis else {},
        equity=snapshot.equity,
        data_client=data_client,
    )
    plan = algorithm.plan(context)
    return replace(
        plan,
        strategy=strategy,
        # A zero weight is a proposal to hold nothing, which is what an absent symbol already
        # says; carrying it would only widen the order set with no-ops.
        intents=[intent for intent in plan.intents if intent.kind != "weight" or intent.value],
        latest_prices=context.latest_prices,
        metadata={**plan.metadata, "requirements": requirements},
        # The moment the context described: live that is now, in a replay it is the signal
        # date. Taken from the context so nothing downstream needs a clock of its own.
        as_of=context.timestamp,
    )


def _cost_basis(brokerage: Brokerage) -> dict[str, float]:
    """What the account paid per held symbol, from the broker's own position marks.

    Best-effort: a brokerage that cannot report a cost basis leaves the symbol out, and the
    algorithm falls back to whatever anchor it used before. A failed read must not take the run
    down with it -- this is context for pricing an exit, not a precondition for trading.
    """
    try:
        rows = brokerage.get_position_details()
    except Exception as exc:  # noqa: BLE001 - an unavailable cost basis is not a failed run
        logger.warning("Could not read cost basis: %s", exc)
        return {}
    return {
        str(row["symbol"]): float(row.get("avg_entry_price") or 0.0)
        for row in rows or []
        if float(row.get("avg_entry_price") or 0.0) > 0
    }


def execute_algorithm(
    plan: AlgorithmPlan,
    config,
    brokerage: Brokerage,
    *,
    algorithm: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Place ``plan``'s orders and let the algorithm record what came back.

    Thin on purpose: the algorithm owns both halves -- ``place_orders`` knows nothing about
    strategies, and what a fill means for an accrued budget is a question only the algorithm
    can answer.

    The pre-trade gates are asserted here rather than in any one driver, because this is the
    single point every order path crosses. Raises :class:`TradingRefused` when one refuses.
    """
    algorithm = algorithm or get_algorithm_class(plan.strategy).from_config(config)
    # Preferred from the plan, since that is what the run was actually built against; re-derived
    # only when a caller hands over a plan that never went through ``run_algorithm`` -- the gate
    # must not be skippable by omitting a metadata key.
    requirements = plan.metadata.get("requirements") or algorithm.requirements(config, {})
    assert_tradable(requirements, config, brokerage)
    return algorithm.execute(plan, config, brokerage, **kwargs)
