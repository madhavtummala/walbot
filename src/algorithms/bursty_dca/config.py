"""Tuning for Bursty DCA: the sizing model, and the per-symbol budgets it deploys.

Two kinds of configuration, both this algorithm's own. :class:`BurstyConfig` is the ordinary
tuning dataclass every algorithm has. The *plan* -- what to buy and how much of it per month --
is the rest, and it needs its own reader because it is a nested structure rather than a list of
scalars, which is also why the Tune screen renders it through a purpose-built editor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...common.config_utils import as_float

#: Where a plan lives inside this algorithm's ordinary config section -- ``algorithms.<id>.plan``,
#: read and written through ``/api/algorithm-config`` like every other algorithm's knobs.

#: Hard ceiling on a per-symbol budget, in dollars per month per symbol.
MAX_ITEM_AMOUNT = 5_000.0

#: The plan carries what to buy and how much, and nothing else -- cadence and enablement live
#: on the binding, not here.
BUCKETS = ("buy", "sell")


@dataclass(frozen=True)
class BurstyConfig:
    """The sizing model: a valuation factor and a backlog factor, multiplied.

        size = budget x conviction(z) x willingness(backlog)

    Two independent questions, deliberately kept as two factors. ``conviction`` asks *how good
    is this price*, ``willingness`` asks *how much of the budget is this symbol entitled to
    spend right now*. Multiplying them is what lets an exceptional dislocation still deploy
    while borrowed ahead of plan, without either factor needing a special case for the other.
    """

    #: Extra multiples of the monthly budget per standard deviation of favourable dislocation
    #: from the moving average: ``conviction = 1 + scaling_factor x z``. Measured in sigma
    #: rather than percent, so one setting means the same thing across symbols of different vol.
    scaling_factor: float = 0.5
    #: Window for the moving average *and* for the standard deviation that normalizes distance
    #: from it. One window, because a z-score whose mean and deviation come from different
    #: periods is not measuring dislocation against anything coherent.
    regime_ma_days: int = 150
    #: Cap on cumulative deployment per symbol per month, in multiples of the monthly budget.
    #: A backstop against a runaway signal, not the primary control -- ``relax_months`` is.
    max_monthly_multiple: float = 3.0
    #: Width of the backlog resistance curve, in months of budget -- a continuous width rather
    #: than a threshold, so falling behind matters gradually rather than in one step.
    relax_months: float = 2.0
    #: How far ``willingness`` swings either side of 1.0 as the backlog saturates. At the
    #: default, a symbol months ahead of plan deploys 0.3x and one months behind deploys 1.7x.
    #: Zero disables backlog resistance entirely and spends at plan regardless.
    relax_depth: float = 0.7

    @property
    def required_daily_bars(self) -> int:
        return self.regime_ma_days + 30


# --------------------------------------------------------------------------------------
# The plan: per-symbol monthly budgets.
# --------------------------------------------------------------------------------------


def sanitize_plan(plan: dict[str, Any] | None, universe: set[str]) -> dict[str, Any]:
    """Normalize a plan and keep only symbols present in the configured universe.

    An absent or empty plan sanitizes to empty buckets -- never a built-in default basket, which
    would let clearing the board silently leave the algorithm still buying.
    """
    sanitized: dict[str, Any] = {}

    for bucket in BUCKETS:
        raw_items = ((plan or {}).get(bucket) or {}).get("items") or []
        seen: set[str] = set()
        items: list[dict[str, Any]] = []
        for item in raw_items:
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol or symbol not in universe or symbol in seen:
                continue
            seen.add(symbol)
            items.append({
                "symbol": symbol,
                "amount": min(as_float(item.get("amount"), default=0.0), MAX_ITEM_AMOUNT),
            })
        sanitized[bucket] = {"amount": sum(item["amount"] for item in items), "items": items}

    return sanitized


def unknown_plan_symbols(plan: dict[str, Any], universe: set[str]) -> list[str]:
    """Plan symbols that are not in the tradable universe.

    :func:`sanitize_plan` drops them, so without this a typo in a bucket is invisible: the row
    simply never appears and the money is silently never spent.
    """
    unknown: list[str] = []
    for bucket in BUCKETS:
        for item in (plan.get(bucket) or {}).get("items", []) or []:
            symbol = str(item.get("symbol", "")).strip().upper()
            if symbol and symbol not in universe and symbol not in unknown:
                unknown.append(symbol)
    return unknown


def plan_budgets(plan: dict[str, Any]) -> dict[str, float]:
    """Monthly dollar budget per symbol, signed: buy items positive, sell items negative."""
    budgets: dict[str, float] = {}
    for bucket in BUCKETS:
        sign = 1.0 if bucket == "buy" else -1.0
        for item in (plan.get(bucket) or {}).get("items", []) or []:
            symbol = str(item.get("symbol", "")).strip().upper()
            if not symbol:
                continue
            budgets[symbol] = budgets.get(symbol, 0.0) + (sign * abs(as_float(item.get("amount"))))
    return budgets
