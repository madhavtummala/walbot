"""What a Bursty DCA run decided about each bucket, and the two factors behind it.

Every amount here restates something ``plan`` already computed -- nothing is recalculated, so
the preview and the order it previews stay identical. Valuation and backlog are continuous
multipliers, so each is reported on its own line with the multiple it contributed; only the
genuinely hard conditions (no data, nothing held to trim, an order too small to clear share
rounding) are ever marked blocking.
"""

from __future__ import annotations

from typing import Any

from ...core.interfaces import (
    ACTION_BLOCKED,
    ACTION_ENTER,
    ACTION_IDLE,
    AlgorithmPlan,
    Check,
    SignalRow,
    SignalView,
)


def signal_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Turn one run's per-symbol working values into the published signal rows.

    Kept as plain dicts because ``plan.signals`` is data rather than presentation: the MCP
    agent reads it and a backtest records it. :func:`signal_view` is what shapes it for a deck.
    """
    return {row["symbol"]: _row(row) for row in rows}


def _row(row: dict[str, Any]) -> dict[str, Any]:
    state = row["state"]
    valuation = row["valuation"]
    buying = bool(row["buying"])
    size = float(row["size"])
    floor_dollars = float(row["floor_dollars"])
    budget = abs(float(row["monthly_budget"]))
    return {
        # Spread first so the explicit keys below win -- the valuation detail's own ``close``
        # is the last bar it scored, not the live price the order was sized at.
        **valuation.get("detail", {}),
        "signal": 1 if buying else -1,
        "side": "LONG" if buying else "SHORT",
        # The order size, signed -- what a reader actually ranks rows by.
        "score": (size if buying else -size),
        "close": row["price"] or None,
        "action": _action(row),
        "monthly_budget": row["monthly_budget"],
        "accrued": round(state.accrued, 2),
        "deployed_this_month": round(state.deployed_this_month, 2),
        "min_executable": round(floor_dollars, 2),
        "backlog_months": round(float(row["backlog_months"]), 2),
        "conviction": round(float(row["conviction"]), 3),
        "willingness": round(float(row["willingness"]), 3),
        # The product of the two, as a multiple of the plain monthly budget.
        "plan_multiple": round(size / budget, 3) if budget > 0 else 0.0,
        "next_order": round(size if buying else -size, 2),
        "notional": round((size if buying else -size) if row["deployed"] else 0.0, 2),
        "reason": _headline(row),
        "checks": _checks(row),
    }


def _action(row: dict[str, Any]) -> str:
    if row["deployed"]:
        return ACTION_ENTER
    # Nothing budgeted at all is idle; wanting to spend and being stopped is blocked.
    return ACTION_BLOCKED if row["monthly_budget"] else ACTION_IDLE


def _headline(row: dict[str, Any]) -> str:
    valuation = row["valuation"]
    size = abs(float(row["size"]))
    floor_dollars = float(row["floor_dollars"])
    budget = abs(float(row["monthly_budget"]))
    selling = float(row["monthly_budget"]) < 0

    if not valuation["ok"]:
        return str(valuation.get("reason") or "No price history")
    if row["deployed"]:
        multiple = size / budget if budget > 0 else 0.0
        return f"Deploying ${size:.0f} ({multiple:.1f}x budget)"
    if selling and float(row.get("held") or 0.0) <= 0.0:
        return "Nothing held to sell -- sells trim a position, they never short it"
    if size <= 0:
        # ``conviction`` floored at zero: say which way, rather than "did not fire".
        return f"Holding off -- {valuation['reason']}"
    if size < floor_dollars:
        # A whole-share floor means a small budget against an expensive name is short for most
        # of the month, only clearing it on the accrued balance -- name the wait.
        accrued = float(row["state"].accrued)
        months = (floor_dollars - accrued) / budget if budget > 0 else 0.0
        wait = f", ~{months:.0f} months away" if not row["fractional"] and months >= 1 else ""
        return f"Sizing ${size:.0f} against a ${floor_dollars:.0f} share, ${accrued:.0f} banked{wait}"
    # Reached only by a preview: ``plan`` deploys whenever the size clears the floor.
    return f"Ready to deploy ${size:.0f} on the next scheduled run"


def _checks(row: dict[str, Any]) -> list[dict[str, Any]]:
    """The two sizing factors, then the hard conditions, in the order ``plan`` applies them.

    Only the last two can be ``blocking``. A factor of 0.4x is not a failure -- it is the model
    working -- and marking it as one turned every ordinary rich-priced day into a red row.
    """
    state = row["state"]
    valuation = row["valuation"]
    buying = bool(row["buying"])
    size = abs(float(row["size"]))
    floor_dollars = float(row["floor_dollars"])
    conviction = float(row["conviction"])
    willingness = float(row["willingness"])
    z = float(valuation.get("detail", {}).get("z") or 0.0)

    checks = [
        Check(
            label=f"Price vs average ({'buy' if buying else 'sell'} side)",
            # A 0.95x factor is the model working, not a failure -- only zeroing the order
            # (conviction <= 0) genuinely blocks.
            ok=conviction > 0.0,
            value=f"{conviction:.2f}x · {abs(z):.1f}σ {'below' if z > 0 else 'above'}",
            limit="",  # No threshold to clear, so no "needs <limit>" to render.
            blocking=conviction <= 0.0,
        ),
        Check(
            label="Budget backlog",
            # ``willingness`` is bounded away from zero by construction, so this never fails --
            # the allowance surfaces as the order size, not as a gate of its own.
            ok=True,
            value=f"{willingness:.2f}x · {_signed_dollars(state.accrued)}",
            limit="",
            blocking=False,
        ),
    ]

    # Only meaningful once the month has deployed something -- a fresh month's "$0 of $1,500"
    # is noise, not a reason.
    budget = abs(float(row["monthly_budget"]))
    if state.deployed_this_month > 0 and budget > 0:
        cap = float(row.get("monthly_cap") or 0.0)
        room = cap - state.deployed_this_month
        checks.append(Check(
            label="Monthly cap room",
            ok=room > 0,
            value=f"${state.deployed_this_month:,.0f} deployed",
            limit=f"≤ ${cap:,.0f}",
            blocking=room <= 0,
        ))

    if not valuation["ok"]:
        checks.append(Check(
            label="Priceable",
            ok=False,
            value=str(valuation.get("reason") or "No price history"),
            limit="a full moving-average window of bars",
            blocking=True,
        ))
        return [check.__dict__ for check in checks]

    # A sell budget trims what is held, never opens a short -- so on the sell side the position
    # itself is a gate the buy side has no counterpart to.
    if not buying:
        held = float(row.get("held") or 0.0)
        checks.append(Check(
            label="Position to trim",
            ok=held > 0.0,
            value=f"{held:g} sh held" if held > 0 else "nothing held",
            limit="> 0 shares",
            blocking=held <= 0.0,
        ))
        if held <= 0.0:
            return [check.__dict__ for check in checks]

    # Only when there is an order to round -- a zeroed conviction already reported its own
    # reason above, so this would otherwise state the same cause twice.
    if size > 0:
        whole_share = not row["fractional"] and float(row["price"]) > 0
        checks.append(Check(
            label="Clears share rounding",
            ok=size >= floor_dollars,
            value=f"${size:,.0f} order",
            limit=f"≥ ${floor_dollars:,.0f}" + (" (one share)" if whole_share else ""),
            blocking=size < floor_dollars,
        ))
    return [check.__dict__ for check in checks]


def signal_view(plan: AlgorithmPlan, *, unknown: list[str]) -> SignalView:
    """Render every configured bucket, whether or not the algorithm is running -- the binding's
    switch controls placing orders, not whether the plan exists."""
    ma_days = int(plan.metadata.get("regime_ma_days") or 0)
    rows = [
        SignalRow(
            symbol=symbol,
            action=str(values["action"]),
            headline=str(values["reason"]),
            metrics=[
                {"label": "Budget", "value": _signed_dollars(values["monthly_budget"]) + "/mo"},
                {"label": "Upcoming", "value": _order_size(values)},
                {"label": "Invested", "value": _invested(values)},
                {"label": "Backlog", "value": _backlog(values)},
                {"label": "Price", "value": _price(values)},
                # The reference the price is scored against, so the gate row's σ figure can be
                # sanity-checked against a chart.
                {"label": _average_label(ma_days), "value": _average(values)},
            ],
            checks=[Check(**check) for check in values["checks"]],
        )
        for symbol, values in plan.signals.items()
    ]
    # No cross-sectional score to sort on -- buckets carry budgets, not scores.
    rows.sort(key=lambda row: (-sum(check.ok for check in row.checks), row.symbol))

    monthly_total = float(plan.metadata.get("monthly_total") or 0.0)
    summary = [
        {"label": "Mode", "value": str(plan.metadata.get("allocation_mode") or "DCA")},
        {"label": "Planned", "value": f"${monthly_total:,.0f}/month"},
        {"label": "Deploying", "value": str(sum(1 for row in rows if row.action == ACTION_ENTER))},
        {"label": "Symbols", "value": str(len(rows))},
        {"label": "Scaling", "value": f"{plan.metadata.get('scaling_factor')}x/σ"},
        {"label": "Relax", "value": f"{plan.metadata.get('relax_months')} months"},
    ]
    if unknown:
        summary.append({"label": "Not tradable", "value": ", ".join(unknown)})
    return SignalView(rows=rows, summary=summary)


def _signed_dollars(amount: float) -> str:
    """``-$60`` rather than ``$-60``: the sign belongs to the amount, not to the currency."""
    return f"-${abs(float(amount)):,.0f}" if float(amount) < 0 else f"${float(amount):,.0f}"


def _order_size(values: dict[str, Any]) -> str:
    size = float(values["next_order"])
    return _signed_dollars(size) if size else "--"


def _invested(values: dict[str, Any]) -> str:
    """What this month has actually deployed, as filled notional -- ``$0`` is a fresh month."""
    return _signed_dollars(float(values["deployed_this_month"]))


def _average_label(ma_days: int) -> str:
    """``150d avg`` rather than ``Average``: the window is the setting a reader would change."""
    return f"{ma_days}d avg" if ma_days > 0 else "Average"


def _average(values: dict[str, Any]) -> str:
    """The moving average the price was scored against, and how far off it the price sits.

    Percent rather than sigma here. The gate row already states the dislocation in sigma --
    the unit the sizing actually uses -- and repeating it would say nothing new; percent is
    the form a reader can check against a chart.
    """
    average = values.get("moving_average")
    if not average:
        return "--"
    distance = float(values.get("distance") or 0.0)
    return f"${float(average):,.2f} ({-distance:+.1%})"


def _backlog(values: dict[str, Any]) -> str:
    """The backlog itself, in budget dollars: positive is lagged, negative is rushed.

    Not the months ratio -- that divides by a setting and reads as a duration -- and not
    words for the sign: sitting between ``Budget`` and ``Invested``, the balance speaks
    for itself and the sign is the whole story.
    """
    return _signed_dollars(float(values["accrued"]))


def _price(values: dict[str, Any]) -> str:
    close = values.get("close")
    if not close:
        return "--"
    return f"${float(close):,.2f}"
