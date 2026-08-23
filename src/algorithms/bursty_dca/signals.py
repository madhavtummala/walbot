"""What a Bursty DCA run decided about each bucket, and the two factors behind it.

Every amount here is a restatement of something ``plan`` already computed -- nothing is
recalculated, which is what keeps the preview and the order it previews identical rather than
merely similar.

The rows have to work harder than they used to. When sizing was gated, a row could say "did
not fire" and that was the whole explanation; now that valuation and backlog are continuous
multipliers, an order of $180 against a $200 budget is the *product* of two numbers, and a
reader who cannot see both has no way to tell a mildly rich price from a symbol that has
already spent its month. So each factor is reported on its own line with the multiple it
contributed, and only the genuinely hard conditions -- no data, nothing held for a sell to
trim, or an order too small to clear share rounding -- are ever marked blocking.
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
        # Spread first so the explicit keys below win. The valuation detail carries a ``close``
        # of its own -- the last *bar* it scored -- and when this trailed the literal it
        # overwrote the live price the order was actually sized at, so the deck quoted one
        # price beside an order built from another.
        **valuation.get("detail", {}),
        "signal": 1 if buying else -1,
        "side": "LONG" if buying else "SHORT",
        # The size this run would order, signed. The accrued balance used to serve as the score
        # because it was the only thing that separated one row from another; now the order size
        # carries both factors and the budget at once, which is what a reader is ranking by.
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
        # The product of the two, as a multiple of the plain monthly budget: the one number
        # that says how far this run departed from straight DCA, and in which direction.
        "plan_multiple": round(size / budget, 3) if budget > 0 else 0.0,
        # What a run would order right now, signed like the budget, whether or not it is the
        # run that trades in this direction.
        "next_order": round(size if buying else -size, 2),
        "notional": round((size if buying else -size) if row["deployed"] else 0.0, 2),
        "reason": _headline(row),
        "checks": _checks(row),
    }


def _action(row: dict[str, Any]) -> str:
    if row["deployed"]:
        return ACTION_ENTER
    # Nothing to deploy at all is idle; wanting to and being stopped is blocked, and the
    # difference is the one a reader cares about.
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
        # The clamp to what is held zeroed the order before the price had any say, so blaming
        # the valuation here ("Holding off -- 2.1σ above its average") would send the reader
        # off to fix the wrong thing.
        return "Nothing held to sell -- sells trim a position, they never short it"
    if size <= 0:
        # ``conviction`` floored at zero: the price is far enough the wrong way that this
        # bucket wants none of its budget. Say which way, rather than "did not fire".
        return f"Holding off -- {valuation['reason']}"
    if size < floor_dollars:
        # On a whole-share broker the floor is one share, so a small budget against an
        # expensive name is short of one for most of the month and only gets there on the
        # accrued balance. Naming the balance and the wait turns "too small" into something a
        # reader can act on -- raise the budget, or accept the months.
        accrued = float(row["state"].accrued)
        months = (floor_dollars - accrued) / budget if budget > 0 else 0.0
        wait = f", ~{months:.0f} months away" if not row["fractional"] and months >= 1 else ""
        return f"Sizing ${size:.0f} against a ${floor_dollars:.0f} share, ${accrued:.0f} banked{wait}"
    # Reached only by a preview: ``plan`` deploys whenever the size clears the floor, so a row
    # that gets here has already passed every gate and is waiting on the binding's cron.
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
    backlog = float(row["backlog_months"])
    z = float(valuation.get("detail", {}).get("z") or 0.0)

    checks = [
        Check(
            label=f"Price vs average ({'buy' if buying else 'sell'} side)",
            # A 0.95x factor is the model working, not a failure. Reading ``ok`` as "clears
            # 1.00x" printed FAIL on every day a symbol traded above its average -- roughly
            # half of them -- which trains a reader to ignore the column. It is false only
            # when the factor actually zeroes the order, which is the one case worth seeing,
            # and that case genuinely blocks.
            ok=conviction > 0.0,
            value=f"{conviction:.2f}x · {abs(z):.1f}σ {'below' if z > 0 else 'above'}",
            # No limit. The deck renders one as "needs <limit>", which is the wrong verb for a
            # factor with no threshold to clear -- "PASS / 0.83x / needs 1.00x" states a pass
            # and a shortfall in the same breath. The label already says 1.00x is the average.
            limit="",
            blocking=conviction <= 0.0,
        ),
        Check(
            label="Budget backlog",
            # ``willingness`` is bounded away from zero by construction, so this row reports a
            # multiple and never fails. The allowance is what can actually stop a run, and it
            # surfaces as the order size rather than as a gate of its own.
            ok=True,
            # Signed months, because "0.8 months behind" and "0.8 months ahead" pull the
            # multiple in opposite directions and read identically without the word.
            value=f"{willingness:.2f}x · {abs(backlog):.1f}mo "
                  + ("banked" if backlog >= 0 else "over"),
            limit="",
            blocking=False,
        ),
    ]

    # Only meaningful once the month has actually deployed something; on a fresh month it is a
    # row saying "$0 of $1,500", which is noise rather than a reason.
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

    # A sell budget trims what is actually held -- it never opens a short -- so on the sell
    # side the position itself is a gate the buy side has no counterpart to. Without this row
    # a sell symbol with no inventory passed every visible test and still went nowhere, which
    # read as a broken deck rather than as an empty book.
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

    # Only when there is an order to round. A zeroed conviction already reported itself as the
    # reason on the row above, and appending "FAIL -- $0 order needs >= $160" beneath it states
    # one cause twice, as though two separate things had gone wrong.
    if size > 0:
        # The floor is one share on a whole-share brokerage, which is the entire reason a small
        # budget against an expensive name cannot trade every run. Saying so on the check itself
        # beats a separate warning row, which rendered as a second failure on symbols trading fine.
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
    """Render every configured bucket, whether or not the algorithm is running.

    The binding's switch controls whether orders are placed, not whether the plan exists -- a
    view that went blank when the algorithm was off would give no way to check what it would do
    before turning it on.
    """
    quotes = dict(plan.metadata.get("price_quotes") or {})
    ma_days = int(plan.metadata.get("regime_ma_days") or 0)
    rows = [
        SignalRow(
            symbol=symbol,
            action=str(values["action"]),
            headline=str(values["reason"]),
            metrics=[
                {"label": "Budget", "value": _signed_dollars(values["monthly_budget"]) + "/mo"},
                {"label": "Next order", "value": _order_size(values)},
                # ``Next order`` over ``Budget``, so the two columns it is derived from sit
                # beside it. Named for the budget rather than "the plan", which in this
                # algorithm also names the whole bubble board.
                {"label": "vs budget", "value": f"{float(values['plan_multiple']):.2f}x"},
                {"label": "Backlog", "value": _backlog(values)},
                {"label": "Price", "value": _price(values, quotes.get(symbol))},
                # The reference the price is scored against. Without it the σ figure on the
                # gate row is unfalsifiable -- a reader can see "0.4σ above average" but not
                # what the average was, so there is no way to sanity-check it against a chart.
                {"label": _average_label(ma_days), "value": _average(values)},
            ],
            checks=[Check(**check) for check in values["checks"]],
        )
        for symbol, values in plan.signals.items()
    ]
    # No cross-sectional score to sort on -- buckets carry budgets, not scores -- so the
    # shared reading order reduces to gates cleared, then the alphabet: rows that passed
    # more of their checks sit nearer the top.
    rows.sort(key=lambda row: (-sum(check.ok for check in row.checks), row.symbol))

    monthly_total = float(plan.metadata.get("monthly_total") or 0.0)
    summary = [
        {"label": "Mode", "value": str(plan.metadata.get("allocation_mode") or "DCA")},
        # The configured monthly total, not what happens to be deployable this minute.
        {"label": "Planned", "value": f"${monthly_total:,.0f}/month"},
        {"label": "Deploying", "value": str(sum(1 for row in rows if row.action == ACTION_ENTER))},
        {"label": "Symbols", "value": str(len(rows))},
        {"label": "Scaling", "value": f"{plan.metadata.get('scaling_factor')}x/σ"},
        {"label": "Relax", "value": f"{plan.metadata.get('relax_months')} months"},
    ]
    # No "Prices" chip. Staleness is a per-symbol fact and it is already marked on the row that
    # is priced by it; a header chip listing the stale names only restated the rows below it.
    if unknown:
        summary.append({"label": "Not tradable", "value": ", ".join(unknown)})
    return SignalView(rows=rows, summary=summary)


def _signed_dollars(amount: float) -> str:
    """``-$60`` rather than ``$-60``: the sign belongs to the amount, not to the currency."""
    return f"-${abs(float(amount)):,.0f}" if float(amount) < 0 else f"${float(amount):,.0f}"


def _order_size(values: dict[str, Any]) -> str:
    size = float(values["next_order"])
    return _signed_dollars(size) if size else "--"


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
    """Signed months of budget, worded rather than signed with a minus.

    ``-0.8`` and ``0.8`` months pull the size in opposite directions, and a bare minus sign in
    a column of dollar amounts reads as a short rather than as a debt.
    """
    months = float(values["backlog_months"])
    return f"{abs(months):.1f}mo " + ("banked" if months >= 0 else "over")


def _price(values: dict[str, Any], quote: dict[str, Any] | None) -> str:
    close = values.get("close")
    if not close:
        return "--"
    # "stale", not "delayed": ``current: False`` means the price came from the DuckDB bar cache
    # via ``prices_from_store`` rather than from a live quote, so it is the last stored close --
    # not a lagging live feed. Marked only when the feed actually said so, because treating
    # *absent* provenance as stale labelled every price on any path that records no quote
    # metadata (a backtest, a replay), which is the same as not marking them at all.
    stale = quote is not None and not quote.get("current")
    return f"${float(close):,.2f}" + (" (stale)" if stale else "")
