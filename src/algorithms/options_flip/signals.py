"""What an Options Flip run decided, per symbol.

Every configured symbol gets a row every run, including the ones doing nothing. That is
deliberate and it is the main thing this view is for: most sessions most symbols will not trade,
and "no direction" and "direction but pre-market disagreed" and "everything confirmed but the
chain had nothing tradable" are three very different silences. A view that only showed the
symbols with orders would render all three identically, as absence.
"""

from __future__ import annotations

from typing import Any

from ...core.interfaces import (
    ACTION_BLOCKED,
    ACTION_ENTER,
    ACTION_HOLD,
    ACTION_IDLE,
    AlgorithmPlan,
    Check,
    SignalRow,
    SignalView,
)
from .lifecycle import BIDDING, HELD


def signal_view(plan: AlgorithmPlan) -> SignalView:
    """One row per symbol, ordered so the positions and the near-misses come first."""
    rows = [_row(symbol, signal) for symbol, signal in sorted(plan.signals.items())]
    rows.sort(key=lambda row: (_ORDER.get(row.action, 9), row.symbol))
    return SignalView(rows=rows, summary=_summary(rows, plan))


_ORDER = {ACTION_HOLD: 0, ACTION_ENTER: 1, ACTION_BLOCKED: 2, ACTION_IDLE: 3}


def _row(symbol: str, signal: dict[str, Any]) -> SignalRow:
    state = str(signal.get("state") or "")
    checks = [_check(raw) for raw in signal.get("checks") or []]
    if state == HELD:
        action = ACTION_HOLD
    elif state == BIDDING:
        action = ACTION_ENTER
    elif any(check.blocking for check in checks):
        action = ACTION_BLOCKED
    else:
        action = ACTION_IDLE
    return SignalRow(
        symbol=symbol,
        action=action,
        headline=str(signal.get("headline") or ""),
        metrics=_metrics(signal),
        checks=checks,
    )


def _metrics(signal: dict[str, Any]) -> list[dict[str, str]]:
    """What the row needs to be *judged*, not merely watched.

    Reported even on a run that places nothing, which is the point: on most sessions this
    algorithm stands down, and "no trade today" tells you nothing about whether the setup was
    close or hopeless. The contract, its cost, and the band it expects to transact in do.

    Kept to four columns, because every extra one squeezes the reason text the row is built
    around. The spread is not among them: it is no longer a gate, and it is already visible on the
    chosen-contract line in the expanded panel.
    """
    estimate = signal.get("estimate") or {}
    metrics: list[dict[str, str]] = []
    # No Direction column: the contract label already reads "$88 call", so a CALL/PUT column
    # beside it says the same word twice. The only case it carried anything extra was a symbol
    # with a direction but no contract, and that row now shows the pre-market reading instead.
    if not estimate.get("contract") and (direction := str(signal.get("direction") or "")):
        metrics.append({"label": "Direction", "value": direction.upper()})

    if premarket := signal.get("premarket_change"):
        metrics.append({"label": "Pre-market", "value": f"{float(premarket):+.2%}"})
    # Guarded on the contract, not on the dict: direction and pre-market are merged in even when
    # no contract was found, so the dict is never empty.
    if not estimate.get("contract"):
        return metrics

    mark = float(estimate.get("mark", 0.0) or 0.0)
    # No "(not tradable)" suffix: the row already carries a BLOCKED badge and the gate list says
    # exactly which floor it missed and by how much. Repeating it here is a third telling.
    metrics.append({"label": "Contract", "value": str(estimate.get("contract_label") or "")})
    # The contract's current mid price, per share. Not multiplied out: ×100 is a fixed property
    # of every listed option and the contract count is a config setting, so the product carried
    # no information the reader did not already have.
    metrics.append({"label": "Price", "value": f"${mark:.2f}"})
    # Low and high in one cell, and without the dollar totals beside each: the totals are the
    # per-contract figure times a round number that is already in the Cost column.
    metrics.append({
        "label": "Est. band",
        "value": f"${float(estimate.get('estimated_low', 0.0)):.2f} – ${float(estimate.get('estimated_high', 0.0)):.2f}",
    })

    # Gross. It assumes *both* ends fill -- entry at the predicted low, exit at the predicted
    # high -- when neither is guaranteed and the exit may hit the stop instead, so read it as the
    # band's width in dollars rather than as an expectation.
    metrics.append({
        "label": "Est Profit",
        "value": f"${float(estimate.get('expected_profit', 0.0)):,.0f}",
    })

    if (fill := float(signal.get("fill_price", 0.0) or 0.0)) > 0:
        metrics.append({"label": "Fill", "value": f"${fill:.2f}"})
    return metrics


def _summary(rows: list[SignalRow], plan: AlgorithmPlan) -> list[dict[str, str]]:
    """The header strip: label/value chips, matching what every other algorithm returns.

    A list rather than a sentence because that is what :class:`SignalView` declares and what the
    deck renders -- it maps over these to build the metric row, and a string silently becomes
    ``payload.summary.map is not a function``, which empties the whole panel rather than just
    the strip.
    """
    held = sum(1 for row in rows if row.action == ACTION_HOLD)
    bidding = sum(1 for row in rows if row.action == ACTION_ENTER)
    blocked = sum(1 for row in rows if row.action == ACTION_BLOCKED)

    summary = [
        {"label": "Symbols", "value": str(len(rows))},
        {"label": "Held", "value": str(held)},
        {"label": "Bidding", "value": str(bidding)},
    ]
    if blocked:
        summary.append({"label": "Blocked", "value": str(blocked)})
    # Deliberately no aggregate "why flat". The strip is counts across the whole book, and one
    # symbol's blocking gate is not a fact about the book -- with several symbols it names
    # whichever happened to sort first. Each row already carries its own reason, measured.
    return summary


def _check(raw: Any) -> Check:
    """Checks travel through ``plan.signals`` as plain dicts, since the plan must stay JSON-able."""
    if isinstance(raw, Check):
        return raw
    data = dict(raw or {})
    return Check(
        label=str(data.get("label", "")),
        ok=bool(data.get("ok", False)),
        value=str(data.get("value", "")),
        limit=str(data.get("limit", "")),
        blocking=bool(data.get("blocking", False)),
    )
