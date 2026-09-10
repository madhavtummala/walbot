"""What an Options Flip run decided, per symbol.

Every configured symbol gets a row every run, including the ones doing nothing -- "no direction",
"direction but gates disagreed" and "confirmed but nothing tradable" are different silences a
positions-only view would render identically.
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
    return SignalView(rows=rows, summary=_summary(rows))


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
    """What the row needs to be judged, not merely watched: contract, price, band, profit."""
    estimate = signal.get("estimate") or {}
    metrics: list[dict[str, str]] = []
    if not estimate.get("contract") and (direction := str(signal.get("direction") or "")):
        metrics.append({"label": "Direction", "value": direction.upper()})
    if not estimate.get("contract"):
        return metrics

    mark = float(estimate.get("mark", 0.0) or 0.0)
    metrics.append({"label": "Contract", "value": str(estimate.get("contract_label") or "")})
    metrics.append({"label": "Price", "value": f"${mark:.2f}"})

    # "Band" is always in the currency the row's Price is quoted in, so the two can be read
    # against each other. For a position already open that is premium -- what it cost and what
    # the resting sell order is asking; before entry it is the underlying, because that is
    # where the entry and target levels live and where the reach probabilities are measured.
    entry_premium = float(estimate.get("entry_premium", 0.0) or 0.0)
    target_premium = float(estimate.get("target_premium", 0.0) or 0.0)
    if entry_premium > 0 or target_premium > 0:
        metrics.append({
            "label": "Band",
            "value": f"${entry_premium:,.2f} → ${target_premium:,.2f}",
        })

    entry = float(estimate.get("entry_underlying", 0.0) or 0.0)
    target = float(estimate.get("target_underlying", 0.0) or 0.0)
    if target > 0:
        # The underlying band the premium one is derived from. Shown alongside rather than
        # instead of it: the premium is what the order asks, this is why it asks that.
        reach = (
            f" ({float(estimate.get('p_touch', 0.0)):.0%}/{float(estimate.get('p_target', 0.0)):.0%})"
            if estimate.get("p_touch") or estimate.get("p_target") else ""
        )
        metrics.append({
            "label": "Band" if not (entry_premium > 0 or target_premium > 0) else "Underlying",
            "value": f"${entry:,.2f} → ${target:,.2f}{reach}",
        })

    # Gross, and assumes both ends fill -- read as the band's width in dollars, not a promise.
    # On a held row it prices the resting ask against what the position cost, which is the
    # question that row exists to answer.
    metrics.append({
        "label": "Est Profit",
        "value": f"${float(estimate.get('expected_profit', 0.0)):,.0f}",
    })

    return metrics


def _summary(rows: list[SignalRow]) -> list[dict[str, str]]:
    """The header strip: label/value chips, matching what every other algorithm returns."""
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
