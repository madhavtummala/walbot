"""What a Rally Rotation run decided about each name, and the gates behind it.

Rows are published on the plan and read by the deck, an MCP agent, and the backtest alike, so
gate results travel as data rather than as a sentence -- a string can be shown but not audited.
Run-level facts are denormalised onto every row, since the hold/rotate pass reads these signals
and nothing else.
"""

from __future__ import annotations

from typing import Any

from ...core.interfaces import (
    ACTION_BLOCKED,
    ACTION_ENTER,
    ACTION_EXIT,
    ACTION_HOLD,
    ACTION_IDLE,
    AlgorithmPlan,
    Check,
    SignalRow,
    SignalView,
)
from .config import RallyRotationConfig
from .gates import blocking, entry_checks, mark_blocking
from ..base import _score_of


def build_signals(
    scored: dict[str, dict[str, Any]],
    data: dict[str, Any],
    defensive_book: dict[str, float],
    config: RallyRotationConfig,
) -> dict[str, dict[str, Any]]:
    """The per-symbol record, before the hold/rotate pass has decided anything.

    Weights and the final action are filled in afterwards by :func:`finalize`, because they are
    not knowable until the book is known.
    """
    return {
        symbol: {
            "signal": 0,
            "score": float(row.get("base_score", 0.0)),
            "base_score": float(row.get("base_score", 0.0)),
            "score_components": {k: float(v) for k, v in (row.get("score_components") or {}).items()},
            "rank": int(row.get("rank") or 0),
            "eligible": 1 if row.get("eligible") else 0,
            "close": float(row.get("close", 0.0)),
            "moving_average": float(row.get("moving_average", 0.0)),
            "ma_distance": float(row.get("ma_distance", 0.0)),
            "annual_volatility": float(row.get("annual_volatility", 0.0)),
            "realized_volatility": float(row.get("annual_volatility", 0.0)),
            "abs_return": float(row.get("abs_return", 0.0)),
            "fast_return": float(row.get("fast_return", 0.0)),
            "daily_bars": int(row.get("daily_bars", 0)),
            "enough_history": bool(row.get("enough_history")),
            "above_moving_average": bool(row.get("above_moving_average")),
            # What this symbol would be worth in the defensive book, so the hold/rotate pass can
            # build one without re-reading market data.
            "defensive_weight": float(defensive_book.get(symbol, 0.0)),
            "target_weight": 0.0,
            # -- run-level, repeated per row so the hold/rotate pass can read them ------------
            "data_ok": 1 if data.get("data_ok") else 0,
            "data_detail": str(data.get("detail", "")),
            "universe_coverage": float(data.get("coverage", 0.0)),
        }
        for symbol, row in scored.items()
    }


def finalize(
    signals: dict[str, dict[str, Any]],
    weights: dict[str, float],
    held: set[str],
    config: RallyRotationConfig,
    notes: dict[str, list[Check]],
) -> dict[str, dict[str, Any]]:
    """Attach the decision and the gates that justify it, once the book is settled.

    ``notes`` carries what the selection pass decided (settling period, slot contest, re-rank
    throttle) -- not recoverable from the market gates alone, since a name can clear every one
    and still be turned away.
    """
    defensive = {name.upper() for name in config.defensive_universe}
    for symbol, row in signals.items():
        weight = float(weights.get(symbol, 0.0))
        was_held = symbol in held
        row["target_weight"] = weight
        row["signal"] = 1 if weight > 0 else 0
        is_defensive = symbol in defensive
        row["defensive"] = is_defensive
        # One gate list for held and unheld alike: a name that fails it is dropped from the
        # ranking and therefore sold, so a held name is explained against the same band. The
        # defensive sleeve gets no gates -- it's held precisely when nothing else qualifies, so
        # a momentum score decides nothing about it.
        if is_defensive:
            checks: list[Check] = []
        else:
            market = entry_checks(row, config)
            checks = mark_blocking([*market, *notes.get(symbol, [])])
        action = _action(weight, was_held, is_defensive, bool(row["eligible"]))
        row["action"] = action
        row["checks"] = [check.__dict__ for check in checks]
        row["reason"] = _headline(action, row, checks, is_defensive)
    return signals


def _action(weight: float, was_held: bool, defensive: bool, eligible: bool) -> str:
    if weight > 0:
        return ACTION_HOLD if was_held else ACTION_ENTER
    if was_held:
        return ACTION_EXIT
    # Wanting in and being stopped is the interesting case; never having qualified is not.
    return ACTION_BLOCKED if eligible and not defensive else ACTION_IDLE


def _headline(action: str, row: dict[str, Any], checks: list[Check], defensive: bool) -> str:
    """One line: the gate that decided it, or -- when nothing was in the way -- what it did.

    Order matters. An unusable universe overrides every per-name verdict below it, and the
    defensive sleeve is not selected by any of these gates at all, so both are answered first.
    """
    if not row["data_ok"]:
        return f"Data gap: {row['data_detail']}"
    if defensive:
        return "Defensive sleeve" if row["target_weight"] > 0 else "Defensive sleeve idle"
    if culprit := blocking(checks):
        return f"{culprit.label}: {culprit.value}"
    rank = int(row["rank"] or 0)
    if action == ACTION_HOLD:
        return f"Rank {rank} - held" if rank else "Held"
    if action == ACTION_ENTER:
        return f"Rank {rank} - opening" if rank else "Opening"
    if action == ACTION_EXIT:
        # Every gate passed and it's still leaving -- only the sizing throttle does that.
        return "Closed below the minimum trade size"
    return "Eligible, not selected"


def signal_view(plan: AlgorithmPlan) -> SignalView:
    """Rows best-first: score, then gates cleared, then the alphabet."""
    rows = [
        SignalRow(
            symbol=symbol,
            action=str(values.get("action") or ACTION_IDLE),
            headline=str(values.get("reason") or ""),
            metrics=_metrics(values, plan.latest_prices.get(symbol)),
            checks=[Check(**check) for check in values.get("checks") or []],
        )
        for symbol, values in plan.signals.items()
    ]
    return SignalView(
        rows=sorted(
            rows,
            key=lambda row: (-_score_of(row), -sum(check.ok for check in row.checks), row.symbol),
        ),
        summary=_summary(plan, rows),
    )


def _metrics(values: dict[str, Any], latest_price: Any = None) -> list[dict[str, str]]:
    """Score, rank and volatility are cross-sectional facts about a *candidate* -- the defensive
    sleeve isn't one, so those three are dashed out for it; its weight is real and stays. Price
    is shown for every row: the plan's live ``latest_prices`` preferred, falling back to the
    row's last daily close for a holding outside the universe.
    """
    weight = {"label": "Weight", "value": f"{float(values['target_weight']):.1%}"}
    priced = float(latest_price or 0.0) or float(values.get("close") or 0.0)
    price = {"label": "Price", "value": f"${priced:,.2f}" if priced > 0 else "--"}
    if values.get("defensive"):
        return [
            {"label": "Score", "value": "--"},
            {"label": "Rank", "value": "--"},
            weight,
            price,
            {"label": "Vol", "value": "--"},
        ]
    return [
        {"label": "Score", "value": f"{float(values['score']):.2f}"},
        {"label": "Rank", "value": str(values["rank"] or "--")},
        weight,
        price,
        {"label": "Vol", "value": f"{float(values['annual_volatility']):.0%}"},
    ]


def _summary(plan: AlgorithmPlan, rows: list[SignalRow]) -> list[dict[str, str]]:
    exposure = sum(weight for weight in plan.target_weights.values() if weight > 0)
    summary = [
        {"label": "Allocation", "value": str(plan.metadata.get("allocation_mode") or "--")},
        {"label": "Exposure", "value": f"{exposure:.0%}"},
        {"label": "Held", "value": str(sum(1 for row in rows if row.action in (ACTION_HOLD, ACTION_ENTER)))},
        {"label": "Eligible", "value": str(int(plan.metadata.get("eligible_count") or 0))},
    ]
    for label, action in (("Entering", ACTION_ENTER), ("Exiting", ACTION_EXIT)):
        if count := sum(1 for row in rows if row.action == action):
            summary.append({"label": label, "value": str(count)})

    data = plan.metadata.get("universe_data") or {}
    if not data.get("data_ok", True):
        summary.append({"label": "Data", "value": str(data.get("detail") or "insufficient")})
    summary.append({"label": "Universe", "value": str(len(rows))})
    return summary
