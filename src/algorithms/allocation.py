"""Turning scored candidates into portfolio weights.

Every score-ranked allocation algorithm here does the same three things: filter a universe down
to what qualifies, spread an exposure budget across the survivors in proportion to score, and
hold the total inside a gross cap. Those were written out separately in each algorithm -- twice
for ranking, twice for allocation, four times for the gross cap -- which is how the two copies
of the allocator came to differ on what happens when a name hits its per-name cap.
"""

from __future__ import annotations

from typing import Any, Iterable


def rank_by_score(
    scores_by_symbol: dict[str, dict[str, Any]],
    symbols: Iterable[str],
    *,
    min_score: float,
    gate_key: str | None = None,
    min_gate: float | None = None,
    require_trend: bool = True,
) -> list[dict[str, Any]]:
    """The rows from ``symbols`` that qualify, best score first.

    ``gate_key``/``min_gate`` is the second, raw-return floor an algorithm may apply alongside
    the cross-sectional score. It is a sanity check rather than a ranking term: it stops the
    best of a uniformly falling universe being bought for being least bad.

    ``require_trend`` reads ``macro_trend_ok``. Defensive sleeves switch it off, because the
    point of holding bills is that nothing else is trending.
    """
    candidates = [
        row
        for symbol in symbols
        if (row := scores_by_symbol.get(symbol)) is not None
        and (not require_trend or bool(row.get("macro_trend_ok")))
        and float(row.get("score", 0.0)) >= min_score
        and (
            min_gate is None
            or gate_key is None
            or float(row.get(gate_key, 0.0)) >= min_gate
        )
    ]
    return sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)


def allocate_by_score(
    candidates: list[dict[str, Any]],
    exposure: float,
    max_positions: int,
    max_weight: float,
) -> dict[str, float]:
    """Spread ``exposure`` across the top ``max_positions`` candidates, in proportion to score.

    A name whose proportional share exceeds ``max_weight`` takes the cap, and the residual is
    re-spread across the rest rather than left as cash -- the cap is a concentration limit, not
    an instruction to under-invest. Repeats until no further name is capped, since capping one
    can push another over.

    Non-positive scores contribute nothing, so a set with no positive score at all is split
    evenly rather than divided by zero.
    """
    remaining = list(candidates)[: max(max_positions, 0)]
    weights: dict[str, float] = {}
    budget = max(exposure, 0.0)

    while remaining and budget > 0:
        scores = [max(float(row.get("score", 0.0)), 0.0) for row in remaining]
        total = sum(scores)
        shares = [
            (row, budget / len(remaining) if total <= 0 else budget * score / total)
            for row, score in zip(remaining, scores)
        ]
        capped = [(row, share) for row, share in shares if share >= max_weight]
        if not capped:
            for row, share in shares:
                weights[str(row["symbol"])] = max(0.0, min(share, max_weight))
            break
        for row, _share in capped:
            weights[str(row["symbol"])] = max_weight
            budget -= max_weight
        remaining = [row for row, share in shares if share < max_weight]

    return weights


def scale_to_gross(weights: dict[str, float], max_gross: float) -> dict[str, float]:
    """Shrink ``weights`` proportionally if their gross exceeds ``max_gross``.

    Gross rather than net, so a short leg counts toward the limit instead of offsetting the
    long one. A non-positive cap means no cap.
    """
    gross = sum(abs(weight) for weight in weights.values())
    if max_gross > 0 and gross > max_gross:
        scale = max_gross / gross
        return {symbol: weight * scale for symbol, weight in weights.items()}
    return dict(weights)
