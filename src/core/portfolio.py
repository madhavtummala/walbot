from __future__ import annotations

from ..common.config_utils import as_float


def _allocate_with_caps(raw_scores: dict[str, float], max_weight: float, target_exposure: float) -> dict[str, float]:
    if not raw_scores or target_exposure <= 0:
        return {symbol: 0.0 for symbol in raw_scores}

    remaining = {symbol: max(score, 0.0) for symbol, score in raw_scores.items() if score > 0}
    weights: dict[str, float] = {}
    remaining_exposure = target_exposure

    while remaining and remaining_exposure > 0:
        total_score = sum(remaining.values())
        if total_score <= 0:
            break

        capped_this_round = False
        for symbol, score in list(remaining.items()):
            proposed_weight = (score / total_score) * remaining_exposure
            if proposed_weight > max_weight:
                weights[symbol] = max_weight
                remaining_exposure -= max_weight
                del remaining[symbol]
                capped_this_round = True

        if not capped_this_round:
            for symbol, score in remaining.items():
                weights[symbol] = (score / total_score) * remaining_exposure
            break

    return weights


def compute_target_weights(
    signals: dict[str, dict[str, float | int]],
    max_weight_per_symbol: float,
    max_portfolio_exposure: float = 1.0,
    max_longs: int | None = None,
    target_annual_vol: float | None = None,
) -> dict[str, float]:
    """Compute score/risk-weighted long targets from composite signals."""
    if not signals:
        return {}

    candidates: list[tuple[str, float, float]] = []
    for symbol, info in signals.items():
        if int(info.get("signal", 0)) != 1:
            continue

        score = max(as_float(info.get("score", info.get("ret_N", 0.0))), 0.0)
        if score <= 0:
            continue

        realized_vol = as_float(info.get("realized_vol"), default=0.20)
        realized_vol = max(realized_vol, 0.05)
        risk_adjusted_score = score / realized_vol
        candidates.append((symbol, risk_adjusted_score, realized_vol))

    if not candidates:
        return {symbol: 0.0 for symbol in signals.keys()}

    candidates.sort(key=lambda item: item[1], reverse=True)
    if max_longs is not None and max_longs > 0:
        candidates = candidates[:max_longs]

    selected_vols = {symbol: realized_vol for symbol, _, realized_vol in candidates}
    raw_scores = {symbol: risk_adjusted_score for symbol, risk_adjusted_score, _ in candidates}
    target_exposure = min(max_portfolio_exposure, 1.0)
    allocated = _allocate_with_caps(raw_scores, max_weight_per_symbol, target_exposure)

    if target_annual_vol is not None and target_annual_vol > 0:
        estimated_vol = sum(allocated.get(symbol, 0.0) * selected_vols[symbol] for symbol in allocated)
        if estimated_vol > target_annual_vol:
            vol_scale = target_annual_vol / estimated_vol
            allocated = {symbol: weight * vol_scale for symbol, weight in allocated.items()}

    return {symbol: allocated.get(symbol, 0.0) for symbol in signals.keys()}


