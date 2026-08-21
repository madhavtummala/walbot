"""Step 1: propose a book from market data alone.

Pure in the ``AlgorithmContext`` sense -- no state store, no clock, no brokerage -- which is
what lets the backtester drive the same call the live runner does.
"""

from __future__ import annotations

from .config import RallyRotationConfig
from .layers import defensive_weights, eligibility, park_residual, score_to_weights, sentiment_adjusted, universe_data_ok
from .scoring import base_scores, compute_features



import logging
from typing import Any

import pandas as pd

from ...core.interfaces import AlgorithmContext

logger = logging.getLogger(__name__)




def rank_candidates(
    scored: dict[str, dict[str, Any]],
    config: RallyRotationConfig,
) -> list[dict[str, Any]]:
    """Eligible risk-on names, best first. Ineligible names are never ranked."""
    eligible = []
    for symbol in config.risk_on_universe:
        row = scored.get(symbol)
        if row and row.get("eligible"):
            eligible.append(row)
    eligible.sort(key=lambda row: float(row.get("base_score", 0.0)), reverse=True)
    for position, row in enumerate(eligible, start=1):
        row["rank"] = position
    return eligible


def _selection_reason(row: dict[str, Any], weight: float, data: dict[str, Any], config: RallyRotationConfig) -> str:
    """One line saying why this symbol is or is not held, for the dashboard and the audit."""
    symbol = str(row.get("symbol", ""))
    if weight > 0:
        if symbol in {name.upper() for name in config.defensive_universe}:
            return "Defensive sleeve"
        return f"Rank {int(row.get('rank') or 0)} - held"
    if symbol in {name.upper() for name in config.defensive_universe}:
        return "Defensive sleeve idle" if data.get("data_ok") else "Not the strongest defensive"
    if not data.get("data_ok"):
        return f"Data gap: {data.get('detail', '')}"
    if not row.get("eligible"):
        return str(row.get("eligibility_reason") or "Not eligible")
    if float(row.get("base_score", 0.0)) < config.min_base_score:
        return "Score below quality floor"
    rank = int(row.get("rank") or 0)
    if rank and rank > config.entry_rank_max:
        return f"Rank {rank}, outside entry rank {config.entry_rank_max}"
    return "No slot"


def build_signals(
    scored: dict[str, dict[str, Any]],
    weights: dict[str, float],
    data: dict[str, Any],
    defensive_book: dict[str, float],
    config: RallyRotationConfig,
) -> dict[str, dict[str, Any]]:
    """Per-symbol rows: the dashboard's view, step 2's input, and the audit record.

    Run-level facts (data sufficiency) are denormalised onto every row because
    ``refine`` receives only these signals -- decision metadata does not travel with them. The
    timestamp is not among them any more: ``refine`` takes it as an argument.
    """
    signals: dict[str, dict[str, Any]] = {}
    for symbol, row in scored.items():
        weight = float(weights.get(symbol, 0.0))
        z = row.get("z", {}) if isinstance(row.get("z"), dict) else {}
        signals[symbol] = {
            "signal": 1 if weight > 0 else 0,
            "score": float(row.get("base_score", 0.0)),
            "base_score": float(row.get("base_score", 0.0)),
            "score_unsmoothed": float(row.get("score_unsmoothed", 0.0)),
            "score_components": {key: float(value) for key, value in (row.get("score_components") or {}).items()},
            "rank": int(row.get("rank") or 0),
            "eligible": 1 if row.get("eligible") else 0,
            "eligibility_reason": str(row.get("eligibility_reason") or ""),
            "nano_return": float(row.get("nano_return", 0.0)),
            "micro_return": float(row.get("micro_return", 0.0)),
            "meso_return": float(row.get("meso_return", 0.0)),
            "macro_return": float(row.get("macro_return", 0.0)),
            "abs_return": float(row.get("abs_return", 0.0)),
            "fast_return": float(row.get("fast_return", 0.0)),
            "nano_z": float(z.get("nano", 0.0)),
            "meso_z": float(z.get("meso", 0.0)),
            "macro_z": float(z.get("macro", 0.0)),
            "annual_volatility": float(row.get("annual_volatility", 0.0)),
            "realized_volatility": float(row.get("annual_volatility", 0.0)),
            "target_weight": weight,
            # What this symbol would be worth in the defensive book, so step 2 can build one
            # without re-reading market data.
            "defensive_weight": float(defensive_book.get(symbol, 0.0)),
            "social_score": float(row.get("sentiment_score", 0.0)),
            "close": float(row.get("close", 0.0)),
            # Trend context for the dashboard: the actual MA value, not just the distance.
            "moving_average": float(row.get("moving_average", 0.0)),
            "daily_bars": int(row.get("daily_bars", 0)),
            # Consumed by the dashboard row subtitle; without it every row renders "Inactive".
            "trend_ok": 1 if row.get("above_moving_average") else 0,
            "ma_distance": float(row.get("ma_distance", 0.0)),
            # Intermediate gate inputs: vol, range, volume, trend signals.
            "vol_5d": float(row.get("vol_5d", 0.0)),
            "range_expansion": float(row.get("range_expansion", 0.0)),
            "volume_ratio": float(row.get("volume_ratio", 0.0)),
            "trend_ma_distance": float(row.get("trend_ma_distance", 0.0)),
            "trend_return": float(row.get("trend_return", 0.0)),
            "reason": _selection_reason(row, weight, data, config),
            # -- run-level, repeated per row so refine can read them ---------------------
            "data_ok": 1 if data.get("data_ok") else 0,
            "data_detail": str(data.get("detail", "")),
            "universe_coverage": float(data.get("coverage", 0.0)),
        }
    return signals


def allocation_mode(weights: dict[str, float], config: RallyRotationConfig) -> str:
    """The one-word summary the dashboard prints for this run."""
    defensive = {name.upper() for name in config.defensive_universe}
    held = {symbol for symbol, weight in weights.items() if weight > 0}
    if not held:
        return "Cash"
    return "Defensive" if held <= defensive else "Risk-on"


def analyze_universe(context: AlgorithmContext, config: RallyRotationConfig) -> dict[str, Any]:
    """The whole read-only pipeline: features, scores, eligibility, ranking, weights.

    Returned as a dict rather than assembled inline so tests and the dashboard can inspect
    each layer's output without going through ``AlgorithmDecision``.
    """
    features = {
        symbol: compute_features(
            symbol,
            context.daily_bars_by_symbol.get(symbol, pd.DataFrame()),
            config,
        )
        for symbol in config.symbols
    }
    scored = base_scores(features, config)
    for symbol, row in scored.items():
        row["sentiment_score"] = float(context.sentiment_scores.get(symbol, 0.0))
        ok, reason = eligibility(row, config)
        row["eligible"] = ok
        row["eligibility_reason"] = reason
        if config.sentiment_weight:
            clip = max(config.sentiment_clip, 0.0)
            tilt = max(-clip, min(clip, row["sentiment_score"])) * config.sentiment_weight
            row["base_score"] = float(row["base_score"]) + tilt
            row["score_components"]["sentiment"] = tilt

    data = universe_data_ok(scored, config)
    ranked = rank_candidates(scored, config)

    qualified = [
        row
        for row in ranked
        if int(row.get("rank") or 0) <= config.entry_rank_max
        and float(row.get("base_score", 0.0)) >= config.min_base_score
    ]
    entries = qualified[: max(config.max_positions, 0)]

    # Always computed, whatever this step proposes: step 2 can decide to go defensive for
    # reasons only it can see -- a holding that broke its exit band, a theme that stopped
    # qualifying -- and it cannot derive a defensive book from a risk-on proposal.
    defensive_book = defensive_weights(scored, config)

    if data.get("data_ok") and entries:
        weights = score_to_weights(entries, config)
        weights = sentiment_adjusted(weights, context.sentiment_scores, config)
        # Anything left undeployed goes to bills rather than sitting as cash.
        weights = park_residual(weights, defensive_book, config)
    else:
        weights = dict(defensive_book)

    full_weights = {symbol: float(weights.get(symbol, 0.0)) for symbol in config.symbols}
    return {
        "features": features,
        "scored": scored,
        "data": data,
        "ranked": ranked,
        "entries": entries,
        "defensive_book": defensive_book,
        "weights": full_weights,
    }


# =========================================================================================
# Step 2 helpers: stateful, position-aware
