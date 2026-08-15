"""Step 1: propose a book from market data alone.

Pure in the ``AlgorithmContext`` sense -- no state store, no clock, no brokerage -- which is
what lets the backtester drive the same call the live runner does.
"""

from __future__ import annotations

from .config import DualMomentumConfig
from .layers import covariance_matrix, defensive_weights, eligibility, market_regime, score_to_weights, sentiment_adjusted, timing, volatility_scale
from .scoring import base_scores, compute_features



import logging
from datetime import datetime
from typing import Any

import pandas as pd

from ...core.interfaces import AlgorithmContext

logger = logging.getLogger(__name__)

STATE_KEY = "dual_momentum_runtime"

EPSILON = 1e-9

#: Trading days per year, for annualising a daily volatility estimate.
TRADING_DAYS = 252




def rank_candidates(
    scored: dict[str, dict[str, Any]],
    config: DualMomentumConfig,
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


def _selection_reason(row: dict[str, Any], weight: float, regime: dict[str, Any], config: DualMomentumConfig) -> str:
    """One line saying why this symbol is or is not held, for the dashboard and the audit."""
    symbol = str(row.get("symbol", ""))
    if weight > 0:
        if symbol in {name.upper() for name in config.defensive_universe}:
            return "Defensive sleeve"
        return f"Rank {int(row.get('rank') or 0)} - {row.get('timing_reason') or 'held'}"
    if symbol == config.benchmark and symbol not in config.risk_on_universe:
        return "Benchmark only"
    if symbol in {name.upper() for name in config.defensive_universe}:
        return "Defensive sleeve idle" if regime.get("risk_on") else "Not the strongest defensive"
    if not regime.get("risk_on"):
        return f"Risk-off: {regime.get('detail', '')}"
    if not row.get("eligible"):
        return str(row.get("eligibility_reason") or "Not eligible")
    if float(row.get("base_score", 0.0)) < config.min_base_score:
        return "Score below quality floor"
    rank = int(row.get("rank") or 0)
    if rank and rank > config.entry_rank_max:
        return f"Rank {rank}, outside entry rank {config.entry_rank_max}"
    if not row.get("timing"):
        return str(row.get("timing_reason") or "Waiting for entry timing")
    return "No slot"


def build_signals(
    scored: dict[str, dict[str, Any]],
    weights: dict[str, float],
    regime: dict[str, Any],
    vol: dict[str, Any],
    covariance: dict[str, dict[str, float]],
    defensive_book: dict[str, float],
    as_of: datetime,
    config: DualMomentumConfig,
) -> dict[str, dict[str, Any]]:
    """Per-symbol rows: the dashboard's view, step 2's input, and the audit record.

    Run-level facts (regime, volatility scale, timestamp) are denormalised onto every row
    because ``refine`` receives only these signals -- decision metadata does not travel with
    them.
    """
    signals: dict[str, dict[str, Any]] = {}
    for symbol, row in scored.items():
        weight = float(weights.get(symbol, 0.0))
        z = row.get("z", {}) if isinstance(row.get("z"), dict) else {}
        signals[symbol] = {
            "signal": 1 if weight > 0 else 0,
            "score": float(row.get("base_score", 0.0)),
            "base_score": float(row.get("base_score", 0.0)),
            "score_components": {key: float(value) for key, value in (row.get("score_components") or {}).items()},
            "rank": int(row.get("rank") or 0),
            "eligible": 1 if row.get("eligible") else 0,
            "eligibility_reason": str(row.get("eligibility_reason") or ""),
            "timing": 1 if row.get("timing") else 0,
            "timing_reason": str(row.get("timing_reason") or ""),
            "momentum_change": float(row.get("momentum_change", 0.0)),
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
            "covariance_row": {
                str(other): float(value) for other, value in (covariance.get(symbol) or {}).items()
            },
            "target_weight": weight,
            # What this symbol would be worth in the defensive book, so step 2 can build one
            # without re-reading market data.
            "defensive_weight": float(defensive_book.get(symbol, 0.0)),
            "social_score": float(row.get("sentiment_score", 0.0)),
            "close": float(row.get("close", 0.0)),
            # Consumed by the dashboard row subtitle; without it every row renders "Inactive".
            "trend_ok": 1 if row.get("above_moving_average") else 0,
            "reason": _selection_reason(row, weight, regime, config),
            # -- run-level, repeated per row so refine can read them ---------------------
            "regime_risk_on": 1 if regime.get("risk_on") else 0,
            "regime_detail": str(regime.get("detail", "")),
            "regime_breadth": float(regime.get("breadth", 0.0)),
            "vol_scale": float(vol.get("scale", 1.0)),
            "portfolio_volatility": float(vol.get("portfolio_volatility", 0.0)),
            "vol_below_floor": 1 if vol.get("below_floor") else 0,
            "as_of": as_of.isoformat(),
        }
    return signals


def allocation_mode(weights: dict[str, float], regime: dict[str, Any], config: DualMomentumConfig) -> str:
    """The one-word summary the dashboard prints for this run."""
    defensive = {name.upper() for name in config.defensive_universe}
    held = {symbol for symbol, weight in weights.items() if weight > 0}
    if not held:
        return "Cash"
    if held <= defensive:
        return "Defensive"
    return "Risk-on" if regime.get("risk_on") else "Risk-on (unconfirmed)"


def analyze_universe(context: AlgorithmContext, config: DualMomentumConfig) -> dict[str, Any]:
    """The whole read-only pipeline: features, regime, eligibility, ranking, timing, weights.

    Returned as a dict rather than assembled inline so tests and the dashboard can inspect
    each layer's output without going through ``AlgorithmDecision``.
    """
    features = {
        symbol: compute_features(
            symbol,
            context.history_bars_by_symbol.get(symbol, pd.DataFrame()),
            context.bars_by_symbol.get(symbol, pd.DataFrame()),
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
        timing_ok, timing_reason = timing(row, config)
        row["timing"] = timing_ok
        row["timing_reason"] = timing_reason
        if config.sentiment_weight:
            clip = max(config.sentiment_clip, 0.0)
            tilt = max(-clip, min(clip, row["sentiment_score"])) * config.sentiment_weight
            row["base_score"] = float(row["base_score"]) + tilt
            row["score_components"]["sentiment"] = tilt

    regime = market_regime(scored, config)
    ranked = rank_candidates(scored, config)

    entries = [
        row
        for row in ranked
        if int(row.get("rank") or 0) <= config.entry_rank_max
        and float(row.get("base_score", 0.0)) >= config.min_base_score
        and row.get("timing")
    ][: max(config.max_positions, 0)]

    covariance = covariance_matrix(context.bars_by_symbol, config.symbols, config)
    # Always computed, whatever the regime says: step 2 can decide to go defensive for
    # reasons only it can see (a pending regime confirmation, the drawdown breaker), and it
    # cannot derive a defensive book from a risk-on proposal.
    defensive_book = defensive_weights(scored, config)

    if regime.get("risk_on") and entries:
        weights = score_to_weights(entries, config)
        weights = sentiment_adjusted(weights, context.sentiment_scores, config)
        vol = volatility_scale(weights, covariance, config)
        if vol["below_floor"]:
            weights = dict(defensive_book)
            vol = {**vol, "scale": 1.0}
        else:
            weights = {symbol: weight * vol["scale"] for symbol, weight in weights.items()}
    else:
        weights = dict(defensive_book)
        vol = volatility_scale(weights, covariance, config)

    full_weights = {symbol: float(weights.get(symbol, 0.0)) for symbol in config.symbols}
    return {
        "features": features,
        "scored": scored,
        "regime": regime,
        "ranked": ranked,
        "entries": entries,
        "covariance": covariance,
        "volatility": vol,
        "defensive_book": defensive_book,
        "weights": full_weights,
    }


# =========================================================================================
# Step 2 helpers: stateful, position-aware
