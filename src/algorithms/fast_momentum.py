from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from .allocation import allocate_by_score, rank_by_score, scale_to_gross
from .base import BaseAlgorithm
from .risk import session_drawdown_breached
from ..common.config_utils import account_sizing_fallbacks, load_tuning, tuning_section
from ..core.interfaces import AlgorithmContext, AlgorithmDecision, AlgorithmRequirements, Schedule
from ..data.bars import closes_of, realized_volatility, return_over_minutes, return_over_periods
from ..data.state_store import load_state, save_state

logger = logging.getLogger(__name__)

STATE_KEY = "defensive_momentum_intraday_risk"


@dataclass(frozen=True)
class DefensiveMomentumConfig:
    """Configurable knobs for the intraday defensive momentum + sentiment strategy."""

    risk_on_universe: list[str] = field(default_factory=lambda: ["QQQ", "VTI", "IWM", "IEMG", "ACWI"])
    defensive_universe: list[str] = field(default_factory=lambda: ["BIL", "IEF", "AGG", "TLT", "GLD"])
    #: Preferred bar resolution, in minutes. Fidelity only: the horizons below are market-time,
    #: so a finer grid resolves them more precisely and a coarser one still answers them.
    #: 0 takes whatever the feed is configured to prefer.
    intraday_bar_minutes: int = 0
    nano_momentum_lookback_minutes: int = 150
    #: Three trading sessions.
    micro_momentum_lookback_minutes: int = 1170
    meso_trend_lookback_days: int = 60
    macro_trend_lookback_days: int = 180
    sentiment_lookback_minutes: int = 60
    max_gross_exposure: float = 1.0
    max_single_position_weight: float = 0.25
    max_positions: int = 4
    min_risk_on_score: float = 0.0
    min_risk_on_micro_return: float = 0.0
    min_defensive_score: float = 0.0
    min_score_delta_to_replace: float = 0.0
    per_trade_value_min: float = 50.0
    rebalance_threshold: float = 0.01
    w_price_nano: float = 0.25
    w_price_micro: float = 0.35
    w_price_meso: float = 0.20
    w_price_macro: float = 0.10
    w_sentiment: float = 0.1
    w_pullback_uptrend: float = 0.1
    pullback_meso_z_threshold: float = 1.0
    pullback_nano_z_threshold: float = -0.5
    pullback_nano_z_cap: float = 3.0
    pullback_min_micro_return: float = -0.02
    volatility_lookback_minutes: int = 195
    #: Quoted per 15-minute bar whatever the feed's grid is -- see ``data/bars.py``.
    max_intraday_volatility: float = 0.06
    high_volatility_weight_scale: float = 0.5
    intraday_drawdown_limit: float = -0.02

    @classmethod
    def from_runtime_config(cls, config: Any) -> "DefensiveMomentumConfig":
        return load_tuning(
            cls,
            tuning_section(config, "fast_momentum"),
            fallbacks=account_sizing_fallbacks(config),
        )

    @property
    def symbols(self) -> list[str]:
        return sorted(set(self.risk_on_universe) | set(self.defensive_universe))

    @property
    def required_history_minutes(self) -> int:
        return max(
            self.nano_momentum_lookback_minutes,
            self.micro_momentum_lookback_minutes,
            self.volatility_lookback_minutes,
        )

    @property
    def required_daily_bars(self) -> int:
        return max(self.meso_trend_lookback_days, self.macro_trend_lookback_days)


def compute_price_features(
    symbol: str,
    history_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    config: DefensiveMomentumConfig,
) -> dict[str, Any]:
    """Calculate multi-horizon momentum, absolute trend, and intraday volatility.

    The two fast horizons are measured in market minutes against ``history_bars``, so they
    mean the same span whether the feed supplied 1-minute, 5-minute, or daily observations.
    The two slow ones stay positional on daily closes: "the 60-day return" is a statement
    about sessions, not about elapsed time.
    """
    daily_closes = closes_of(daily_bars)

    volatility = realized_volatility(history_bars, config.volatility_lookback_minutes)
    nano_return = return_over_minutes(history_bars, config.nano_momentum_lookback_minutes)
    micro_return = return_over_minutes(history_bars, config.micro_momentum_lookback_minutes)
    meso_return = return_over_periods(daily_closes, config.meso_trend_lookback_days)
    macro_return = return_over_periods(daily_closes, config.macro_trend_lookback_days)

    closes = closes_of(history_bars)
    return {
        "symbol": symbol.upper(),
        "nano_return": nano_return,
        "micro_return": micro_return,
        "meso_return": meso_return,
        "macro_return": macro_return,
        "macro_trend_ok": macro_return > 0.0,
        "realized_volatility": 0.0 if math.isnan(volatility) else volatility,
        "close": float(closes.iloc[-1]) if not closes.empty else 0.0,
    }


def zscores_by_feature(features_by_symbol: dict[str, dict[str, Any]], keys: list[str]) -> dict[str, dict[str, float]]:
    """Compute simple cross-sectional z-scores for the requested feature keys."""
    zscores: dict[str, dict[str, float]] = {symbol: {} for symbol in features_by_symbol}
    for key in keys:
        values = [float(features.get(key, 0.0)) for features in features_by_symbol.values()]
        mean = sum(values) / len(values) if values else 0.0
        variance = sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0
        std = math.sqrt(variance)
        for symbol, features in features_by_symbol.items():
            zscores[symbol][key] = 0.0 if std == 0 else (float(features.get(key, 0.0)) - mean) / std
    return zscores


def compute_composite_scores(
    features_by_symbol: dict[str, dict[str, Any]],
    sentiment_scores: dict[str, float],
    config: DefensiveMomentumConfig,
) -> dict[str, dict[str, Any]]:
    """Combine z-scored price momentum and normalized sentiment into one score."""
    zscores = zscores_by_feature(features_by_symbol, ["nano_return", "micro_return", "meso_return", "macro_return"])
    scored: dict[str, dict[str, Any]] = {}
    for symbol, features in features_by_symbol.items():
        sentiment = max(-1.0, min(1.0, float(sentiment_scores.get(symbol, 0.0))))
        nano_z = zscores[symbol]["nano_return"]
        micro_z = zscores[symbol]["micro_return"]
        meso_z = zscores[symbol]["meso_return"]
        micro_return = float(features.get("micro_return", 0.0))
        pullback_uptrend = 0.0
        if (
            config.w_pullback_uptrend > 0
            and bool(features.get("macro_trend_ok"))
            and meso_z >= config.pullback_meso_z_threshold
            and nano_z <= config.pullback_nano_z_threshold
            and micro_return >= config.pullback_min_micro_return
        ):
            pullback_depth = min(abs(nano_z), max(config.pullback_nano_z_cap, 0.0))
            pullback_uptrend = config.w_pullback_uptrend * meso_z * pullback_depth
        components = {
            "price_nano": config.w_price_nano * nano_z,
            "price_micro": config.w_price_micro * micro_z,
            "price_meso": config.w_price_meso * meso_z,
            "price_macro": config.w_price_macro * zscores[symbol]["macro_return"],
            "sentiment": config.w_sentiment * sentiment,
            "pullback_uptrend": pullback_uptrend,
        }
        score = sum(components.values())
        scored[symbol] = {**features, "symbol": symbol, "score": score, "sentiment_score": sentiment, "components": components}
    return scored





def _defensive_momentum_reason(
    row: dict[str, Any], weight: float, config: DefensiveMomentumConfig
) -> str:
    """Explain why a symbol was or was not selected, for the dashboard and the MCP agent."""
    if weight > 0:
        return "Top Rank"

    symbol = str(row.get("symbol", "")).upper()
    is_defensive = symbol in {item.upper() for item in config.defensive_universe}
    if not is_defensive and not bool(row.get("macro_trend_ok", False)):
        return "Macro negative"

    score = float(row.get("score", 0.0) or 0.0)
    min_score = config.min_defensive_score if is_defensive else config.min_risk_on_score
    if score < min_score:
        return "Score too low"

    if not is_defensive and float(row.get("micro_return", 0.0) or 0.0) < config.min_risk_on_micro_return:
        return "Micro too low"

    return "No rank slot"


def decide_target_weights(
    scores_by_symbol: dict[str, dict[str, Any]],
    config: DefensiveMomentumConfig,
    current_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Rank eligible risk-on and defensive ETFs, then allocate dynamically by score.

    Current holdings can be protected from minor score-based turnover by requiring
    a candidate symbol to exceed existing positions by config.min_score_delta_to_replace.
    """
    current_weights = dict(current_weights or {})
    current_symbol_positions = {symbol for symbol, weight in current_weights.items() if weight > 0.0}
    score_delta = max(config.min_score_delta_to_replace, 0.0)
    weights = {symbol: 0.0 for symbol in config.symbols}

    def ranked(
        symbols: list[str],
        min_score: float,
        min_micro_return: float | None = None,
        *,
        require_macro_trend: bool = True,
    ) -> list[dict[str, Any]]:
        return rank_by_score(
            scores_by_symbol,
            symbols,
            min_score=min_score,
            gate_key="micro_return",
            min_gate=min_micro_return,
            require_trend=require_macro_trend,
        )

    candidates = ranked(config.risk_on_universe, config.min_risk_on_score, config.min_risk_on_micro_return)
    candidates.extend(ranked(config.defensive_universe, config.min_defensive_score))
    candidates.sort(
        key=lambda item: float(item.get("score", 0.0)) + (score_delta if item["symbol"] in current_symbol_positions else 0.0),
        reverse=True,
    )
    selected = candidates[: max(config.max_positions, 0)]
    if score_delta > 0 and current_symbol_positions:
        effective_top = sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)[: max(config.max_positions, 0)]
        retained_by_bonus = [s for s in selected if s["symbol"] in current_symbol_positions and s not in effective_top]
        for item in retained_by_bonus:
            logger.info(
                "Stickiness retained %s: score=%.3f + delta=%.2f kept it in top %d",
                item["symbol"],
                float(item.get("score", 0.0)),
                score_delta,
                config.max_positions,
            )

    if not selected:
        selected = ranked(config.defensive_universe, config.min_defensive_score, require_macro_trend=False)[: max(config.max_positions, 0)]

    weights.update(
        allocate_by_score(
            selected,
            config.max_gross_exposure,
            config.max_positions,
            config.max_single_position_weight,
        )
    )
    return scale_to_gross(weights, config.max_gross_exposure)


def apply_risk_guards(
    target_weights: dict[str, float],
    scores_by_symbol: dict[str, dict[str, Any]],
    current_weights: dict[str, float],
    equity: float,
    config: DefensiveMomentumConfig,
    as_of: datetime,
) -> dict[str, float]:
    """Apply position caps, volatility filters, turnover threshold, and drawdown kill-switch."""
    if intraday_kill_switch_triggered(equity, config, as_of):
        logger.warning("Fast Momentum kill-switch active; target weights set to zero")
        return {symbol: 0.0 for symbol in target_weights}

    guarded: dict[str, float] = {}
    for symbol, weight in target_weights.items():
        capped = min(max(weight, 0.0), config.max_single_position_weight)
        realized_vol = float(scores_by_symbol.get(symbol, {}).get("realized_volatility", 0.0))
        if config.max_intraday_volatility > 0 and realized_vol > config.max_intraday_volatility:
            capped *= max(0.0, min(1.0, config.high_volatility_weight_scale))
        if capped <= 0.0:
            guarded[symbol] = 0.0
            continue
        if abs(capped - current_weights.get(symbol, 0.0)) < max(config.rebalance_threshold, 0.0):
            capped = current_weights.get(symbol, 0.0)
        if abs(capped - current_weights.get(symbol, 0.0)) * equity < config.per_trade_value_min:
            capped = current_weights.get(symbol, 0.0)
        guarded[symbol] = capped

    return scale_to_gross(guarded, config.max_gross_exposure)


def intraday_kill_switch_triggered(
    equity: float, config: DefensiveMomentumConfig, as_of: datetime
) -> bool:
    """Track account equity from the first run of the session and flatten after a breach.

    ``as_of`` is the moment the algorithm is reasoning about, and is required: it used to be
    ``date.today()``, which is right live and wrong under replay, where every simulated date is
    "today". See ``algorithms/risk.py``.
    """
    state = load_state(STATE_KEY, {})
    if not isinstance(state, dict):
        state = {}
    breached = session_drawdown_breached(state, equity, config.intraday_drawdown_limit, as_of)
    save_state(STATE_KEY, state)
    return breached


def rows_from_scores(
    scores_by_symbol: dict[str, dict[str, Any]],
    target_weights: dict[str, float],
    allocation_mode: str,
) -> list[dict[str, Any]]:
    """Build dashboard/log signal rows from scored symbols and final weights."""
    rows = []
    for symbol, row in scores_by_symbol.items():
        weight = float(target_weights.get(symbol, 0.0))
        components = row.get("components", {}) if isinstance(row.get("components"), dict) else {}
        rows.append(
            {
                "symbol": symbol,
                "signal": 1 if weight > 0 else 0,
                "score": float(row.get("score", 0.0)),
                "price_score": float(row.get("macro_return", 0.0)),
                "social_score": float(row.get("sentiment_score", 0.0)),
                "sentiment_component": float(components.get("sentiment", 0.0)),
                "pullback_score": float(components.get("pullback_uptrend", 0.0)),
                "score_components": {key: float(value) for key, value in components.items()},
                "volume_score": 0.0,
                "ret_N": float(row.get("meso_return", 0.0)),
                "ret_short": float(row.get("nano_return", 0.0)),
                "macro_return": float(row.get("macro_return", 0.0)),
                "meso_return": float(row.get("meso_return", 0.0)),
                "micro_return": float(row.get("micro_return", 0.0)),
                "nano_return": float(row.get("nano_return", 0.0)),
                "sma_long": 0.0,
                "allocation_mode": allocation_mode,
            }
        )
    return rows


def allocation_mode(target_weights: dict[str, float]) -> str:
    return "Dynamic rank" if any(weight > 0 for weight in target_weights.values()) else "Cash"


def score_universe(
    context: AlgorithmContext, strategy_config: DefensiveMomentumConfig
) -> dict[str, dict[str, Any]]:
    """Score every symbol from the bars already in ``context``.

    Takes the context rather than fetching, so the live runner and the backtester score
    identically -- the backtester supplies a point-in-time slice and gets the real algorithm
    instead of a reimplementation of it.
    """
    return compute_composite_scores(
        {
            symbol: compute_price_features(
                symbol,
                context.history_bars_by_symbol.get(symbol, pd.DataFrame()),
                context.bars_by_symbol.get(symbol, pd.DataFrame()),
                strategy_config,
            )
            for symbol in strategy_config.symbols
        },
        context.sentiment_scores,
        strategy_config,
    )


def apply_stickiness(
    target_weights: dict[str, float],
    scores_by_symbol: dict[str, dict[str, Any]],
    current_weights: dict[str, float],
    config: DefensiveMomentumConfig,
) -> dict[str, float]:
    """Retain a held symbol the proposal drops unless a challenger beats it by the score delta.

    The churn guard from ``decide_target_weights``, restated against an already-chosen set so
    it also protects incumbents when the proposal came from a reviewing agent rather than
    from the ranking itself.
    """
    score_delta = max(config.min_score_delta_to_replace, 0.0)
    incumbents = {symbol for symbol, weight in current_weights.items() if weight > 0.0}
    proposed = {symbol: weight for symbol, weight in target_weights.items() if weight > 0.0}
    dropped = incumbents - set(proposed)
    if not score_delta or not dropped or len(proposed) < max(config.max_positions, 0):
        return dict(target_weights)

    def score_of(symbol: str) -> float:
        return float(scores_by_symbol.get(symbol, {}).get("score", 0.0))

    weakest = min(proposed, key=score_of)
    kept = dict(target_weights)
    for symbol in sorted(dropped, key=score_of, reverse=True):
        if score_of(symbol) + score_delta <= score_of(weakest):
            continue
        logger.info(
            "Stickiness retained %s: score=%.3f + delta=%.2f beats %s at %.3f",
            symbol,
            score_of(symbol),
            score_delta,
            weakest,
            score_of(weakest),
        )
        kept[symbol] = current_weights[symbol]
        kept[weakest] = 0.0
        proposed.pop(weakest, None)
        if not proposed:
            break
        weakest = min(proposed, key=score_of)
    return kept


def signals_from_scores(
    scores_by_symbol: dict[str, dict[str, Any]],
    target_weights: dict[str, float],
    config: "DefensiveMomentumConfig",
) -> dict[str, dict[str, Any]]:
    """Flatten scores into the per-symbol signal rows the dashboard and step 2 both read.

    ``refine`` reads ``score`` and ``realized_volatility`` back out of these rows, so this is
    the contract that lets step 2 work without re-deriving anything from market data.
    """
    rows = rows_from_scores(scores_by_symbol, target_weights, allocation_mode(target_weights))
    return {
        row["symbol"]: {
            "signal": int(row["signal"]),
            "score": float(row["score"]),
            "price_score": float(row["price_score"]),
            "social_score": float(row["social_score"]),
            "sentiment_component": float(row["sentiment_component"]),
            "pullback_score": float(row["pullback_score"]),
            "score_components": row["score_components"],
            "volume_score": 0.0,
            "ret_N": float(row["ret_N"]),
            "ret_short": float(row["ret_short"]),
            "macro_return": float(row["macro_return"]),
            "meso_return": float(row["meso_return"]),
            "micro_return": float(row["micro_return"]),
            "nano_return": float(row["nano_return"]),
            "realized_volatility": float(scores_by_symbol.get(row["symbol"], {}).get("realized_volatility", 0.0)),
            # Consumed by the dashboard row subtitle (web/static/app.js) -- dropping it renders
            # every symbol as "Inactive".
            "trend_ok": 1 if scores_by_symbol.get(row["symbol"], {}).get("macro_trend_ok") else 0,
            "reason": _defensive_momentum_reason(
                scores_by_symbol.get(row["symbol"], {}),
                float(target_weights.get(row["symbol"], 0.0)),
                config,
            ),
            "sma_long": 0.0,
        }
        for row in rows
    }


class FastMomentumAlgorithm(BaseAlgorithm):
    algorithm_id = "fast_momentum"

    #: Hourly through the session. Nano and micro momentum are computed from intraday bars,
    #: so a once-a-day look would discard the signal this algorithm exists to trade.
    schedule = Schedule()

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        strategy_config = DefensiveMomentumConfig.from_runtime_config(config)
        return AlgorithmRequirements(
            price_symbols=sorted(set(strategy_config.symbols) | set(current_positions)),
            daily_lookback_days=strategy_config.required_daily_bars,
            history_lookback_minutes=strategy_config.required_history_minutes,
            preferred_bar_minutes=strategy_config.intraday_bar_minutes,
            needs_sentiment=True,
            paper_only=True,
        )

    def sizing(self, config: Any) -> dict[str, float]:
        """No cash buffer: gross exposure is already capped inside the weights."""
        strategy_config = DefensiveMomentumConfig.from_runtime_config(config)
        return {
            "cash_buffer": 0.0,
            "min_trade_dollars": strategy_config.per_trade_value_min,
            "rebalance_threshold": strategy_config.rebalance_threshold,
        }

    def analyze(self, context: AlgorithmContext) -> AlgorithmDecision:
        """Score the universe and rank it with no knowledge of what is held.

        Stickiness and the risk guards are deliberately absent -- both need current weights
        and equity, so they run in ``refine``.
        """
        strategy_config = DefensiveMomentumConfig.from_runtime_config(context.config)
        scores = score_universe(context, strategy_config)
        raw_weights = decide_target_weights(scores, strategy_config)
        return AlgorithmDecision(
            target_weights=raw_weights,
            signals=signals_from_scores(scores, raw_weights, strategy_config),
            metadata={
                "allocation_mode": allocation_mode(raw_weights),
                "market_sentiment": context.market_sentiment,
            },
        )

    def refine_weights(
        self,
        target_weights: dict[str, float],
        signals: dict[str, dict[str, Any]],
        snapshot: Any,
        latest_prices: dict[str, float],
        config: Any,
        as_of: datetime,
    ) -> dict[str, float]:
        """Protect incumbents from churn, then apply the position-aware risk guards."""
        strategy_config = DefensiveMomentumConfig.from_runtime_config(config)
        current_weights = snapshot.weights(latest_prices)
        scores = {symbol: dict(row) for symbol, row in signals.items()}
        for symbol, row in scores.items():
            row.setdefault("symbol", symbol)

        kept = apply_stickiness(target_weights, scores, current_weights, strategy_config)
        return apply_risk_guards(
            kept, scores, current_weights, snapshot.equity, strategy_config, as_of
        )
