from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import math
import pandas as pd

from .allocation import allocate_by_score, rank_by_score
from .base import BaseAlgorithm
from .fast_momentum import apply_risk_guards, compute_composite_scores
from ..common.config_utils import account_sizing_fallbacks, load_tuning, tuning_section
from ..core.interfaces import DAILY_AT_OPEN, AlgorithmContext, AlgorithmDecision, AlgorithmRequirements
from ..data.bars import closes_of, realized_volatility, return_over_minutes, return_over_periods


@dataclass(frozen=True)
class InvestSpyConfig:
    """SPY-specific state strategy with separate growth, flat, falling, and crisis behavior."""

    spy_symbol: str = field(default="SPY", metadata={"coerce": "symbol"})
    equity_income_universe: list[str] = field(default_factory=lambda: ["XYLD"])
    defensive_universe: list[str] = field(default_factory=lambda: ["BIL"])
    crisis_hedge_universe: list[str] = field(default_factory=lambda: ["SH", "VXX"])
    #: Preferred bar resolution, in minutes. Fidelity only: the horizons below are market-time,
    #: so a finer grid resolves them more precisely and a coarser one still answers them.
    #: 0 takes whatever the feed is configured to prefer.
    intraday_bar_minutes: int = 0
    micro_momentum_lookback_minutes: int = 45
    #: One trading session.
    meso_momentum_lookback_minutes: int = 390
    macro_trend_lookback_days: int = 60
    sentiment_lookback_minutes: int = 60
    max_gross_exposure: float = 1.0
    max_single_position_weight: float = 1.0
    max_defensive_positions: int = 2
    max_crisis_hedge_positions: int = 1
    flat_equity_income_exposure: float = 0.50
    max_crisis_hedge_exposure: float = 0.15
    max_crisis_hedge_weight: float = 0.10
    min_income_score: float = 0.0
    min_defensive_score: float = 0.0
    min_crisis_hedge_score: float = 0.0
    min_crisis_hedge_meso_return: float = 0.0
    growth_macro_return: float = 0.02
    growth_meso_return: float = 0.0
    pullback_micro_return: float = -0.005
    falling_macro_return: float = 0.0
    falling_meso_return: float = -0.01
    crisis_macro_return: float = -0.05
    crisis_meso_return: float = -0.02
    crisis_micro_return: float = -0.01
    sentiment_positive: float = 0.10
    sentiment_negative: float = -0.20
    sentiment_crisis: float = -0.40
    per_trade_value_min: float = 50.0
    rebalance_threshold: float = 0.01
    w_price_nano: float = 0.0
    w_price_micro: float = 0.35
    w_price_meso: float = 0.40
    w_price_macro: float = 0.15
    w_sentiment: float = 0.10
    w_pullback_uptrend: float = 0.0
    pullback_macro_z_threshold: float = 1.0
    pullback_micro_z_threshold: float = -0.5
    pullback_micro_z_cap: float = 3.0
    pullback_min_meso_return: float = 0.0
    volatility_lookback_minutes: int = 300
    #: Quoted per 15-minute bar whatever the feed's grid is -- see ``data/bars.py``.
    max_intraday_volatility: float = 0.08
    high_volatility_weight_scale: float = 0.7
    intraday_drawdown_limit: float = -0.03

    @classmethod
    def from_runtime_config(cls, config: Any) -> "InvestSpyConfig":
        # Read every id this algorithm has had: it is ``spy_rotation`` now, but tuning saved
        # earlier is filed under ``regime_rotation`` or ``invest_spy``, and the key on disk is
        # still the oldest of the three.
        return load_tuning(
            cls,
            tuning_section(config, "spy_rotation", "regime_rotation", "invest_spy"),
            fallbacks=account_sizing_fallbacks(config),
        )

    @property
    def symbols(self) -> list[str]:
        return sorted(
            {self.spy_symbol}
            | set(self.equity_income_universe)
            | set(self.defensive_universe)
            | set(self.crisis_hedge_universe)
        )

    @property
    def required_history_minutes(self) -> int:
        return max(
            self.micro_momentum_lookback_minutes,
            self.meso_momentum_lookback_minutes,
            self.volatility_lookback_minutes,
        )


def compute_invest_spy_price_features(
    symbol: str,
    history_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    config: InvestSpyConfig,
) -> dict[str, Any]:
    """State-detection inputs: two wall-clock horizons plus the daily macro trend."""
    daily_closes = closes_of(daily_bars)
    closes = closes_of(history_bars)
    volatility = realized_volatility(history_bars, config.volatility_lookback_minutes)
    macro_return = return_over_periods(daily_closes, config.macro_trend_lookback_days)
    micro_return = return_over_minutes(history_bars, config.micro_momentum_lookback_minutes)
    meso_return = return_over_minutes(history_bars, config.meso_momentum_lookback_minutes)
    return {
        "symbol": symbol.upper(),
        "nano_return": micro_return,
        "micro_return": micro_return,
        "meso_return": meso_return,
        "macro_return": macro_return,
        "macro_trend_ok": macro_return > 0.0,
        "realized_volatility": 0.0 if math.isnan(volatility) else volatility,
        "close": float(closes.iloc[-1]) if not closes.empty else 0.0,
    }


def classify_spy_state(spy_score: dict[str, Any], spy_sentiment: float, config: InvestSpyConfig) -> str:
    macro_return = float(spy_score.get("macro_return", 0.0))
    meso_return = float(spy_score.get("meso_return", 0.0))
    micro_return = float(spy_score.get("micro_return", 0.0))
    sentiment = float(spy_sentiment)

    if (
        macro_return <= config.crisis_macro_return
        or meso_return <= config.crisis_meso_return
        or (micro_return <= config.crisis_micro_return and sentiment <= config.sentiment_negative)
        or sentiment <= config.sentiment_crisis
    ):
        return "CRISIS"
    if macro_return < config.falling_macro_return or meso_return <= config.falling_meso_return or sentiment <= config.sentiment_negative:
        return "FALLING"
    if macro_return >= config.growth_macro_return and meso_return >= config.growth_meso_return:
        if micro_return <= config.pullback_micro_return:
            return "PULLBACK"
        return "GROWING"
    return "FLAT"


def _ranked(
    scores_by_symbol: dict[str, dict[str, Any]],
    symbols: list[str],
    min_score: float,
    min_meso_return: float | None = None,
    *,
    require_macro_trend: bool = True,
) -> list[dict[str, Any]]:
    return rank_by_score(
        scores_by_symbol,
        symbols,
        min_score=min_score,
        gate_key="meso_return",
        min_gate=min_meso_return,
        require_trend=require_macro_trend,
    )


def _allocate_dynamic(
    weights: dict[str, float],
    candidates: list[dict[str, Any]],
    exposure: float,
    max_positions: int,
    max_weight: float,
) -> None:
    weights.update(allocate_by_score(candidates, exposure, max_positions, max_weight))


def decide_invest_spy_weights(
    scores_by_symbol: dict[str, dict[str, Any]],
    spy_state: str,
    config: InvestSpyConfig,
) -> dict[str, float]:
    weights = {symbol: 0.0 for symbol in config.symbols}
    if spy_state in {"GROWING", "PULLBACK"}:
        weights[config.spy_symbol] = min(config.max_gross_exposure, config.max_single_position_weight)
        return weights

    if spy_state == "FLAT":
        income = _ranked(scores_by_symbol, config.equity_income_universe, config.min_income_score, 0.0)
        _allocate_dynamic(weights, income, min(config.flat_equity_income_exposure, config.max_gross_exposure), 1, config.max_single_position_weight)
        remaining = max(config.max_gross_exposure - sum(weights.values()), 0.0)
        defensive = _ranked(scores_by_symbol, config.defensive_universe, config.min_defensive_score, require_macro_trend=False)
        _allocate_dynamic(weights, defensive, remaining, config.max_defensive_positions, config.max_single_position_weight)
        return weights

    if spy_state == "CRISIS":
        hedge = _ranked(
            scores_by_symbol,
            config.crisis_hedge_universe,
            config.min_crisis_hedge_score,
            config.min_crisis_hedge_meso_return,
            require_macro_trend=False,
        )
        _allocate_dynamic(
            weights,
            hedge,
            min(config.max_crisis_hedge_exposure, config.max_gross_exposure),
            config.max_crisis_hedge_positions,
            config.max_crisis_hedge_weight,
        )

    remaining = max(config.max_gross_exposure - sum(weights.values()), 0.0)
    defensive = _ranked(scores_by_symbol, config.defensive_universe, config.min_defensive_score, require_macro_trend=False)
    _allocate_dynamic(weights, defensive, remaining, config.max_defensive_positions, config.max_single_position_weight)
    return weights


def score_universe(
    context: AlgorithmContext, strategy_config: InvestSpyConfig
) -> tuple[dict[str, dict[str, Any]], str]:
    """Score the universe and classify SPY, from the bars already in ``context``.

    Takes the context rather than fetching, so the live runner and the backtester classify
    the same regime from the same inputs instead of two separate implementations.
    """
    features = {
        symbol: compute_invest_spy_price_features(
            symbol,
            context.history_bars_by_symbol.get(symbol, pd.DataFrame()),
            context.bars_by_symbol.get(symbol, pd.DataFrame()),
            strategy_config,
        )
        for symbol in strategy_config.symbols
    }
    scores = compute_composite_scores(features, context.sentiment_scores, strategy_config)
    state = classify_spy_state(
        scores.get(strategy_config.spy_symbol, {}),
        context.sentiment_scores.get(strategy_config.spy_symbol, context.market_sentiment),
        strategy_config,
    )
    return scores, state


def signals_from_scores(
    scores_by_symbol: dict[str, dict[str, Any]],
    target_weights: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Per-symbol signal rows the dashboard and step 2 both read."""
    return {
        symbol: {
            "signal": 1 if float(target_weights.get(symbol, 0.0)) > 0 else 0,
            "score": float(row.get("score", 0.0)),
            "price_score": float(row.get("macro_return", 0.0)),
            "social_score": float(row.get("sentiment_score", 0.0)),
            "realized_volatility": float(row.get("realized_volatility", 0.0)),
            "trend_ok": 1 if row.get("macro_trend_ok") else 0,
            "reason": (
                "Selected"
                if float(target_weights.get(symbol, 0.0)) > 0
                else "Macro negative"
                if not row.get("macro_trend_ok")
                else "No rank slot"
            ),
            "volume_score": 0.0,
            "ret_N": float(row.get("meso_return", 0.0)),
            "sma_long": 0.0,
        }
        for symbol, row in scores_by_symbol.items()
    }


class InvestSpyAlgorithm(BaseAlgorithm):
    algorithm_id = "invest_spy"

    #: Once per session. Every input is a daily bar, so a second run the same day reads the
    #: same closes and can only churn the portfolio.
    schedule = DAILY_AT_OPEN

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        strategy_config = InvestSpyConfig.from_runtime_config(config)
        return AlgorithmRequirements(
            price_symbols=sorted(set(strategy_config.symbols) | set(current_positions)),
            daily_lookback_days=strategy_config.macro_trend_lookback_days,
            history_lookback_minutes=strategy_config.required_history_minutes,
            preferred_bar_minutes=strategy_config.intraday_bar_minutes,
            needs_sentiment=True,
            paper_only=True,
        )

    def sizing(self, config: Any) -> dict[str, float]:
        """No cash buffer: gross exposure is already capped inside the weights."""
        strategy_config = InvestSpyConfig.from_runtime_config(config)
        return {
            "cash_buffer": 0.0,
            "min_trade_dollars": strategy_config.per_trade_value_min,
            "rebalance_threshold": strategy_config.rebalance_threshold,
        }

    def analyze(self, context: AlgorithmContext) -> AlgorithmDecision:
        """Score the universe and allocate, with no knowledge of what is held."""
        strategy_config = InvestSpyConfig.from_runtime_config(context.config)
        scores, state = score_universe(context, strategy_config)
        raw_weights = decide_invest_spy_weights(scores, state, strategy_config)
        return AlgorithmDecision(
            target_weights=raw_weights,
            signals=signals_from_scores(scores, raw_weights),
            metadata={
                "allocation_mode": state,
                "market_sentiment": context.market_sentiment,
                "spy_state": state,
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
        """Apply the position-aware risk guards to a reviewed set of weights."""
        strategy_config = InvestSpyConfig.from_runtime_config(config)
        current_weights = snapshot.weights(latest_prices)
        scores = {symbol: dict(row) for symbol, row in signals.items()}
        return apply_risk_guards(
            target_weights,
            scores,
            current_weights,
            snapshot.equity,
            strategy_config,
            as_of,
        )
