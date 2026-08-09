from __future__ import annotations

from typing import Any

import pandas as pd

from .base import BaseAlgorithm, json_number
from ..connectors import fetch_latest_news_sentiment, merge_social_frames, news_records_to_social_frames
from ..core.interfaces import AlgorithmContext, AlgorithmDecision, AlgorithmRequirements
from ..core.portfolio import compute_target_weights
from ..core.strategy_models import strategy_signal_rows, weights_from_strategy_rows
from ..data.social import load_social_trends_csv


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_live_exposure(config: Any) -> float:
    return min(
        max(float(config.max_portfolio_exposure), 0.0),
        max(1.0 - max(float(config.cash_buffer), 0.0), 0.0),
        1.0,
    )


def _template_signal_map(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    return {
        str(row["symbol"]): {
            "signal": int(row.get("signal", 0)),
            "score": _as_float(row.get("score")),
            "price_score": _as_float(row.get("price_score", row.get("ret_N"))),
            "social_score": _as_float(row.get("social_score")),
            "sentiment": _as_float(row.get("sentiment")),
            "volume_score": _as_float(row.get("volume_score")),
            "ret_N": _as_float(row.get("ret_N")),
            "sma_long": _as_float(row.get("sma_long")),
        }
        for row in rows
    }



def _with_reasons(
    signals: dict[str, dict[str, Any]],
    target_weights: dict[str, float],
    config: Any,
) -> dict[str, dict[str, Any]]:
    """Attach a human-readable selection reason to each signal row.

    The dashboard and the MCP agent both surface ``reason``; without it every row renders
    blank.
    """
    annotated: dict[str, dict[str, Any]] = {}
    for symbol, row in signals.items():
        values = dict(row)
        weight = float(target_weights.get(symbol, 0.0) or 0.0)
        score = float(values.get("score", values.get("composite_score", 0.0)) or 0.0)
        if weight > 0:
            reason = "Selected"
        elif int(values.get("signal", 0) or 0) <= 0:
            reason = "No entry signal"
        elif score < float(getattr(config, "min_composite_score", 0.0) or 0.0):
            reason = "Score too low"
        else:
            reason = "No rank slot"
        values["reason"] = reason
        annotated[symbol] = values
    return annotated


class TemplateStrategyAlgorithm(BaseAlgorithm):
    algorithm_id = "momentum_social"

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        daily_lookback_days = config.momentum_lookback_days
        if self.algorithm_id == "dual_momentum":
            daily_lookback_days = max(daily_lookback_days, 252)
        return AlgorithmRequirements(
            price_symbols=sorted(set(config.symbols) | set(current_positions)),
            daily_lookback_days=daily_lookback_days,
            daily_ma_days=config.long_ma_days,
            daily_extra_buffer_days=config.history_extra_buffer_days,
            include_latest_daily=True,
            needs_sentiment=self.algorithm_id in {"momentum_social", "dual_momentum"},
        )

    def decide(self, context: AlgorithmContext) -> AlgorithmDecision:
        config = context.config
        if self.algorithm_id == "momentum_social":
            signals = self._momentum_social_signals(context)
            target_weights = compute_target_weights(
                signals,
                config.max_weight_per_symbol,
                max_portfolio_exposure=_max_live_exposure(config),
                max_longs=config.max_longs,
                target_annual_vol=config.target_annual_vol,
            )
        else:
            social_weight = config.social_momentum_weight if self.algorithm_id == "dual_momentum" else 0.0
            rows = strategy_signal_rows(
                self.algorithm_id,
                context.bars_by_symbol,
                social_by_symbol=context.sentiment_by_symbol or None,
                social_lookback_days=config.social_lookback_days,
                social_weight=social_weight,
            )
            signals = _template_signal_map(rows)
            target_weights = weights_from_strategy_rows(
                rows,
                config.symbols,
                max_longs=config.max_longs,
                max_weight_per_symbol=config.max_weight_per_symbol,
                max_portfolio_exposure=_max_live_exposure(config),
            )
        return AlgorithmDecision(
            target_weights=target_weights,
            signals=_with_reasons(signals, target_weights, config),
            metadata={"strategy": self.algorithm_id},
            cash_buffer=config.cash_buffer,
            min_trade_dollars=config.min_trade_dollars,
            rebalance_threshold=config.rebalance_threshold,
        )

    def _momentum_social_signals(self, context: AlgorithmContext) -> dict[str, dict[str, float | int]]:
        config = context.config
        social_by_symbol = context.sentiment_by_symbol
        if not social_by_symbol:
            social_by_symbol = merge_social_frames(
                load_social_trends_csv(config.social_trends_csv, config.symbols),
                news_records_to_social_frames(fetch_latest_news_sentiment(config.symbols, config)),
            )
        from ..data.signals.signals import compute_signals_for_universe

        return compute_signals_for_universe(
            context.bars_by_symbol,
            config.momentum_lookback_days,
            config.long_ma_days,
            short_lookback_days=config.short_momentum_lookback_days,
            volume_lookback_days=config.volume_lookback_days,
            social_by_symbol=social_by_symbol,
            social_lookback_days=config.social_lookback_days,
            price_momentum_weight=config.price_momentum_weight,
            social_momentum_weight=config.social_momentum_weight,
            volume_momentum_weight=config.volume_momentum_weight,
            min_composite_score=config.min_composite_score,
        )

def _latest_sma(df: pd.DataFrame, window: int) -> float | None:
    if df.empty or "close" not in df or len(df) < window:
        return None
    close = pd.to_numeric(df["close"], errors="coerce")
    return json_number(close.rolling(window).mean().iloc[-1])


class MomentumSocialAlgorithm(TemplateStrategyAlgorithm):
    algorithm_id = "momentum_social"


class TrendFollowingAlgorithm(TemplateStrategyAlgorithm):
    algorithm_id = "trend_following"


class MeanReversionAlgorithm(TemplateStrategyAlgorithm):
    algorithm_id = "mean_reversion"


class BreakoutAlgorithm(TemplateStrategyAlgorithm):
    algorithm_id = "breakout"


class RiskParityAlgorithm(TemplateStrategyAlgorithm):
    algorithm_id = "risk_parity"


class DualMomentumAlgorithm(TemplateStrategyAlgorithm):
    algorithm_id = "dual_momentum"
