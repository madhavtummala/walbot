from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math
import pandas as pd

from .fast_momentum import (
    apply_risk_guards,
    compute_composite_scores,
    get_daily_bars,
    get_intraday_bars,
    get_sentiment_snapshot,
    weights_from_positions,
)


@dataclass(frozen=True)
class InvestSpyConfig:
    """SPY-specific state strategy with separate growth, flat, falling, and crisis behavior."""

    spy_symbol: str = "SPY"
    equity_income_universe: list[str] = field(default_factory=lambda: ["XYLD"])
    defensive_universe: list[str] = field(default_factory=lambda: ["BIL"])
    crisis_hedge_universe: list[str] = field(default_factory=lambda: ["SH", "VXX"])
    micro_momentum_lookback_bars: int = 3
    meso_momentum_lookback_bars: int = 26
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
    volatility_lookback_bars: int = 20
    max_intraday_volatility: float = 0.08
    high_volatility_weight_scale: float = 0.7
    intraday_drawdown_limit: float = -0.03

    @classmethod
    def from_runtime_config(cls, config: Any) -> "InvestSpyConfig":
        raw = {}
        if isinstance(getattr(config, "algorithm_configs", None), dict):
            raw = config.algorithm_configs.get("invest_spy", {}) or {}
        if not isinstance(raw, dict):
            raw = {}

        def number(key: str, default: float) -> float:
            try:
                return float(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        def integer(key: str, default: int) -> int:
            try:
                return int(raw.get(key, default))
            except (TypeError, ValueError):
                return default

        def symbols(key: str, default: list[str]) -> list[str]:
            value = raw.get(key, default)
            if isinstance(value, str):
                parsed = [item.strip().upper() for item in value.split(",") if item.strip()]
            elif isinstance(value, list):
                parsed = [str(item).strip().upper() for item in value if str(item).strip()]
            else:
                parsed = []
            return parsed or list(default)

        defaults = cls()
        return cls(
            spy_symbol=str(raw.get("spy_symbol", defaults.spy_symbol) or defaults.spy_symbol).strip().upper(),
            equity_income_universe=symbols("equity_income_universe", defaults.equity_income_universe),
            defensive_universe=symbols("defensive_universe", defaults.defensive_universe),
            crisis_hedge_universe=symbols("crisis_hedge_universe", defaults.crisis_hedge_universe),
            micro_momentum_lookback_bars=integer("micro_momentum_lookback_bars", defaults.micro_momentum_lookback_bars),
            meso_momentum_lookback_bars=integer("meso_momentum_lookback_bars", defaults.meso_momentum_lookback_bars),
            macro_trend_lookback_days=integer("macro_trend_lookback_days", defaults.macro_trend_lookback_days),
            sentiment_lookback_minutes=integer("sentiment_lookback_minutes", defaults.sentiment_lookback_minutes),
            max_gross_exposure=number("max_gross_exposure", defaults.max_gross_exposure),
            max_single_position_weight=number("max_single_position_weight", defaults.max_single_position_weight),
            max_defensive_positions=integer("max_defensive_positions", defaults.max_defensive_positions),
            max_crisis_hedge_positions=integer("max_crisis_hedge_positions", defaults.max_crisis_hedge_positions),
            flat_equity_income_exposure=number("flat_equity_income_exposure", defaults.flat_equity_income_exposure),
            max_crisis_hedge_exposure=number("max_crisis_hedge_exposure", defaults.max_crisis_hedge_exposure),
            max_crisis_hedge_weight=number("max_crisis_hedge_weight", defaults.max_crisis_hedge_weight),
            min_income_score=number("min_income_score", defaults.min_income_score),
            min_defensive_score=number("min_defensive_score", defaults.min_defensive_score),
            min_crisis_hedge_score=number("min_crisis_hedge_score", defaults.min_crisis_hedge_score),
            min_crisis_hedge_meso_return=number("min_crisis_hedge_meso_return", defaults.min_crisis_hedge_meso_return),
            growth_macro_return=number("growth_macro_return", defaults.growth_macro_return),
            growth_meso_return=number("growth_meso_return", defaults.growth_meso_return),
            pullback_micro_return=number("pullback_micro_return", defaults.pullback_micro_return),
            falling_macro_return=number("falling_macro_return", defaults.falling_macro_return),
            falling_meso_return=number("falling_meso_return", defaults.falling_meso_return),
            crisis_macro_return=number("crisis_macro_return", defaults.crisis_macro_return),
            crisis_meso_return=number("crisis_meso_return", defaults.crisis_meso_return),
            crisis_micro_return=number("crisis_micro_return", defaults.crisis_micro_return),
            sentiment_positive=number("sentiment_positive", defaults.sentiment_positive),
            sentiment_negative=number("sentiment_negative", defaults.sentiment_negative),
            sentiment_crisis=number("sentiment_crisis", defaults.sentiment_crisis),
            per_trade_value_min=number("per_trade_value_min", getattr(config, "min_trade_dollars", defaults.per_trade_value_min)),
            rebalance_threshold=number("rebalance_threshold", getattr(config, "rebalance_threshold", defaults.rebalance_threshold)),
            w_price_nano=number("w_price_nano", defaults.w_price_nano),
            w_price_micro=number("w_price_micro", defaults.w_price_micro),
            w_price_meso=number("w_price_meso", defaults.w_price_meso),
            w_price_macro=number("w_price_macro", defaults.w_price_macro),
            w_sentiment=number("w_sentiment", defaults.w_sentiment),
            volatility_lookback_bars=integer("volatility_lookback_bars", defaults.volatility_lookback_bars),
            max_intraday_volatility=number("max_intraday_volatility", defaults.max_intraday_volatility),
            high_volatility_weight_scale=number("high_volatility_weight_scale", defaults.high_volatility_weight_scale),
            intraday_drawdown_limit=number("intraday_drawdown_limit", defaults.intraday_drawdown_limit),
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
    def required_intraday_bars(self) -> int:
        return max(
            self.micro_momentum_lookback_bars,
            self.meso_momentum_lookback_bars,
            self.volatility_lookback_bars,
        ) + 1


def _return_over(closes: pd.Series, bars: int) -> float:
    if len(closes) <= bars or bars <= 0:
        return 0.0
    start = float(closes.iloc[-bars - 1])
    end = float(closes.iloc[-1])
    return end / start - 1.0 if start > 0 else 0.0


def compute_invest_spy_price_features(
    symbol: str,
    intraday_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    config: InvestSpyConfig,
) -> dict[str, Any]:
    intraday = intraday_bars.copy() if isinstance(intraday_bars, pd.DataFrame) else pd.DataFrame()
    daily = daily_bars.copy() if isinstance(daily_bars, pd.DataFrame) else pd.DataFrame()
    closes = pd.to_numeric(intraday.get("close", pd.Series(dtype=float)), errors="coerce").dropna()
    daily_closes = pd.to_numeric(daily.get("close", pd.Series(dtype=float)), errors="coerce").dropna()
    intraday_returns = closes.pct_change().dropna().tail(config.volatility_lookback_bars)
    realized_volatility = float(intraday_returns.std()) if not intraday_returns.empty else 0.0
    macro_return = _return_over(daily_closes, config.macro_trend_lookback_days)
    micro_return = _return_over(closes, config.micro_momentum_lookback_bars)
    meso_return = _return_over(closes, config.meso_momentum_lookback_bars)
    return {
        "symbol": symbol.upper(),
        "nano_return": micro_return,
        "micro_return": micro_return,
        "meso_return": meso_return,
        "macro_return": macro_return,
        "macro_trend_ok": macro_return > 0.0,
        "realized_volatility": 0.0 if math.isnan(realized_volatility) else realized_volatility,
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
    candidates = [
        scores_by_symbol[symbol]
        for symbol in symbols
        if symbol in scores_by_symbol
        and (not require_macro_trend or bool(scores_by_symbol[symbol].get("macro_trend_ok")))
        and float(scores_by_symbol[symbol].get("score", 0.0)) >= min_score
        and (
            min_meso_return is None
            or float(scores_by_symbol[symbol].get("meso_return", 0.0)) >= min_meso_return
        )
    ]
    return sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)


def _allocate_dynamic(
    weights: dict[str, float],
    candidates: list[dict[str, Any]],
    exposure: float,
    max_positions: int,
    max_weight: float,
) -> None:
    selected = candidates[: max(max_positions, 0)]
    if not selected or exposure <= 0:
        return
    scores = [max(float(row.get("score", 0.0)), 0.0) for row in selected]
    total_score = sum(scores)
    for row, score in zip(selected, scores):
        weight = exposure / len(selected) if total_score <= 0 else exposure * score / total_score
        weights[str(row["symbol"])] = max(0.0, min(weight, max_weight))


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


def build_invest_spy_targets(
    runtime_config: Any,
    data_client: Any,
    current_positions: dict[str, int],
    latest_prices: dict[str, float],
    equity: float,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]], dict[str, Any]]:
    strategy_config = InvestSpyConfig.from_runtime_config(runtime_config)
    symbols = strategy_config.symbols
    intraday_bars = get_intraday_bars(symbols, strategy_config.required_intraday_bars, runtime_config, data_client)
    daily_bars = get_daily_bars(symbols, strategy_config.macro_trend_lookback_days, runtime_config, data_client)
    sentiment, market_sentiment = get_sentiment_snapshot(symbols, strategy_config.sentiment_lookback_minutes, runtime_config)
    features = {
        symbol: compute_invest_spy_price_features(symbol, intraday_bars.get(symbol, pd.DataFrame()), daily_bars.get(symbol, pd.DataFrame()), strategy_config)
        for symbol in symbols
    }
    scores = compute_composite_scores(features, sentiment, strategy_config)
    state = classify_spy_state(scores.get(strategy_config.spy_symbol, {}), sentiment.get(strategy_config.spy_symbol, market_sentiment), strategy_config)
    raw_weights = decide_invest_spy_weights(scores, state, strategy_config)
    current_weights = weights_from_positions(current_positions, latest_prices, equity)
    target_weights = apply_risk_guards(raw_weights, scores, current_weights, equity, strategy_config)
    signals = {
        symbol: {
            "signal": 1 if float(target_weights.get(symbol, 0.0)) > 0 else 0,
            "score": float(row.get("score", 0.0)),
            "price_score": float(row.get("macro_return", 0.0)),
            "social_score": float(row.get("sentiment_score", 0.0)),
            "volume_score": 0.0,
            "ret_N": float(row.get("meso_return", 0.0)),
            "sma_long": 0.0,
        }
        for symbol, row in scores.items()
    }
    return target_weights, signals, {
        "spy_state": state,
        "market_sentiment": market_sentiment,
        "scores": scores,
        "raw_target_weights": raw_weights,
    }
