from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from .base import BaseAlgorithm
from ..connectors import fetch_intraday_market_bars, fetch_latest_news_sentiment
from ..core.interfaces import AlgorithmContext, AlgorithmDecision, AlgorithmRequirements
from ..data import fetch_daily_bars
from ..data.state_store import load_state, save_state

logger = logging.getLogger(__name__)

STATE_KEY = "defensive_momentum_intraday_risk"


@dataclass(frozen=True)
class DefensiveMomentumConfig:
    """Configurable knobs for the intraday defensive momentum + sentiment strategy."""

    risk_on_universe: list[str] = field(default_factory=lambda: ["QQQ", "VTI", "IWM", "IEMG", "ACWI"])
    defensive_universe: list[str] = field(default_factory=lambda: ["BIL", "IEF", "AGG", "TLT", "GLD"])
    nano_momentum_lookback_bars: int = 10
    micro_momentum_lookback_bars: int = 78
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
    volatility_lookback_bars: int = 13
    max_intraday_volatility: float = 0.06
    high_volatility_weight_scale: float = 0.5
    intraday_drawdown_limit: float = -0.02

    @classmethod
    def from_runtime_config(cls, config: Any) -> "DefensiveMomentumConfig":
        raw = {}
        if isinstance(getattr(config, "algorithm_configs", None), dict):
            raw = config.algorithm_configs.get("fast_momentum", {}) or {}
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
            risk_on_universe=symbols("risk_on_universe", defaults.risk_on_universe),
            defensive_universe=symbols("defensive_universe", defaults.defensive_universe),
            nano_momentum_lookback_bars=integer("nano_momentum_lookback_bars", defaults.nano_momentum_lookback_bars),
            micro_momentum_lookback_bars=integer("micro_momentum_lookback_bars", defaults.micro_momentum_lookback_bars),
            meso_trend_lookback_days=integer("meso_trend_lookback_days", defaults.meso_trend_lookback_days),
            macro_trend_lookback_days=integer("macro_trend_lookback_days", defaults.macro_trend_lookback_days),
            sentiment_lookback_minutes=integer("sentiment_lookback_minutes", defaults.sentiment_lookback_minutes),
            max_gross_exposure=number("max_gross_exposure", defaults.max_gross_exposure),
            max_single_position_weight=number("max_single_position_weight", defaults.max_single_position_weight),
            max_positions=integer("max_positions", defaults.max_positions),
            min_risk_on_score=number("min_risk_on_score", defaults.min_risk_on_score),
            min_risk_on_micro_return=number("min_risk_on_micro_return", defaults.min_risk_on_micro_return),
            min_defensive_score=number("min_defensive_score", defaults.min_defensive_score),
            min_score_delta_to_replace=number("min_score_delta_to_replace", defaults.min_score_delta_to_replace),
            per_trade_value_min=number("per_trade_value_min", getattr(config, "min_trade_dollars", defaults.per_trade_value_min)),
            rebalance_threshold=number("rebalance_threshold", getattr(config, "rebalance_threshold", defaults.rebalance_threshold)),
            w_price_nano=number("w_price_nano", defaults.w_price_nano),
            w_price_micro=number("w_price_micro", defaults.w_price_micro),
            w_price_meso=number("w_price_meso", defaults.w_price_meso),
            w_price_macro=number("w_price_macro", defaults.w_price_macro),
            w_sentiment=number("w_sentiment", defaults.w_sentiment),
            w_pullback_uptrend=number("w_pullback_uptrend", defaults.w_pullback_uptrend),
            pullback_meso_z_threshold=number("pullback_meso_z_threshold", defaults.pullback_meso_z_threshold),
            pullback_nano_z_threshold=number("pullback_nano_z_threshold", defaults.pullback_nano_z_threshold),
            pullback_nano_z_cap=number("pullback_nano_z_cap", defaults.pullback_nano_z_cap),
            pullback_min_micro_return=number("pullback_min_micro_return", defaults.pullback_min_micro_return),
            volatility_lookback_bars=integer("volatility_lookback_bars", defaults.volatility_lookback_bars),
            max_intraday_volatility=number("max_intraday_volatility", defaults.max_intraday_volatility),
            high_volatility_weight_scale=number("high_volatility_weight_scale", defaults.high_volatility_weight_scale),
            intraday_drawdown_limit=number("intraday_drawdown_limit", defaults.intraday_drawdown_limit),
        )

    @property
    def symbols(self) -> list[str]:
        return sorted(set(self.risk_on_universe) | set(self.defensive_universe))

    @property
    def required_intraday_bars(self) -> int:
        return max(
            self.nano_momentum_lookback_bars,
            self.micro_momentum_lookback_bars,
            self.volatility_lookback_bars,
        ) + 1

    @property
    def required_daily_bars(self) -> int:
        return max(self.meso_trend_lookback_days, self.macro_trend_lookback_days)


def _return_over(closes: pd.Series, bars: int) -> float:
    if len(closes) <= bars or bars <= 0:
        return 0.0
    start = float(closes.iloc[-bars - 1])
    end = float(closes.iloc[-1])
    return end / start - 1.0 if start > 0 else 0.0


def compute_price_features(
    symbol: str,
    intraday_bars: pd.DataFrame,
    daily_bars: pd.DataFrame,
    config: DefensiveMomentumConfig,
) -> dict[str, Any]:
    """Calculate multi-horizon momentum, absolute trend, and intraday volatility."""
    intraday = intraday_bars.copy() if isinstance(intraday_bars, pd.DataFrame) else pd.DataFrame()
    daily = daily_bars.copy() if isinstance(daily_bars, pd.DataFrame) else pd.DataFrame()
    closes = pd.to_numeric(intraday.get("close", pd.Series(dtype=float)), errors="coerce").dropna()
    daily_closes = pd.to_numeric(daily.get("close", pd.Series(dtype=float)), errors="coerce").dropna()

    intraday_returns = closes.pct_change().dropna().tail(config.volatility_lookback_bars)
    realized_volatility = float(intraday_returns.std()) if not intraday_returns.empty else 0.0
    nano_return = _return_over(closes, config.nano_momentum_lookback_bars)
    micro_return = _return_over(closes, config.micro_momentum_lookback_bars)
    meso_return = _return_over(daily_closes, config.meso_trend_lookback_days)
    macro_return = _return_over(daily_closes, config.macro_trend_lookback_days)

    return {
        "symbol": symbol.upper(),
        "nano_return": nano_return,
        "micro_return": micro_return,
        "meso_return": meso_return,
        "macro_return": macro_return,
        "macro_trend_ok": macro_return > 0.0,
        "realized_volatility": 0.0 if math.isnan(realized_volatility) else realized_volatility,
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


def get_intraday_bars(symbols: list[str], lookback_bars: int, config: Any, data_client: Any = None) -> dict[str, pd.DataFrame]:
    """Use Finnhub to fetch the last N 15-minute bars per symbol."""
    return fetch_intraday_market_bars(
        symbols,
        config,
        lookback_bars=lookback_bars,
        bar_minutes=15,
    )


def get_daily_bars(symbols: list[str], lookback_days: int, config: Any, data_client: Any = None) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV bars for the absolute momentum trend filter."""
    return fetch_daily_bars(
        symbols,
        lookback_days=lookback_days,
        ma_days=0,
        extra_buffer_days=5,
        alpaca_data_client=data_client,
        data_feed=getattr(config, "alpaca_data_feed", "iex"),
        include_latest=True,
        config=config,
    )


def get_sentiment_snapshot(
    symbols: list[str],
    lookback_minutes: int,
    config: Any,
) -> tuple[dict[str, float], float]:
    """Normalize recent provider sentiment records into symbol and market scores."""
    records: list[dict[str, Any]] = []
    providers = [str(item).lower() for item in getattr(config, "sentiment_data_provider_order", [])]
    if not providers:
        providers = [str(item).lower() for item in getattr(config, "news_sentiment_provider_order", [])]
    if not providers:
        providers = [""]
    for provider in providers:
        provider_config = config
        if provider and dataclasses.is_dataclass(config):
            provider_config = dataclasses.replace(config, news_sentiment_provider_order=[provider])
        try:
            records.extend(fetch_latest_news_sentiment(symbols, provider_config))
        except Exception as exc:
            logger.warning("Sentiment provider %s unavailable; continuing with neutral fallback: %s", provider or "default", exc)

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=max(lookback_minutes, 1))
    by_symbol: dict[str, list[float]] = {symbol.upper(): [] for symbol in symbols}
    for record in records:
        symbol = str(record.get("symbol", "")).upper()
        if symbol not in by_symbol:
            continue
        timestamp = pd.to_datetime(record.get("timestamp"), utc=True, errors="coerce")
        if not pd.isna(timestamp) and timestamp < cutoff:
            continue
        try:
            sentiment = float(record.get("social_score", record.get("sentiment", 0.0)))
        except (TypeError, ValueError):
            sentiment = 0.0
        by_symbol[symbol].append(max(-1.0, min(1.0, sentiment)))

    symbol_sentiment = {
        symbol: (sum(values) / len(values) if values else 0.0)
        for symbol, values in by_symbol.items()
    }
    market_sentiment = symbol_sentiment.get("SPY")
    if market_sentiment is None:
        values = list(symbol_sentiment.values())
        market_sentiment = sum(values) / len(values) if values else 0.0
    return symbol_sentiment, float(market_sentiment)


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
        candidates = [
            scores_by_symbol[symbol]
            for symbol in symbols
            if symbol in scores_by_symbol
            and (not require_macro_trend or bool(scores_by_symbol[symbol].get("macro_trend_ok")))
            and float(scores_by_symbol[symbol].get("score", 0.0)) >= min_score
            and (
                min_micro_return is None
                or float(scores_by_symbol[symbol].get("micro_return", 0.0)) >= min_micro_return
            )
        ]
        return sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)

    candidates = ranked(config.risk_on_universe, config.min_risk_on_score, config.min_risk_on_micro_return)
    candidates.extend(ranked(config.defensive_universe, config.min_defensive_score))
    candidates.sort(
        key=lambda item: float(item.get("score", 0.0)) + (score_delta if item["symbol"] in current_symbol_positions else 0.0),
        reverse=True,
    )
    selected = candidates[: max(config.max_positions, 0)]

    if not selected:
        selected = ranked(config.defensive_universe, config.min_defensive_score, require_macro_trend=False)[: max(config.max_positions, 0)]

    remaining = list(selected)
    remaining_exposure = max(config.max_gross_exposure, 0.0)
    while remaining and remaining_exposure > 0:
        positive_scores = [max(float(row.get("score", 0.0)), 0.0) for row in remaining]
        total_score = sum(positive_scores)
        allocated_any = False
        next_remaining: list[dict[str, Any]] = []
        for row, score in zip(remaining, positive_scores):
            raw_weight = (
                remaining_exposure / len(remaining)
                if total_score <= 0
                else remaining_exposure * score / total_score
            )
            if raw_weight >= config.max_single_position_weight:
                weights[str(row["symbol"])] = config.max_single_position_weight
                remaining_exposure -= config.max_single_position_weight
                allocated_any = True
            else:
                next_remaining.append(row)
        if not allocated_any:
            for row, score in zip(remaining, positive_scores):
                weight = (
                    remaining_exposure / len(remaining)
                    if total_score <= 0
                    else remaining_exposure * score / total_score
                )
                weights[str(row["symbol"])] = max(0.0, min(weight, config.max_single_position_weight))
            break
        remaining = next_remaining

    gross = sum(abs(weight) for weight in weights.values())
    if gross > config.max_gross_exposure > 0:
        scale = config.max_gross_exposure / gross
        weights = {symbol: weight * scale for symbol, weight in weights.items()}
    return weights


def apply_risk_guards(
    target_weights: dict[str, float],
    scores_by_symbol: dict[str, dict[str, Any]],
    current_weights: dict[str, float],
    equity: float,
    config: DefensiveMomentumConfig,
) -> dict[str, float]:
    """Apply position caps, volatility filters, turnover threshold, and drawdown kill-switch."""
    if intraday_kill_switch_triggered(equity, config):
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

    gross = sum(abs(weight) for weight in guarded.values())
    if gross > config.max_gross_exposure > 0:
        scale = config.max_gross_exposure / gross
        guarded = {symbol: weight * scale for symbol, weight in guarded.items()}
    return guarded


def intraday_kill_switch_triggered(equity: float, config: DefensiveMomentumConfig) -> bool:
    """Track account equity from the first run of the day and stop entries after a drawdown breach."""
    today = date.today().isoformat()
    state = load_state(STATE_KEY, {})
    if state.get("date") != today:
        state = {"date": today, "start_equity": equity, "halted": False}
    start_equity = float(state.get("start_equity") or equity)
    drawdown = equity / start_equity - 1.0 if start_equity > 0 else 0.0
    if drawdown <= config.intraday_drawdown_limit:
        state["halted"] = True
    state["last_equity"] = equity
    state["drawdown"] = drawdown
    save_state(STATE_KEY, state)
    return bool(state.get("halted"))


def weights_from_positions(current_positions: dict[str, int], latest_prices: dict[str, float], equity: float) -> dict[str, float]:
    """Convert current share positions to portfolio weights for turnover control."""
    if equity <= 0:
        return {}
    return {
        symbol: (shares * latest_prices.get(symbol, 0.0)) / equity
        for symbol, shares in current_positions.items()
    }


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
    return "FLAT_RANK" if any(weight > 0 for weight in target_weights.values()) else "CASH"


def build_defensive_momentum_targets(
    runtime_config: Any,
    data_client: Any,
    current_positions: dict[str, int],
    latest_prices: dict[str, float],
    equity: float,
) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, Any]]:
    """Run the full 30-minute strategy decision loop and return weights plus signal logs."""
    strategy_config = DefensiveMomentumConfig.from_runtime_config(runtime_config)
    symbols = strategy_config.symbols
    intraday_bars = get_intraday_bars(symbols, strategy_config.required_intraday_bars, runtime_config, data_client)
    daily_bars = get_daily_bars(symbols, strategy_config.required_daily_bars, runtime_config, data_client)
    sentiment, market_sentiment = get_sentiment_snapshot(symbols, strategy_config.sentiment_lookback_minutes, runtime_config)

    features = {
        symbol: compute_price_features(symbol, intraday_bars.get(symbol, pd.DataFrame()), daily_bars.get(symbol, pd.DataFrame()), strategy_config)
        for symbol in symbols
    }
    scores = compute_composite_scores(features, sentiment, strategy_config)
    current_weights = weights_from_positions(current_positions, latest_prices, equity)
    raw_weights = decide_target_weights(scores, strategy_config, current_weights)
    target_weights = apply_risk_guards(raw_weights, scores, current_weights, equity, strategy_config)
    mode = allocation_mode(target_weights)

    rows = rows_from_scores(scores, target_weights, mode)
    signals = {
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
            "sma_long": 0.0,
        }
        for row in rows
    }
    metadata = {
        "allocation_mode": mode,
        "market_sentiment": market_sentiment,
        "scores": scores,
        "raw_target_weights": raw_weights,
    }
    logger.info(
        "Fast Momentum decision mode=%s market_sentiment=%s targets=%s",
        mode,
        market_sentiment,
        target_weights,
    )
    return target_weights, signals, metadata


class FastMomentumAlgorithm(BaseAlgorithm):
    algorithm_id = "fast_momentum"

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        strategy_config = DefensiveMomentumConfig.from_runtime_config(config)
        return AlgorithmRequirements(
            price_symbols=sorted(set(strategy_config.symbols) | set(current_positions)),
            daily_lookback_days=strategy_config.required_daily_bars,
            paper_only=True,
        )

    def decide(self, context: AlgorithmContext) -> AlgorithmDecision:
        strategy_config = DefensiveMomentumConfig.from_runtime_config(context.config)
        target_weights, signals, metadata = build_defensive_momentum_targets(
            context.config,
            context.extra.get("data_client"),
            context.positions,
            context.latest_prices,
            context.equity,
        )
        return AlgorithmDecision(
            target_weights=target_weights,
            signals=signals,
            metadata=metadata,
            cash_buffer=0.0,
            min_trade_dollars=strategy_config.per_trade_value_min,
            rebalance_threshold=strategy_config.rebalance_threshold,
        )
