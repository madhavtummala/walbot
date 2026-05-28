from __future__ import annotations

import dataclasses
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from .alpaca_client import get_historical_intraday_bars
from .connectors import fetch_latest_news_sentiment
from .data import fetch_daily_bars
from .state_store import load_state, save_state

logger = logging.getLogger(__name__)

STATE_KEY = "defensive_momentum_intraday_risk"


@dataclass(frozen=True)
class DefensiveMomentumConfig:
    """Configurable knobs for the intraday defensive momentum + sentiment strategy."""

    risk_on_universe: list[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "VTI", "IWM", "IJH", "IJR", "IEFA", "IEMG", "ACWI", "ACWX"]
    )
    defensive_universe: list[str] = field(
        default_factory=lambda: ["BIL", "SHY", "SPTS", "IEF", "GOVT", "AGG", "BND", "IUSB", "STIP", "TLT", "GLD"]
    )
    regime_symbol: str = "SPY"
    price_lookback_short_bars: int = 1
    price_lookback_medium_bars: int = 6
    price_lookback_daily_bars: int = 13
    daily_abs_momentum_lookback_days: int = 20
    sentiment_lookback_minutes: int = 60
    max_gross_exposure: float = 1.0
    max_single_position_weight: float = 0.25
    max_risk_on_positions: int = 4
    max_defensive_positions: int = 2
    cautious_max_risk_on_exposure: float = 0.30
    cautious_min_risk_on_score: float = 1.0
    per_trade_value_min: float = 50.0
    w_price_short: float = 0.4
    w_price_medium: float = 0.3
    w_price_daily: float = 0.2
    w_sentiment: float = 0.1
    regime_bull_price_threshold: float = 0.01
    regime_bear_price_threshold: float = -0.01
    regime_sentiment_positive: float = 0.2
    regime_sentiment_negative: float = -0.2
    volatility_lookback_bars: int = 13
    max_intraday_volatility: float = 0.06
    high_volatility_weight_scale: float = 0.5
    intraday_drawdown_limit: float = -0.02

    @classmethod
    def from_runtime_config(cls, config: Any) -> "DefensiveMomentumConfig":
        raw = {}
        if isinstance(getattr(config, "algorithm_configs", None), dict):
            raw = config.algorithm_configs.get("defensive_momentum", {}) or {}
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
            regime_symbol=str(raw.get("regime_symbol", defaults.regime_symbol)).strip().upper() or defaults.regime_symbol,
            price_lookback_short_bars=integer("price_lookback_short_bars", defaults.price_lookback_short_bars),
            price_lookback_medium_bars=integer("price_lookback_medium_bars", defaults.price_lookback_medium_bars),
            price_lookback_daily_bars=integer("price_lookback_daily_bars", defaults.price_lookback_daily_bars),
            daily_abs_momentum_lookback_days=integer("daily_abs_momentum_lookback_days", defaults.daily_abs_momentum_lookback_days),
            sentiment_lookback_minutes=integer("sentiment_lookback_minutes", defaults.sentiment_lookback_minutes),
            max_gross_exposure=number("max_gross_exposure", defaults.max_gross_exposure),
            max_single_position_weight=number("max_single_position_weight", defaults.max_single_position_weight),
            max_risk_on_positions=integer("max_risk_on_positions", defaults.max_risk_on_positions),
            max_defensive_positions=integer("max_defensive_positions", defaults.max_defensive_positions),
            cautious_max_risk_on_exposure=number("cautious_max_risk_on_exposure", defaults.cautious_max_risk_on_exposure),
            cautious_min_risk_on_score=number("cautious_min_risk_on_score", defaults.cautious_min_risk_on_score),
            per_trade_value_min=number("per_trade_value_min", getattr(config, "min_trade_dollars", defaults.per_trade_value_min)),
            w_price_short=number("w_price_short", defaults.w_price_short),
            w_price_medium=number("w_price_medium", defaults.w_price_medium),
            w_price_daily=number("w_price_daily", defaults.w_price_daily),
            w_sentiment=number("w_sentiment", defaults.w_sentiment),
            regime_bull_price_threshold=number("regime_bull_price_threshold", defaults.regime_bull_price_threshold),
            regime_bear_price_threshold=number("regime_bear_price_threshold", defaults.regime_bear_price_threshold),
            regime_sentiment_positive=number("regime_sentiment_positive", defaults.regime_sentiment_positive),
            regime_sentiment_negative=number("regime_sentiment_negative", defaults.regime_sentiment_negative),
            volatility_lookback_bars=integer("volatility_lookback_bars", defaults.volatility_lookback_bars),
            max_intraday_volatility=number("max_intraday_volatility", defaults.max_intraday_volatility),
            high_volatility_weight_scale=number("high_volatility_weight_scale", defaults.high_volatility_weight_scale),
            intraday_drawdown_limit=number("intraday_drawdown_limit", defaults.intraday_drawdown_limit),
        )

    @property
    def symbols(self) -> list[str]:
        return sorted(set(self.risk_on_universe) | set(self.defensive_universe) | {self.regime_symbol})

    @property
    def required_intraday_bars(self) -> int:
        return max(
            self.price_lookback_short_bars,
            self.price_lookback_medium_bars,
            self.price_lookback_daily_bars,
            self.volatility_lookback_bars,
        ) + 1


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

    daily_return = _return_over(daily_closes, config.daily_abs_momentum_lookback_days)
    intraday_returns = closes.pct_change().dropna().tail(config.volatility_lookback_bars)
    realized_volatility = float(intraday_returns.std()) if not intraday_returns.empty else 0.0

    return {
        "symbol": symbol.upper(),
        "short_return": _return_over(closes, config.price_lookback_short_bars),
        "medium_return": _return_over(closes, config.price_lookback_medium_bars),
        "daily_bar_return": _return_over(closes, config.price_lookback_daily_bars),
        "daily_return": daily_return,
        "daily_trend_ok": daily_return > 0.0,
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
    zscores = zscores_by_feature(features_by_symbol, ["short_return", "medium_return", "daily_return"])
    scored: dict[str, dict[str, Any]] = {}
    for symbol, features in features_by_symbol.items():
        sentiment = max(-1.0, min(1.0, float(sentiment_scores.get(symbol, 0.0))))
        components = {
            "price_short": config.w_price_short * zscores[symbol]["short_return"],
            "price_medium": config.w_price_medium * zscores[symbol]["medium_return"],
            "price_daily": config.w_price_daily * zscores[symbol]["daily_return"],
            "sentiment": config.w_sentiment * sentiment,
        }
        score = sum(components.values())
        scored[symbol] = {**features, "symbol": symbol, "score": score, "sentiment_score": sentiment, "components": components}
    return scored


def compute_market_regime(
    spy_price_features: dict[str, Any],
    market_sentiment: float,
    config: DefensiveMomentumConfig,
) -> tuple[str, dict[str, float]]:
    """Classify the current market as RISK_ON, RISK_OFF, or CAUTIOUS."""
    daily_return = float(spy_price_features.get("daily_return", 0.0))
    sentiment = float(market_sentiment)
    if daily_return > config.regime_bull_price_threshold and sentiment > config.regime_sentiment_positive:
        regime = "RISK_ON"
    elif daily_return < config.regime_bear_price_threshold and sentiment < config.regime_sentiment_negative:
        regime = "RISK_OFF"
    else:
        regime = "CAUTIOUS"
    return regime, {"daily_return": daily_return, "market_sentiment": sentiment}


def get_intraday_bars(symbols: list[str], lookback_bars: int, config: Any, data_client: Any = None) -> dict[str, pd.DataFrame]:
    """Use Alpaca to fetch the last N 30-minute bars per symbol."""
    return get_historical_intraday_bars(
        symbols,
        lookback_bars=lookback_bars,
        bar_minutes=30,
        data_client=data_client,
        data_feed=getattr(config, "alpaca_data_feed", "iex"),
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
    regime: str,
    config: DefensiveMomentumConfig,
) -> dict[str, float]:
    """Apply dual momentum ranking and regime rules to produce target weights."""
    weights = {symbol: 0.0 for symbol in config.symbols}

    def ranked(symbols: list[str], min_score: float) -> list[dict[str, Any]]:
        candidates = [
            scores_by_symbol[symbol]
            for symbol in symbols
            if symbol in scores_by_symbol
            and bool(scores_by_symbol[symbol].get("daily_trend_ok"))
            and float(scores_by_symbol[symbol].get("score", 0.0)) >= min_score
        ]
        return sorted(candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)

    def assign(candidates: list[dict[str, Any]], total_cap: float, position_limit: int) -> None:
        selected = candidates[: max(position_limit, 0)]
        if not selected:
            return
        per_symbol = min(config.max_single_position_weight, total_cap / len(selected))
        for row in selected:
            weights[str(row["symbol"])] = max(0.0, per_symbol)

    if regime == "RISK_ON":
        assign(ranked(config.risk_on_universe, 0.0), config.max_gross_exposure, config.max_risk_on_positions)
    elif regime == "RISK_OFF":
        assign(ranked(config.defensive_universe, 0.0), config.max_gross_exposure, config.max_defensive_positions)
    else:
        assign(
            ranked(config.risk_on_universe, config.cautious_min_risk_on_score),
            config.cautious_max_risk_on_exposure,
            config.max_risk_on_positions,
        )
        remaining = max(0.0, config.max_gross_exposure - sum(weights.values()))
        assign(ranked(config.defensive_universe, 0.0), remaining, config.max_defensive_positions)

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
        logger.warning("Defensive Momentum kill-switch active; target weights set to zero")
        return {symbol: 0.0 for symbol in target_weights}

    guarded: dict[str, float] = {}
    for symbol, weight in target_weights.items():
        capped = min(max(weight, 0.0), config.max_single_position_weight)
        realized_vol = float(scores_by_symbol.get(symbol, {}).get("realized_volatility", 0.0))
        if config.max_intraday_volatility > 0 and realized_vol > config.max_intraday_volatility:
            capped *= max(0.0, min(1.0, config.high_volatility_weight_scale))
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
    regime: str,
) -> list[dict[str, Any]]:
    """Build dashboard/log signal rows from scored symbols and final weights."""
    rows = []
    for symbol, row in scores_by_symbol.items():
        weight = float(target_weights.get(symbol, 0.0))
        rows.append(
            {
                "symbol": symbol,
                "signal": 1 if weight > 0 else 0,
                "score": float(row.get("score", 0.0)),
                "price_score": float(row.get("daily_return", 0.0)),
                "social_score": float(row.get("sentiment_score", 0.0)),
                "volume_score": 0.0,
                "ret_N": float(row.get("medium_return", 0.0)),
                "sma_long": 0.0,
                "regime": regime,
            }
        )
    return rows


def build_defensive_momentum_targets(
    runtime_config: Any,
    data_client: Any,
    current_positions: dict[str, int],
    latest_prices: dict[str, float],
    equity: float,
) -> tuple[dict[str, float], dict[str, dict[str, float | int]], dict[str, Any]]:
    """Run the full 30-minute strategy decision loop and return weights plus signal logs."""
    strategy_config = DefensiveMomentumConfig.from_runtime_config(runtime_config)
    symbols = strategy_config.symbols
    intraday_bars = get_intraday_bars(symbols, strategy_config.required_intraday_bars, runtime_config, data_client)
    daily_bars = get_daily_bars(symbols, strategy_config.daily_abs_momentum_lookback_days, runtime_config, data_client)
    sentiment, market_sentiment = get_sentiment_snapshot(symbols, strategy_config.sentiment_lookback_minutes, runtime_config)

    features = {
        symbol: compute_price_features(symbol, intraday_bars.get(symbol, pd.DataFrame()), daily_bars.get(symbol, pd.DataFrame()), strategy_config)
        for symbol in symbols
    }
    scores = compute_composite_scores(features, sentiment, strategy_config)
    regime, regime_inputs = compute_market_regime(scores.get(strategy_config.regime_symbol, {}), market_sentiment, strategy_config)
    raw_weights = decide_target_weights(scores, regime, strategy_config)
    current_weights = weights_from_positions(current_positions, latest_prices, equity)
    target_weights = apply_risk_guards(raw_weights, scores, current_weights, equity, strategy_config)

    rows = rows_from_scores(scores, target_weights, regime)
    signals = {
        row["symbol"]: {
            "signal": int(row["signal"]),
            "score": float(row["score"]),
            "price_score": float(row["price_score"]),
            "social_score": float(row["social_score"]),
            "volume_score": 0.0,
            "ret_N": float(row["ret_N"]),
            "sma_long": 0.0,
        }
        for row in rows
    }
    metadata = {
        "regime": regime,
        "regime_inputs": regime_inputs,
        "scores": scores,
        "raw_target_weights": raw_weights,
    }
    logger.info(
        "Defensive Momentum decision regime=%s inputs=%s targets=%s",
        regime,
        regime_inputs,
        target_weights,
    )
    return target_weights, signals, metadata
