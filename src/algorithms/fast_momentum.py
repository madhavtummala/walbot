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
from ..core.interfaces import AlgorithmContext, AlgorithmDecision, AlgorithmRequirements, Schedule
from ..data import fetch_daily_bars
from ..data.state_store import load_state, save_state

logger = logging.getLogger(__name__)

STATE_KEY = "defensive_momentum_intraday_risk"


@dataclass(frozen=True)
class DefensiveMomentumConfig:
    """Configurable knobs for the intraday defensive momentum + sentiment strategy."""

    risk_on_universe: list[str] = field(default_factory=lambda: ["QQQ", "VTI", "IWM", "IEMG", "ACWI"])
    defensive_universe: list[str] = field(default_factory=lambda: ["BIL", "IEF", "AGG", "TLT", "GLD"])
    #: The grid the ``*_lookback_bars`` knobs below are counted on. 15 minutes is what they
    #: were fitted against -- micro 78 is exactly three sessions of 15m bars -- so changing it
    #: rescales those horizons in wall-clock terms. Backtest before moving it.
    intraday_bar_minutes: int = 15
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
            intraday_bar_minutes=integer("intraday_bar_minutes", defaults.intraday_bar_minutes),
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


def get_intraday_bars(
    symbols: list[str],
    lookback_bars: int,
    config: Any,
    data_client: Any = None,
    bar_minutes: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch the last N intraday bars per symbol, on this strategy's own grid."""
    return fetch_intraday_market_bars(
        symbols,
        config,
        lookback_bars=lookback_bars,
        bar_minutes=bar_minutes or DefensiveMomentumConfig.from_runtime_config(config).intraday_bar_minutes,
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

    symbol_sentiment, market_sentiment, _metadata, _providers = sentiment_scores_from_records(
        symbols, records, lookback_minutes
    )
    return symbol_sentiment, market_sentiment


def sentiment_scores_from_records(
    symbols: list[str],
    records: list[dict[str, Any]],
    lookback_minutes: int,
) -> tuple[dict[str, float], float, dict[str, Any], list[str]]:
    """Normalise raw provider sentiment records into per-symbol and market scores.

    Pure: takes already-fetched records so callers control provider selection and fetching.
    Returns ``(by_symbol, market_sentiment, metadata, providers_seen)``. Market sentiment
    falls back to the universe average when SPY is not covered.
    """
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=max(lookback_minutes, 1))
    by_symbol: dict[str, list[float]] = {symbol.upper(): [] for symbol in symbols}
    providers_seen: list[str] = []
    used = 0

    for record in records or []:
        provider = str(record.get("provider", "")).lower()
        if provider and provider not in providers_seen:
            providers_seen.append(provider)
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
        used += 1

    symbol_sentiment = {
        symbol: (sum(values) / len(values) if values else 0.0)
        for symbol, values in by_symbol.items()
    }
    market_sentiment = symbol_sentiment.get("SPY")
    if market_sentiment is None:
        values = list(symbol_sentiment.values())
        market_sentiment = sum(values) / len(values) if values else 0.0

    metadata = {
        "records_seen": len(records or []),
        "records_used": used,
        "covered_symbols": sum(1 for values in by_symbol.values() if values),
    }
    return symbol_sentiment, float(market_sentiment), metadata, providers_seen


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
                context.intraday_bars_by_symbol.get(symbol, pd.DataFrame()),
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
            intraday_lookback_bars=strategy_config.required_intraday_bars,
            intraday_bar_minutes=strategy_config.intraday_bar_minutes,
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
    ) -> dict[str, float]:
        """Protect incumbents from churn, then apply the position-aware risk guards."""
        strategy_config = DefensiveMomentumConfig.from_runtime_config(config)
        current_weights = snapshot.weights(latest_prices)
        scores = {symbol: dict(row) for symbol, row in signals.items()}
        for symbol, row in scores.items():
            row.setdefault("symbol", symbol)

        kept = apply_stickiness(target_weights, scores, current_weights, strategy_config)
        return apply_risk_guards(kept, scores, current_weights, snapshot.equity, strategy_config)
