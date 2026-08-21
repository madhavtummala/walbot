"""Intraday pick: direction from macro trend, name from intraday setup."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

from ...core.interfaces import (
    AlgorithmContext,
    AlgorithmDecision,
    AlgorithmRequirements,
    Intent,
    Schedule,
)
from ..base import BaseAlgorithm
from .config import IntradayPickConfig

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────


def _pct_change(series: pd.Series, periods: int) -> float:
    if series is None or len(series) <= periods:
        return 0.0
    current = float(series.iloc[-1])
    prior = float(series.iloc[-periods - 1])
    return (current / prior) - 1.0 if prior else 0.0


def _true_range(bars: pd.DataFrame) -> pd.Series:
    high = bars["high"]
    low = bars["low"]
    prev_close = bars["close"].shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(bars: pd.DataFrame, period: int) -> float:
    if len(bars) < period + 1:
        return 0.0
    tr = _true_range(bars)
    return float(tr.tail(period).mean())


def _realized_vol(closes: pd.Series, window: int) -> float:
    if closes is None or len(closes) < window + 1:
        return 0.0
    returns = closes.pct_change().dropna().tail(window)
    return float(returns.std() * math.sqrt(252)) if len(returns) >= 2 else 0.0


def _volume_ratio(volumes: pd.Series, window: int = 20) -> float:
    if volumes is None or len(volumes) < window + 1:
        return 0.0
    avg_vol = float(volumes.tail(window + 1).head(window).mean())
    today = float(volumes.iloc[-1])
    return today / avg_vol if avg_vol > 0 else 0.0


def _range_today(bars: pd.DataFrame) -> float:
    if bars is None or bars.empty:
        return 0.0
    last = bars.iloc[-1]
    close = float(last["close"])
    if close <= 0:
        return 0.0
    return (float(last["high"]) - float(last["low"])) / close


def _vwap(bars: pd.DataFrame) -> float:
    if bars is None or bars.empty:
        return 0.0
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    cum_tv = (typical * bars["volume"]).cumsum()
    cum_vol = bars["volume"].cumsum()
    vwap_series = cum_tv / cum_vol.replace(0, float("nan"))
    val = vwap_series.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _intraday_momentum(bars: pd.DataFrame, lookback: int) -> float:
    if bars is None or len(bars) < lookback * 2:
        return 0.0
    recent = bars["close"].tail(lookback).mean()
    prior = bars["close"].iloc[-(lookback * 2):-lookback].mean()
    return (recent / prior) - 1.0 if prior > 0 else 0.0


# ── macro trend ─────────────────────────────────────────────────────────────


def detect_macro_trend(
    benchmark_bars: pd.DataFrame, cfg: IntradayPickConfig
) -> str:
    """Return 'bull', 'bear', or 'flat' based on the benchmark daily bars."""
    if benchmark_bars is None or len(benchmark_bars) < cfg.trend_ma_period:
        return "flat"

    closes = benchmark_bars["close"]
    ma = closes.tail(cfg.trend_ma_period).mean()
    last_price = float(closes.iloc[-1])
    recent_return = _pct_change(closes, cfg.trend_lookback_days)
    above_ma = last_price > ma
    trend_return_ok = abs(recent_return) >= abs(cfg.trend_min_return)

    if above_ma and recent_return > 0 and trend_return_ok:
        return "bull"
    if not above_ma and recent_return < 0 and trend_return_ok:
        return "bear"
    return "flat"


# ── candidate scoring ────────────────────────────────────────────────────────


def score_candidate(
    daily_bars: pd.DataFrame,
    intraday_bars: pd.DataFrame,
    cfg: IntradayPickConfig,
    macro_direction: str,
) -> Dict[str, Any]:
    """Score a single symbol for suitability as an options pick."""
    if daily_bars is None or daily_bars.empty:
        return {"score": 0.0, "reason": "no daily bars"}

    closes = daily_bars["close"]
    last_price = float(closes.iloc[-1])

    if last_price < cfg.min_price or last_price > cfg.max_price:
        return {"score": 0.0, "reason": f"price {last_price:.2f} outside range"}

    vol_short = _realized_vol(closes, cfg.vol_short_window)
    vol_long = _realized_vol(closes, cfg.vol_long_window)
    vol_regime = vol_short / vol_long if vol_long > 0 else 0.0

    atr = _atr(daily_bars, cfg.atr_period)
    today_range = _range_today(daily_bars)
    atr_pct = atr / last_price if last_price > 0 else 0.0
    range_expansion = today_range / atr_pct if atr_pct > 0 else 0.0

    vol_ratio = _volume_ratio(daily_bars["volume"])

    momentum = 0.0
    above_vwap = False
    intraday_range_left = 0.0
    if intraday_bars is not None and not intraday_bars.empty and len(intraday_bars) >= cfg.intraday_momentum_bars:
        momentum = _intraday_momentum(intraday_bars, cfg.intraday_momentum_bars)
        vwap = _vwap(intraday_bars)
        last_close = float(intraday_bars["close"].iloc[-1])
        above_vwap = last_close > vwap if vwap > 0 else False
        intraday_range = _range_today(intraday_bars)
        if atr_pct > 0:
            intraday_range_left = max(0.0, 1.0 - (intraday_range / atr_pct))

    if macro_direction == "bull":
        direction_aligned = momentum >= 0 and above_vwap
    elif macro_direction == "bear":
        direction_aligned = momentum <= 0 and not above_vwap
    else:
        direction_aligned = False

    score = 0.0
    reasons: list[str] = []

    if vol_regime >= cfg.vol_regime_threshold:
        score += min(vol_regime, 3.0)
        reasons.append(f"vol regime {vol_regime:.2f}")
    else:
        reasons.append(f"vol regime low {vol_regime:.2f}")

    if range_expansion >= cfg.range_expansion_threshold:
        score += min(range_expansion, 3.0)
        reasons.append(f"range expansion {range_expansion:.2f}")

    if direction_aligned:
        score += 2.0
        reasons.append("direction aligned")

    if intraday_range_left >= cfg.intraday_range_budget_pct:
        score += 1.5
        reasons.append(f"range budget {intraday_range_left:.0%} remaining")

    if vol_ratio >= cfg.min_volume_ratio:
        score += 0.5
        reasons.append(f"vol ratio {vol_ratio:.2f}")

    return {
        "score": score,
        "reason": "; ".join(reasons),
        "last_price": last_price,
        "vol_short": vol_short,
        "vol_long": vol_long,
        "vol_regime": vol_regime,
        "atr": atr,
        "atr_pct": atr_pct,
        "range_expansion": range_expansion,
        "volume_ratio": vol_ratio,
        "intraday_momentum": momentum,
        "above_vwap": above_vwap,
        "intraday_range_left": intraday_range_left,
        "direction_aligned": direction_aligned,
    }


# ── option pricing estimation ────────────────────────────────────────────────


def estimate_option_range(
    underlying_price: float,
    atr: float,
    vol_annualized: float,
    dte: int,
    cfg: IntradayPickConfig,
    option_type: str = "call",
) -> Dict[str, Any]:
    """Estimate the fair-value range of an option for the day.

    Uses a simplified approach: intrinsic + time value based on expected move.
    The entry limit is placed below fair value (discount), the exit limit above.

    Returns: {fair_value, entry_limit, exit_limit, expected_move}
    """
    if underlying_price <= 0 or atr <= 0:
        return {"fair_value": 0.0, "entry_limit": 0.0, "exit_limit": 0.0, "expected_move": 0.0}

    daily_move = underlying_price * (vol_annualized / math.sqrt(252)) if vol_annualized > 0 else atr
    expected_move = daily_move * math.sqrt(max(dte, 1))

    if option_type == "call":
        intrinsic = max(underlying_price * 0.02, 0.50)
    else:
        intrinsic = max(underlying_price * 0.02, 0.50)

    time_value = expected_move * 0.40
    fair_value = intrinsic + time_value

    entry_limit = round(max(fair_value * (1.0 - cfg.entry_discount_pct), 0.01), 2)
    exit_limit = round(fair_value * (1.0 + cfg.exit_target_pct), 2)

    return {
        "fair_value": round(fair_value, 2),
        "entry_limit": entry_limit,
        "exit_limit": exit_limit,
        "expected_move": round(expected_move, 2),
        "daily_move": round(daily_move, 2),
    }


# ── option contract selection ────────────────────────────────────────────────


def _option_type_for_trend(macro_direction: str) -> str:
    return "call" if macro_direction == "bull" else "put"


def _select_strike(last_price: float, option_type: str, cfg: IntradayPickConfig) -> float:
    """Pick a strike near ATM based on delta targets."""
    if option_type == "call":
        target_delta = (cfg.call_delta_min + cfg.call_delta_max) / 2.0
    else:
        target_delta = (cfg.put_delta_min + cfg.put_delta_max) / 2.0

    if option_type == "call":
        offset_pct = 1.0 - target_delta
    else:
        offset_pct = abs(target_delta)

    return round(last_price * offset_pct, 2)


def _pick_expiration(dte_min: int, dte_max: int) -> int:
    """Pick the nearest expiration within the DTE window."""
    return max(dte_min, min(1, dte_max))


# ── algorithm ────────────────────────────────────────────────────────────────


STATE_KEY = "intraday_pick_runtime"


class IntradayPickAlgorithm(BaseAlgorithm):
    """Directional options: buy calls in bull trends, puts in bear trends, when the
    intraday setup confirms the move.  Entry and exit are both limit orders."""

    algorithm_id = "intraday_pick"

    #: Runs every 15 minutes during regular market hours.
    schedule = Schedule(
        refresh_minutes=15,
        jitter_minutes=2,
        start_time="09:30",
        end_time="15:45",
    )

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        cfg = IntradayPickConfig.from_runtime_config(config)
        symbols = [cfg.benchmark_symbol]
        if cfg.candidate_symbols:
            symbols.extend(s for s in cfg.candidate_symbols if s not in symbols)
        return AlgorithmRequirements(
            price_symbols=symbols,
            daily_lookback_days=cfg.required_daily_bars,
            daily_ma_days=cfg.trend_ma_period,
            intraday_lookback_minutes=cfg.required_intraday_bars * cfg.intraday_bar_minutes,
            preferred_bar_minutes=cfg.intraday_bar_minutes,
            paper_only=True,
        )

    def analyze(self, context: AlgorithmContext) -> AlgorithmDecision:
        cfg = IntradayPickConfig.from_runtime_config(context.config)

        benchmark_bars = context.daily_bars_by_symbol.get(cfg.benchmark_symbol)
        if benchmark_bars is None or benchmark_bars.empty:
            return AlgorithmDecision(
                metadata={"reason": f"no data for benchmark {cfg.benchmark_symbol}", "macro": "flat"},
            )

        macro = detect_macro_trend(benchmark_bars, cfg)
        if macro == "flat":
            return AlgorithmDecision(
                metadata={"reason": "macro trend flat; sitting in cash", "macro": macro},
            )

        candidates = []
        allowed = set(cfg.candidate_symbols) if cfg.candidate_symbols else None
        for symbol in context.daily_bars_by_symbol:
            if symbol == cfg.benchmark_symbol:
                continue
            if symbol in context.positions:
                continue
            if allowed is not None and symbol not in allowed:
                continue
            daily_bars = context.daily_bars_by_symbol[symbol]
            intraday_bars = context.intraday_bars_by_symbol.get(symbol)
            result = score_candidate(daily_bars, intraday_bars, cfg, macro)
            result["symbol"] = symbol
            candidates.append(result)

        candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
        top = [c for c in candidates[: cfg.top_n_candidates] if c.get("score", 0) > 0]

        if not top:
            return AlgorithmDecision(
                metadata={"reason": "no qualifying candidates", "macro": macro, "candidates_evaluated": len(candidates)},
            )

        option_type = cfg.force_option_type if cfg.force_option_type else _option_type_for_trend(macro)
        best = top[0]
        last_price = best["last_price"]
        atr = best["atr"]
        vol = best.get("vol_short", 0.0)
        dte = _pick_expiration(cfg.option_dte_min, cfg.option_dte_max)
        strike = _select_strike(last_price, option_type, cfg)
        pricing = estimate_option_range(last_price, atr, vol, dte, cfg, option_type)

        contracts = cfg.contracts_per_signal
        cost_per_contract = pricing["entry_limit"] * 100
        total_cost = cost_per_contract * contracts
        if total_cost > cfg.max_notional_per_trade and cost_per_contract > 0:
            contracts = max(1, int(cfg.max_notional_per_trade / cost_per_contract))

        intent = Intent(
            symbol=best["symbol"],
            kind="option",
            value=contracts,
        )

        signals = {best["symbol"]: {
            "symbol": best["symbol"],
            "option_type": option_type,
            "strike": strike,
            "dte": dte,
            "contracts": contracts,
            "entry_limit": pricing["entry_limit"],
            "exit_limit": pricing["exit_limit"],
            "fair_value": pricing["fair_value"],
            "macro": macro,
            "reason": best["reason"],
            "score": best["score"],
            "vol_regime": best.get("vol_regime", 0.0),
            "range_expansion": best.get("range_expansion", 0.0),
        }}

        return AlgorithmDecision(
            intents=[intent],
            signals=signals,
            metadata={
                "macro": macro,
                "option_type": option_type,
                "strike": strike,
                "dte": dte,
                "contracts": contracts,
                "entry_limit": pricing["entry_limit"],
                "exit_limit": pricing["exit_limit"],
                "pricing": pricing,
                "candidates_evaluated": len(candidates),
            },
            mode="incremental",
        )

    def refine(
        self,
        intents: list[Intent],
        signals: dict[str, dict[str, Any]],
        snapshot: Any,
        latest_prices: dict[str, float],
        config: Any,
        as_of: datetime,
    ) -> list[Intent]:
        """Skip refinement: options intents go directly to the order pipeline."""
        return list(intents)

    def sizing(self, config: Any) -> dict[str, float]:
        return {"min_trade_dollars": 0.0, "rebalance_threshold": 0.0}
