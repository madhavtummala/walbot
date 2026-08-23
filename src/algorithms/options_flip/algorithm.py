"""Options Flip: one contract at a time, bought cheap and flipped dearer.

Two gates in series, and the first is the one that usually closes. The benchmark's trend decides
whether to buy calls, buy puts, or sit out entirely; only if it has a direction are candidates
scored on volatility regime, range expansion and intraday alignment. Entry and exit are both
limit orders, placed together, so the position is sized for the loss the moment it is opened.

The name is the whole intent: a *flip* rather than a holding. It opens one contract at
``entry_discount_pct`` below fair value and closes it at ``exit_target_pct`` above, and the
direction it takes is the only thing the macro trend decides -- a bear reading buys puts on the
same reasoning a bull reading buys calls, so "cheap" and "dearer" mean the same thing either way.

Deliberately options-only, which is what the rename settled. Choosing *which* contract -- strike,
expiry, the delta band the config already names -- is the part still to be written; today it
sizes against the underlying and treats ``contracts_per_signal`` as the whole position.

Two things the docstring should not let you assume, because they are not implemented:

* **The exit is a price target, not a clock.** The sell limit rests until it fills. Nothing here
  closes at the bell or on the next session, so a contract can be held far longer than "flip"
  suggests -- or never sold at all.
* **Nothing enforces one position at a time.** ``contracts_per_signal`` bounds a single order,
  not the book, so a second signal on a second candidate opens alongside the first.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict

import pandas as pd

from ...core.interfaces import (
    MODE_INCREMENTAL,
    AlgorithmContext,
    AlgorithmPlan,
    AlgorithmRequirements,
    Check,
    Intent,
    SignalView,
)
from ..base import BaseAlgorithm
from .config import OptionsFlipConfig
from .signals import macro_view, signal_rows, signal_view

logger = logging.getLogger(__name__)


# ── bar maths ────────────────────────────────────────────────────────────────


def _pct_change(series: pd.Series, periods: int) -> float:
    if series is None or len(series) <= periods:
        return 0.0
    current = float(series.iloc[-1])
    prior = float(series.iloc[-periods - 1])
    return (current / prior) - 1.0 if prior else 0.0


def _true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    return pd.concat(
        [bars["high"] - bars["low"], (bars["high"] - prev_close).abs(), (bars["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def _atr(bars: pd.DataFrame, period: int) -> float:
    if len(bars) < period + 1:
        return 0.0
    return float(_true_range(bars).tail(period).mean())


def _realized_vol(closes: pd.Series, window: int) -> float:
    if closes is None or len(closes) < window + 1:
        return 0.0
    returns = closes.pct_change().dropna().tail(window)
    return float(returns.std() * math.sqrt(252)) if len(returns) >= 2 else 0.0


def _volume_ratio(volumes: pd.Series, window: int = 20) -> float:
    if volumes is None or len(volumes) < window + 1:
        return 0.0
    avg_vol = float(volumes.tail(window + 1).head(window).mean())
    return float(volumes.iloc[-1]) / avg_vol if avg_vol > 0 else 0.0


def _range_today(bars: pd.DataFrame) -> float:
    if bars is None or bars.empty:
        return 0.0
    last = bars.iloc[-1]
    close = float(last["close"])
    return (float(last["high"]) - float(last["low"])) / close if close > 0 else 0.0


def _vwap(bars: pd.DataFrame) -> float:
    if bars is None or bars.empty:
        return 0.0
    typical = (bars["high"] + bars["low"] + bars["close"]) / 3.0
    vwap_series = (typical * bars["volume"]).cumsum() / bars["volume"].cumsum().replace(0, float("nan"))
    value = vwap_series.iloc[-1]
    return float(value) if pd.notna(value) else 0.0


def _intraday_momentum(bars: pd.DataFrame, lookback: int) -> float:
    if bars is None or len(bars) < lookback * 2:
        return 0.0
    recent = bars["close"].tail(lookback).mean()
    prior = bars["close"].iloc[-(lookback * 2):-lookback].mean()
    return (recent / prior) - 1.0 if prior > 0 else 0.0


# ── macro trend: the gate that decides whether to trade at all ────────────────


def detect_macro_trend(benchmark_bars: pd.DataFrame, cfg: OptionsFlipConfig) -> tuple[str, list[dict[str, Any]]]:
    """'bull', 'bear' or 'flat' from the benchmark's daily bars, with the gates it applied.

    Returns the checks alongside the verdict because "flat" is the most common answer this
    algorithm gives and the least informative on its own: the deck needs to say whether the
    benchmark was below its average, or above it but not moving fast enough to bother.
    """
    if benchmark_bars is None or len(benchmark_bars) < cfg.trend_ma_period:
        have = 0 if benchmark_bars is None else len(benchmark_bars)
        return "flat", [_check(
            f"{cfg.trend_ma_period}-day benchmark history", False,
            f"{have} bars", f">= {cfg.trend_ma_period} bars", blocking=True,
        )]

    closes = benchmark_bars["close"]
    ma = float(closes.tail(cfg.trend_ma_period).mean())
    last_price = float(closes.iloc[-1])
    recent_return = _pct_change(closes, cfg.trend_lookback_days)
    above_ma = last_price > ma
    trend_return_ok = abs(recent_return) >= abs(cfg.trend_min_return)

    if above_ma and recent_return > 0 and trend_return_ok:
        macro = "bull"
    elif not above_ma and recent_return < 0 and trend_return_ok:
        macro = "bear"
    else:
        macro = "flat"

    directional = macro != "flat"
    return macro, [
        _check(
            f"Benchmark vs {cfg.trend_ma_period}-day average", True,
            f"${last_price:,.2f} {'above' if above_ma else 'below'} ${ma:,.2f}",
            "either side sets the direction",
        ),
        _check(
            f"{cfg.trend_lookback_days}-day move has conviction", trend_return_ok,
            f"{recent_return:+.2%}", f"|move| >= {abs(cfg.trend_min_return):.2%}",
            blocking=not trend_return_ok,
        ),
        _check(
            "Trend and position agree", directional,
            f"{'above' if above_ma else 'below'} average, moving {recent_return:+.2%}",
            "both pointing the same way",
            blocking=trend_return_ok and not directional,
        ),
    ]


def _check(label: str, ok: bool, value: str = "", limit: str = "", *, blocking: bool = False) -> dict[str, Any]:
    return Check(label=label, ok=ok, value=value, limit=limit, blocking=blocking).__dict__


# ── candidate scoring ────────────────────────────────────────────────────────


def score_candidate(
    daily_bars: pd.DataFrame,
    intraday_bars: pd.DataFrame,
    cfg: OptionsFlipConfig,
    macro_direction: str,
) -> Dict[str, Any]:
    """Score one symbol's suitability, keeping every input the score was built from.

    The inputs are kept rather than reduced to a number because the deck explains the decision
    from them: a name that lost is worth a row saying which condition it missed, and by how much.
    """
    if daily_bars is None or daily_bars.empty:
        return {"score": 0.0, "no_daily_bars": True, "last_price": 0.0}

    closes = daily_bars["close"]
    last_price = float(closes.iloc[-1])
    if last_price < cfg.min_price or last_price > cfg.max_price:
        return {"score": 0.0, "price_outside_range": True, "last_price": last_price}

    vol_short = _realized_vol(closes, cfg.vol_short_window)
    vol_long = _realized_vol(closes, cfg.vol_long_window)
    vol_regime = vol_short / vol_long if vol_long > 0 else 0.0

    atr = _atr(daily_bars, cfg.atr_period)
    atr_pct = atr / last_price if last_price > 0 else 0.0
    range_expansion = _range_today(daily_bars) / atr_pct if atr_pct > 0 else 0.0
    vol_ratio = _volume_ratio(daily_bars["volume"])

    momentum = 0.0
    above_vwap = False
    intraday_range_left = 0.0
    if intraday_bars is not None and not intraday_bars.empty and len(intraday_bars) >= cfg.intraday_momentum_bars:
        momentum = _intraday_momentum(intraday_bars, cfg.intraday_momentum_bars)
        vwap = _vwap(intraday_bars)
        above_vwap = float(intraday_bars["close"].iloc[-1]) > vwap if vwap > 0 else False
        if atr_pct > 0:
            intraday_range_left = max(0.0, 1.0 - (_range_today(intraday_bars) / atr_pct))

    if macro_direction == "bull":
        direction_aligned = momentum >= 0 and above_vwap
    elif macro_direction == "bear":
        direction_aligned = momentum <= 0 and not above_vwap
    else:
        direction_aligned = False

    # Additive rather than gated: a name can win while missing one condition, which is why the
    # deck shows these as conditions with their measurements rather than as vetoes.
    score = 0.0
    if vol_regime >= cfg.vol_regime_threshold:
        score += min(vol_regime, 3.0)
    if range_expansion >= cfg.range_expansion_threshold:
        score += min(range_expansion, 3.0)
    if direction_aligned:
        score += 2.0
    if intraday_range_left >= cfg.intraday_range_budget_pct:
        score += 1.5
    if vol_ratio >= cfg.min_volume_ratio:
        score += 0.5

    return {
        "score": score,
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


# ── the contract to buy ──────────────────────────────────────────────────────


def estimate_option_range(
    underlying_price: float,
    atr: float,
    vol_annualized: float,
    dte: int,
    cfg: OptionsFlipConfig,
    option_type: str = "call",
) -> Dict[str, Any]:
    """Fair value for the day, and the two limits either side of it.

    Simplified deliberately: intrinsic plus a time value scaled off the expected move. The entry
    limit sits below fair value and the exit above, so a fill is already a discount and the
    target is set before the position exists.
    """
    if underlying_price <= 0 or atr <= 0:
        return {"fair_value": 0.0, "entry_limit": 0.0, "exit_limit": 0.0, "expected_move": 0.0, "daily_move": 0.0}

    daily_move = underlying_price * (vol_annualized / math.sqrt(252)) if vol_annualized > 0 else atr
    expected_move = daily_move * math.sqrt(max(dte, 1))
    fair_value = max(underlying_price * 0.02, 0.50) + expected_move * 0.40

    return {
        "fair_value": round(fair_value, 2),
        "entry_limit": round(max(fair_value * (1.0 - cfg.entry_discount_pct), 0.01), 2),
        "exit_limit": round(fair_value * (1.0 + cfg.exit_target_pct), 2),
        "expected_move": round(expected_move, 2),
        "daily_move": round(daily_move, 2),
    }


def _select_strike(last_price: float, option_type: str, cfg: OptionsFlipConfig) -> float:
    """A strike near the money, from the delta band this algorithm is willing to hold."""
    if option_type == "call":
        target_delta = (cfg.call_delta_min + cfg.call_delta_max) / 2.0
        offset_pct = 1.0 - target_delta
    else:
        offset_pct = abs((cfg.put_delta_min + cfg.put_delta_max) / 2.0)
    return round(last_price * offset_pct, 2)


def _pick_expiration(dte_min: int, dte_max: int) -> int:
    """The nearest expiration inside the DTE window."""
    return max(dte_min, min(1, dte_max))


class OptionsFlipAlgorithm(BaseAlgorithm):
    """Directional options: calls in a bull trend, puts in a bear one, when intraday confirms."""

    algorithm_id = "options_flip"
    tuning_class = OptionsFlipConfig

    #: Contracts are sized by count, not by portfolio drift, so neither account floor applies.
    min_trade_dollars = 0.0
    rebalance_threshold = 0.0

    #: Every 15 minutes through the session, stopping before the close so a fill has time to
    #: work rather than being opened into the last print. The two firings before 09:30 are
    #: refused by the market-open check in ``run_once``, which is the one place that question
    #: is asked -- expressing the half-hour offset in cron would take two expressions.
    cron = "*/15 9-15 * * 1-5"

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        cfg = self.tuning(config)
        symbols = [cfg.benchmark_symbol, *(s for s in cfg.candidate_symbols if s != cfg.benchmark_symbol)]
        return AlgorithmRequirements(
            price_symbols=symbols,
            daily_lookback_days=cfg.required_daily_bars,
            daily_ma_days=cfg.trend_ma_period,
            intraday_lookback_minutes=cfg.required_intraday_bars * cfg.intraday_bar_minutes,
            preferred_bar_minutes=cfg.intraday_bar_minutes,
            # Unproven: keep it on paper until walk-forward results say otherwise.
            paper_only=True,
        )

    def plan(self, context: AlgorithmContext) -> AlgorithmPlan:
        cfg = self.tuning(context.config)
        macro, macro_checks = detect_macro_trend(
            context.daily_bars_by_symbol.get(cfg.benchmark_symbol), cfg
        )
        if macro == "flat":
            # The common outcome, and the one worth explaining: no candidate was ever scored,
            # so the macro gates are the entire account of the run.
            return AlgorithmPlan(metadata={
                "macro": macro,
                "benchmark": cfg.benchmark_symbol,
                "reason": _macro_reason(macro_checks, cfg),
                "checks": macro_checks,
                "candidates_evaluated": 0,
            })

        candidates = self._score_universe(context, cfg, macro)
        top = [row for row in candidates if row.get("score", 0) > 0][: max(cfg.top_n_candidates, 1)]
        metadata: dict[str, Any] = {
            "macro": macro,
            "benchmark": cfg.benchmark_symbol,
            "checks": macro_checks,
            "candidates_evaluated": len(candidates),
        }

        if not top:
            return AlgorithmPlan(
                signals=signal_rows(candidates, "", {}, cfg, macro),
                metadata={**metadata, "reason": "No candidate scored above zero"},
            )

        best = top[0]
        contract = self._contract(best, cfg, macro)
        return AlgorithmPlan(
            intents=[Intent(symbol=str(best["symbol"]), kind="option", value=contract["contracts"])],
            signals=signal_rows(candidates, str(best["symbol"]), contract, cfg, macro),
            metadata={**metadata, **contract},
            mode=MODE_INCREMENTAL,
        )

    def _score_universe(
        self, context: AlgorithmContext, cfg: OptionsFlipConfig, macro: str
    ) -> list[dict[str, Any]]:
        """Every eligible name, scored and sorted. Losers are kept, so the deck can explain them."""
        allowed = set(cfg.candidate_symbols) if cfg.candidate_symbols else None
        candidates = []
        for symbol in context.daily_bars_by_symbol:
            if symbol == cfg.benchmark_symbol or symbol in context.positions:
                continue
            if allowed is not None and symbol not in allowed:
                continue
            scored = score_candidate(
                context.daily_bars_by_symbol[symbol],
                context.intraday_bars_by_symbol.get(symbol),
                cfg,
                macro,
            )
            # A name with no bars at all has nothing to explain; it is a data gap, not a
            # rejection, and a row saying "0.00" would read as a verdict on the symbol.
            if scored.get("no_daily_bars"):
                continue
            candidates.append({**scored, "symbol": symbol})
        candidates.sort(key=lambda row: -float(row.get("score", 0.0)))
        return candidates

    def _contract(self, best: dict[str, Any], cfg: OptionsFlipConfig, macro: str) -> dict[str, Any]:
        """Type, strike, expiry and the two limits, sized to the per-trade notional cap."""
        option_type = cfg.force_option_type or ("call" if macro == "bull" else "put")
        last_price = float(best["last_price"])
        dte = _pick_expiration(cfg.option_dte_min, cfg.option_dte_max)
        pricing = estimate_option_range(
            last_price, float(best["atr"]), float(best.get("vol_short", 0.0)), dte, cfg, option_type
        )

        contracts = cfg.contracts_per_signal
        cost_per_contract = pricing["entry_limit"] * 100
        if cost_per_contract > 0 and cost_per_contract * contracts > cfg.max_notional_per_trade:
            contracts = max(1, int(cfg.max_notional_per_trade / cost_per_contract))

        return {
            "option_type": option_type,
            "strike": _select_strike(last_price, option_type, cfg),
            "dte": dte,
            "contracts": contracts,
            "entry_limit": pricing["entry_limit"],
            "exit_limit": pricing["exit_limit"],
            "fair_value": pricing["fair_value"],
            "pricing": pricing,
        }

    def signal_view(self, plan: AlgorithmPlan) -> SignalView:
        return signal_view(plan) if plan.signals else macro_view(plan)


def _macro_reason(checks: list[dict[str, Any]], cfg: OptionsFlipConfig) -> str:
    """The first failing macro gate, stated as the reason nothing was traded."""
    for check in checks:
        if check.get("blocking"):
            return f"{check['label']}: {check['value']}"
    return f"{cfg.benchmark_symbol} trend is flat; sitting in cash"
