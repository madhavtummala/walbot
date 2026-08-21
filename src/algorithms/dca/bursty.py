"""Bursty DCA: daily buys sized proportionally to drawdown from peak.

  drawdown = max(0, (peak - price) / peak)
  size = budget × (1 + drawdown × scaling_factor)

Two fixed runs per day, US Eastern: buys on the 11:00 run, sells on the 15:00 run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import pandas as pd

from ...core.interfaces import (
    MARKET_TZ,
    Schedule,
    SignalView,
    AlgorithmContext,
    AlgorithmRequirements,
    Intent,
    PortfolioSnapshot,
)
from .accrual import SymbolState, load_accrual_state
from .bot import DCAAlgorithm, plan_budgets
from . import unknown_plan_symbols, raw_plan_from_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BurstyConfig:
    """Tuning for the drawdown-based sizing model."""

    #: How much to scale buy size per unit of drawdown.
    #: size = budget × (1 + drawdown × scaling_factor)
    scaling_factor: float = 10.0
    #: Cap on cumulative deployment per symbol per month, in multiples of the monthly budget.
    max_monthly_multiple: float = 3.0
    #: Reference period for peak calculation (rolling high-water mark).
    regime_ma_days: int = 150
    #: How much to scale the monthly cap with drawdown.
    #: effective_cap = max_monthly_multiple × budget × (1 + drawdown × cap_boost)
    cap_boost: float = 0.0

    @classmethod
    def from_runtime_config(cls, config: Any) -> BurstyConfig:
        section = (getattr(config, "algorithm_configs", {}) or {}).get("bursty_dca") or {}
        settings: dict[str, Any] = {}
        for name, field in cls.__dataclass_fields__.items():
            if name not in section:
                continue
            try:
                settings[name] = type(field.default)(section[name])
            except (TypeError, ValueError):
                logger.warning("Ignoring unusable bursty_dca.%s: %r", name, section[name])
        return cls(**settings)

    @property
    def required_daily_bars(self) -> int:
        return self.regime_ma_days + 30


def _latest(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return float("nan")
    value = series.iloc[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_trigger(
    bars: pd.DataFrame | None,
    buying: bool,
    months_behind: float = 0.0,
) -> dict[str, Any]:
    """Compute drawdown from peak and decide whether to buy."""
    if bars is None or bars.empty or "close" not in bars:
        return {"fires": False, "reason": "No price history", "detail": {}}

    closes = pd.to_numeric(bars["close"], errors="coerce").dropna()
    if closes.empty:
        return {"fires": False, "reason": "No price history", "detail": {}}

    close = float(closes.iloc[-1])
    peak = float(closes.cummax().iloc[-1])

    drawdown = max(0.0, (peak - close) / peak) if peak > 0 else 0.0

    detail = {
        "close": close,
        "peak": round(peak, 4),
        "drawdown": round(drawdown, 4),
        "months_behind": round(months_behind, 2),
    }

    if buying:
        return {"fires": True, "reason": f"Drawdown {drawdown*100:.1f}%", "detail": detail}

    if drawdown > 0:
        return {"fires": False, "reason": f"Still below peak (-{drawdown*100:.1f}%)", "detail": detail}
    return {"fires": True, "reason": "At peak", "detail": detail}


def planned_order_size(
    monthly_budget: float,
    drawdown: float,
    deployed_this_month: float,
    settings: BurstyConfig,
) -> float:
    """Dollar size ``refine`` would place for one symbol, before position/broker clamps.

    size = |budget| × (1 + scaling_factor × drawdown), clamped to the month's remaining
    cap room. Shared by step 2 and the live-signal view so the dashboard previews exactly
    what a run would order rather than a parallel approximation of it.
    """
    budget = abs(float(monthly_budget))
    if budget <= 0:
        return 0.0
    drawdown = max(float(drawdown), 0.0)
    desired = budget * (1.0 + settings.scaling_factor * drawdown)
    effective_cap = settings.max_monthly_multiple * (1.0 + settings.cap_boost * drawdown)
    remaining = max(effective_cap * budget - max(float(deployed_this_month), 0.0), 0.0)
    return min(desired, remaining)


def _rsi(series: pd.Series, period: int = 2) -> float:
    if len(series) < period + 1:
        return float("nan")
    deltas = series.diff().dropna()
    gains = deltas.clip(lower=0)
    losses = (-deltas.clip(upper=0))
    avg_gain = gains.rolling(period).mean().iloc[-1]
    avg_loss = losses.rolling(period).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _percent_b(close: float, sma_20: float, std_20: float) -> float:
    if std_20 <= 0:
        return float("nan")
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    band_width = upper - lower
    if band_width <= 0:
        return float("nan")
    return float((close - lower) / band_width)


def _compute_indicators(bars: pd.DataFrame | None, settings: BurstyConfig) -> dict[str, Any]:
    """Compute technical indicators from daily bars for the signal view."""
    result: dict[str, Any] = {}
    if bars is None or bars.empty or "close" not in bars:
        return result

    closes = pd.to_numeric(bars["close"], errors="coerce").dropna()
    if closes.empty:
        return result

    close = float(closes.iloc[-1])
    result["close"] = round(close, 2)

    # RSI(2)
    rsi2 = _rsi(closes, 2)
    result["rsi_2"] = round(rsi2, 1) if not pd.isna(rsi2) else None

    # Bollinger %B (20-day)
    if len(closes) >= 20:
        sma_20 = float(closes.rolling(20).mean().iloc[-1])
        std_20 = float(closes.rolling(20).std().iloc[-1])
        pct_b = _percent_b(close, sma_20, std_20)
        result["pct_b"] = round(pct_b, 3) if not pd.isna(pct_b) else None
        result["sma_20"] = round(sma_20, 2)
    else:
        result["pct_b"] = None
        result["sma_20"] = None

    # 150-day MA distance
    ma_period = settings.regime_ma_days
    if len(closes) >= ma_period:
        ma = float(closes.rolling(ma_period).mean().iloc[-1])
        result["ma_distance"] = round((close - ma) / ma, 4) if ma > 0 else None
        result[f"ma_{ma_period}"] = round(ma, 2)
    else:
        result["ma_distance"] = None
        result[f"ma_{ma_period}"] = None

    # Peak and drawdown
    peak = float(closes.cummax().iloc[-1])
    drawdown = max(0.0, (peak - close) / peak) if peak > 0 else 0.0
    result["peak"] = round(peak, 2)
    result["drawdown"] = round(drawdown, 4)

    # Realized volatility (20-day annualized)
    if len(closes) >= 21:
        daily_ret = closes.pct_change().dropna()
        vol = float(daily_ret.rolling(20).std().iloc[-1]) * (252 ** 0.5)
        result["volatility"] = round(vol, 4)
    else:
        result["volatility"] = None

    return result


class BurstyDCAAlgorithm(DCAAlgorithm):
    algorithm_id = "bursty_dca"

    #: Two fixed runs per day, US Eastern: the 11:00 run places buys, the 15:00 run places
    #: sells. ``refresh_minutes=240`` from the 11:00 start yields exactly those two buckets,
    #: so a binding on any scheduled frequency (``1d`` included) fires at both times.
    schedule = Schedule(
        start_time="11:00",
        end_time="15:30",
        refresh_minutes=240,
        jitter_minutes=15,
    )

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        settings = BurstyConfig.from_runtime_config(config)
        return AlgorithmRequirements(
            price_symbols=sorted(plan_budgets(self.plan(config))),
            daily_lookback_days=settings.required_daily_bars,
            daily_ma_days=settings.regime_ma_days,
        )

    def trigger(
        self,
        symbol: str,
        context: AlgorithmContext,
        plan: dict[str, Any],
        months_behind: float = 0.0,
    ) -> dict[str, Any]:
        buying = plan_budgets(plan).get(symbol, 0.0) >= 0
        return evaluate_trigger(context.daily_bars_by_symbol.get(symbol), buying, months_behind)

    def reason(
        self,
        state: SymbolState,
        floor_dollars: float,
        trigger: dict[str, Any],
        ready: bool,
    ) -> str:
        return super().reason(state, floor_dollars, trigger, ready)

    def signal_view(self, config: Any, *, data_client: Any = None) -> SignalView:
        from ...core.market_context import build_algorithm_context
        from ...api.api_payloads import universe_payload
        from .bot import plan_budgets

        context = build_algorithm_context(config, self.requirements(config, {}), data_client=data_client)
        decision = self.analyze(context)

        settings = BurstyConfig.from_runtime_config(config)
        quote_meta = dict(context.extra.get("price_quotes") or {})

        leaders = []
        for symbol, values in decision.signals.items():
            bars = context.daily_bars_by_symbol.get(symbol)
            indicators = _compute_indicators(bars, settings)

            monthly_budget = float(values.get("monthly_budget") or 0.0)
            drawdown = float(indicators.get("drawdown") or 0.0)
            size = planned_order_size(
                monthly_budget,
                drawdown,
                float(values.get("deployed_this_month") or 0.0),
                settings,
            )
            quote = quote_meta.get(symbol) or {}
            leaders.append({
                **values,
                "symbol": symbol,
                "signal": "LONG" if monthly_budget >= 0 else "SHORT",
                "side": "LONG" if monthly_budget >= 0 else "SHORT",
                "target_weight": None,
                **indicators,
                # What a run would order right now: the refine math applied to the same
                # state the dashboard is looking at, signed like the budget.
                "next_order": round(size if monthly_budget >= 0 else -size, 2),
                # Provenance of the price this row is priced at, so a stored-bar
                # fallback never masquerades as a live print.
                "price_time": quote.get("timestamp"),
                "price_current": bool(quote.get("current")),
            })

        leaders.sort(key=lambda row: (row["reason"] != "Ready", row["symbol"]))
        monthly_total = float(decision.metadata.get("monthly_total") or 0.0)
        summary = [
            {"label": "Mode", "value": str(decision.metadata.get("allocation_mode") or "DCA")},
            {"label": "Planned", "value": f"${monthly_total:.0f}/month"},
            {"label": "Symbols", "value": str(len(leaders))},
            {"label": "Scaling", "value": f"{settings.scaling_factor}x"},
            {"label": "Cap", "value": f"{settings.max_monthly_multiple}x monthly"},
        ]
        if quote_meta:
            stale = sorted(symbol for symbol, quote in quote_meta.items() if not quote.get("current"))
            summary.append({
                "label": "Prices",
                "value": "live" if not stale else f"delayed: {', '.join(stale)}",
            })
        unknown = unknown_plan_symbols(
            raw_plan_from_config(config, self.algorithm_id), universe_payload()["rows"]
        )
        if unknown:
            summary.append({"label": "Not tradable", "value": ", ".join(unknown)})
        return SignalView(leaders=leaders, summary=summary)

    def refine(
        self,
        intents: list[Intent],
        signals: dict[str, dict[str, Any]],
        snapshot: PortfolioSnapshot,
        latest_prices: dict[str, float],
        config: Any,
        as_of: datetime,
    ) -> list[Intent]:
        """Size trades proportionally to drawdown from peak.

        size = budget × (1 + drawdown × scaling_factor)

        Buys execute on the 11:00 ET run, sells on the 15:00 ET run.
        """
        # ``as_of`` is a UTC instant from the context; the buy/sell split is stated in
        # market time, so convert before comparing hours. A naive timestamp (backtest
        # date-only bars) is taken as market time already, which keeps hour at 0 and
        # disables the filter there.
        local_hour = (as_of.astimezone(MARKET_TZ) if as_of.tzinfo else as_of).hour
        buy_hour = 11
        sell_hour = 15
        # Outside the two run windows -- backtests, or an off-schedule manual run --
        # process all intents rather than dropping them.
        is_live_buy = local_hour == buy_hour
        is_live_sell = local_hour == sell_hour
        filter_by_hour = local_hour in (buy_hour, sell_hour)

        settings = BurstyConfig.from_runtime_config(config)

        account_id = getattr(config, "account_id", "") or ""
        state = load_accrual_state(self.algorithm_id, account_id)

        refined: list[Intent] = []
        for intent in intents:
            price = float(latest_prices.get(intent.symbol, 0.0) or 0.0)
            if price <= 0:
                continue
            budget = abs(float(signals.get(intent.symbol, {}).get("monthly_budget") or 0.0))
            if budget <= 0:
                continue
            symbol_state = state.get(intent.symbol, SymbolState())

            buying = intent.value >= 0

            # Buys at 11 AM, sells at 3 PM only (skip filter in backtest where hour=0)
            if filter_by_hour and buying and not is_live_buy:
                continue
            if filter_by_hour and not buying and not is_live_sell:
                continue

            # Drawdown-based sizing
            sig = signals.get(intent.symbol, {})
            drawdown = float(sig.get("drawdown", 0.0) or 0.0)
            desired = planned_order_size(budget, drawdown, symbol_state.deployed_this_month, settings)
            if not buying:
                desired = min(desired, float(snapshot.positions.get(intent.symbol, 0.0)) * price)
            if desired <= 0:
                continue

            refined.append(replace(intent, value=desired if buying else -desired))

        return refined
