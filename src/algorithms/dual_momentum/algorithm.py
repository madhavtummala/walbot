"""The plugin the registry resolves.
"""

from __future__ import annotations

from .config import DualMomentumConfig
from .layers import score_to_weights, volatility_scale
from .proposal import allocation_mode, analyze_universe, build_signals
from .stateful import _defensive_book, _in_cooldown, _minutes_between, _parse_time, _record_exits, _resolve_replacements, _run_facts, apply_turnover_filters, confirm_regime, intraday_drawdown_breached



import logging
from typing import Any


from ..base import BaseAlgorithm
from ...core.interfaces import AlgorithmContext, AlgorithmDecision, AlgorithmRequirements, Schedule
from ...data.state_store import load_state, save_state

logger = logging.getLogger(__name__)

STATE_KEY = "dual_momentum_runtime"

EPSILON = 1e-9

#: Trading days per year, for annualising a daily volatility estimate.
TRADING_DAYS = 252


class DualMomentumAlgorithm(BaseAlgorithm):
    """Dual momentum with a regime gate, split timing, and a volatility target."""

    algorithm_id = "dual_momentum"

    #: Every 15 minutes: that is the risk and timing cadence. Re-ranking is throttled
    #: separately to ``signal_refresh_minutes`` inside ``refine``, so the expensive decision
    #: (which names to own) stays slow while the cheap ones (scale, stop, time an entry)
    #: stay fast.
    schedule = Schedule(refresh_minutes=15, jitter_minutes=2)

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        strategy_config = DualMomentumConfig.from_runtime_config(config)
        return AlgorithmRequirements(
            price_symbols=sorted(set(strategy_config.symbols) | set(current_positions)),
            daily_lookback_days=strategy_config.required_daily_bars,
            daily_ma_days=strategy_config.etf_ma_days,
            history_lookback_minutes=strategy_config.required_history_minutes,
            preferred_bar_minutes=strategy_config.intraday_bar_minutes,
            needs_sentiment=strategy_config.uses_sentiment,
            # Unproven: keep it on paper until walk-forward results say otherwise.
            paper_only=True,
        )

    def sizing(self, config: Any) -> dict[str, float]:
        """No cash buffer: gross exposure is already bounded inside the weights."""
        strategy_config = DualMomentumConfig.from_runtime_config(config)
        return {
            "cash_buffer": 0.0,
            "min_trade_dollars": strategy_config.minimum_trade_notional,
            "rebalance_threshold": strategy_config.rebalance_weight_threshold,
        }

    def analyze(self, context: AlgorithmContext) -> AlgorithmDecision:
        strategy_config = DualMomentumConfig.from_runtime_config(context.config)
        outcome = analyze_universe(context, strategy_config)
        weights = outcome["weights"]
        regime = outcome["regime"]
        return AlgorithmDecision(
            target_weights=weights,
            signals=build_signals(
                outcome["scored"],
                weights,
                regime,
                outcome["volatility"],
                outcome["covariance"],
                outcome["defensive_book"],
                context.timestamp,
                strategy_config,
            ),
            metadata={
                "allocation_mode": allocation_mode(weights, regime, strategy_config),
                "market_sentiment": context.market_sentiment,
                "regime": regime,
                "portfolio_volatility": outcome["volatility"]["portfolio_volatility"],
                "vol_scale": outcome["volatility"]["scale"],
                "eligible_count": len(outcome["ranked"]),
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
        """Position-aware layer: regime hysteresis, hold/exit asymmetry, cooldown, risk stops.

        Everything here needs either what is currently held or memory of previous runs, which
        is exactly what ``analyze`` is forbidden to touch.
        """
        strategy_config = DualMomentumConfig.from_runtime_config(config)
        account = str(getattr(config, "account_id", "") or "")
        state_key = f"{STATE_KEY}:{account}" if account else STATE_KEY
        state = load_state(state_key, {})
        if not isinstance(state, dict):
            state = {}

        facts = _run_facts(signals)
        now = _parse_time(facts["as_of"])
        current = snapshot.weights(latest_prices)
        held = {symbol for symbol, weight in current.items() if weight > 0}
        defensive = {name.upper() for name in strategy_config.defensive_universe}

        state.update(confirm_regime(state, facts["regime_risk_on"], strategy_config))
        risk_on = bool(state["regime_risk_on"])

        rows = {symbol: dict(row, symbol=symbol) for symbol, row in signals.items()}
        book = _defensive_book(rows)

        def settle(weights: dict[str, float]) -> dict[str, float]:
            """Apply the turnover filters, record exits, and persist state."""
            result = apply_turnover_filters(
                {symbol: float(weights.get(symbol, 0.0)) for symbol in target_weights},
                current,
                float(snapshot.equity or 0.0),
                strategy_config,
            )
            _record_exits(state, held, {s for s, weight in result.items() if weight > 0}, facts["as_of"])
            save_state(state_key, state)
            return result

        session = now.date().isoformat() if now else ""
        if intraday_drawdown_breached(state, float(snapshot.equity or 0.0), strategy_config, session):
            logger.warning(
                "Dual Momentum drawdown breaker active (%.2f%%); holding the defensive sleeve only",
                100 * float(state.get("session_drawdown", 0.0)),
            )
            return settle(book)

        if not risk_on:
            # The raw gate may read risk-on while the confirmation is still pending; the
            # risk-on proposal must not be acted on until it is confirmed. Deliberately not
            # stamping last_selection_at: no selection happened, and pretending one did would
            # throttle the first real one.
            return settle(book)

        proposed = {symbol for symbol, weight in target_weights.items() if weight > 0} - defensive

        keep: set[str] = set()
        for symbol in held - defensive:
            row = rows.get(symbol, {})
            rank = int(row.get("rank") or 0)
            if not int(row.get("eligible", 0)):
                logger.info("Dual Momentum exiting %s: %s", symbol, row.get("eligibility_reason"))
                continue
            if rank and rank > strategy_config.exit_rank_max:
                logger.info("Dual Momentum exiting %s: rank %d beyond exit rank", symbol, rank)
                continue
            # An incumbent is held while it stays eligible and ranked, even if the timing flag
            # is false: timing decides when to *enter*, not whether to stay.
            keep.add(symbol)

        # Re-ranking is throttled: between selection refreshes the book keeps its membership
        # and only the risk layer acts, which is what "slow selection, fast timing" means.
        # A free slot is exempt -- the throttle exists to stop churn between comparable names,
        # and sitting in cash while a qualified name waits out the hour is lag, not risk
        # management.
        since_selection = _minutes_between(now, _parse_time(str(state.get("last_selection_at", ""))))
        may_reselect = (
            since_selection >= max(strategy_config.signal_refresh_minutes, 0)
            or len(keep) < max(strategy_config.max_positions, 0)
        )

        if may_reselect:
            entrants = {symbol for symbol in proposed if symbol not in keep}
            entrants = {symbol for symbol in entrants if not _in_cooldown(state, symbol, now, strategy_config)}
            selection = _resolve_replacements(keep, entrants, rows, strategy_config)
            state["last_selection_at"] = facts["as_of"]
        else:
            selection = set(keep)

        chosen_rows = [rows[symbol] for symbol in selection if symbol in rows]
        weights = score_to_weights(chosen_rows, strategy_config) if chosen_rows else {}

        covariance = {symbol: dict(rows[symbol].get("covariance_row") or {}) for symbol in selection if symbol in rows}
        vol = volatility_scale(weights, covariance, strategy_config)
        if vol["below_floor"]:
            logger.info("Dual Momentum de-risking to defensive: ex-ante vol %.1f%%", 100 * vol["portfolio_volatility"])
            weights = dict(book)
        else:
            weights = {symbol: weight * vol["scale"] for symbol, weight in weights.items()}

        if not any(weights.values()):
            # Nothing qualifies: sit in the defensive sleeve rather than in the least-bad name.
            weights = dict(book)

        return settle(weights)
