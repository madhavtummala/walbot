"""The plugin the registry resolves.
"""

from __future__ import annotations

from .config import STATE_KEY, DualMomentumConfig
from .layers import hold_eligibility, park_residual, theme_allocation, theme_of, volatility_scale
from .proposal import allocation_mode, analyze_universe, build_signals
from .stateful import track_eligibility, resolve_themes, _defensive_book, _in_cooldown, _record_exits, _run_facts, apply_turnover_filters, confirm_regime, intraday_drawdown_breached



import logging
from datetime import datetime
from typing import Any


from ..base import BaseAlgorithm
from ...core.interfaces import DAILY_AT_OPEN, AlgorithmContext, AlgorithmDecision, AlgorithmRequirements
from ...data.state_store import load_state, save_state

logger = logging.getLogger(__name__)


class DualMomentumAlgorithm(BaseAlgorithm):
    """Dual momentum with a regime gate, split timing, and a volatility target."""

    algorithm_id = "dual_momentum"

    #: Once per session. Every feature is computed from daily bars now, so a second look
    #: before the close reads the same closes and cannot produce a different answer -- which
    #: is exactly what ``DAILY_AT_OPEN`` exists for.
    #:
    #: This was every 15 minutes, from when the score was built on intraday bars. Left there
    #: it would re-run the same decision ~26 times a session; a live binding at ``1hr`` was
    #: doing roughly that, against a backtest that only ever steps once a day.
    schedule = DAILY_AT_OPEN

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        strategy_config = DualMomentumConfig.from_runtime_config(config)
        return AlgorithmRequirements(
            price_symbols=sorted(set(strategy_config.symbols) | set(current_positions)),
            daily_lookback_days=strategy_config.required_daily_bars,
            daily_ma_days=strategy_config.etf_ma_days,
            # Nothing intraday: every feature comes from the daily bars above. Asking for a
            # history window would make the live path fetch one on each run and the replay
            # build a HistoryCache over it, both for data no layer reads.
            history_lookback_minutes=strategy_config.required_history_minutes,
            needs_sentiment=strategy_config.uses_sentiment,
            # Unproven: keep it on paper until walk-forward results say otherwise.
            paper_only=True,
        )

    def sizing(self, config: Any) -> dict[str, float]:
        """Exposure comes from ``risk_on_gross_max``; affording it is the funding stage's job.

        This used to carry the account cash buffer so a plan built on top of a deliberately
        retained sub-threshold position could still be paid for. Order funding now checks that
        directly against buying power, so the buffer no longer has to be guessed at here.
        """
        strategy_config = DualMomentumConfig.from_runtime_config(config)
        return {
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
                strategy_config,
            ),
            metadata={
                "allocation_mode": allocation_mode(weights, regime, strategy_config),
                # Reported only when it is actually an input, so the dashboard does not show a
                # sentiment reading for a run that never asked for one.
                **({"market_sentiment": context.market_sentiment}
                   if strategy_config.uses_sentiment else {}),
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
        as_of: datetime,
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
        # ``as_of`` is the only clock this layer reads, and it arrives as an argument rather
        # than from a lookup: every elapsed-time decision below -- the re-entry cooldown, the
        # selection throttle, the session breaker -- has to measure against the moment step 1
        # described, which in a replay is a date months ago.
        stamp = as_of.isoformat()
        current = snapshot.weights(latest_prices)
        held = {symbol for symbol, weight in current.items() if weight > 0}
        defensive = {name.upper() for name in strategy_config.defensive_universe}

        state.update(confirm_regime(state, facts["regime_risk_on"], strategy_config))
        risk_on = bool(state["regime_risk_on"])

        rows = {symbol: dict(row, symbol=symbol) for symbol, row in signals.items()}
        book = _defensive_book(rows)

        def settle(weights: dict[str, float]) -> dict[str, float]:
            """Apply the turnover filters, record exits, and persist state.

            The symbol set is the union of what step 1 proposed, what is currently held, and
            what this layer decided to hold. It used to be ``target_weights`` alone, which
            silently discarded every decision this layer had just made about a name step 1 did
            not re-propose -- and ``pipeline.place_orders`` drops zero-weight intents before
            ``refine`` runs, so ``target_weights`` only ever contains the current proposal.

            An incumbent that ``refine`` kept therefore vanished from the returned weights, and
            ``MODE_TARGET`` reads an absent symbol as a target of zero: the position was sold.
            Over a 12-month replay that force-sold a still-qualifying holding on 73% of
            decisions, left roughly half the book in raw cash, and was the single largest
            source of turnover.
            """
            symbols = set(target_weights) | set(current) | set(weights)
            result = apply_turnover_filters(
                {symbol: float(weights.get(symbol, 0.0)) for symbol in symbols},
                current,
                float(snapshot.equity or 0.0),
                strategy_config,
            )
            _record_exits(state, held, {s for s, weight in result.items() if weight > 0}, stamp)
            save_state(state_key, state)
            return result

        if intraday_drawdown_breached(state, float(snapshot.equity or 0.0), strategy_config, as_of):
            logger.warning(
                "Dual Momentum drawdown breaker active (%.2f%%); holding the defensive sleeve only",
                100 * float(state.get("session_drawdown", 0.0)),
            )
            return settle(book)

        if not risk_on:
            # The raw gate may read risk-on while the confirmation is still pending; the
            # risk-on proposal must not be acted on until it is confirmed.
            return settle(book)

        # Eligibility is tracked per *theme*, not per name: the book's slow decision is which
        # exposures to hold, and a theme counts as qualifying on a day when any of its ETFs
        # does. Which ETF inside it is a sizing question, settled every day by
        # ``theme_allocation`` with no confirmation at all.
        theme_rows: dict[str, dict[str, Any]] = {}
        for symbol, row in rows.items():
            if symbol in defensive:
                continue
            theme = theme_of(symbol, strategy_config)
            prior = theme_rows.get(theme)
            better = prior is None or (
                int(row.get("eligible", 0)),
                float(row.get("base_score", 0.0)),
            ) > (int(prior.get("eligible", 0)), float(prior.get("base_score", 0.0)))
            if better:
                theme_rows[theme] = row
        history = track_eligibility(state, theme_rows, strategy_config)
        window = max(strategy_config.eligibility_window, 1)
        theme_score = {t: float(r.get("base_score", 0.0)) for t, r in theme_rows.items()}

        def eligible_days(theme: str) -> int:
            return sum(history.get(theme, []))

        def watched_long_enough(theme: str) -> bool:
            return len(history.get(theme, [])) >= window

        held_themes = {theme_of(symbol, strategy_config) for symbol in held - defensive}

        keep: set[str] = set()
        for theme in held_themes:
            row = theme_rows.get(theme, {})
            # Two ways out, answering different questions. The count is the slow one: this
            # theme has stopped qualifying for long enough to mean something. The band, via
            # hold_eligibility, is the fast one -- including the single-session crash stop.
            days = eligible_days(theme)
            # Only once the window is full: a cold state store has no history, and reading
            # that as "ineligible" would sell everything on the first run.
            if watched_long_enough(theme) and days <= strategy_config.exit_max_eligible_days:
                logger.info(
                    "Dual Momentum exiting theme %s: eligible on only %d of the last %d runs",
                    theme, days, window,
                )
                continue
            stays, why = hold_eligibility(row, strategy_config)
            if not stays:
                logger.info("Dual Momentum exiting theme %s: %s", theme, why)
                continue
            keep.add(theme)

        entrant_themes = {
            theme
            for theme, row in theme_rows.items()
            if theme not in keep
            and eligible_days(theme) >= strategy_config.entry_min_eligible_days
            and int(row.get("rank") or 0)
            and int(row["rank"]) <= strategy_config.entry_rank_max
            and float(row.get("base_score", 0.0)) >= strategy_config.min_base_score
            and int(row.get("timing", 1))
            and not _in_cooldown(state, theme, as_of, strategy_config)
        }
        selection = resolve_themes(keep, entrant_themes, theme_score, strategy_config)
        state["last_selection_at"] = stamp

        # Budget per theme, split inside it by today's scores -- no confirmation, every day.
        weights = theme_allocation(selection, rows, strategy_config)

        covariance = {symbol: dict(rows[symbol].get("covariance_row") or {}) for symbol in weights}
        vol = volatility_scale(weights, covariance, strategy_config)
        if vol["below_floor"]:
            logger.info("Dual Momentum de-risking to defensive: ex-ante vol %.1f%%", 100 * vol["portfolio_volatility"])
            weights = dict(book)
        else:
            weights = {symbol: weight * vol["scale"] for symbol, weight in weights.items()}
            # Same rule as step 1: undeployed gross belongs in bills, not in cash.
            weights = park_residual(weights, book, strategy_config)

        if not any(weights.values()):
            # Nothing qualifies: sit in the defensive sleeve rather than in the least-bad name.
            weights = dict(book)

        return settle(weights)
