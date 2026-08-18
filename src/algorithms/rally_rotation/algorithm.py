"""The plugin the registry resolves.
"""

from __future__ import annotations

from .config import STATE_KEY, RallyRotationConfig
from .layers import crash_stop, climax_top, hold_eligibility, park_residual, score_to_weights
from .proposal import allocation_mode, analyze_universe, build_signals
from .stateful import track_eligibility, track_ranking, resolve_positions, _defensive_book, _record_exits, _run_facts, apply_turnover_filters, partial_adjustment, action_due, advance_run, record_action



import logging
from datetime import datetime
from typing import Any


from ..base import BaseAlgorithm
from ...core.interfaces import DAILY_AT_OPEN, AlgorithmContext, AlgorithmDecision, AlgorithmRequirements
from ...data.state_store import load_state, save_state

logger = logging.getLogger(__name__)


class RallyRotationAlgorithm(BaseAlgorithm):
    """Dual momentum: cross-sectional rank, per-name absolute momentum, theme-level rotation."""

    algorithm_id = "rally_rotation"

    #: Once per session. Every feature is computed from daily bars now, so a second look
    #: before the close reads the same closes and cannot produce a different answer -- which
    #: is exactly what ``DAILY_AT_OPEN`` exists for.
    #:
    #: This was every 15 minutes, from when the score was built on intraday bars. Left there
    #: it would re-run the same decision ~26 times a session; a live binding at ``1hr`` was
    #: doing roughly that, against a backtest that only ever steps once a day.
    schedule = DAILY_AT_OPEN

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        strategy_config = RallyRotationConfig.from_runtime_config(config)
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
        strategy_config = RallyRotationConfig.from_runtime_config(config)
        return {
            "min_trade_dollars": strategy_config.minimum_trade_notional,
            "rebalance_threshold": strategy_config.rebalance_weight_threshold,
        }

    def analyze(self, context: AlgorithmContext) -> AlgorithmDecision:
        strategy_config = RallyRotationConfig.from_runtime_config(context.config)
        outcome = analyze_universe(context, strategy_config)
        weights = outcome["weights"]
        data = outcome["data"]
        return AlgorithmDecision(
            target_weights=weights,
            signals=build_signals(
                outcome["scored"],
                weights,
                data,
                outcome["defensive_book"],
                strategy_config,
            ),
            metadata={
                "allocation_mode": allocation_mode(weights, strategy_config),
                # Reported only when it is actually an input, so the dashboard does not show a
                # sentiment reading for a run that never asked for one.
                **({"market_sentiment": context.market_sentiment}
                   if strategy_config.uses_sentiment else {}),
                "universe_data": data,
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
        """Position-aware layer: hold/exit asymmetry, replacement margin, cooldown, risk stops.

        Everything here needs either what is currently held or memory of previous runs, which
        is exactly what ``analyze`` is forbidden to touch.
        """
        strategy_config = RallyRotationConfig.from_runtime_config(config)
        account = str(getattr(config, "account_id", "") or "")
        state_key = f"{STATE_KEY}:{account}" if account else STATE_KEY
        state = load_state(state_key, {})
        if not isinstance(state, dict):
            state = {}

        advance_run(state)
        facts = _run_facts(signals)
        # ``as_of`` is the only clock this layer reads, and it arrives as an argument rather
        # than from a lookup: every elapsed-time decision below -- the re-entry cooldown, the
        # selection throttle, the session breaker -- has to measure against the moment step 1
        # described, which in a replay is a date months ago.
        stamp = as_of.isoformat()
        current = snapshot.weights(latest_prices)
        held = {symbol for symbol, weight in current.items() if weight > 0}
        defensive = {name.upper() for name in strategy_config.defensive_universe}

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
            stepped = partial_adjustment(
                {symbol: float(weights.get(symbol, 0.0)) for symbol in symbols},
                current,
                strategy_config,
            )
            result = apply_turnover_filters(
                stepped,
                current,
                float(snapshot.equity or 0.0),
                strategy_config,
            )
            _record_exits(state, held, {s for s, weight in result.items() if weight > 0}, stamp)
            save_state(state_key, state)
            return result

        if not facts["data_ok"]:
            # Not a bearish reading -- an unusable one. See ``universe_data_ok``.
            logger.warning("[%s] Rally Rotation holding the defensive sleeve: %s", stamp[:10], facts["data_detail"])
            return settle(book)

        risk_rows = {symbol: row for symbol, row in rows.items() if symbol not in defensive}
        history = track_eligibility(state, risk_rows, strategy_config)
        rank_history = track_ranking(state, risk_rows, strategy_config)
        window = max(strategy_config.eligibility_window, 1)

        def eligible_days(symbol: str) -> int:
            return sum(history.get(symbol, []))

        def ranked_top_days(symbol: str) -> int:
            return sum(rank_history.get(symbol, []))

        def watched_long_enough(symbol: str) -> bool:
            return len(history.get(symbol, [])) >= window

        # The crash stop answers to no clock. Everything else -- ranking, entering, replacing,
        # and the considered exits -- happens on ``rerank_interval_days``, because they are all
        # the same decision seen from different sides and the score they rest on has a twelve
        # session horizon. Re-ranking every session asked it a question it cannot answer that
        # fast, and the book paid the spread for the noise.
        crashed = {
            symbol for symbol in held - defensive
            if not crash_stop(rows.get(symbol, {}), strategy_config)[0]
        }
        for symbol in crashed:
            logger.warning(
                "[%s] Rally Rotation crash stop on %s: %s", stamp[:10], symbol,
                crash_stop(rows.get(symbol, {}), strategy_config)[1],
            )

        # Climax top exit: fires every session like crash_stop, outside normal cadence.
        climax_exits = {
            symbol for symbol in held - defensive - crashed
            if not climax_top(rows.get(symbol, {}), strategy_config)[0]
        }
        for symbol in climax_exits:
            logger.warning(
                "[%s] Rally Rotation climax top exit on %s: %s", stamp[:10], symbol,
                climax_top(rows.get(symbol, {}), strategy_config)[1],
            )

        if not action_due(state, "rerank", strategy_config.rerank_interval_days):
            # Between re-rankings the book may only shrink, and only for a crash. Everything
            # else stays exactly where it is, which is the whole point of the throttle.
            survivors = (held - defensive) - crashed - climax_exits
            weights = {symbol: float(current.get(symbol, 0.0)) for symbol in survivors}
        else:
            record_action(state, "rerank")
            keep: set[str] = set()
            for symbol in (held - defensive) - crashed - climax_exits:
                row = rows.get(symbol, {})
                # Two ways out, answering different questions. The count is the slow one: this
                # name has stopped qualifying for long enough to mean something. The band, via
                # hold_eligibility, is the fast one.
                days = eligible_days(symbol)
                # Only once the window is full: a cold state store has no history, and reading
                # that as "ineligible" would sell everything on the first run.
                if watched_long_enough(symbol) and days <= strategy_config.exit_max_eligible_days:
                    logger.info(
                        "[%s] Rally Rotation exiting %s: eligible on only %d of the last %d runs",
                        stamp[:10], symbol, days, window,
                    )
                    continue
                stays, why = hold_eligibility(row, strategy_config)
                if not stays:
                    logger.info("[%s] Rally Rotation exiting %s: %s", stamp[:10], symbol, why)
                    continue
                keep.add(symbol)

            candidates = [
                dict(row, symbol=symbol)
                for symbol, row in sorted(
                    risk_rows.items(), key=lambda item: -float(item[1].get("base_score", 0.0))
                )
                if int(row.get("eligible", 0))
                and (
                    # Held names face eligibility alone. The quality floor and the settling
                    # period are all *entry* conditions: a holding that would not be bought
                    # today is not thereby worth selling, and applying them symmetrically
                    # sells a name the moment it stops being a purchase.
                    symbol in keep
                    or (
                        float(row.get("base_score", 0.0)) >= strategy_config.min_base_score
                        and ranked_top_days(symbol) >= strategy_config.entry_min_eligible_days
                    )
                )
            ]
            for position, row in enumerate(candidates, start=1):
                row["rank"] = position
            selection = resolve_positions(keep, candidates, strategy_config, as_of=stamp)
            chosen = [row for row in candidates if str(row["symbol"]) in selection]
            weights = score_to_weights(chosen, strategy_config) if chosen else {}

        # Same rule as step 1: undeployed gross belongs in bills, not in cash.
        weights = park_residual(weights, book, strategy_config)

        if not any(weights.values()):
            # Nothing qualifies: sit in the defensive sleeve rather than in the least-bad name.
            weights = dict(book)

        return settle(weights)
