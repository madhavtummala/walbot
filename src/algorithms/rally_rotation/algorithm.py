"""Dual momentum: relative strength selects, absolute momentum permits.

One ``plan`` call, two passes over the universe. The first reads only bars -- features, scores,
eligibility, a proposed book. The second reads what is actually held and what previous runs saw,
and turns that proposal into a decision: which holdings stay, which are rotated out, which
challengers have earned a slot.

They used to be two calls made at separate moments, which is what let the second one reach into
the state store on its own. They are one pass now, and the memory arrives on the context.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ...core.interfaces import (
    Check,
    AlgorithmContext,
    AlgorithmPlan,
    AlgorithmRequirements,
    PortfolioSnapshot,
    SignalView,
    intents_from_weights,
)
from ..base import BaseAlgorithm
from .config import RallyRotationConfig
from .gates import climax_check, crash_stop, entry_checks, passes, universe_data_ok
from .memory import (
    action_due,
    observed_days,
    qualifying_days,
    record_action,
    resolve_positions,
    sessions_since,
    track_eligibility,
    track_ranking,
)
from .scoring import base_scores, compute_features
from .signals import build_signals, finalize, signal_view
from .sizing import (
    apply_turnover_filters,
    defensive_weights,
    park_residual,
    partial_adjustment,
    score_to_weights,
    sentiment_adjusted,
)

logger = logging.getLogger(__name__)


class RallyRotationAlgorithm(BaseAlgorithm):
    """Cross-sectional rank, per-name absolute momentum, position-aware rotation."""

    algorithm_id = "rally_rotation"
    tuning_class = RallyRotationConfig

    #: Once per session, at the open. Every feature is computed from daily bars, so a second
    #: look before the close reads the same closes and cannot produce a different answer. This
    #: was every 15 minutes, from when the score was built on intraday bars: left there it
    #: re-ran the same decision ~26 times a session, against a backtest that only steps daily.
    cron = "30 9 * * 1-5"

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        settings = self.tuning(config)
        return AlgorithmRequirements(
            price_symbols=sorted(set(settings.symbols) | set(current_positions)),
            daily_lookback_days=settings.required_daily_bars,
            daily_ma_days=settings.etf_ma_days,
            # Nothing intraday: every feature comes from the daily bars above. Asking for an
            # intraday window would make the live path fetch one on each run and the replay build
            # a HistoryCache over it, both for data no gate reads.
            intraday_lookback_minutes=settings.required_history_minutes,
            needs_sentiment=settings.uses_sentiment,
            # Eligibility history and the re-rank throttle both measure elapsed market days,
            # which means both need what previous runs recorded and the day they recorded it on.
            needs_state=True,
            # Unproven: keep it on paper until walk-forward results say otherwise.
            paper_only=True,
        )

    def sizing(self, config: Any) -> dict[str, float]:
        """From this algorithm's own tuning rather than the account's.

        The floors are part of the strategy here -- they are what the turnover brake is stated
        in -- so they cannot come from an account-level default the way they do elsewhere.
        """
        settings = self.tuning(config)
        return {
            "min_trade_dollars": settings.minimum_trade_notional,
            "rebalance_threshold": settings.rebalance_weight_threshold,
        }

    def plan(self, context: AlgorithmContext) -> AlgorithmPlan:
        settings = self.tuning(context.config)
        universe = self.rank_universe(context, settings)
        proposed = universe["weights"]
        signals = build_signals(universe["scored"], universe["data"], universe["defensive_book"], settings)

        current = PortfolioSnapshot(
            positions=context.positions, equity=context.equity
        ).weights(context.latest_prices)
        held = {symbol for symbol, weight in current.items() if weight > 0}

        state = dict(context.state)
        # ``notes`` is how the selection pass explains itself. Without it the rows could only be
        # re-derived from the market gates, and a name turned away by the settling period or the
        # replacement margin -- having passed every market gate -- got an invented reason.
        notes: dict[str, list[Check]] = {}
        weights = self._hold_or_rotate(context, settings, proposed, signals, current, held, state, notes)

        return AlgorithmPlan(
            intents=intents_from_weights(weights),
            signals=finalize(signals, weights, held, settings, notes),
            metadata={
                "allocation_mode": self._allocation_mode(weights, settings),
                # Reported only when it is actually an input, so the deck does not show a
                # sentiment reading for a run that never asked for one.
                **({"market_sentiment": context.market_sentiment} if settings.uses_sentiment else {}),
                "universe_data": universe["data"],
                "eligible_count": len(universe["ranked"]),
            },
            state=state,
        )

    # ----------------------------------------------------------------------------------
    # Pass one: market data only.
    # ----------------------------------------------------------------------------------

    def rank_universe(self, context: AlgorithmContext, settings: RallyRotationConfig) -> dict[str, Any]:
        """Features, scores, eligibility, ranking, and the book this pass would propose.

        Pure in the ``AlgorithmContext`` sense -- no state, no clock, no brokerage -- which is
        what lets the backtester drive the identical call the live runner does.

        Returns each layer rather than only the weights, so a test or an audit can ask which step
        rejected a name without re-running the four that came before it.
        """
        features = {
            symbol: compute_features(symbol, context.daily_bars_by_symbol.get(symbol, pd.DataFrame()), settings)
            for symbol in settings.symbols
        }
        scored = base_scores(features, settings)
        for symbol, row in scored.items():
            row["sentiment_score"] = float(context.sentiment_scores.get(symbol, 0.0))
            row["eligible"] = passes(entry_checks(row, settings))
            if settings.sentiment_weight:
                clip = max(settings.sentiment_clip, 0.0)
                tilt = max(-clip, min(clip, row["sentiment_score"])) * settings.sentiment_weight
                row["base_score"] = float(row["base_score"]) + tilt
                row["score_components"]["sentiment"] = tilt

        data = universe_data_ok(scored, settings)
        ranked = self._rank(scored, settings)
        # Always computed, whatever this pass proposes: the second half can decide to go
        # defensive for reasons only it can see, and it cannot derive a defensive book from a
        # risk-on proposal.
        defensive_book = defensive_weights(scored, settings)

        qualified = [
            row for row in ranked
            if int(row.get("rank") or 0) <= settings.entry_rank_max
            and float(row.get("base_score", 0.0)) >= settings.min_base_score
        ]
        entries = qualified[: max(settings.max_positions, 0)]

        if data["data_ok"] and entries:
            weights = score_to_weights(entries, settings)
            weights = sentiment_adjusted(weights, context.sentiment_scores, settings)
            weights = park_residual(weights, defensive_book, settings)
        else:
            weights = dict(defensive_book)

        return {
            "scored": scored,
            "data": data,
            "ranked": ranked,
            "entries": entries,
            "defensive_book": defensive_book,
            "weights": {symbol: float(weights.get(symbol, 0.0)) for symbol in settings.symbols},
        }

    @staticmethod
    def _rank(scored: dict[str, dict[str, Any]], settings: RallyRotationConfig) -> list[dict[str, Any]]:
        """Eligible risk-on names, best first. Ineligible names are never ranked."""
        eligible = [
            scored[symbol] for symbol in settings.risk_on_universe
            if symbol in scored and scored[symbol].get("eligible")
        ]
        eligible.sort(key=lambda row: float(row.get("base_score", 0.0)), reverse=True)
        for position, row in enumerate(eligible, start=1):
            row["rank"] = position
        return eligible

    # ----------------------------------------------------------------------------------
    # Pass two: the book, and the memory of previous runs.
    # ----------------------------------------------------------------------------------

    def _hold_or_rotate(
        self,
        context: AlgorithmContext,
        settings: RallyRotationConfig,
        proposed: dict[str, float],
        signals: dict[str, dict[str, Any]],
        current: dict[str, float],
        held: set[str],
        state: dict[str, Any],
        notes: dict[str, list[Check]],
    ) -> dict[str, float]:
        """Hold/exit asymmetry, replacement margin, risk stops.

        Mutates ``state`` in place, and records into ``notes`` every selection decision that the
        market gates cannot account for on their own.
        """
        # ``context.timestamp`` is the only clock this pass reads. Every elapsed-time decision
        # below has to measure against the moment this run describes, which in a replay is a
        # date months ago.
        as_of = context.timestamp
        stamp = as_of.isoformat()
        defensive = {name.upper() for name in settings.defensive_universe}
        rows = {symbol: dict(row, symbol=symbol) for symbol, row in signals.items()}
        book = {
            symbol: float(row["defensive_weight"])
            for symbol, row in rows.items() if float(row["defensive_weight"]) > 0
        }

        if not signals or not next(iter(signals.values()))["data_ok"]:
            # Not a bearish reading -- an unusable one. See ``universe_data_ok``.
            detail = next(iter(signals.values()))["data_detail"] if signals else "no signals"
            logger.warning("[%s] Rally Rotation holding the defensive sleeve: %s", stamp[:10], detail)
            return self._settle(book, proposed, current, context, settings)

        risk_rows = {symbol: row for symbol, row in rows.items() if symbol not in defensive}
        history = track_eligibility(state, risk_rows, settings, as_of)
        rank_history = track_ranking(state, risk_rows, settings, as_of)
        window = max(settings.eligibility_window, 1)

        # The stops answer to no clock. Everything else -- ranking, entering, replacing and the
        # considered exits -- happens on ``rerank_interval_days``, because they are the same
        # decision seen from different sides and the score they rest on has a twelve-session
        # horizon. Re-ranking every session asked it a question it cannot answer that fast, and
        # the book paid the spread for the noise.
        stopped = set()
        for symbol in held - defensive:
            row = rows.get(symbol, {})
            # The crash test is already among the exit gates, so only the climax one is news
            # here; appending both listed it twice on every holding.
            climax = climax_check(row, settings)
            if climax is not None:
                notes.setdefault(symbol, []).append(climax)
            for check in (crash_stop(row, settings), climax):
                if check is not None and not check.ok:
                    logger.warning("[%s] Rally Rotation stopping out %s: %s", stamp[:10], symbol, check.value)
                    stopped.add(symbol)
                    break

        due = action_due(state, "rerank", settings.rerank_interval_days, as_of)
        if not due:
            # Between re-rankings the book may only shrink, and only for a stop. Everything else
            # stays exactly where it is, which is the whole point of the throttle -- and is worth
            # saying, because otherwise a qualifying name looks unaccountably passed over.
            # Reachable only when the clock is set and readable -- that is what ``action_due``
            # answering False means -- so this cannot be the cold-start case.
            waiting = Check(
                label="Re-rank due",
                ok=False,
                value=f"{sessions_since(str(state['last_rerank_day']), as_of)} sessions since",
                limit=f"≥ {settings.rerank_interval_days} sessions",
            )
            for symbol in risk_rows:
                if symbol not in held:
                    notes.setdefault(symbol, []).append(waiting)
            survivors = (held - defensive) - stopped
            weights = {symbol: float(current.get(symbol, 0.0)) for symbol in survivors}
        else:
            record_action(state, "rerank", as_of)
            keep = self._survivors(rows, held - defensive - stopped, history, window, settings, stamp, notes)
            candidates = self._candidates(risk_rows, keep, rank_history, settings, notes)
            selection = resolve_positions(keep, candidates, settings, as_of=stamp)
            self._record_slots(candidates, selection, settings, notes)
            chosen = [row for row in candidates if str(row["symbol"]) in selection]
            weights = score_to_weights(chosen, settings) if chosen else {}
            # A holding that survived every exit test and still lost its place. Which of the two
            # ways that happened is the whole content of the row, and neither is derivable from
            # the gates: one is about the field, the other about there being no field at all.
            for symbol in keep - set(weights):
                notes.setdefault(symbol, []).append(Check(
                    label="Kept its slot",
                    ok=False,
                    value="displaced by a higher-scoring name" if chosen else "no name qualified; moved to the defensive sleeve",
                    limit=f"top {settings.max_positions}",
                ))

        # Same rule as pass one: undeployed gross belongs in bills, not in cash.
        weights = park_residual(weights, book, settings)
        if not any(weights.values()):
            # Nothing qualifies: sit in the defensive sleeve rather than in the least-bad name.
            weights = dict(book)
        return self._settle(weights, proposed, current, context, settings)

    @staticmethod
    def _survivors(
        rows: dict[str, dict[str, Any]],
        candidates: set[str],
        history: dict[str, dict[str, int]],
        window: int,
        settings: RallyRotationConfig,
        stamp: str,
        notes: dict[str, list[Check]],
    ) -> set[str]:
        """Which holdings stay. Two ways out, answering different questions.

        The count is the slow one: this name has stopped qualifying for long enough to mean
        something. The exit band is the fast one. Both are checked only once the window is full,
        because a cold state store has no history and reading that as "ineligible" would sell
        everything on the first run.
        """
        keep: set[str] = set()
        for symbol in candidates:
            watched = observed_days(history.get(symbol))
            days = qualifying_days(history.get(symbol))
            # Only once the window is full: a cold state store has no history, and reading that
            # as "ineligible" would sell everything on the first run after a restart.
            still_qualifying = watched < window or days > settings.exit_max_eligible_days
            notes.setdefault(symbol, []).append(Check(
                label="Still qualifying recently",
                ok=still_qualifying,
                value=f"eligible on {days} of the last {watched} days",
                limit=f"> {settings.exit_max_eligible_days}, once {window} days are on record",
            ))
            if not still_qualifying:
                logger.info(
                    "[%s] Rally Rotation exiting %s: eligible on only %d of the last %d days",
                    stamp[:10], symbol, days, window,
                )
                continue
            keep.add(symbol)
        return keep

    @staticmethod
    def _candidates(
        risk_rows: dict[str, dict[str, Any]],
        keep: set[str],
        rank_history: dict[str, dict[str, int]],
        settings: RallyRotationConfig,
        notes: dict[str, list[Check]],
    ) -> list[dict[str, Any]]:
        """Everything selectable this run, ranked. Holdings face eligibility alone.

        The quality floor and the settling period are *entry* conditions: a holding that would
        not be bought today is not thereby worth selling, and applying them symmetrically sells a
        name the moment it stops being a purchase.
        """
        candidates: list[dict[str, Any]] = []
        ordered = sorted(risk_rows.items(), key=lambda item: -float(item[1].get("base_score", 0.0)))
        for symbol, row in ordered:
            if not int(row.get("eligible", 0)):
                continue
            if symbol in keep:
                candidates.append(dict(row, symbol=symbol))
                continue
            score = float(row.get("base_score", 0.0))
            settled = qualifying_days(rank_history.get(symbol))
            entry = [
                Check(
                    label="Score above the quality floor",
                    ok=score >= settings.min_base_score,
                    value=f"{score:.2f}",
                    limit=f"≥ {settings.min_base_score:.2f}",
                ),
                Check(
                    label=f"Ranked in the top {settings.entry_rank_max} for long enough",
                    ok=settled >= settings.entry_min_eligible_days,
                    value=f"{settled} of the last {observed_days(rank_history.get(symbol))} days",
                    limit=f"≥ {settings.entry_min_eligible_days} days",
                ),
            ]
            notes.setdefault(symbol, []).extend(entry)
            if passes(entry):
                candidates.append(dict(row, symbol=symbol))
        for position, row in enumerate(candidates, start=1):
            row["rank"] = position
        return candidates

    @staticmethod
    def _record_slots(
        candidates: list[dict[str, Any]],
        selection: set[str],
        settings: RallyRotationConfig,
        notes: dict[str, list[Check]],
    ) -> None:
        """Why a qualifying candidate did not get a slot: the book was full of better names.

        The last gate, and the only competitive one. Everything above it is a statement about the
        name itself; this one is a statement about the field it was in, which is why it has to be
        recorded here rather than inferred from the row.
        """
        for row in candidates:
            symbol = str(row["symbol"])
            if symbol in selection:
                continue
            notes.setdefault(symbol, []).append(Check(
                label="Won a position slot",
                ok=False,
                value=f"rank {int(row.get('rank') or 0)} of {len(candidates)} qualifying",
                limit=(
                    f"top {settings.max_positions}, or beat a holding by "
                    f"{settings.min_score_delta_to_replace:.2f}"
                ),
            ))

    @staticmethod
    def _settle(
        weights: dict[str, float],
        proposed: dict[str, float],
        current: dict[str, float],
        context: AlgorithmContext,
        settings: RallyRotationConfig,
    ) -> dict[str, float]:
        """Damp the move to ``weights`` and return the book to aim at.

        The symbol set is the union of what pass one proposed, what is currently held, and what
        pass two decided to hold. It used to be the proposal alone, which silently discarded
        every decision pass two had just made about a name pass one did not re-propose: an
        incumbent that was *kept* vanished from the returned weights, and ``MODE_TARGET`` reads
        an absent symbol as a target of zero, so the position was sold. Over a 12-month replay
        that force-sold a still-qualifying holding on 73% of decisions and was the single largest
        source of turnover.
        """
        symbols = set(proposed) | set(current) | set(weights)
        stepped = partial_adjustment(
            {symbol: float(weights.get(symbol, 0.0)) for symbol in symbols}, current, settings
        )
        return apply_turnover_filters(stepped, current, float(context.equity or 0.0), settings)

    @staticmethod
    def _allocation_mode(weights: dict[str, float], settings: RallyRotationConfig) -> str:
        """The one-word summary the deck prints for this run."""
        defensive = {name.upper() for name in settings.defensive_universe}
        held = {symbol for symbol, weight in weights.items() if weight > 0}
        if not held:
            return "Cash"
        return "Defensive" if held <= defensive else "Risk-on"

    def signal_view(self, plan: AlgorithmPlan) -> SignalView:
        return signal_view(plan)
