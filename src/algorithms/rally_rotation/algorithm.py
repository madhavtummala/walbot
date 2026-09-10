"""Dual momentum: relative strength selects, absolute momentum permits.

One ``plan`` call, two passes over the universe. The first reads only bars -- features, scores,
eligibility, a proposed book. The second reads what is actually held and what previous runs saw,
and turns that proposal into a decision: which holdings stay, which are rotated out, which
challengers have earned a slot.
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
from .config import MIN_ENTRY_SCORE, RallyRotationConfig
from .gates import crash_stop, entry_checks, passes, universe_data_ok
from .memory import (
    action_due,
    observed_days,
    qualifying_days,
    record_action,
    resolve_positions,
    sessions_since,
    track_ranking,
)
from .scoring import base_scores, compute_features
from .signals import build_signals, finalize, signal_view
from .sizing import (
    apply_turnover_filters,
    defensive_weights,
    park_residual,
    score_to_weights,
)

logger = logging.getLogger(__name__)


class RallyRotationAlgorithm(BaseAlgorithm):
    """Cross-sectional rank, per-name absolute momentum, position-aware rotation."""

    algorithm_id = "rally_rotation"
    tuning_class = RallyRotationConfig

    #: Once per session, at the open. Every feature is computed from daily bars, so a second
    #: look before the close would read the same closes and cannot produce a different answer.
    cron = "30 9 * * 1-5"

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        settings = self.tuning(config)
        return AlgorithmRequirements(
            price_symbols=sorted(set(settings.symbols) | set(current_positions)),
            daily_lookback_days=settings.required_daily_bars,
            daily_ma_days=settings.etf_ma_days,
            # Nothing intraday: every feature comes from the daily bars above.
            intraday_lookback_minutes=settings.required_history_minutes,
            # Eligibility history and the re-rank throttle both measure elapsed market days.
            needs_state=True,
            # Unproven: keep it on paper until walk-forward results say otherwise.
            paper_only=True,
        )

    def sizing(self, config: Any) -> dict[str, float]:
        """From this algorithm's own tuning, not the account's -- the floors are part of the
        strategy here, the turnover brake's own unit."""
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
        # How the selection pass explains itself -- not re-derivable from the market gates
        # alone, since a name can be turned away by the settling period or replacement margin
        # having passed every one of them.
        notes: dict[str, list[Check]] = {}
        weights = self._hold_or_rotate(context, settings, proposed, signals, current, held, state, notes)

        return AlgorithmPlan(
            intents=intents_from_weights(weights),
            signals=finalize(signals, weights, held, settings, notes),
            metadata={
                "allocation_mode": self._allocation_mode(weights, settings),
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

        Pure in the ``AlgorithmContext`` sense -- no state, no clock, no brokerage. Returns each
        layer rather than only the weights, so an audit can ask which step rejected a name.
        """
        features = {
            symbol: compute_features(symbol, context.daily_bars_by_symbol.get(symbol, pd.DataFrame()), settings)
            for symbol in settings.symbols
        }
        scored = base_scores(features, settings)
        for row in scored.values():
            row["eligible"] = passes(entry_checks(row, settings))

        data = universe_data_ok(scored, settings)
        ranked = self._rank(scored, settings)
        # Always computed: the second pass can decide to go defensive for reasons only it can
        # see, and cannot derive a defensive book from a risk-on proposal.
        defensive_book = defensive_weights(scored, settings)

        qualified = [
            row for row in ranked
            if int(row.get("rank") or 0) <= settings.entry_rank_max
            and float(row.get("base_score", 0.0)) >= MIN_ENTRY_SCORE
        ]
        entries = qualified[: max(settings.max_positions, 0)]

        if data["data_ok"] and entries:
            weights = score_to_weights(entries, settings)
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

        Mutates ``state`` in place, and records into ``notes`` every selection decision the
        market gates cannot account for on their own.
        """
        # The only clock this pass reads -- in a replay, a date months ago.
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
        rank_history = track_ranking(state, risk_rows, settings, as_of)

        # The stops answer to no clock; ranking, entering, replacing and considered exits all sit
        # on ``rerank_interval_days`` instead, since the score they rest on has a slow horizon.
        stopped = set()
        for symbol in held - defensive:
            check = crash_stop(rows.get(symbol, {}), settings)
            if not check.ok:
                logger.warning("[%s] Rally Rotation stopping out %s: %s", stamp[:10], symbol, check.value)
                stopped.add(symbol)

        due = action_due(state, "rerank", settings.rerank_interval_days, as_of)
        if not due:
            # Between re-rankings the book may only shrink, and only for a stop.
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
            keep = (held - defensive) - stopped
            candidates = self._candidates(risk_rows, keep, rank_history, settings, notes)
            selection = resolve_positions(keep, candidates, settings, as_of=stamp)
            self._record_slots(candidates, selection, settings, notes)
            chosen = [row for row in candidates if str(row["symbol"]) in selection]
            weights = score_to_weights(chosen, settings) if chosen else {}
            # A holding that survived every exit test and still lost its place -- neither reason
            # is derivable from the gates alone.
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
    def _candidates(
        risk_rows: dict[str, dict[str, Any]],
        keep: set[str],
        rank_history: dict[str, dict[str, int]],
        settings: RallyRotationConfig,
        notes: dict[str, list[Check]],
    ) -> list[dict[str, Any]]:
        """Everything selectable this run, ranked. Holdings face eligibility alone.

        The settling period is an *entry* condition: a holding that wouldn't be bought today
        isn't thereby worth selling. Eligibility is not exempt though -- an ineligible name never
        reaches the ranked list, so ``resolve_positions`` cannot retain it and it's sold;
        ``exit_rank_max`` protects a holding that slipped in rank, not one that failed a gate.
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
                    label="Scores at or above the universe median",
                    ok=score >= MIN_ENTRY_SCORE,
                    value=f"{score:.2f}",
                    limit=f"≥ {MIN_ENTRY_SCORE:.2f}",
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

        The last gate, and the only competitive one -- a statement about the field, not the
        name, so it has to be recorded here rather than inferred from the row.
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
        """Filter out trades too small to be worth their costs, and return the book to aim at.

        The symbol set is the union of what pass one proposed, what is currently held, and what
        pass two decided to hold -- not the proposal alone, which would silently discard a
        pass-two decision about a name pass one didn't re-propose (``MODE_TARGET`` reads an
        absent symbol as a target of zero, selling a still-qualifying holding).
        """
        symbols = set(proposed) | set(current) | set(weights)
        target = {symbol: float(weights.get(symbol, 0.0)) for symbol in symbols}
        return apply_turnover_filters(target, current, float(context.equity or 0.0), settings)

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
