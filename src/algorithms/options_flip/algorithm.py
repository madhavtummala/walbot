"""Options Flip: buy a predicted intraday low, sell into strength, let the exchange hold the stop.

The strategy in one paragraph. Read a multi-day trend per symbol and require this morning's
pre-market to confirm it -- if it does not, that symbol does not trade today. Pick the nearest
contract at least ``min_dte`` out inside a delta band, and rest a limit buy at today's predicted
low -- an absolute price level derived from how far comparable past sessions pulled back before
going the right way. Walk that bid in toward the contract's midpoint as the session runs down, at
a pace ``entry_decay_power`` sets; it never crosses the spread, so a day the market never comes
to simply does not trade. On a fill, an OCO goes to the exchange: a profit limit that only ever
ratchets up, and a stop that never moves. Be flat within ``max_hold_sessions``.

**Why the orders live at the broker.** This runs on a five-minute cron, and the day's low lasts
about ninety seconds. A poller cannot buy it -- it can only ever transact at whatever the mark
happens to be when it wakes. A resting limit order captures a price the algorithm never observes,
which is the entire reason the entry is an order rather than a decision. The same argument holds
harder for the stop: a stop breached at 10:07 and recovered by 10:09 must fire, and only the
exchange is watching. So the cron is a *revision* cadence -- it re-prices a prediction -- and
never an execution cadence. That is also why streaming market data would buy nothing here.

**Everything is re-derived, nothing is remembered that can be observed.** Positions come from the
broker, session extremes come from today's bars, direction is recomputed each run. State carries
only what genuinely cannot be re-read: the fill price the stop is anchored to, the contract's
delta, and how many sessions a position has been held.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from datetime import date, datetime, time
from typing import Any

import pandas as pd

from ...core.interfaces import (
    MARKET_TZ,
    AlgorithmContext,
    AlgorithmPlan,
    AlgorithmRequirements,
    Check,
    SignalView,
)
from ...core.options import CALL, is_osi_symbol, parse_osi
from ...data.state_store import algorithm_state_key, save_state
from ..base import BaseAlgorithm
from ..rally_rotation.memory import market_day, sessions_since
from ..reconcile import ORDER_IDS_KEY, broker_supports_oco, reconcile_orders
from .config import OptionsFlipConfig
from .contracts import affordable_contracts, fill_missing_deltas, select_contract
from .direction import premarket_confirms, trend_direction, typical_daily_move
from .excursion import (
    expected_excursion,
    entry_underlying_target,
    option_price_for,
    predicted_extreme,
    session_fraction_remaining,
)
from .lifecycle import BIDDING, HELD, plan_symbol
from .signals import signal_view

logger = logging.getLogger(__name__)

REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)


class OptionsFlipAlgorithm(BaseAlgorithm):
    """One option contract per symbol, bought at a predicted low and bracketed at the exchange."""

    algorithm_id = "options_flip"
    tuning_class = OptionsFlipConfig

    #: Contracts are sized by count against a notional cap, so neither portfolio floor applies.
    min_trade_dollars = 0.0
    rebalance_threshold = 0.0

    #: The whole strategy is a resting limit that waits for a price the algorithm never observes,
    #: and a stop the exchange watches between runs. The replay's paper book fills at the mark and
    #: holds no orders, so there is nothing here for it to simulate -- and a book that filled
    #: every bid at the mark would report an edge that came entirely from the simulation.
    backtestable = False
    not_backtestable_reason = (
        "Options Flip works by leaving orders resting at the broker -- a limit waiting for a "
        "price, and a stop the exchange watches between runs. The backtester fills at the mark "
        "and keeps no resting orders, so it cannot represent either. Simulating it would report "
        "an edge that came from the simulation rather than the strategy."
    )

    #: Every five minutes through the session. Each fire re-prices the resting bid as the walk-in
    #: progresses and the underlying moves; between fires the resting order and the exchange-side
    #: stop are doing the actual work.
    cron = "*/5 9-15 * * 1-5"

    @staticmethod
    def _symbols(cfg: OptionsFlipConfig, config: Any) -> list[str]:
        """The symbols to run, falling back to the configured universe when none are named.

        Resolved in one place because ``requirements`` and ``plan`` must agree: if they disagree
        the context loads bars for one set and the algorithm iterates another, and the difference
        shows up as symbols that silently never trade.
        """
        return [str(symbol).upper() for symbol in (cfg.symbols or getattr(config, "symbols", []) or [])]

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        cfg = self.tuning(config)
        return AlgorithmRequirements(
            price_symbols=self._symbols(cfg, config),
            daily_lookback_days=cfg.required_daily_bars,
            daily_ma_days=cfg.trend_ma_period,
            intraday_lookback_minutes=cfg.required_intraday_minutes,
            preferred_bar_minutes=cfg.intraday_bar_minutes,
            needs_state=True,
            needs_option_chains=True,
            needs_premarket=True,
        )

    def plan(self, context: AlgorithmContext) -> AlgorithmPlan:
        cfg = self.tuning(context.config)
        state = dict(context.state or {})
        symbols_memory = dict(state.get("symbols") or {})
        session = _session_facts(context.timestamp, cfg)
        held = _held_contracts(context.positions)

        symbols = self._symbols(cfg, context.config)
        orders, signals, memory = [], {}, {}
        for symbol in symbols:
            outcome = self._plan_one(
                symbol, context, cfg, session,
                memory=dict(symbols_memory.get(symbol) or {}),
                held_contract=held.get(symbol, ""),
            )
            orders.extend(outcome.orders)
            signals[symbol] = _signal(outcome)
            if outcome.memory:
                memory[symbol] = outcome.memory

        return AlgorithmPlan(
            desired_orders=orders,
            signals=signals,
            state={**state, "symbols": memory, "last_run_day": session["market_day"]},
            metadata={
                "session": session,
                "reason": _reason(signals),
                "symbols_evaluated": len(symbols),
            },
        )

    def _plan_one(self, symbol, context, cfg, session, *, memory, held_contract):
        """One symbol, start to finish: direction, contract, budget, orders."""
        daily = context.daily_bars_by_symbol.get(symbol)
        intraday = context.intraday_bars_by_symbol.get(symbol)
        underlying_now = float(context.latest_prices.get(symbol, 0.0) or 0.0)

        # A held position is managed on the memory it was opened with. Re-deriving its direction
        # or contract each run would let a trend that has since turned rewrite the bracket around
        # a position it did not open -- the stop exists precisely for that case.
        if held_contract:
            memory = _refresh_held(memory, held_contract, context, session, cfg)
            return plan_symbol(
                symbol, memory=memory, held_contract=held_contract,
                direction=str(memory.get("direction") or ""), contract=None,
                contracts=int(memory.get("contracts", 1) or 1),
                underlying_now=underlying_now,
                entry_target=0.0,
                exit_budget=_budget(daily, intraday, memory.get("direction"), cfg, session, exit_side=True),
                checks=[], config=cfg, session=session,
                oco=broker_supports_oco(getattr(context.config, "account_id", "")),
            )

        direction, checks = trend_direction(daily, cfg)
        vol_ok, vol_check, annual_vol = _volatility_gate(daily, cfg)
        checks = checks + [vol_check]
        if not direction or not vol_ok:
            return plan_symbol(
                symbol, memory=memory, held_contract="", direction="", contract=None, contracts=0,
                underlying_now=underlying_now, entry_target=0.0, exit_budget=0.0,
                checks=checks, config=cfg, session=session,
            )

        premarket = _premarket(context, symbol)
        confirms, premarket_checks = premarket_confirms(
            premarket, direction, typical_daily_move(daily, cfg.volatility_window), cfg
        )
        checks = checks + premarket_checks

        # The contract is chosen even when pre-market has vetoed the trade. It costs one chain
        # request and it is the difference between a deck that says "no trade today" and one that
        # says which contract, at what price, against what estimated range -- which is what makes
        # the strategy assessable on the days it does not fire. Only the *orders* are gated.
        contract, candidate, contract_checks = _pick_contract(
            context, symbol, direction, session, cfg,
            spot=underlying_now, annual_volatility=annual_vol,
        )
        checks = checks + contract_checks
        contracts = affordable_contracts(contract, cfg) if contract else 0
        if contract and not contracts:
            checks = checks + [Check(
                label="Affordable at the notional cap",
                ok=False,
                value=f"${contract.ask * 100:,.0f} per contract",
                limit=f"≤ ${float(cfg.max_notional_per_trade):,.0f}",
                blocking=True,
            )]

        levels = _entry_levels(daily, intraday, direction, cfg, session, underlying_now)
        exit_budget = _budget(daily, intraday, direction, cfg, session, exit_side=True)
        # Estimated against whatever the chain's best in-band contract was, traded or not: a
        # rejected near-miss is exactly what has to be visible to judge the strategy.
        estimate = _estimate(
            contract or candidate, direction, underlying_now, levels, exit_budget, cfg,
            date.fromisoformat(session["market_day"]),
        )

        outcome = plan_symbol(
            symbol, memory=memory, held_contract="",
            # Pre-market is what gates the order, not the analysis: passing no direction is how
            # the state machine is told to stand down while the deck keeps the reasoning above.
            direction=direction if confirms else "",
            contract=contract, contracts=contracts, underlying_now=underlying_now,
            entry_target=levels["bid"], exit_budget=0.0,
            checks=checks, config=cfg, session=session,
            oco=broker_supports_oco(getattr(context.config, "account_id", "")),
        )
        if estimate and contract is None:
            estimate["tradable"] = False

        # The economic gate, and the last one applied: everything above asks whether the setup is
        # *right*, this asks whether it is worth doing at all. A contract whose predicted move is
        # a few dollars is a losing trade after any realistic round trip, even when every other
        # gate passed and the direction was called correctly -- which is exactly the trade that
        # is easiest to keep making without noticing.
        if contract and contracts > 0 and estimate:
            # Judged **per contract**, not on the position total. A floor on the total is
            # satisfiable by buying more -- ten contracts predicted to move two cents each clears
            # a $25 position floor -- so raising ``contracts_per_position`` would quietly loosen
            # the quality bar. Whether the setup is worth trading cannot depend on how much of it
            # you buy; size is a separate decision, made by the notional cap.
            per_contract = float(estimate.get("expected_profit_per_contract", 0.0))
            floor = float(cfg.min_expected_profit)
            worth_it = per_contract >= floor
            total = float(estimate.get("expected_profit", 0.0))
            checks = checks + [Check(
                label="Worth trading",
                ok=worth_it,
                value=(
                    f"${per_contract:,.0f} per contract"
                    + (f" (${total:,.0f} for {contracts})" if contracts > 1 else "")
                ),
                limit=f"≥ ${floor:,.0f} per contract",
                blocking=not worth_it,
            )]
            if not worth_it:
                contracts = 0
        if outcome.memory and contract:
            # The delta is the one input the exit needs and cannot re-read: after a fill this
            # symbol's chain is no longer fetched, since the contract is already chosen.
            outcome.memory["delta"] = contract.delta
        return replace(outcome, estimate={
            **estimate,
            "direction": direction,
            "premarket_change": (premarket or {}).get("change_pct"),
        })

    def execute(self, plan: AlgorithmPlan, config: Any, brokerage: Any, **kwargs: Any) -> dict[str, Any]:
        """Reconcile the resting orders, then save state. Never calls ``place_orders``.

        The base implementation sizes intents into share orders through the funding ladder and
        the rebalance floor, which are statements about a portfolio. This algorithm's output is
        an order book, so it goes to :func:`reconcile_orders` instead.

        **State is written unconditionally**, which is the one deliberate difference from the
        base class. There, state is committed only after orders go out, which is right for an
        algorithm whose memory is its order history. Here the memory includes the fill price the
        stop is anchored to and how long a position has been held -- facts that must survive a
        run which correctly did nothing, and which a cron would otherwise lose on every quiet
        fire. (Rally Rotation has the opposite bug for the same reason: its eligibility window is
        empty because dashboard previews never reach ``execute``.)
        """
        state = dict(plan.state or {})
        outcome = reconcile_orders(
            plan.desired_orders, brokerage, dict(state.get(ORDER_IDS_KEY) or {})
        )
        state[ORDER_IDS_KEY] = outcome["order_ids"]
        save_state(algorithm_state_key(self.algorithm_id, getattr(config, "account_id", "")), state)
        return {
            "strategy": plan.strategy,
            "mode": "lifecycle",
            "status": "ok",
            "equity": 0.0,
            "final_weights": {},
            "order_results": outcome["results"],
            "working_orders": outcome["working"],
            "state": state,
        }

    def signal_view(self, plan: AlgorithmPlan) -> SignalView:
        return signal_view(plan)


def _session_facts(now: datetime, cfg: OptionsFlipConfig) -> dict[str, Any]:
    """The run's market-time facts, so nothing downstream needs a clock."""
    moment = now.astimezone(MARKET_TZ) if now.tzinfo else now.replace(tzinfo=MARKET_TZ)
    today = moment.date()
    open_at = pd.Timestamp(datetime.combine(today, REGULAR_OPEN, tzinfo=MARKET_TZ))
    close_at = pd.Timestamp(datetime.combine(today, REGULAR_CLOSE, tzinfo=MARKET_TZ))
    return {
        "market_day": market_day(now),
        "fraction_remaining": session_fraction_remaining(
            pd.Timestamp(moment), open_time=open_at, close_time=close_at
        ),
    }


def _parse_time(value: str) -> time:
    hour, _, minute = str(value or "").partition(":")
    return time(int(hour or 0), int(minute or 0))


def _held_contracts(positions: dict[str, Any]) -> dict[str, str]:
    """Option positions keyed by their underlying.

    Schwab reports options and equities in one map keyed by symbol, with nothing else marking
    the difference -- the shape of an OSI string is the only signal, which is why
    :func:`is_osi_symbol` exists.
    """
    held: dict[str, str] = {}
    for symbol, quantity in (positions or {}).items():
        if float(quantity or 0) <= 0 or not is_osi_symbol(symbol):
            continue
        try:
            held[parse_osi(symbol)["underlying"]] = str(symbol).upper()
        except ValueError:
            logger.warning("Options Flip ignoring unparseable position symbol %r", symbol)
    return held


def _volatility_gate(daily: pd.DataFrame, cfg: OptionsFlipConfig) -> tuple[bool, Check, float]:
    """This strategy needs movement, and not too much of it.

    Below the floor there is no excursion to predict and the premium is pure theta. Above the
    ceiling the premium already prices in a bigger move than the model is forecasting, so a
    correct prediction still loses.
    """
    annual = 0.0
    if daily is not None and len(daily) > cfg.volatility_window:
        returns = daily["close"].astype(float).pct_change().dropna().tail(cfg.volatility_window)
        annual = float(returns.std() * (252 ** 0.5)) if len(returns) >= 2 else 0.0
    ok = annual <= float(cfg.max_annual_volatility)
    return ok, Check(
        label="Volatility not excessive",
        ok=ok,
        value=f"{annual:.0%} annualised",
        limit=f"≤ {float(cfg.max_annual_volatility):.0%}",
        blocking=not ok,
    ), annual


def _premarket(context: AlgorithmContext, symbol: str) -> dict[str, Any] | None:
    """This morning's pre-market for one symbol, or ``None`` if it could not be read.

    Failure is not fatal and not silent: a symbol whose pre-market cannot be read simply does not
    trade today, because confirmation is required and absent confirmation is not confirmation.
    """
    reader = (context.extra or {}).get("premarket")
    if reader is None:
        return None
    try:
        return reader(symbol)
    except Exception as exc:
        logger.warning("Options Flip could not read pre-market for %s: %s", symbol, exc)
        return None


def _pick_contract(context, symbol, direction, session, cfg, spot=0.0, annual_volatility=0.0):
    reader = (context.extra or {}).get("option_chain")
    if reader is None:
        return None, None, [Check(
            label="Option chain available",
            ok=False,
            value="no chain provider bound",
            limit="required to choose a contract",
            blocking=True,
        )]
    try:
        chain = reader(symbol, option_type=direction, min_dte=cfg.min_dte, max_dte=cfg.max_dte)
    except Exception as exc:
        logger.warning("Options Flip could not read the chain for %s: %s", symbol, exc)
        return None, None, [Check(
            label="Option chain available",
            ok=False, value=str(exc)[:80], limit="chain request succeeded", blocking=True,
        )]
    as_of = datetime.fromisoformat(session["market_day"]).date()
    chain, estimated = fill_missing_deltas(
        chain, spot=spot, annual_volatility=annual_volatility, as_of=as_of,
    )
    best, candidate, checks = select_contract(chain, direction=direction, as_of=as_of, config=cfg)
    if estimated:
        # Said out loud, because it changes how much the chosen strike should be trusted.
        checks = [Check(
            label="Greeks estimated",
            ok=True,
            value="the chain returned none; delta computed from realised volatility",
            limit="provider greeks preferred",
        )] + checks
    return best, candidate, checks


def _estimate(
    contract, direction: str, underlying_now: float, levels: dict[str, float], exit_budget: float,
    cfg, as_of: date,
) -> dict[str, Any]:
    """The contract, its cost, and the price band this run expects to transact in.

    The whole point of reporting it is the last line: ``edge`` is the gap between the estimated
    exit and the estimated entry, and ``spread`` is what the market charges to make the round
    trip. If the spread is the larger of the two the strategy cannot pay for itself on that
    contract however well the direction is called, and no other number on the deck says so.
    """
    if contract is None or underlying_now <= 0:
        return {}

    units = max(int(cfg.contracts_per_position), 1)
    # The band's low is the *predicted* low, not the walk-in's current position. The band is a
    # statement about the session -- "this is the range the model expects" -- and it should not
    # collapse to the current price simply because the session is nearly over or has not started.
    # Where the bid actually sits is reported separately, on the Entry bid gate.
    entry = option_price_for(
        levels.get("predicted") or underlying_now,
        underlying_now=underlying_now, option_mark=contract.midpoint, delta=contract.delta,
    )
    bid_now = option_price_for(
        levels.get("bid") or underlying_now,
        underlying_now=underlying_now, option_mark=contract.midpoint, delta=contract.delta,
    )
    exit_underlying = (
        underlying_now * (1.0 + exit_budget) if direction == CALL
        else underlying_now * (1.0 - exit_budget)
    )
    exit_price = option_price_for(
        exit_underlying, underlying_now=underlying_now,
        option_mark=contract.midpoint, delta=contract.delta,
    )
    low, high = round(entry, 2), round(exit_price, 2)
    return {
        "contract": contract.osi_symbol,
        "contract_label": (
            f"${contract.strike:g} {contract.option_type} · {contract.expiry:%d %b}"
            f" · {contract.dte(as_of)}d · delta {contract.delta:.2f}"
        ),
        "strike": contract.strike,
        "expiry": contract.expiry.isoformat(),
        "delta": contract.delta,
        "units": units,
        "mark": round(contract.midpoint, 2),
        "bid": contract.bid,
        "ask": contract.ask,
        "spread_pct": contract.spread_pct,
        "cost_per_contract": round(contract.midpoint * 100.0, 2),
        "cost_total": round(contract.midpoint * 100.0 * units, 2),
        "estimated_low": low,
        "estimated_high": high,
        "estimated_low_total": round(entry * 100.0 * units, 2),
        "estimated_high_total": round(exit_price * 100.0 * units, 2),
        # Derived from the *rounded* band, so "Est Profit" is always exactly the displayed band
        # times 100. Computing it from the unrounded prices left the two disagreeing by up to a
        # dollar, which invites the reader to go looking for a difference that is not there.
        "edge": round(high - low, 2),
        #: Gross, per the whole position. Commissions are not modelled anywhere in this codebase,
        #: so netting them here would have been the one place claiming a precision the rest of
        #: the system does not have.
        "expected_profit": round((high - low) * 100.0 * units, 2),
        #: The same figure for a single contract. This is what the economics of the *setup* are
        #: judged on -- see the "Worth trading" gate.
        "expected_profit_per_contract": round((high - low) * 100.0, 2),
        "spread": round(max(contract.ask - contract.bid, 0.0), 2),
        "predicted_underlying": round(float(levels.get("predicted") or 0.0), 4),
        "bid_underlying": round(float(levels.get("bid") or 0.0), 4),
        "bid_now": round(bid_now, 2),
        "exit_budget": exit_budget,
    }


def _budget(daily, intraday, direction, cfg, session, *, exit_side: bool) -> float:
    """The exit's favourable excursion, as a fraction of the underlying.

    The direction is **inverted** for the exit, and that is the whole subtlety. ``excursions``
    measures the move *against* the trade -- for a call it is ``(open - low) / open``, the dip an
    entry waits for. A profit target is the opposite move, so it has to be measured off the
    opposite tail: how far the underlying typically *rises* above its open. Passing the trade's
    own direction here priced every call's target off the distribution of falls, and then applied
    it upward. On a symmetric name the two are close enough to look right, which is exactly why
    it survived.
    """
    if not direction or daily is None or daily.empty:
        return 0.0
    probability = float(cfg.exit_fill_probability if exit_side else cfg.target_fill_probability)
    measured = ("put" if direction == CALL else CALL) if exit_side else direction
    expected = expected_excursion(
        daily, direction=measured, lookback=cfg.excursion_lookback_days,
        fill_probability=probability,
        # The exit is held for up to ``max_hold_sessions``, so it is priced off the move available
        # over that many sessions rather than over one. The entry waits for a dip today.
        horizon=max(int(cfg.max_hold_sessions), 1) if exit_side else 1,
    )["excursion"]
    # Exit only. It names a favourable excursion still to come and should not shrink just
    # because the day is late, so there is no decay here -- only the entry walks in.
    return expected


def _session_open(intraday, market_day: str, fallback: float) -> float:
    """Today's opening price, from today's bars only.

    The lookback is stated in market minutes, so a 390-minute window at 09:35 spans yesterday's
    afternoon as well as this morning -- and ``bars.iloc[0]`` is then *yesterday's* open. Anchoring
    a prediction about today to that is not a rounding error: on a gap it moves the predicted low
    by the size of the gap.

    Falls back to the current price when today has no bars yet, which is right at the open: the
    first print of the session is the open.
    """
    if intraday is None or intraday.empty:
        return fallback
    stamps = pd.to_datetime(intraday["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    today = intraday[stamps.dt.date.astype(str) == market_day]
    if today.empty:
        return fallback
    return float(today["open"].astype(float).iloc[0])


def _entry_levels(daily, intraday, direction, cfg, session, underlying_now: float) -> dict[str, float]:
    """``{"predicted", "bid"}`` -- the session's predicted low, and where the bid sits right now.

    Two different statements, and conflating them is what made the deck's band start at the
    current price outside market hours. ``predicted`` is a claim about the *day*: ``open × (1 -
    expected)``, fixed once the session opens. ``bid`` is where the walk-in has reached at this
    moment, which converges on the market as the session runs down -- and collapses onto it
    entirely when there is no session left, which is correct for an order and meaningless as a
    forecast.

    A price rather than a fraction, because the prediction is about today's low and that is a
    level. Handing a fraction downstream to be applied to whatever the price is at the moment of
    the run made the bid drift up with a rally.
    """
    if not direction or daily is None or daily.empty or underlying_now <= 0:
        return {"predicted": 0.0, "bid": 0.0}
    expected = expected_excursion(
        daily, direction=direction, lookback=cfg.excursion_lookback_days,
        fill_probability=float(cfg.target_fill_probability),
    )["excursion"]
    session_open = _session_open(intraday, session["market_day"], underlying_now)
    predicted = predicted_extreme(session_open, expected, direction=direction)
    return {
        "predicted": predicted,
        "bid": entry_underlying_target(
            underlying_now, predicted, direction=direction,
            fraction_remaining=session["fraction_remaining"],
            decay_power=float(cfg.entry_decay_power),
        ),
    }


def _refresh_held(memory, held_contract, context, session, cfg) -> dict[str, Any]:
    """Bring a held position's memory up to date: its mark, and how long it has been held.

    ``sessions_held`` is counted in market days from the fill rather than in runs, for the same
    reason every other clock in this codebase is: a run count makes the deadline a function of
    the cron expression, so editing the schedule would silently redefine "two sessions".
    """
    memory = dict(memory)
    memory.setdefault("contract", held_contract)
    memory.setdefault("contracts", 1)
    mark = float(context.latest_prices.get(held_contract, 0.0) or 0.0)
    if mark > 0:
        memory["mark"] = mark
    if not memory.get("fill_price"):
        # Opened outside this algorithm, or state was lost. The mark is the only anchor left, and
        # anchoring the stop to it is conservative: it sets the floor from here rather than
        # pretending to know a cost basis we do not have.
        memory["fill_price"] = mark
        memory.setdefault("filled_day", session["market_day"])
        logger.warning(
            "Options Flip found %s held with no recorded fill; anchoring the stop to the mark",
            held_contract,
        )
    filled_day = str(memory.get("filled_day") or session["market_day"])
    memory["filled_day"] = filled_day
    try:
        memory["sessions_held"] = sessions_since(filled_day, datetime.fromisoformat(session["market_day"]))
    except ValueError:
        memory["sessions_held"] = 0
    memory["state"] = HELD
    return memory


def _signal(outcome) -> dict[str, Any]:
    """One symbol's row, as JSON-able plain data -- the plan is persisted and rendered."""
    memory = outcome.memory or {}
    estimate = outcome.estimate or {}
    return {
        "state": outcome.state,
        "headline": outcome.headline,
        "checks": [asdict(check) for check in outcome.checks],
        "direction": estimate.get("direction") or memory.get("direction", ""),
        "contract": estimate.get("contract") or memory.get("contract", ""),
        "fill_price": memory.get("fill_price", 0.0),
        "mark": memory.get("mark") or estimate.get("mark", 0.0),
        "premarket_change": estimate.get("premarket_change"),
        "estimate": estimate,
    }


def _reason(signals: dict[str, dict[str, Any]]) -> str:
    """Why nothing happened, when nothing did -- the first blocking gate across the symbols."""
    if any(signal.get("state") in (BIDDING, HELD) for signal in signals.values()):
        return ""
    for signal in signals.values():
        for check in signal.get("checks") or []:
            if check.get("blocking"):
                # The measurement, not just the gate's name. A bare label reads as a statement
                # of fact -- "Pre-market has enough prints" -- when it is the thing that failed.
                label, value = str(check.get("label", "")), str(check.get("value", ""))
                return f"{label}: {value}" if value else label
    return "no symbol confirmed a direction"
