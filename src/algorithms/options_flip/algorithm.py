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
from .candidates import scoring_parameters, trend_strength
from .indicators import average_true_range, quote_age_seconds
from .levels import conditional_levels
from .pricing import expected_profit, max_debit, scenarios
from .regime import bull_regime
from .excursion import option_price_for, session_fraction_remaining
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
        named = [str(symbol).upper() for symbol in (cfg.symbols or [])]
        if named:
            return named
        # This algorithm's own list, or the account's tradable universe. Rally Rotation's
        # configured universe is deliberately *not* consulted: the two run different symbol
        # lists on different accounts, and reading its section let one strategy's tuning
        # silently decide the other's candidates. Only its scoring function is borrowed.
        return [str(s).upper() for s in (getattr(config, "symbols", []) or [])]

    def requirements(self, config: Any, current_positions: dict[str, int]) -> AlgorithmRequirements:
        cfg = self.tuning(config)
        return AlgorithmRequirements(
            price_symbols=self._symbols(cfg, config),
            daily_lookback_days=cfg.required_daily_bars,
            daily_ma_days=cfg.regime_slow_ma_days,
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

        universe = self._symbols(cfg, context.config)
        # No ranking, and nothing shared between symbols. Each is scored from its own bars
        # inside ``_plan_one``, so adding or removing a name cannot change what the others do.
        symbols = sorted({*universe, *held}) if universe else sorted(held)
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
                "universe": len(universe),
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
                exit_target=_held_exit_target(intraday, session, cfg, underlying_now, daily),
                checks=[], config=cfg, session=session,
                oco=broker_supports_oco(getattr(context.config, "account_id", "")),
            )

        history, today = _split_sessions(intraday, session["market_day"])
        # ── gate 1: is the bull thesis intact today? ──────────────────────────────────
        eligible, regime, checks = bull_regime(
            daily, today, price=underlying_now, config=cfg,
        )
        # Scored from this symbol's own bars, against a threshold in its own sigma. A held
        # position is managed out whatever it scores today.
        strength = trend_strength(daily, scoring_parameters())
        trending = bool(strength >= float(cfg.min_trend_strength)) or bool(held_contract)
        checks = [Check(
            label="Trend strength",
            ok=trending,
            value=f"{strength:+.2f}σ" + ("" if trending else " — not trending enough today"),
            limit=f"≥ {float(cfg.min_trend_strength):+.2f}σ, measured on this symbol alone",
            blocking=not trending,
        )] + checks
        atr = float(regime.get("atr", 0.0) or 0.0)

        vol_ok, vol_check, annual_vol = _volatility_gate(daily, cfg)
        checks = checks + [vol_check]

        # ── gate 2: where would we buy and sell, and how often is each reached? ───────
        # Levels are sampled at the minute a decision could still be *made*, not at the clock's
        # current minute. Past the entry cutoff there is no remaining session to travel through,
        # so every past session contributes an empty "after" window and the sample comes back
        # zero -- which reads as "no comparable history" when the truth is "too late to arm one".
        # Outside the session entirely this shows the first fire, so an overnight deck previews
        # the decision tomorrow morning rather than a degenerate one.
        levels = conditional_levels(
            history, minute=_decision_minute(session, cfg), price=underlying_now,
            session_open=_session_open(intraday, session["market_day"], underlying_now),
            atr=atr, config=cfg,
        )
        in_time = float(session.get("fraction_remaining", 0.0)) >= float(cfg.entry_cutoff_fraction)
        # The entry is placed *at* the ``entry_reach`` quantile, so its own reach is that
        # number by construction and needs no separate floor. What still has to be checked is the
        # conditional half: of the days that dipped this far, how many then paid.
        # ``target_reach`` places the target at the reach it asks for, so the only remaining
        # question is whether a sample exists at all, and whether there is session left to work.
        levels_ok = levels["p_target"] > 0.0 and levels["entry"] > 0.0 and in_time
        checks = checks + [Check(
            label="Entry reachable",
            ok=levels["p_touch"] > 0.0,
            value=(
                f"{levels['p_touch']:.0%} of {levels['sample']} comparable sessions reached "
                f"${levels['entry']:,.2f} ({levels['k_entry']:.2f} ATR below)"
                + ("" if levels["conditional"] else " — unconditional sample, bucket was thin")
            ),
            limit=f"placed at the {float(cfg.entry_reach):.0%} reach quantile",
            blocking=levels["p_touch"] <= 0.0,
        ), Check(
            label="Target reachable after entry",
            ok=levels["p_target"] > 0.0,
            value=(
                f"{levels['p_target']:.0%} of those went on to ${levels['target']:,.2f} "
                f"({levels['k_target']:.2f} ATR above the entry)"
            ),
            limit=f"placed at the {float(cfg.target_reach):.0%} reach quantile of the days that dipped",
            blocking=levels["p_target"] <= 0.0,
        ), Check(
            label="Time to work",
            ok=in_time,
            value=f"{float(session.get('fraction_remaining', 0.0)):.0%} of the session left",
            limit=f"≥ {float(cfg.entry_cutoff_fraction):.0%} to arm a new entry",
            blocking=not in_time,
        )]

        # The contract is priced whatever the gates said, so a stand-down row still names it.
        contract, candidate, contract_checks = _pick_contract(
            context, symbol, CALL, session,  cfg,
            spot=underlying_now, annual_volatility=annual_vol,
        )
        checks = checks + contract_checks
        priced = contract or candidate

        # ── gate 3: does the base case clear its costs, through the full greeks? ──────
        estimate: dict[str, Any] = {}
        contracts = 0
        if priced is not None and atr > 0 and levels["entry"] > 0:
            outcomes = scenarios(
                priced, entry_underlying=levels["entry"], target_underlying=levels["target"],
                spot=underlying_now, config=cfg,
            )
            ceiling = max_debit(priced, outcomes, config=cfg)
            contracts = affordable_contracts(priced, cfg) if contract else 0
            profit = expected_profit(outcomes, contracts or 1, config=cfg)
            worth_it = profit["per_contract"] >= float(cfg.min_profit_per_contract)
            estimate = _estimate_row(
                priced, levels, outcomes, profit, ceiling, contracts, regime, cfg,
            )
            checks = checks + [Check(
                label="Worth trading",
                ok=worth_it,
                value=(
                    f"${profit['per_contract']:,.0f} per contract base case "
                    f"(bad ${profit['bad']:,.0f}, good ${profit['good']:,.0f})"
                ),
                limit=f"≥ ${float(cfg.min_profit_per_contract):,.0f} per contract, gross",
                blocking=not worth_it,
            ), Check(
                label="Max debit",
                ok=ceiling > 0,
                value=f"${ceiling:.2f} — the entry limit is never raised past this",
                limit="derived from the base case, not chosen",
            )]
            if not worth_it:
                contracts = 0
            age = quote_age_seconds(priced, session.get("now_ms", 0.0))
            if age > float(cfg.max_quote_age_seconds):
                checks = checks + [Check(
                    label="Quote fresh", ok=False, value=f"{age:,.0f}s old",
                    limit=f"≤ {float(cfg.max_quote_age_seconds):,.0f}s", blocking=True,
                )]
                contracts = 0

        if not (eligible and vol_ok and levels_ok and trending):
            contracts = 0
        direction = CALL if (eligible and vol_ok and levels_ok and trending and contracts > 0) else ""

        outcome = plan_symbol(
            symbol, memory=memory, held_contract="",
            # Momentum gates the order, not the analysis. Passing the *real* direction -- empty
            # when nothing is trending -- is how the state machine is told to stand down, while
            # the deck above keeps the previewed contract and the reasoning that goes with it.
            direction=direction,
            contract=contract, contracts=contracts, underlying_now=underlying_now,
            entry_target=levels["entry"], exit_target=float(levels.get("target", 0.0)),
            checks=checks, config=cfg, session=session,
            oco=broker_supports_oco(getattr(context.config, "account_id", "")),
        )
        if estimate and contract is None:
            estimate["tradable"] = False

        if outcome.memory and contract:
            # The delta is the one input the exit needs and cannot re-read: after a fill this
            # symbol's chain is no longer fetched, since the contract is already chosen.
            outcome.memory["delta"] = contract.delta
        return replace(outcome, estimate={
            **estimate,
            "direction": direction,
            "provisional": not direction,
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
    open_minute = open_at.hour * 60 + open_at.minute
    close_minute = close_at.hour * 60 + close_at.minute
    raw_minute = moment.hour * 60 + moment.minute
    in_session = open_minute <= raw_minute <= close_minute
    return {
        "market_day": market_day(now),
        # Minute of the market day, **clamped into session hours**. The band is keyed on it,
        # because intraday volatility is a function of the clock -- large at the open, small at
        # midday -- not of elapsed time.
        #
        # Clamped because the raw clock is meaningless outside the session and silently emptied
        # the sample: at 00:21 the minute is 21, ``session_sigma`` keeps bars at or before minute
        # 21, every bar sits between 570 and 960, and so sixteen sessions of history produced a
        # sample of zero and the deck reported "no band" while the data sat right there. Outside
        # the session the band is shown as of the last close, which is the most recent complete
        # picture rather than a degenerate one.
        "minute": raw_minute if in_session else close_minute,
        # So the deck can say the band is a review of the last close rather than a live reading.
        "in_session": in_session,
        #: Epoch milliseconds, for comparing against a chain quote's own stamp.
        "now_ms": pd.Timestamp(moment).timestamp() * 1000.0,
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
    contract, direction: str, underlying_now: float, levels: dict[str, float], exit_target: float,
    cfg, as_of: date, contracts: int = 1,
) -> dict[str, Any]:
    """The contract, its cost, and the price band this run expects to transact in.

    The whole point of reporting it is the last line: ``edge`` is the gap between the estimated
    exit and the estimated entry, and ``spread`` is what the market charges to make the round
    trip. If the spread is the larger of the two the strategy cannot pay for itself on that
    contract however well the direction is called, and no other number on the deck says so.
    """
    if contract is None or underlying_now <= 0:
        return {}

    units = max(int(contracts), 1)
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
        underlying_now * (1.0 + exit_target) if direction == CALL
        else underlying_now * (1.0 - exit_target)
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
        "exit_target": exit_target,
    }


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


def _decision_minute(session: dict[str, Any], cfg: Any) -> int:
    """The minute the level sample is drawn at: never past the entry cutoff, never before 10:00."""
    close_minute = 16 * 60
    first_fire = 10 * 60
    cutoff = int(close_minute - float(cfg.entry_cutoff_fraction) * 390)
    if not session.get("in_session", True):
        return first_fire
    return max(min(int(session.get("minute") or first_fire), cutoff), first_fire)


def _estimate_row(contract, levels, outcomes, profit, ceiling, contracts, regime, cfg) -> dict[str, Any]:
    """What the deck needs to judge the setup, on the days it trades and the days it does not.

    Every field is reported whether or not an order goes out. "No trade today" says nothing about
    whether the setup was close or hopeless; the contract, its price, the levels it would have
    transacted between and the modelled profit say which.
    """
    return {
        "contract": contract.osi_symbol,
        "contract_label": (
            f"${contract.strike:g} {contract.option_type} · {contract.expiry:%d %b} · "
            f"delta {contract.delta:+.2f}"
        ),
        "mark": contract.midpoint,
        "spread_pct": contract.spread_pct,
        "entry_underlying": levels["entry"],
        "target_underlying": levels["target"],
        "p_touch": levels["p_touch"],
        "p_target": levels["p_target"],
        "max_debit": ceiling,
        "expected_profit_per_contract": profit["per_contract"],
        "expected_profit": profit["total"],
        "bad_case": profit["bad"],
        "good_case": profit["good"],
        "contracts": contracts,
        "atr": regime.get("atr", 0.0),
        "vwap": regime.get("vwap", 0.0),
        "gap_atr": regime.get("gap_atr", 0.0),
    }


def _split_sessions(intraday, market_day: str):
    """``(history, today)`` -- past sessions and this one, in the shape :mod:`.band` wants.

    Sigma is a claim about *other* sessions. Leaving today in the history measures the day
    against itself, which drags the band toward whatever has already happened this morning and
    makes a genuine breakout read as ordinary.
    """
    if intraday is None or intraday.empty:
        empty = pd.DataFrame(columns=["ts", "open", "close", "day", "minute"])
        return empty, empty
    frame = intraday.copy()
    stamps = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    frame["ts"] = stamps
    frame["day"] = stamps.dt.date
    frame["minute"] = stamps.dt.hour * 60 + stamps.dt.minute
    today_key = date.fromisoformat(market_day)
    return frame[frame["day"] < today_key], frame[frame["day"] == today_key]


def _previous_close(history) -> float:
    """The last print of the most recent completed session, for the gap."""
    if history is None or history.empty:
        return 0.0
    last_day = history["day"].max()
    session = history[history["day"] == last_day].sort_values("ts")
    return float(session.iloc[-1]["close"]) if len(session) else 0.0


def _held_exit_target(intraday, session, cfg, underlying_now: float, daily) -> float:
    """The target *level* for a position already open, from the same model that opened it.

    Recomputed each run rather than remembered, so a position taken in a quiet session is not
    still asking a quiet session's price two days later. It reuses ``conditional_levels`` for the
    same reason the entry does: one model, one set of quantiles, and no second definition of
    "how far can this travel" that could drift away from the first.
    """
    history, _today = _split_sessions(intraday, session["market_day"])
    atr = average_true_range(daily, int(cfg.atr_days)) if daily is not None else 0.0
    if atr <= 0:
        return 0.0
    levels = conditional_levels(
        history, minute=_decision_minute(session, cfg), price=underlying_now,
        session_open=_session_open(intraday, session["market_day"], underlying_now),
        atr=atr, config=cfg,
    )
    return float(levels.get("target", 0.0))


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
