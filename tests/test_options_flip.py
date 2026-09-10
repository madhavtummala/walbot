"""Options Flip: regime gating, level probabilities, greeks pricing and the state machine.

The bar fixtures are built rather than recorded, so each one states the fact it is testing --
"thirty sessions that each dipped 2% and closed up" is a distribution with a known quantile, and
a test against it fails for one reason.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.algorithms.options_flip.config import ENTRY_MAX_REPRICE_PCT, OptionsFlipConfig
from src.algorithms.options_flip.contracts import affordable_contracts, select_contract
from src.algorithms.options_flip.indicators import (
    average_true_range,
    directional_volume,
    ma_slope,
    moving_average,
    opening_range,
    session_vwap,
)
from src.algorithms.options_flip.levels import conditional_levels, excursion_samples
from src.algorithms.options_flip.pricing import Scenario, max_debit, option_change, scenarios
from src.algorithms.options_flip.regime import bull_regime
from src.algorithms.options_flip.excursion import (
    option_price_for,
    session_fraction_remaining,
    target_price,
)
from src.algorithms.options_flip.lifecycle import BIDDING, FLAT, HELD, plan_symbol
from src.algorithms.options_flip.option_band import choose_band, option_daily_atr, prepare_option_bars
from src.core.interfaces import MARKET_TZ
from src.core.options import CALL, PUT, OptionContract, is_osi_symbol, osi_symbol, parse_osi


# ── fixtures ─────────────────────────────────────────────────────────────────


def bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Daily OHLC from ``(open, high, low, close)`` tuples, with a plausible index."""
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    frame["volume"] = 1_000_000.0
    frame["timestamp"] = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="UTC")
    return frame


def trending_bars(n: int = 60, dip: float = 0.02, gain: float = 0.01) -> pd.DataFrame:
    """``n`` sessions that each dip ``dip`` below the open and close ``gain`` above it."""
    return bars([(100.0, 100.0 * (1 + gain), 100.0 * (1 - dip), 100.0 * (1 + gain))] * n)


def contract(**kwargs) -> OptionContract:
    defaults = dict(
        osi_symbol="QQQM  260220C00100000", underlying="QQQM", option_type=CALL,
        strike=100.0, expiry=date(2026, 2, 20), bid=2.00, ask=2.10, mark=2.05,
        delta=0.45, open_interest=1000, volume=500, implied_volatility=0.25,
    )
    return OptionContract(**{**defaults, **kwargs})


def cfg(**kwargs) -> OptionsFlipConfig:
    # Most lifecycle fixtures were written against a live stop; the deployed default is now off.
    kwargs.setdefault("stop_loss_pct", 0.10)
    # The contract fixtures below are built at delta 0.45, so the tests state the target that
    # matches them rather than tracking the deployed default, which is a tuning decision and not
    # what any of these assertions are about.
    kwargs.setdefault("target_delta", 0.45)
    # No ``symbols`` argument: which symbols trade is the board's statement now, not a knob.
    return OptionsFlipConfig(**kwargs)


def session(**kwargs) -> dict:
    return {
        "market_day": "2026-02-10",
        "fraction_remaining": 0.5,
        **kwargs,
    }


# ── OSI symbols ──────────────────────────────────────────────────────────────


class TestOsiSymbols:
    def test_round_trip(self) -> None:
        symbol = osi_symbol("AAPL", date(2025, 1, 17), "call", 150.0)
        assert symbol == "AAPL  250117C00150000"
        assert len(symbol) == 21
        assert parse_osi(symbol) == {
            "underlying": "AAPL", "expiry": date(2025, 1, 17),
            "option_type": CALL, "strike": 150.0,
        }

    def test_fractional_strike_survives(self) -> None:
        symbol = osi_symbol("SPY", date(2026, 3, 20), "put", 512.5)
        assert parse_osi(symbol)["strike"] == 512.5
        assert parse_osi(symbol)["option_type"] == PUT

    def test_root_is_right_padded(self) -> None:
        # A left-padded root parses without error and names a contract that does not exist.
        assert osi_symbol("F", date(2026, 3, 20), "call", 12.0).startswith("F     ")

    def test_equity_symbol_is_not_an_option(self) -> None:
        assert not is_osi_symbol("AAPL")
        assert is_osi_symbol("AAPL  250117C00150000")

    def test_malformed_symbol_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_osi("NOTANOPTION")


class TestOptionContract:
    def test_midpoint_and_spread(self) -> None:
        assert contract(bid=2.0, ask=2.10).midpoint == pytest.approx(2.05)
        assert contract(bid=2.0, ask=2.10).spread_pct == pytest.approx(0.0488, abs=1e-3)

    def test_one_sided_quote_falls_back_and_reads_as_untradable(self) -> None:
        one_sided = contract(bid=0.0, ask=2.10)
        assert one_sided.midpoint == 2.10
        # Infinite rather than zero: this is compared against a ceiling, and zero would make the
        # widest possible market look like the tightest.
        assert one_sided.spread_pct == float("inf")


# ── the excursion model ──────────────────────────────────────────────────────


class TestTargetAndDeltaTranslation:
    def test_a_call_waits_below_the_market_and_a_put_above(self) -> None:
        assert target_price(100.0, 0.02, direction=CALL) == pytest.approx(98.0)
        assert target_price(100.0, 0.02, direction=PUT) == pytest.approx(102.0)

    def test_call_bid_is_below_the_mark(self) -> None:
        price = option_price_for(98.0, underlying_now=100.0, option_mark=2.05, delta=0.45)
        assert price == pytest.approx(2.05 - 0.90)

    def test_put_adverse_move_lowers_the_bid_too(self) -> None:
        # The underlying rising is adverse for a put, and the negative delta must carry that.
        # With abs(delta) this would price the adverse move as a gain.
        price = option_price_for(102.0, underlying_now=100.0, option_mark=2.05, delta=-0.45)
        assert price == pytest.approx(2.05 - 0.90)

    def test_price_never_goes_non_positive(self) -> None:
        assert option_price_for(50.0, underlying_now=100.0, option_mark=1.0, delta=0.9) == 0.01


# ── direction ────────────────────────────────────────────────────────────────


class TestContractSelection:
    def chain(self) -> list[OptionContract]:
        return [
            contract(osi_symbol="Q1", strike=95.0, delta=0.70, expiry=date(2026, 2, 20)),
            contract(osi_symbol="Q2", strike=100.0, delta=0.45, expiry=date(2026, 2, 20)),
            contract(osi_symbol="Q3", strike=105.0, delta=0.30, expiry=date(2026, 2, 20)),
        ]

    def test_picks_the_mid_band_delta(self) -> None:
        best, _candidate, checks = select_contract(
            self.chain(), direction=CALL, as_of=date(2026, 2, 1), config=cfg()
        )
        assert best is not None and best.osi_symbol == "Q2"
        assert all(check.ok for check in checks)

    def test_rejects_expiries_inside_the_dte_floor(self) -> None:
        soon = [contract(expiry=date(2026, 2, 3))]
        best, _candidate, checks = select_contract(soon, direction=CALL, as_of=date(2026, 2, 1), config=cfg(min_dte=10))
        assert best is None
        assert checks[0].blocking

    def test_prefers_the_nearest_qualifying_expiry(self) -> None:
        chain = [
            contract(osi_symbol="NEAR", expiry=date(2026, 2, 20), delta=0.45),
            contract(osi_symbol="FAR", expiry=date(2026, 3, 20), delta=0.45),
        ]
        best, _candidate, _checks = select_contract(chain, direction=CALL, as_of=date(2026, 2, 1), config=cfg())
        assert best is not None and best.osi_symbol == "NEAR"

    def test_a_wide_spread_is_disqualifying(self) -> None:
        """A wide market is refused, and the reason is the stop rather than the entry.

        Resting as a limit means the entry never *crosses* the spread -- but the stop is
        denominated in premium, so a contract quoted 8.2% wide against a 10% stop has spent four
        fifths of the risk budget before the direction call is tested. Measured live on IAU's
        Sep-4 84 call, which carried 164 open interest and so cleared the floor comfortably.
        """
        wide = [contract(bid=0.90, ask=1.30, delta=0.45)]   # 36% wide

        best, candidate, _checks = select_contract(
            wide, direction=CALL, as_of=date(2026, 2, 1), config=cfg()
        )

        assert best is None
        # Still reported, so the deck can say the name was close rather than silently absent.
        assert candidate is not None

    def test_open_interest_is_a_liquidity_gate_of_its_own(self) -> None:
        thin = [contract(bid=2.00, ask=2.10, delta=0.45, open_interest=5)]

        best, candidate, checks = select_contract(
            thin, direction=CALL, as_of=date(2026, 2, 1), config=cfg()
        )

        assert best is None
        assert candidate is not None          # still reported, so the deck can explain
        liquidity = next(c for c in checks if c.label == "Liquid enough to trade")
        assert liquidity.blocking and "open interest" in liquidity.value

    def test_thin_open_interest_is_rejected(self) -> None:
        thin = [contract(open_interest=5, delta=0.45)]
        best, _candidate, _checks = select_contract(thin, direction=CALL, as_of=date(2026, 2, 1), config=cfg())
        assert best is None

    def test_puts_use_the_put_band(self) -> None:
        puts = [contract(option_type=PUT, delta=-0.45, osi_symbol="P1")]
        best, _candidate, _checks = select_contract(puts, direction=PUT, as_of=date(2026, 2, 1), config=cfg())
        assert best is not None and best.osi_symbol == "P1"

# ── the state machine ────────────────────────────────────────────────────────


class TestLifecycle:
    """The state machine. The exit target steps DOWN with the days held, not up."""

    def test_the_exit_target_steps_down_as_the_days_run_out(self) -> None:
        """The reversal of the old ratchet, and the reason for it.

        A target that only ever rose asked more of a position the longer it failed to deliver,
        which is how a winner becomes a deadline exit at the bid. The schedule seeks 70% of the
        modelled gain on the day of entry and gives up a step each session, which is what makes
        the order executable rather than theoretical.
        """
        day0 = self.held(memory={"sessions_held": 0})
        day1 = self.held(memory={"sessions_held": 1})
        t0 = self._target(day0)
        t1 = self._target(day1)
        assert t1 < t0, "the ask must give ground as the deadline approaches"

    def _target(self, outcome) -> float:
        leg = next(
            o for o in outcome.orders
            if (o.request.children or [o.request])[0].order_type == "limit"
        )
        request = leg.request
        child = next((c for c in request.children if c.order_type == "limit"), request)
        return float(child.limit_price)

    def bidding(self, **overrides):
        kwargs = dict(
            memory={}, held_contract="", direction=CALL, contract=contract(), contracts=1,
            underlying_now=100.0, entry_target=98.0, exit_target=0.0, checks=[],
            config=cfg(), session=session(),
        )
        kwargs.update(overrides)
        return plan_symbol("QQQM", **kwargs)

    def test_no_direction_means_no_orders(self) -> None:
        outcome = self.bidding(direction="", contract=None, contracts=0)
        assert outcome.state == FLAT and outcome.orders == []

    def test_the_entry_is_a_pullback_limit_below_the_mark(self) -> None:
        """It waits for the dip the level model predicts; it does not buy at the market.

        Resting at the mid fills on almost every armed session and pays whatever is asked. On
        real option bars a static limit at the entry level filled 4 of 11 armed sessions, and the
        seven misses sat 5-6.5% below the mark and never traded there. That miss rate is the
        strategy working, not failing -- it is what ``entry_reach`` prices.
        """
        outcome = self.bidding()
        assert outcome.state == BIDDING
        request = outcome.orders[0].request
        assert request.asset_type == "option"
        assert request.order_type == "limit"
        assert request.action == "buy"
        # entry_target is 2% below spot; through a 0.45 delta the floor is
        # 2.05 - (100*0.02*0.45) = 1.15, and the ratchet lifts it toward the 2.05 mark as the
        # session runs down. It must sit between the two and never above the mark.
        assert 1.15 <= float(request.limit_price) < 2.05

    def test_the_bid_never_crosses_the_spread(self) -> None:
        """It converges toward the midpoint, not the ask.

        Paying the offer would guarantee a fill and in doing so discard the entire edge on
        exactly the days the prediction was wrong.
        """
        late = self.bidding(entry_target=99.9, session=session(fraction_remaining=0.02))
        assert late.orders[0].request.limit_price <= contract().midpoint

    def test_a_thin_contracts_jumpy_quote_does_not_move_the_bid_in_one_step(self) -> None:
        """ENTRY_MAX_REPRICE_PCT caps how far one run can move the resting bid, independent of
        entry_patience -- verified live on USO, where the ratchet's own given_up moved by 0.01
        between two runs while the resting bid jumped $2.20, which given_up cannot explain; the
        contract's own thinly-traded quote had jumped instead."""
        # Late in the session, given_up is close to 1 -- without a cap this would walk almost
        # all the way to the $2.05 mid in one step from the $1.00 previous bid.
        late = self.bidding(
            memory={"bid": 1.00}, entry_target=99.9, session=session(fraction_remaining=0.02),
        )
        max_step = 1.00 * ENTRY_MAX_REPRICE_PCT
        assert float(late.orders[0].request.limit_price) <= 1.00 + max_step + 1e-9

    def test_the_reprice_cap_never_overrides_the_never_above_mid_rule(self) -> None:
        """The step cap could in principle push the price up to its ceiling even when the mid
        sits below that -- re-asserting the mid cap after the clamp is what this guards."""
        cheap_contract = contract(bid=1.00, ask=1.02, mark=1.01)
        outcome = self.bidding(
            memory={"bid": 0.90}, contract=cheap_contract, entry_target=99.9,
            session=session(fraction_remaining=0.02),
        )
        assert float(outcome.orders[0].request.limit_price) <= cheap_contract.midpoint

    def test_never_bids_through_the_offer(self) -> None:
        # Target at the market, so the translated price is the mark -- still under the offer.
        outcome = self.bidding(entry_target=100.0)
        assert outcome.orders[0].request.limit_price <= contract().ask

    def test_the_order_key_names_the_role_not_the_submission(self) -> None:
        # A bid re-priced through the day must stay one order to the reconciler.
        assert self.bidding().orders[0].key == "QQQM:entry"
        assert self.bidding(entry_target=97.0).orders[0].key == "QQQM:entry"

    def held(self, **overrides):
        memory = {
            "state": HELD, "direction": CALL, "contracts": 1, "fill_price": 2.00,
            "mark": 2.40, "delta": 0.45, "sessions_held": 0,
            **overrides.pop("memory", {}),
        }
        kwargs = dict(
            memory=memory, held_contract="QQQM 260220C00100000", direction=CALL,
            contract=None, contracts=1, underlying_now=101.0, entry_target=0.0,
            # An absolute target LEVEL, ~2% above the 101.0 spot -- not a fraction. It was a
            # fraction and the caller passed a price, which put the target 400x spot.
            exit_target=103.0, checks=[], config=cfg(), session=session(),
        )
        kwargs.update(overrides)
        return plan_symbol("QQQM", **kwargs)

    def test_a_held_position_gets_two_independent_orders(self) -> None:
        """Target and stop rest as two separate orders, not one broker-side OCO/bracket -- the
        lifecycle already re-derives what should be resting from the broker's positions every
        run, which is the invariant an OCO would otherwise exist to hold."""
        outcome = self.held()
        assert outcome.state == HELD
        requests = {order.key.split(":")[-1]: order.request for order in outcome.orders}
        assert set(requests) == {"target", "stop"}
        for request in requests.values():
            assert request.strategy == "single" and not request.children
            assert request.action == "sell" and request.time_in_force == "gtc"
        assert requests["target"].order_type == "limit"
        assert requests["stop"].order_type == "stop"

    def _stop(self, outcome):
        return next(o.request for o in outcome.orders if o.key.endswith(":stop"))

    def test_the_stop_is_anchored_to_the_fill_not_the_mark(self) -> None:
        stop = self._stop(self.held())
        # The fill was 2.00 and the cap is stop_loss_pct below it -- of the *premium*, not of the
        # underlying, and measured from the fill rather than from wherever the mark now sits.
        assert stop.stop_price == pytest.approx(2.00 * (1 - cfg().stop_loss_pct), abs=0.01)

    def test_the_stop_does_not_move_when_the_mark_rises(self) -> None:
        low = self._stop(self.held(memory={"mark": 2.10}))
        high = self._stop(self.held(memory={"mark": 3.50}))
        assert low.stop_price == high.stop_price

    def test_the_target_ratchets_up(self) -> None:
        raised = self.held(memory={"target": 2.50}, underlying_now=110.0, exit_target=115.5)
        assert raised.orders[0].request.limit_price > 2.50

    def test_the_deadline_is_reported_as_blocking(self) -> None:
        # The deadline is the session after max_hold_sessions have elapsed.
        outcome = self.held(memory={"sessions_held": OptionsFlipConfig().max_hold_sessions})
        assert any(check.blocking for check in outcome.checks)

    def test_at_the_deadline_the_ask_converges_on_the_market(self) -> None:
        # The last fire of the final session: nothing of the walk-in is left, so the ask sits at
        # the mark rather than at a price the position is no longer allowed to wait for.
        #
        # ``max_hold_sessions``/``exit_patience`` are pinned here rather than left on the module
        # default -- this is a mechanism test (does the deadline branch converge to the mark at
        # all), and it should keep passing however the deployed tuning is set. It broke silently
        # the first time the default ``exit_patience`` was tuned past the value this scenario
        # was written against, which is exactly what pinning now prevents.
        outcome = self.held(
            memory={"sessions_held": 2, "mark": 2.40, "target": 0.0},
            session=session(fraction_remaining=0.0),
            config=cfg(max_hold_sessions=2, exit_patience=0.7),
        )
        # The exact number follows the decay step, which is derived from the hold length; what
        # this pins is that the deadline ask sits between the mark and the undecayed target.
        assert 2.05 < float(outcome.orders[0].request.limit_price) < 2.45

    def test_the_deadline_still_asks_for_a_profit_early_in_the_session(self) -> None:
        # Being on the last session does not mean giving up at 09:30 -- it converges over the day.
        outcome = self.held(
            memory={"sessions_held": 2, "mark": 2.40, "target": 0.0},
            session=session(fraction_remaining=1.0), underlying_now=110.0, exit_target=115.5,
        )
        assert outcome.orders[0].request.limit_price > 2.40

    def test_a_held_position_with_no_price_says_so_instead_of_crashing(self) -> None:
        # No recorded fill and no mark: there is nothing to size a bracket from, so the run
        # reports the gap and rests nothing rather than raising on a zero limit price.
        outcome = self.held(
            memory={"state": HELD, "direction": CALL, "contracts": 1,
                    "fill_price": 0.0, "mark": 0.0, "delta": 0.45, "sessions_held": 0},
        )
        assert outcome.state == HELD
        assert outcome.orders == []
        assert any(check.blocking for check in outcome.checks)

    def test_an_unfilled_bid_is_abandoned_the_next_morning(self) -> None:
        # Yesterday's budget described a session that has ended, so it is not re-priced.
        outcome = plan_symbol(
            "QQQM", memory={"state": BIDDING, "bid": 1.15, "market_day": "2026-02-09"},
            held_contract="", direction="", contract=None, contracts=0,
            underlying_now=100.0, entry_target=0.0, exit_target=0.0, checks=[],
            config=cfg(), session=session(day_changed=True),
        )
        assert outcome.state == FLAT and outcome.orders == []


class TestSessionOpen:
    """Today's open must come from today's bars, not from whatever the window starts with."""

    def intraday(self, days: list[str]) -> pd.DataFrame:
        rows, stamps = [], []
        for i, day in enumerate(days):
            rows.append({"open": 100.0 + i, "high": 101.0 + i, "low": 99.0 + i,
                         "close": 100.5 + i, "volume": 1000.0})
            stamps.append(pd.Timestamp(f"{day} 09:30", tz=MARKET_TZ))
        frame = pd.DataFrame(rows)
        frame["timestamp"] = stamps
        return frame

    def test_yesterdays_bars_are_not_todays_open(self) -> None:
        from src.algorithms.options_flip.algorithm import _session_open

        # A 390-minute lookback at 09:35 spans yesterday afternoon; iloc[0] is yesterday's open.
        frame = self.intraday(["2026-02-09", "2026-02-10"])

        assert _session_open(frame, "2026-02-10", fallback=0.0) == pytest.approx(101.0)

    def test_no_bars_today_falls_back_to_the_current_price(self) -> None:
        from src.algorithms.options_flip.algorithm import _session_open

        # Right at the bell the first print of the session *is* the open.
        frame = self.intraday(["2026-02-09"])

        assert _session_open(frame, "2026-02-10", fallback=123.0) == pytest.approx(123.0)

    def test_no_bars_at_all_falls_back(self) -> None:
        from src.algorithms.options_flip.algorithm import _session_open

        assert _session_open(None, "2026-02-10", fallback=55.0) == pytest.approx(55.0)


class TestStrikeSelection:
    """Delta chooses the strike; moneyness is only a backstop on how far it may drift."""

    def chain(self) -> list[OptionContract]:
        # Spot 86.79. Deltas fall as the strike rises, as they do on a real chain.
        return [
            contract(osi_symbol="C82", strike=82.0, delta=0.72),
            contract(osi_symbol="C86", strike=86.0, delta=0.52),
            contract(osi_symbol="C88", strike=88.0, delta=0.45),
            contract(osi_symbol="C95", strike=95.0, delta=0.18),
        ]

    def test_the_target_delta_picks_the_strike_not_the_distance_from_spot(self) -> None:
        # target_delta 0.45 picks the 88 strike even though 86 is nearer to spot.
        best, _c, _k = select_contract(
            self.chain(), direction=CALL, as_of=date(2026, 2, 1), config=cfg()
        )
        assert best is not None and best.strike == 88.0

    def test_the_chosen_contract_reports_what_decided_it(self) -> None:
        _b, _c, checks = select_contract(
            self.chain(), direction=CALL, as_of=date(2026, 2, 1), config=cfg()
        )
        chosen = next(c for c in checks if c.label == "Contract chosen")
        # Delta, and the liquidity the tie was broken on -- the three inputs to the ranking.
        assert "delta" in chosen.value and "vol" in chosen.value and "OI" in chosen.value

    def test_the_target_moves_the_strike(self) -> None:
        best, _c, _k = select_contract(
            self.chain(), direction=CALL, as_of=date(2026, 2, 1), config=cfg(target_delta=0.52),
        )
        assert best is not None and best.strike == 86.0

    def test_moneyness_signs_are_normalised_across_calls_and_puts(self) -> None:
        # Still on OptionContract for anyone who wants it; no longer shown on the deck.
        # Positive is out of the money on both sides, so one threshold reads the same way.
        assert contract(strike=88.0, option_type=CALL).moneyness(86.79) > 0
        assert contract(strike=85.0, option_type=CALL).moneyness(86.79) < 0
        assert contract(strike=85.0, option_type=PUT).moneyness(86.79) > 0
        assert contract(strike=88.0, option_type=PUT).moneyness(86.79) < 0




class TestRepriceTolerance:
    """Anti-churn has to scale with the spread, not just the price."""

    def tolerance(self, bid: float, ask: float, price: float) -> float:
        from src.algorithms.options_flip.lifecycle import _reprice_tolerance

        return _reprice_tolerance(contract(bid=bid, ask=ask), price, cfg())

    def test_a_tight_market_uses_the_price_floor(self) -> None:
        # SPY-like: 0.5% wide. Two percent of the price is the larger term, so nothing changes.
        assert self.tolerance(2.00, 2.01, 2.005) == pytest.approx(0.02)

    def test_a_wide_market_uses_the_spread(self) -> None:
        # 1.20/1.35 on a $1.27 mid: 2% is 2.5c, a sixth of the spread and inside the quote's
        # own noise. The spread term takes over so the order is not re-placed for it.
        assert self.tolerance(1.20, 1.35, 1.27) > 0.02
        assert self.tolerance(1.20, 1.35, 1.27) == pytest.approx((0.15 * 0.25) / 1.27)

    def test_an_unpriced_contract_falls_back(self) -> None:
        assert self.tolerance(1.20, 1.35, 0.0) == pytest.approx(0.02)


class TestExpiryFallthrough:
    """The nearest expiry that yields a tradable contract -- not the nearest expiry full stop."""

    def chain(self) -> list[OptionContract]:
        # Measured shape: IBIT's 2 Sep weekly carries 92 open interest in band, 4 Sep carries
        # 13,252. Committing to the nearer one and giving up throws the trade away for nothing.
        return [
            contract(osi_symbol="NEAR", expiry=date(2026, 9, 2), delta=0.45, open_interest=92),
            contract(osi_symbol="FAR", expiry=date(2026, 9, 4), delta=0.45, open_interest=13252),
        ]

    def test_it_falls_through_a_thin_expiry(self) -> None:
        best, _candidate, checks = select_contract(
            self.chain(), direction=CALL, as_of=date(2026, 8, 23), config=cfg()
        )

        assert best is not None and best.osi_symbol == "FAR"
        assert not any(check.blocking for check in checks)

    def test_the_nearest_liquid_expiry_still_wins(self) -> None:
        """Ascending, so the preference for less premium is intact."""
        both_liquid = [
            contract(osi_symbol="NEAR", expiry=date(2026, 9, 2), delta=0.45, open_interest=5000),
            contract(osi_symbol="FAR", expiry=date(2026, 9, 4), delta=0.45, open_interest=90000),
        ]

        best, _candidate, _checks = select_contract(
            both_liquid, direction=CALL, as_of=date(2026, 8, 23), config=cfg()
        )

        assert best is not None and best.osi_symbol == "NEAR"

    def test_every_expiry_thin_still_reports_the_best_it_saw(self) -> None:
        thin = [
            contract(osi_symbol="A", expiry=date(2026, 9, 2), delta=0.45, open_interest=10),
            contract(osi_symbol="B", expiry=date(2026, 9, 4), delta=0.45, open_interest=20),
        ]

        best, candidate, checks = select_contract(
            thin, direction=CALL, as_of=date(2026, 8, 23), config=cfg()
        )

        assert best is None
        assert candidate is not None                     # the deck can still explain the miss
        liquidity = next(c for c in checks if c.label == "Liquid enough to trade")
        assert liquidity.blocking and "across 2 expiries" in liquidity.value


class TestWorthTrading:
    """A predicted move of a few dollars is not a trade, however right the setup looks."""

    def estimate(self, low: float, high: float, config, contracts: int = 1) -> dict:
        from src.algorithms.options_flip.algorithm import _estimate

        # delta 1.0 makes the underlying move translate one-for-one, so the band is exactly
        # (high - low) and the arithmetic under test is the sizing, not the translation.
        c = contract(bid=low, ask=low + 0.02, mark=low + 0.01, delta=1.0)
        return _estimate(
            c, CALL, 100.0,
            {"predicted": 100.0 - (c.midpoint - low), "bid": 100.0},
            (high - c.midpoint) / 100.0, config, date(2026, 2, 1), contracts=contracts,
        )

    def test_expected_profit_is_the_band_in_dollars(self) -> None:
        est = self.estimate(2.00, 3.00, cfg())

        assert est["expected_profit"] == pytest.approx(est["edge"] * 100)
        # Gross: commissions are not modelled anywhere, so they are not netted here either.
        assert "commission" not in est and "net_profit" not in est

    def test_a_bigger_position_scales_the_total_but_not_the_per_contract_figure(self) -> None:
        one = self.estimate(2.00, 3.00, cfg(), contracts=1)
        three = self.estimate(2.00, 3.00, cfg(), contracts=3)

        assert three["expected_profit"] == pytest.approx(one["expected_profit"] * 3)
        # The per-contract figure is what the gate judges, and it does not move with size.
        assert three["expected_profit_per_contract"] == pytest.approx(
            one["expected_profit_per_contract"]
        )

    def test_buying_more_cannot_satisfy_the_floor(self) -> None:
        """A floor on the position total is satisfiable by scaling up a marginal trade.

        Ten contracts predicted to move two cents each is $25 in total and clears a $25 position
        floor -- while being ten times the risk for an edge that was never there. Judged per
        contract, size cannot buy its way past the quality bar. This matters more now that the
        notional cap alone sets the count: a cheap contract buys a lot of them.
        """
        thin_one = self.estimate(2.00, 2.02, cfg(), contracts=1)
        thin_ten = self.estimate(2.00, 2.02, cfg(), contracts=10)

        assert thin_ten["expected_profit"] >= 20         # the total looks respectable
        assert thin_ten["expected_profit_per_contract"] == pytest.approx(
            thin_one["expected_profit_per_contract"]
        )
        assert thin_ten["expected_profit_per_contract"] < cfg().min_profit_per_contract

    def test_a_thin_move_is_a_small_number(self) -> None:
        # $0.05 predicted on one contract is $5 -- under any sane floor.
        assert self.estimate(2.00, 2.05, cfg())["expected_profit_per_contract"] < 10

    def test_the_floor_is_a_different_axis_from_the_direction_gate(self) -> None:
        """``trend_min_return`` gates the underlying's momentum, this gates the contract's move.

        A symbol can trend hard enough to clear the first while its contract stands to move a
        few dollars -- the trade that looks right on every other gate and still loses.
        """
        thin = self.estimate(2.00, 2.05, cfg())

        assert thin["expected_profit_per_contract"] < cfg().min_profit_per_contract



def _intraday(sessions: dict) -> pd.DataFrame:
    """Five-minute bars for several sessions: ``{date: [(minute, open, close), ...]}``."""
    rows = []
    for day, bars in sessions.items():
        for minute, open_, close in bars:
            rows.append({
                "ts": pd.Timestamp(f"{day} {minute // 60:02d}:{minute % 60:02d}", tz=MARKET_TZ),
                "open": open_, "close": close, "day": pd.Timestamp(day).date(), "minute": minute,
            })
    return pd.DataFrame(rows)



class TestIndicators:
    """The measurements the regime gate and the level model are stated in."""

    def _bars(self, rows):
        return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"])

    def test_true_range_counts_the_gap(self) -> None:
        """A symbol that opens 2% below yesterday and trades quietly has moved 2.5%, not 0.5%."""
        gapped = self._bars([
            [100.0, 100.0, 100.0, 100.0, 1],
            [98.0, 98.5, 98.0, 98.5, 1],       # 0.5 of range, but 2.0 of gap
        ])
        assert average_true_range(gapped, window=1) == pytest.approx(2.0, abs=0.01)

    def test_vwap_weights_by_volume(self) -> None:
        frame = pd.DataFrame({
            "high": [10.0, 20.0], "low": [10.0, 20.0], "close": [10.0, 20.0],
            "volume": [9.0, 1.0], "minute": [570, 575],
        })
        assert session_vwap(frame) == pytest.approx(11.0)

    def test_vwap_without_volume_falls_back_to_the_close(self) -> None:
        """Zero would read as 'price is above VWAP' to every comparison downstream."""
        frame = pd.DataFrame({"high": [10.0], "low": [10.0], "close": [10.0],
                              "volume": [0.0], "minute": [570]})
        assert session_vwap(frame) == pytest.approx(10.0)

    def test_a_short_history_reports_no_average_rather_than_a_short_one(self) -> None:
        """A name below 'the 50-day average' computed from 30 bars is a data gap, not a fact."""
        assert moving_average(pd.Series([1.0] * 30), 50) == 0.0

    def test_directional_volume_splits_by_each_bars_own_open_to_close(self) -> None:
        """No tick data exists, so a bar that closed above its own open votes 'buy'."""
        frame = pd.DataFrame({
            "open": [10.0, 10.0, 10.0], "close": [10.5, 9.5, 10.0],
            "volume": [7.0, 3.0, 5.0],
        })
        split = directional_volume(frame)
        assert split["buy_volume"] == pytest.approx(7.0)
        assert split["sell_volume"] == pytest.approx(3.0)
        # A flat bar's volume counts toward neither, so the denominator is buy + sell only.
        assert split["imbalance"] == pytest.approx(0.4)

    def test_directional_volume_with_no_volume_column_is_neutral(self) -> None:
        empty = pd.DataFrame({"open": [], "close": []})
        assert directional_volume(empty) == {"buy_volume": 0.0, "sell_volume": 0.0, "imbalance": 0.0}
        assert directional_volume(None) == {"buy_volume": 0.0, "sell_volume": 0.0, "imbalance": 0.0}


class TestScenarioPricing:
    """Delta alone prices a small move and nothing else."""

    def _contract(self, **kw):
        return contract(delta=0.60, gamma=0.15, theta=-0.06, vega=0.02, **kw)

    def test_gamma_bends_the_move(self) -> None:
        """Pricing a whole move at the starting delta understates a winner."""
        c = self._contract()
        linear = 0.60 * 2.0
        modelled = option_change(c, Scenario("base", 2.0, 0.0, 0.0))
        assert modelled > linear
        assert modelled == pytest.approx(0.60 * 2.0 + 0.5 * 0.15 * 4.0)

    def test_theta_is_charged_for_the_hold(self) -> None:
        """4% of premium a day on a near-dated contract is not a rounding error."""
        c = self._contract()
        flat = option_change(c, Scenario("bad", 0.0, 0.0, 2.0))
        assert flat == pytest.approx(-0.12)

    def test_vega_can_lose_money_on_a_correct_direction(self) -> None:
        """The failure a delta-only model cannot represent."""
        c = self._contract()
        crushed = option_change(c, Scenario("base", 0.10, -5.0, 1.0))
        assert crushed < 0

    def test_max_debit_is_zero_when_the_base_case_cannot_pay(self) -> None:
        c = self._contract(bid=2.00, ask=2.10)
        outcomes = {"base": {"change": 0.01}}
        assert max_debit(c, outcomes, config=cfg(min_profit_per_contract=25.0)) == 0.0


class TestStopDisabled:
    """A long call cannot lose more than its premium, so the unit is the loss cap."""

    def _held(self, **over):
        memory = {"state": HELD, "direction": CALL, "contracts": 1, "fill_price": 2.00,
                  "bid": 2.00, "mark": 2.20, "delta": 0.45, "sessions_held": 0}
        memory.update(over.pop("memory", {}))
        return plan_symbol(
            "QQQM", memory=memory, held_contract="QQQM  260220C00100000",
            direction=CALL, contract=None, contracts=1, underlying_now=100.0,
            entry_target=0.0, exit_target=102.0, checks=[],
            config=cfg(**over), session={"fraction_remaining": 0.5, "market_day": "2026-02-02"},
        )

    def test_no_stop_order_ever_reaches_the_exchange(self) -> None:
        """Zero means no stop leg is constructed at all."""
        outcome = self._held(stop_loss_pct=0.0)
        legs = [order.request for order in outcome.orders]
        assert all(leg.order_type == "limit" for leg in legs), legs
        assert all(not leg.stop_price for leg in legs), legs
        assert [o.key for o in outcome.orders] == ["QQQM:target"]

    def test_the_stop_key_appears_and_disappears_with_the_setting(self) -> None:
        """Turning the stop off drops the ``:stop`` key entirely, which the reconciler reads as
        'no longer wanted' -- it cancels the resting stop rather than editing it in place. The
        target's own key never changes either way, since there is no longer a second shape."""
        assert [o.key for o in self._held(stop_loss_pct=0.0).orders] == ["QQQM:target"]
        assert {o.key for o in self._held(stop_loss_pct=0.10).orders} == {"QQQM:target", "QQQM:stop"}

    def test_no_stop_rests_the_target_alone(self) -> None:
        outcome = self._held(stop_loss_pct=0.0)
        assert len(outcome.orders) == 1
        request = outcome.orders[0].request
        assert request.strategy == "single"
        assert request.action == "sell" and request.order_type == "limit"
        assert not request.children

    def test_a_live_stop_rests_as_two_independent_orders(self) -> None:
        outcome = self._held(stop_loss_pct=0.10)
        requests = {order.key.split(":")[-1]: order.request for order in outcome.orders}
        assert set(requests) == {"target", "stop"}
        assert requests["target"].order_type == "limit"
        assert requests["stop"].order_type == "stop"
        assert requests["target"].strategy == "single" and requests["stop"].strategy == "single"

    def test_the_deck_says_the_premium_is_the_cap(self) -> None:
        outcome = self._held(stop_loss_pct=0.0)
        stop_check = next(c for c in outcome.checks if c.label == "Protective stop")
        assert "loss cap" in stop_check.value
        assert "deadline" in (stop_check.limit or "")


class TestExitTargetIsALevel:
    """``exit_target`` is an absolute underlying price, not a fraction of spot."""

    def test_the_target_is_read_as_a_price(self) -> None:
        """It was a fraction and the caller passed a price.

        ``underlying_now * (1 + 403.75)`` put the profit target four hundred times spot, so it
        could never fill and every position ran to its deadline instead. The backtest never
        caught it because the backtest does not go through this module -- which is the argument
        for the two testing the same thing.
        """
        outcome = plan_symbol(
            "QQQM",
            memory={"state": HELD, "direction": CALL, "contracts": 1, "fill_price": 2.00,
                    "bid": 2.00, "mark": 2.20, "delta": 0.50, "sessions_held": 0},
            held_contract="QQQM  260220C00100000", direction=CALL, contract=None, contracts=1,
            underlying_now=100.0,
            entry_target=0.0,
            exit_target=104.0,          # a level: 4 dollars above spot
            checks=[], config=cfg(stop_loss_pct=0.0),
            session={"fraction_remaining": 0.5, "market_day": "2026-02-02"},
        )
        ask = float(outcome.orders[0].request.limit_price)
        # 4 dollars of underlying at a 0.50 delta is 2.00 of premium on top of the 2.20 mark;
        # the schedule then asks for a fraction of that gain rather than all of it.
        assert 2.00 < ask < 4.20, f"target {ask} is not a plausible premium"

    def test_a_zero_target_asks_for_nothing_impossible(self) -> None:
        outcome = plan_symbol(
            "QQQM",
            memory={"state": HELD, "direction": CALL, "contracts": 1, "fill_price": 2.00,
                    "bid": 2.00, "mark": 2.20, "delta": 0.50, "sessions_held": 0},
            held_contract="QQQM  260220C00100000", direction=CALL, contract=None, contracts=1,
            underlying_now=100.0, entry_target=0.0, exit_target=0.0, checks=[],
            config=cfg(stop_loss_pct=0.0),
            session={"fraction_remaining": 0.5, "market_day": "2026-02-02"},
        )
        for order in outcome.orders:
            assert float(order.request.limit_price or 0) < 100.0


class TestEveryConfiguredSymbolGetsARow:
    """The trend gate decides what may trade; it does not decide what is shown."""

    def test_the_trend_gate_blocks_the_order_not_the_analysis(self) -> None:
        """The three silences a filtered list renders identically.

        "not trending", "trending but the regime failed" and "everything passed but the chain
        had nothing tradable" are very different, and a view that showed only the qualifying
        symbols collapsed all three into absence -- which reads as a broken deck.
        """
        import inspect
        from src.algorithms.options_flip.algorithm import OptionsFlipAlgorithm

        source = inspect.getsource(OptionsFlipAlgorithm._plan_one)
        # the gate zeroes the size; it does not short-circuit the analysis below it
        assert "trending" in source and "contracts = 0" in source

    def test_no_symbol_limit_remains(self) -> None:
        """Every symbol clearing its own threshold trades. There is no top-N."""
        from dataclasses import fields

        assert "max_candidates" not in {f.name for f in fields(OptionsFlipConfig())}


class TestOptionBandFeedsTheEntry:
    """The band computed for the deck must actually reach the resting bid, not just the sell side.

    Measured on real Sep-18 option history (``tools/options_flip_contract_backtest.py --band``),
    the option band beat the static delta translation by +5.8pp mean return on the sell side --
    the reason to wire it into the entry too. A prior version computed ``band`` purely as a deck
    diagnostic and never passed it into ``plan_symbol``, so the entry silently stayed on the
    cruder translation regardless of how good the contract's own history was.
    """

    def test_plan_one_passes_the_bands_entry_as_entry_premium(self) -> None:
        import inspect
        from src.algorithms.options_flip.algorithm import OptionsFlipAlgorithm

        source = inspect.getsource(OptionsFlipAlgorithm._plan_one)
        assert "entry_premium=" in source


class TestAsymmetricPatience:
    """The two sides face different risks, so they concede on different curves."""

    def test_the_entry_holds_its_price_early_and_concedes_late(self) -> None:
        cfg_ = cfg(entry_patience=1.5)
        early = (1.0 - 0.75) ** cfg_.entry_patience     # 75% of the session left
        late = (1.0 - 0.25) ** cfg_.entry_patience      # 25% left
        assert early < 0.2, "an entry that chases early is a momentum trade wearing a pullback label"
        assert late > 0.5

    def test_the_exit_concedes_early_rather_than_at_gunpoint(self) -> None:
        """With the stop disabled the deadline is the only thing that ends a losing trade, so a
        position reaching it unsold is sold at whatever the market offers."""
        cfg_ = cfg(exit_patience=0.7, exit_gain_share=0.70, max_hold_sessions=5)
        asked = []
        for held in range(cfg_.max_hold_sessions + 1):
            elapsed = min(held / cfg_.max_hold_sessions, 1.0)
            asked.append(cfg_.exit_gain_share * (1.0 - elapsed ** cfg_.exit_patience))
        assert asked[0] == pytest.approx(0.70)
        assert asked[-1] == pytest.approx(0.0, abs=1e-9)
        # a patient (linear) schedule would still be asking 42% at the midpoint
        assert asked[2] < 0.70 * (1 - 2 / 5)
        assert all(a >= b for a, b in zip(asked, asked[1:])), "it must only ever give ground"

    def test_the_two_sides_are_configured_independently(self) -> None:
        # Was ``entry_patience > 1.0 > exit_patience`` -- exit deliberately impatient, entry
        # deliberately patient. A combo sweep over August (see
        # ``tools/options_flip_config_combo_sweep.py``) found holding the exit firmer, not
        # conceding it faster, was the single most reliable lever measured ($230 vs $89 solo,
        # $484 across a real 17-trade sample when combined with a longer hold): both sides are
        # patient now, by tuning rather than by original design. What this test still pins is
        # only that they are set apart, not which one is larger -- if a future month's data
        # argues the old asymmetry back, that assumption is the one to revisit.
        c = OptionsFlipConfig()
        assert c.entry_patience != c.exit_patience


class TestRunHorizon:
    """The target is priced over the whole hold; the entry only has today."""

    def test_the_run_window_grows_with_the_hold_and_the_dip_does_not(self) -> None:
        """Charging five sessions of theta against a one-session target is what made the
        modelled profit negative on setups whose direction was right."""
        rows = []
        for day in (2, 3, 4, 5, 6, 9, 10, 11):
            for minute, close in ((570, 100.0), (600, 100.0), (900, 100.0 + day)):
                rows.append({
                    "ts": pd.Timestamp(f"2026-02-{day:02d} "
                                       f"{minute // 60:02d}:{minute % 60:02d}", tz=MARKET_TZ),
                    "open": 100.0, "high": close, "low": min(close, 100.0), "close": close,
                    "day": pd.Timestamp(f"2026-02-{day:02d}").date(), "minute": minute,
                })
        bars = pd.DataFrame(rows)
        one = excursion_samples(bars, minute=600, atr=1.0, run_horizon=1)
        three = excursion_samples(bars, minute=600, atr=1.0, run_horizon=3)
        assert three["run"].median() > one["run"].median()
        # the dip is a single-session measure either way
        assert three["dip"].median() == pytest.approx(one["dip"].median())


class TestAbsoluteTrendScore:
    """The score is a symbol's own move in its own sigma. No universe is consulted."""

    def _rising(self, drift: float, noise: float, n: int = 60):
        import numpy as np
        rng = np.random.default_rng(0)
        closes = [100.0]
        for i in range(n):
            closes.append(closes[-1] * (1 + drift + noise * rng.standard_normal()))
        return pd.DataFrame({"close": closes})

    def test_a_score_is_computed_from_one_symbol_s_bars_alone(self) -> None:
        """The defect this replaced: a z-score's mean is zero by construction, so the best name
        always scored positive and the rest always negative, whatever the market did. A symbol up
        13% over twenty days was excluded because another was up 23%.

        There is now no batch call to get this wrong -- the function takes one symbol's bars.
        """
        from src.algorithms.options_flip.candidates import scoring_parameters, trend_strength

        params = scoring_parameters()
        strong = self._rising(0.004, 0.004)
        assert trend_strength(strong, params) == pytest.approx(trend_strength(strong, params))
        # A weaker name existing changes nothing, because it is never passed in.
        assert trend_strength(strong, params) > trend_strength(self._rising(0.001, 0.004), params)

    def test_both_names_can_be_positive_at_once(self) -> None:
        """A cross-sectional score cannot express "everything is rallying"; this can."""
        from src.algorithms.options_flip.candidates import scoring_parameters, trend_strength

        params = scoring_parameters()
        scores = [trend_strength(self._rising(d, 0.003), params) for d in (0.004, 0.003)]
        assert all(v > 0 for v in scores), scores

    def test_and_both_can_be_negative(self) -> None:
        """Nothing trending means nothing to trade -- which a ranking could never say."""
        from src.algorithms.options_flip.candidates import scoring_parameters, trend_strength

        params = scoring_parameters()
        scores = [trend_strength(self._rising(d, 0.003), params) for d in (-0.004, -0.003)]
        assert all(v < 0 for v in scores), scores

    def test_the_threshold_is_in_sigma_not_rank(self) -> None:
        """Volatility-scaled, so one number works on a quiet ETF and a violent one alike."""
        from src.algorithms.options_flip.candidates import scoring_parameters, trend_strength

        params = scoring_parameters()
        quiet = trend_strength(self._rising(0.002, 0.001), params)
        wild = trend_strength(self._rising(0.002, 0.010), params)
        assert quiet > wild, "the same drift is a larger move for the quieter name"


class TestStopDisabledRecordsNoStop:
    """Zero means disabled, and nothing is left behind that could act like a stop later."""

    def _bid(self, pct: float):
        from datetime import date as _date
        from src.core.options import OptionContract

        contract_ = OptionContract(
            osi_symbol="Q  260220C00100000", underlying="Q", option_type="call",
            strike=100.0, expiry=_date(2026, 2, 20), bid=2.0, ask=2.1, mark=2.05, delta=0.45,
        )
        return plan_symbol(
            "Q", memory={}, held_contract="", direction=CALL, contract=contract_, contracts=1,
            underlying_now=100.0, entry_target=98.0, exit_target=0.0, checks=[],
            config=cfg(stop_loss_pct=pct),
            session={"fraction_remaining": 1.0, "market_day": "2026-02-02"},
        )

    def test_a_disabled_stop_records_zero_not_the_entry_price(self) -> None:
        """It recorded ``limit x 1.0`` -- the entry price itself.

        The held path re-checks the setting, so nothing acted on it. But a position carrying a
        recorded stop equal to its own fill would be closed the instant the stop was enabled.
        """
        memory = self._bid(0.0).memory
        assert memory["stop"] == 0.0
        assert memory["bid"] > 0.0, "the entry price is still recorded"

    def test_a_live_stop_records_a_real_level(self) -> None:
        memory = self._bid(0.10).memory
        assert memory["stop"] == pytest.approx(memory["bid"] * 0.90, abs=0.01)


def test_the_tune_page_order_matches_the_config_dataclass() -> None:
    """The dashboard renders in the explainer's order, so a drift here reshuffles the form.

    ``plan`` is documented first and is deliberately not a dataclass field: the board is a
    nested structure living in the config section, exactly as Bursty DCA's is, so it cannot
    ride on a flat dataclass of scalars. It is the headline control and reads first.
    """
    from dataclasses import fields as _fields
    from src.algorithms.explainers import EXPLAINERS

    documented = list(EXPLAINERS["options_flip"]["parameters"])
    assert documented[0] == "plan"
    declared = [f.name for f in _fields(OptionsFlipConfig())]
    assert documented[1:] == declared


def _option_bars(sessions: dict) -> pd.DataFrame:
    """Raw per-session 5m option OHLC bars, keyed on ``timestamp`` like the Schwab fetch.

    ``sessions`` maps an ISO date to ``[(minute, open, close), ...]``; high/low are derived so a
    session that dips then runs reads the way real option bars do.
    """
    rows = []
    for day, bars in sessions.items():
        for minute, open_, close in bars:
            low, high = min(open_, close), max(open_, close)
            rows.append({
                "timestamp": pd.Timestamp(f"{day} {minute // 60:02d}:{minute % 60:02d}",
                                          tz=MARKET_TZ).tz_convert("UTC"),
                "open": open_, "high": high, "low": low, "close": close, "volume": 10,
            })
    return pd.DataFrame(rows)


def _dip_then_run_sessions(n: int, dip: float = 0.04, run: float = 0.03, base: float = 10.0):
    """``n`` sessions that each dip below the open and then rally above it, in one session."""
    sessions = {}
    for i in range(n):
        day = pd.Timestamp("2026-02-01") + pd.Timedelta(days=i)
        open_ = base
        low = open_ * (1 - dip)
        high = open_ * (1 + run)
        sessions[str(day.date())] = [
            (585, open_, high), (600, high, low), (615, low, open_ * (1 + run)),
            (660, open_ * (1 + run), open_ * (1 + run)),
        ]
    return sessions


class TestOptionBand:
    """Predicting the chosen contract's low and run from its own premium history."""

    def test_prepare_option_bars_adds_market_time_columns(self) -> None:
        frame = _option_bars(_dip_then_run_sessions(3))
        bars = prepare_option_bars(frame)
        assert {"ts", "minute", "day", "open", "high", "low", "close"} <= set(bars.columns)
        assert bars["minute"].iloc[0] == 585
        assert bars["ts"].dt.tz is not None

    def test_daily_atr_is_computed_in_premium_from_the_contracts_own_bars(self) -> None:
        frame = _option_bars(_dip_then_run_sessions(20, dip=0.04, run=0.03, base=10.0))
        bars = prepare_option_bars(frame)
        atr = option_daily_atr(bars, window=14)
        assert atr > 0.0

    def test_a_sane_option_band_is_preferred_over_the_translation(self) -> None:
        frame = _option_bars(_dip_then_run_sessions(30, dip=0.06, run=0.08, base=10.0))
        bars = prepare_option_bars(frame)
        mark = 10.0
        from src.algorithms.options_flip.option_band import _sane
        band = choose_band(
            bars, minute=600, option_mark=mark, session_open=mark, config=cfg(),
            max_hold=1,
            underlying_translation={"entry": 9.5, "target": 11.0}, contract=contract(),
            underlying_now=100.0,
        )
        assert band["source"] == "option"
        assert band["entry"] <= mark < band["target"]
        assert _sane(band, mark)

    def test_absurd_target_then_falls_back_to_the_translation(self) -> None:
        # A target six times the premium is a thin-sample tail, not a forecast: it must not be
        # rested as an order when a sane underlying translation exists.
        band = choose_band(
            None, minute=600, option_mark=10.0, session_open=10.0, config=cfg(),
            max_hold=1,
            underlying_translation={"entry": 9.5, "target": 11.0}, contract=contract(),
            underlying_now=100.0,
        )
        assert band["source"] == "underlying"
        assert band["target"] == pytest.approx(11.0)
        assert band["entry"] == pytest.approx(9.5)

    def test_no_history_and_no_translation_returns_an_empty_band(self) -> None:
        band = choose_band(
            None, minute=600, option_mark=10.0, session_open=10.0, config=cfg(),
            max_hold=1, underlying_translation=None, contract=None, underlying_now=100.0,
        )
        assert band["entry"] == 0.0 and band["source"] == "none"


class TestSellBand:
    """The sales side re-predicts each run and sells at the mark when the band gives nothing."""

    def held(self, **overrides):
        memory = {
            "state": HELD, "direction": CALL, "contracts": 1, "fill_price": 2.00,
            "mark": 2.40, "delta": 0.45, "sessions_held": 0,
            **overrides.pop("memory", {}),
        }
        kwargs = dict(
            memory=memory, held_contract="QQQM 260220C00100000", direction=CALL,
            contract=None, contracts=1, underlying_now=101.0, entry_target=0.0,
            exit_target=103.0, checks=[], config=cfg(), session=session(),
        )
        kwargs.update(overrides)
        return plan_symbol("QQQM", **kwargs)

    def _target(self, outcome) -> float:
        leg = next(o for o in outcome.orders if (o.request.children or [o.request])[0].order_type == "limit")
        request = leg.request
        child = next((c for c in request.children if c.order_type == "limit"), request)
        return float(child.limit_price)

    def test_a_sane_sell_band_ratchets_toward_it(self) -> None:
        # A target premium comfortably above the mark is asked at the ratchet, not given away.
        outcome = self.held(memory={"target": 2.50}, target_premium=3.00)
        assert outcome.state == HELD
        assert 2.40 < self._target(outcome) <= 3.00

    def test_a_single_closed_read_does_not_collapse_the_target(self) -> None:
        """A gate reading closed for one run only takes one concession step -- the streak-driven
        decay (see ``SELL_GATE_CONCESSION_RATE``) only reaches the mark after several consecutive
        closed reads, so a brief flicker never prices the exit as if the gate had been closed all
        along."""
        outcome = self.held(memory={"target": 3.00}, sell_ok=False)
        assert 2.40 < self._target(outcome) < 3.00
        assert outcome.memory["gate_failed_streak"] == 1

    def test_the_bull_gate_closed_converges_gradually_once_confirmed(self) -> None:
        """Converges toward the mark as the closed-read streak grows -- the same "converge
        across what's left of the clock" mechanism the deadline uses, driven by consecutive
        closed reads instead of the session's fraction remaining."""
        memory = {"target": 3.00}
        outcome = self.held(memory=memory, sell_ok=False)
        # One run in: a step taken, nowhere near the mark yet.
        assert 2.40 < self._target(outcome) < 3.00
        memory = dict(outcome.memory)
        # Each step closes a fixed *fraction* of the remaining gap (an exponential approach,
        # not a linear one), so it only ever gets arbitrarily close, never exactly there --
        # 40 more runs is comfortably enough to call that "converged" for this assertion.
        for _ in range(40):
            outcome = self.held(memory=memory, sell_ok=False)
            memory = dict(outcome.memory)
        assert self._target(outcome) == pytest.approx(2.40, abs=0.05)
        assert outcome.memory["gate_failed_streak"] == 41

    def test_a_reopened_gate_resumes_the_normal_ratchet_immediately(self) -> None:
        """Once sell_ok is true again the streak (and the concession) resets to zero and the
        ratchet resumes right away -- asking for more is never the risky direction, only giving
        ground is, so there is nothing to debounce on the way back up."""
        memory = {"target": 3.00}
        outcome = self.held(memory=memory, sell_ok=False)
        memory = dict(outcome.memory)
        conceded = self._target(outcome)
        assert 2.40 < conceded < 3.00
        outcome = self.held(memory=memory, target_premium=3.00, sell_ok=True)
        assert self._target(outcome) > 2.40
        assert outcome.memory["gate_failed_streak"] == 0

    def test_a_target_at_or_below_the_mark_also_reads_at_the_mark(self) -> None:
        outcome = self.held(memory={"target": 3.00}, target_premium=2.20)
        assert self._target(outcome) == pytest.approx(2.40, abs=0.02)

    def test_no_sell_band_keeps_the_underlying_translation(self) -> None:
        # Reverted to the existing behaviour: translate the underlying exit target.
        outcome = self.held(memory={"target": 2.50}, target_premium=None, sell_ok=True)
        target = self._target(outcome)
        assert target > 2.40  # a sane translation still reaches for a profit


def test_a_held_position_prices_against_what_it_actually_cost() -> None:
    """A limit buy fills at or below its price, so the order we placed is an upper bound on the
    cost and never the cost itself.

    The anchor used to be the bid, and ``fill_price`` was set to whatever the mark happened to
    be on the first poll after the fill -- so the deck reported an unrealised P&L against a
    price the account never paid, and the stop was struck off it too.
    """
    from src.algorithms.options_flip.algorithm import _refresh_held
    from src.algorithms.options_flip.config import OptionsFlipConfig

    osi = "GLD   260918C00400000"
    bidding_memory = {
        "state": "bidding", "contract": osi, "direction": "call",
        "contracts": 1, "bid": 15.00, "stop": 10.0, "market_day": "2026-09-10",
    }

    class Context:
        latest_prices = {osi: 15.40}   # the mark has moved since the fill
        cost_basis = {osi: 14.80}      # what the broker says we actually paid

    memory = _refresh_held(
        dict(bidding_memory), osi, Context(), {"market_day": "2026-09-10"}, OptionsFlipConfig()
    )

    assert memory["fill_price"] == 14.80


def test_a_held_position_falls_back_when_the_broker_reports_no_cost() -> None:
    """A brokerage that cannot report a cost basis must still be tradable -- the position is
    anchored to the mark, and that is worth a warning rather than silence."""
    from src.algorithms.options_flip.algorithm import _refresh_held
    from src.algorithms.options_flip.config import OptionsFlipConfig

    osi = "GLD   260918C00400000"

    class Context:
        latest_prices = {osi: 15.40}
        cost_basis: dict[str, float] = {}

    memory = _refresh_held(
        {"contract": osi, "bid": 15.00}, osi, Context(),
        {"market_day": "2026-09-10"}, OptionsFlipConfig(),
    )

    assert memory["fill_price"] == 15.40


def test_the_cost_basis_is_refreshed_rather_than_frozen() -> None:
    """A partial fill or a second lot moves the average, so it is re-read every run instead of
    being recorded once and kept."""
    from src.algorithms.options_flip.algorithm import _refresh_held
    from src.algorithms.options_flip.config import OptionsFlipConfig

    osi = "GLD   260918C00400000"

    class Context:
        latest_prices = {osi: 15.40}
        cost_basis = {osi: 15.10}      # averaged up by a second lot

    memory = _refresh_held(
        {"contract": osi, "fill_price": 14.80}, osi, Context(),
        {"market_day": "2026-09-10"}, OptionsFlipConfig(),
    )

    assert memory["fill_price"] == 15.10


# --------------------------------------------------------------------------------------
# Per-symbol budgets. One global contract count meant a six-fold difference in money at
# risk between a $17 premium and a $2.50 one -- a position-sizing decision nobody made.
# --------------------------------------------------------------------------------------


class _Quote:
    def __init__(self, ask: float) -> None:
        self.ask = ask
        self.midpoint = ask


def test_the_board_sizes_a_position_in_dollars() -> None:
    """The budget is the loss cap -- a long option cannot lose more than its premium -- so the
    same dollar figure means the same risk on a $17 premium and a $2.50 one."""
    from src.algorithms.options_flip.config import OptionsFlipConfig
    from src.algorithms.options_flip.contracts import affordable_contracts

    config = OptionsFlipConfig()

    assert affordable_contracts(_Quote(17.50), config, budget=3_500.0) == 2    # $3,500
    assert affordable_contracts(_Quote(2.50), config, budget=3_500.0) == 14    # $3,500


def test_a_budget_below_one_contract_opens_nothing() -> None:
    """Rounded down, never up: a budget that cannot cover one contract is not a position."""
    from src.algorithms.options_flip.config import OptionsFlipConfig
    from src.algorithms.options_flip.contracts import affordable_contracts

    assert affordable_contracts(_Quote(17.50), OptionsFlipConfig(), budget=1_000.0) == 0


def test_a_symbol_absent_from_the_board_opens_nothing() -> None:
    """No bubble, no position. The board is the whole statement of what may be traded, so
    there is no global unit left for an unfunded symbol to fall back to."""
    from src.algorithms.options_flip.config import OptionsFlipConfig, sanitize_plan, symbol_budget
    from src.algorithms.options_flip.contracts import affordable_contracts

    plan = sanitize_plan({"call": {"items": [{"symbol": "GLD", "amount": 3_500}]}}, {"GLD", "USO"})

    assert symbol_budget(plan, "GLD", "call") == 3_500.0
    assert symbol_budget(plan, "USO", "call") == 0.0
    assert affordable_contracts(_Quote(8.00), OptionsFlipConfig(), budget=0.0) == 0


def test_the_board_declares_call_and_put_buckets() -> None:
    """The put bucket exists so adding puts is config rather than a new code path -- nothing
    reads it until the level model and the regime gate are mirrored."""
    from src.algorithms.options_flip.algorithm import OptionsFlipAlgorithm
    from src.algorithms.options_flip.config import sanitize_plan

    assert OptionsFlipAlgorithm.tune_buckets == ("call", "put")
    assert set(sanitize_plan({}, {"GLD"})) == {"call", "put"}


def test_editing_a_budget_invalidates_the_cached_signal_view() -> None:
    """The board sizes every position, so a snapshot computed under the old amounts must not be
    served for the new ones."""
    from dataclasses import replace

    from src.algorithms.options_flip.algorithm import OptionsFlipAlgorithm
    from src.core.config import get_config

    algorithm = OptionsFlipAlgorithm({})
    base = get_config()
    symbol = sorted(base.symbols)[0]

    def with_budget(amount: float):
        return replace(base, algorithm_configs={
            **(base.algorithm_configs or {}),
            "options_flip": {"plan": {"call": {"items": [{"symbol": symbol, "amount": amount}]}}},
        })

    assert algorithm.config_fingerprint(with_budget(3_500)) != algorithm.config_fingerprint(with_budget(1_000))
