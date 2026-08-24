"""Options Flip: the excursion maths, contract choice, direction gating and the state machine.

The bar fixtures are built rather than recorded, so each one states the fact it is testing --
"thirty sessions that each dipped 2% and closed up" is a distribution with a known quantile, and
a test against it fails for one reason.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.algorithms.options_flip.config import OptionsFlipConfig
from src.algorithms.options_flip.contracts import affordable_contracts, select_contract
from src.algorithms.options_flip.direction import (
    premarket_confirms,
    trend_direction,
    typical_daily_move,
)
from src.algorithms.options_flip.excursion import (
    entry_underlying_target,
    excursions,
    expected_excursion,
    observed_excursion,
    option_price_for,
    predicted_extreme,
    session_fraction_remaining,
    target_price,
)
from src.algorithms.options_flip.lifecycle import BIDDING, FLAT, HELD, plan_symbol
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
    return OptionsFlipConfig(symbols=["QQQM"], **kwargs)


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


class TestExcursions:
    def test_call_measures_the_dip_below_the_open(self) -> None:
        series = excursions(bars([(100.0, 103.0, 98.0, 102.0)]), direction=CALL, lookback=10)
        assert series.iloc[0] == pytest.approx(0.02)

    def test_put_measures_the_rise_above_the_open(self) -> None:
        series = excursions(bars([(100.0, 103.0, 98.0, 97.0)]), direction=PUT, lookback=10)
        assert series.iloc[0] == pytest.approx(0.03)

    def test_a_zero_range_bar_is_not_a_session(self) -> None:
        """The bar store appends the latest quote as open==high==low==close.

        Left in the sample it contributes a fake 0% excursion, dragging the quantile down on
        exactly the run that depends on it.
        """
        real = bars([(100.0, 101.0, 98.0, 100.5)] * 10)
        with_synthetic = bars([(100.0, 101.0, 98.0, 100.5)] * 10 + [(102.0, 102.0, 102.0, 102.0)])
        assert len(excursions(real, direction=CALL, lookback=60)) == 10
        assert len(excursions(with_synthetic, direction=CALL, lookback=60)) == 10

    def test_quantile_is_a_fill_probability(self) -> None:
        # Every session dipped exactly 2%, so any quantile of the distribution is 2%.
        result = expected_excursion(
            trending_bars(40), direction=CALL, lookback=60, fill_probability=0.6
        )
        assert result["excursion"] == pytest.approx(0.02)
        assert result["conditional"] is True
        assert result["quantile"] == pytest.approx(0.4)

    def test_higher_fill_probability_gives_a_shallower_bid(self) -> None:
        mixed = bars([(100.0, 101.0, 100.0 - d, 101.0) for d in (0.5, 1.0, 1.5, 2.0, 2.5)] * 6)
        shallow = expected_excursion(mixed, direction=CALL, lookback=60, fill_probability=0.9)
        deep = expected_excursion(mixed, direction=CALL, lookback=60, fill_probability=0.2)
        assert shallow["excursion"] < deep["excursion"]

    def test_conditioning_uses_only_agreeing_sessions(self) -> None:
        # Up days dip 1%; down days dip 5%. A call should budget for the up-day pullback.
        frame = bars([(100.0, 102.0, 99.0, 101.0)] * 20 + [(100.0, 100.0, 95.0, 96.0)] * 20)
        result = expected_excursion(frame, direction=CALL, lookback=60, fill_probability=0.5)
        assert result["conditional"] is True
        assert result["excursion"] == pytest.approx(0.01)

    def test_thin_conditional_sample_falls_back_to_unconditional(self) -> None:
        frame = bars([(100.0, 102.0, 99.0, 101.0)] * 3 + [(100.0, 100.0, 95.0, 96.0)] * 30)
        result = expected_excursion(frame, direction=CALL, lookback=60, fill_probability=0.5)
        assert result["conditional"] is False

    def test_no_bars_is_zero_not_an_error(self) -> None:
        assert expected_excursion(None, direction=CALL, lookback=60, fill_probability=0.6) == {
            "excursion": 0.0, "sample": 0, "conditional": False, "quantile": 0.0
        }


class TestPredictedLow:
    """The prediction is a price level for the session, not an offset from the last print."""

    def test_the_predicted_low_is_anchored_to_the_open(self) -> None:
        assert predicted_extreme(100.0, 0.01, direction=CALL) == pytest.approx(99.0)
        # For a put the adverse move is upward, so the predicted extreme is above the open.
        assert predicted_extreme(100.0, 0.01, direction=PUT) == pytest.approx(101.0)

    def test_the_bid_does_not_chase_a_rally(self) -> None:
        """The bug this model replaced: a fraction applied to the current price drifts upward.

        Open 100 with a 1% expected dip predicts a low of 99.00. However far the price runs
        during the morning, that prediction does not move -- the bid waits at 99, it does not
        follow the market to 102.96.
        """
        predicted = predicted_extreme(100.0, 0.01, direction=CALL)
        for current in (100.0, 102.0, 104.0):
            target = entry_underlying_target(
                current, predicted, direction=CALL, fraction_remaining=1.0,
            )
            assert target == pytest.approx(99.0)

    def test_the_bid_may_sit_below_the_current_market(self) -> None:
        # The whole point: it is waiting for a price the market has not reached.
        target = entry_underlying_target(
            104.0, predicted_extreme(100.0, 0.01, direction=CALL),
            direction=CALL, fraction_remaining=1.0,
        )
        assert target < 104.0

    def test_it_converges_on_the_market_as_the_session_ends(self) -> None:
        predicted = predicted_extreme(100.0, 0.02, direction=CALL)
        morning = entry_underlying_target(100.0, predicted, direction=CALL, fraction_remaining=1.0)
        midday = entry_underlying_target(100.0, predicted, direction=CALL, fraction_remaining=0.5)
        close = entry_underlying_target(100.0, predicted, direction=CALL, fraction_remaining=0.0)
        assert morning < midday < close
        assert close == pytest.approx(100.0)

    def test_a_market_already_through_the_prediction_takes_the_market(self) -> None:
        # The dip happened. Bidding below the market again would be waiting for a second one.
        predicted = predicted_extreme(100.0, 0.01, direction=CALL)     # 99.00
        assert entry_underlying_target(
            98.0, predicted, direction=CALL, fraction_remaining=1.0,
        ) == pytest.approx(98.0)

    def test_a_put_never_bids_below_the_market(self) -> None:
        predicted = predicted_extreme(100.0, 0.01, direction=PUT)      # 101.00
        assert entry_underlying_target(
            103.0, predicted, direction=PUT, fraction_remaining=1.0,
        ) == pytest.approx(103.0)

    def test_observed_excursion_reads_todays_low(self) -> None:
        intraday = bars([(100.0, 100.5, 98.0, 99.0), (99.0, 99.5, 97.5, 98.0)])
        assert observed_excursion(intraday, direction=CALL, session_open=100.0) == pytest.approx(0.025)

    def test_session_fraction_remaining(self) -> None:
        open_at = pd.Timestamp("2026-02-10 09:30", tz=MARKET_TZ)
        close_at = pd.Timestamp("2026-02-10 16:00", tz=MARKET_TZ)
        noon = pd.Timestamp("2026-02-10 12:45", tz=MARKET_TZ)
        assert session_fraction_remaining(noon, open_time=open_at, close_time=close_at) == pytest.approx(0.5, abs=0.01)


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


class TestTrendDirection:
    def test_call_when_above_average_and_rising(self) -> None:
        rising = bars([(90.0 + i, 91.0 + i, 89.0 + i, 90.5 + i) for i in range(40)])
        direction, checks = trend_direction(rising, cfg())
        assert direction == CALL
        assert all(check.ok for check in checks)

    def test_put_when_below_average_and_falling(self) -> None:
        falling = bars([(130.0 - i, 131.0 - i, 129.0 - i, 129.5 - i) for i in range(40)])
        assert trend_direction(falling, cfg())[0] == PUT

    def test_flat_when_the_move_lacks_conviction(self) -> None:
        flat = bars([(100.0, 100.5, 99.5, 100.0)] * 40)
        direction, checks = trend_direction(flat, cfg(trend_min_return=0.01))
        assert direction == ""
        assert any(check.blocking for check in checks)

    def test_short_history_blocks_rather_than_guessing(self) -> None:
        direction, checks = trend_direction(bars([(100.0, 101.0, 99.0, 100.0)] * 3), cfg())
        assert direction == ""
        assert checks[0].blocking


class TestPremarketVeto:
    def test_agreement_confirms(self) -> None:
        ok, checks = premarket_confirms(
            {"change_pct": 0.004, "bars": 60}, CALL, typical_move=0.008, config=cfg()
        )
        assert ok and not any(c.blocking for c in checks)

    def test_disagreement_vetoes_and_never_reverses(self) -> None:
        # A large contradicting gap abstains like any other -- it must not become a put signal.
        ok, checks = premarket_confirms(
            {"change_pct": -0.020, "bars": 60}, CALL, typical_move=0.008, config=cfg()
        )
        assert not ok
        assert any(c.blocking for c in checks)

    def test_flat_premarket_is_not_confirmation(self) -> None:
        ok, _ = premarket_confirms({"change_pct": 0.0005, "bars": 60}, CALL, 0.008, cfg())
        assert not ok

    def test_thin_premarket_fails_closed(self) -> None:
        ok, checks = premarket_confirms(
            {"change_pct": 0.02, "bars": 2}, CALL, typical_move=0.008, config=cfg()
        )
        assert not ok
        assert "prints" in checks[0].label

    def test_missing_premarket_fails_closed(self) -> None:
        assert premarket_confirms(None, CALL, 0.008, cfg())[0] is False

    def test_threshold_is_normalised_by_the_symbols_own_volatility(self) -> None:
        # The same 0.4% gap: decisive on a quiet name, unremarkable on a volatile one.
        quiet, _ = premarket_confirms({"change_pct": 0.004, "bars": 60}, CALL, 0.008, cfg())
        wild, _ = premarket_confirms({"change_pct": 0.004, "bars": 60}, CALL, 0.05, cfg())
        assert quiet and not wild

    def test_typical_daily_move(self) -> None:
        frame = bars([(100.0, 102.0, 98.0, 101.0)] * 10)
        assert typical_daily_move(frame) == pytest.approx(0.01)


# ── contract selection ───────────────────────────────────────────────────────


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

    def test_a_wide_spread_is_not_disqualifying(self) -> None:
        """Both legs rest as limits and never cross the spread, so width alone is not a veto.

        A wide market is if anything an opportunity for a patient limit -- it rests inside the
        spread and is paid by whoever is impatient. There is no spread ceiling.
        """
        wide = [contract(bid=0.90, ask=1.30, delta=0.45)]   # 36% wide

        best, _candidate, _checks = select_contract(
            wide, direction=CALL, as_of=date(2026, 2, 1), config=cfg()
        )

        assert best is not None

    def test_open_interest_is_the_only_liquidity_gate(self) -> None:
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

    def test_notional_cap_reduces_the_contract_count(self) -> None:
        rich = contract(ask=6.00)
        assert affordable_contracts(rich, cfg(contracts_per_position=3, max_notional_per_trade=1000)) == 1
        assert affordable_contracts(rich, cfg(contracts_per_position=3, max_notional_per_trade=2000)) == 3


# ── the state machine ────────────────────────────────────────────────────────


class TestLifecycle:
    def bidding(self, **overrides):
        kwargs = dict(
            memory={}, held_contract="", direction=CALL, contract=contract(), contracts=1,
            underlying_now=100.0, entry_target=98.0, exit_budget=0.0, checks=[],
            config=cfg(), session=session(),
        )
        kwargs.update(overrides)
        return plan_symbol("QQQM", **kwargs)

    def test_no_direction_means_no_orders(self) -> None:
        outcome = self.bidding(direction="", contract=None, contracts=0)
        assert outcome.state == FLAT and outcome.orders == []

    def test_bid_sits_below_the_mark(self) -> None:
        outcome = self.bidding()
        assert outcome.state == BIDDING
        request = outcome.orders[0].request
        assert request.asset_type == "option"
        assert request.order_type == "limit"
        assert request.action == "buy"
        # 2% below spot, translated through a 0.45 delta: 2.05 - (100 * 0.02 * 0.45) = 1.15
        assert request.limit_price == pytest.approx(1.15, abs=0.01)

    def test_the_bid_never_crosses_the_spread(self) -> None:
        """It converges toward the midpoint, not the ask.

        Paying the offer would guarantee a fill and in doing so discard the entire edge on
        exactly the days the prediction was wrong.
        """
        late = self.bidding(entry_target=99.9, session=session(fraction_remaining=0.02))
        assert late.orders[0].request.limit_price <= contract().midpoint

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
            exit_budget=0.02, checks=[], config=cfg(), session=session(),
        )
        kwargs.update(overrides)
        return plan_symbol("QQQM", **kwargs)

    def test_a_held_position_gets_an_oco_bracket(self) -> None:
        outcome = self.held()
        assert outcome.state == HELD
        request = outcome.orders[0].request
        assert request.strategy == "oco"
        assert len(request.children) == 2
        kinds = {child.order_type for child in request.children}
        assert kinds == {"limit", "stop"}
        assert all(child.action == "sell" for child in request.children)
        assert all(child.time_in_force == "gtc" for child in request.children)

    def test_the_stop_is_anchored_to_the_fill_not_the_mark(self) -> None:
        stop = next(c for c in self.held().orders[0].request.children if c.order_type == "stop")
        # The fill was 2.00 and the cap is stop_loss_pct below it -- of the *premium*, not of the
        # underlying, and measured from the fill rather than from wherever the mark now sits.
        assert stop.stop_price == pytest.approx(2.00 * (1 - cfg().stop_loss_pct), abs=0.01)

    def test_the_stop_does_not_move_when_the_mark_rises(self) -> None:
        low = next(c for c in self.held(memory={"mark": 2.10}).orders[0].request.children if c.order_type == "stop")
        high = next(c for c in self.held(memory={"mark": 3.50}).orders[0].request.children if c.order_type == "stop")
        assert low.stop_price == high.stop_price

    def test_the_target_ratchets_up(self) -> None:
        raised = self.held(memory={"target": 2.50}, underlying_now=110.0, exit_budget=0.05)
        assert raised.orders[0].request.limit_price > 2.50

    def test_the_target_never_ratchets_down(self) -> None:
        held = self.held(memory={"target": 5.00}, underlying_now=95.0, exit_budget=0.01)
        assert held.orders[0].request.limit_price == pytest.approx(5.00)

    def test_a_trivial_improvement_does_not_move_the_target(self) -> None:
        # Below ratchet_min_improvement the previous target stands, so the bracket is not churned.
        held = self.held(memory={"target": 2.80}, underlying_now=101.0, exit_budget=0.001)
        assert held.orders[0].request.limit_price == pytest.approx(2.80)

    def test_the_deadline_is_reported_as_blocking(self) -> None:
        outcome = self.held(memory={"sessions_held": 2})
        assert any(check.blocking for check in outcome.checks)

    def test_at_the_deadline_the_ask_converges_on_the_market(self) -> None:
        # The last fire of the final session: nothing of the walk-in is left, so the ask sits at
        # the mark rather than at a price the position is no longer allowed to wait for.
        outcome = self.held(
            memory={"sessions_held": 2, "mark": 2.40, "target": 0.0},
            session=session(fraction_remaining=0.0),
        )
        assert outcome.orders[0].request.limit_price == pytest.approx(2.40)

    def test_the_deadline_still_asks_for_a_profit_early_in_the_session(self) -> None:
        # Being on the last session does not mean giving up at 09:30 -- it converges over the day.
        outcome = self.held(
            memory={"sessions_held": 2, "mark": 2.40, "target": 0.0},
            session=session(fraction_remaining=1.0), underlying_now=110.0, exit_budget=0.05,
        )
        assert outcome.orders[0].request.limit_price > 2.40

    def test_an_unfilled_bid_is_abandoned_the_next_morning(self) -> None:
        # Yesterday's budget described a session that has ended, so it is not re-priced.
        outcome = plan_symbol(
            "QQQM", memory={"state": BIDDING, "bid": 1.15, "market_day": "2026-02-09"},
            held_contract="", direction="", contract=None, contracts=0,
            underlying_now=100.0, entry_target=0.0, exit_budget=0.0, checks=[],
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




class TestConvergence:
    """The walk-in shape, and the promise that it never crosses the spread."""

    def bid_at(self, fraction: float, power: float) -> float:
        predicted = predicted_extreme(100.0, 0.01, direction=CALL)   # 99.00
        return entry_underlying_target(
            100.0, predicted, direction=CALL,
            fraction_remaining=fraction, decay_power=power,
        )

    def test_power_one_reaches_the_market_by_the_close(self) -> None:
        assert self.bid_at(1.0, 1.0) == pytest.approx(99.00)
        assert self.bid_at(0.5, 1.0) == pytest.approx(99.50)
        assert self.bid_at(0.0, 1.0) == pytest.approx(100.00)

    def test_lower_power_is_more_patient(self) -> None:
        """Lower stays nearer the predicted low, so it effectively never converges.

        With five-minute fires the last one of the day is at ~1% of the session remaining, and at
        power 0.25 the bid is still well short of the mid there -- which is the point: fewer
        fills, every one at a price that was chosen.
        """
        last_fire = 5 / 390
        assert self.bid_at(last_fire, 1.0) > 99.98      # effectively at the market
        assert self.bid_at(last_fire, 0.25) < 99.70     # still holding out

    def test_higher_power_gives_up_the_edge_sooner(self) -> None:
        assert self.bid_at(0.5, 2.0) > self.bid_at(0.5, 1.0) > self.bid_at(0.5, 0.25)

    def test_the_bid_is_never_on_the_wrong_side_of_the_market(self) -> None:
        # However late it gets, a call bids at or below spot and a put at or above it.
        for fraction in (1.0, 0.5, 0.1, 0.0):
            assert self.bid_at(fraction, 1.0) <= 100.0
            put = entry_underlying_target(
                100.0, predicted_extreme(100.0, 0.01, direction=PUT),
                direction=PUT, fraction_remaining=fraction, decay_power=1.0,
            )
            assert put >= 100.0


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


class TestEstimateBand:
    """The band describes the session; the bid describes this instant. They are not the same."""

    def levels(self, fraction: float) -> dict:
        from src.algorithms.options_flip.algorithm import _entry_levels

        rising = bars([(90.0 + i, 91.0 + i, 89.0 + i, 90.5 + i) for i in range(70)])
        intraday = bars([(100.0, 100.5, 99.0, 99.5)])
        intraday["timestamp"] = [pd.Timestamp("2026-02-10 09:30", tz=MARKET_TZ)]
        return _entry_levels(
            rising, intraday, CALL, cfg(), session(fraction_remaining=fraction), 100.0
        )

    def test_the_predicted_low_does_not_move_with_the_clock(self) -> None:
        assert self.levels(1.0)["predicted"] == pytest.approx(self.levels(0.0)["predicted"])

    def test_the_bid_does_move_with_the_clock(self) -> None:
        assert self.levels(1.0)["bid"] < self.levels(0.0)["bid"]

    def test_with_no_session_left_the_bid_is_the_market_but_the_prediction_is_not(self) -> None:
        """Outside market hours ``fraction_remaining`` is 0, so the bid collapses onto spot.

        That is correct for an order and meaningless as a forecast -- and reading the band off
        the bid is what made it start at the current price on a closed market, as though no dip
        were expected at all.
        """
        levels = self.levels(0.0)
        assert levels["bid"] == pytest.approx(100.0)
        assert levels["predicted"] < 100.0


class TestWorthTrading:
    """A predicted move of a few dollars is not a trade, however right the setup looks."""

    def estimate(self, low: float, high: float, config) -> dict:
        from src.algorithms.options_flip.algorithm import _estimate

        # delta 1.0 makes the underlying move translate one-for-one, so the band is exactly
        # (high - low) and the arithmetic under test is the sizing, not the translation.
        c = contract(bid=low, ask=low + 0.02, mark=low + 0.01, delta=1.0)
        return _estimate(
            c, CALL, 100.0,
            {"predicted": 100.0 - (c.midpoint - low), "bid": 100.0},
            (high - c.midpoint) / 100.0, config, date(2026, 2, 1),
        )

    def test_expected_profit_is_the_band_in_dollars(self) -> None:
        est = self.estimate(2.00, 3.00, cfg())

        assert est["expected_profit"] == pytest.approx(est["edge"] * 100)
        # Gross: commissions are not modelled anywhere, so they are not netted here either.
        assert "commission" not in est and "net_profit" not in est

    def test_a_bigger_position_scales_the_total_but_not_the_per_contract_figure(self) -> None:
        one = self.estimate(2.00, 3.00, cfg(contracts_per_position=1))
        three = self.estimate(2.00, 3.00, cfg(contracts_per_position=3))

        assert three["expected_profit"] == pytest.approx(one["expected_profit"] * 3)
        # The per-contract figure is what the gate judges, and it does not move with size.
        assert three["expected_profit_per_contract"] == pytest.approx(
            one["expected_profit_per_contract"]
        )

    def test_buying_more_cannot_satisfy_the_floor(self) -> None:
        """A floor on the position total is satisfiable by scaling up a marginal trade.

        Ten contracts predicted to move two cents each is $25 in total and clears a $25 position
        floor -- while being ten times the risk for an edge that was never there. Judged per
        contract, size cannot buy its way past the quality bar.
        """
        thin_one = self.estimate(2.00, 2.02, cfg(contracts_per_position=1))
        thin_ten = self.estimate(2.00, 2.02, cfg(contracts_per_position=10))

        assert thin_ten["expected_profit"] >= 20         # the total looks respectable
        assert thin_ten["expected_profit_per_contract"] == pytest.approx(
            thin_one["expected_profit_per_contract"]
        )
        assert thin_ten["expected_profit_per_contract"] < cfg().min_expected_profit

    def test_a_thin_move_is_a_small_number(self) -> None:
        # $0.05 predicted on one contract is $5 -- under any sane floor.
        assert self.estimate(2.00, 2.05, cfg())["expected_profit_per_contract"] < 10

    def test_the_floor_is_a_different_axis_from_the_trend_gate(self) -> None:
        """``trend_min_return`` gates the underlying's momentum, this gates the contract's move.

        A symbol can trend hard enough to clear the first while its contract stands to move a
        few dollars -- the trade that looks right on every other gate and still loses.
        """
        thin = self.estimate(2.00, 2.05, cfg(trend_min_return=0.0))

        assert thin["expected_profit_per_contract"] < cfg().min_expected_profit


class TestExitHorizon:
    """The exit is held for max_hold_sessions, so it is priced over that many sessions."""

    def rising(self) -> pd.DataFrame:
        # Each session opens where the last closed and grinds up, so a two-session window reaches
        # visibly higher above an open than a one-session window can.
        rows, price = [], 100.0
        for _ in range(60):
            rows.append((price, price * 1.01, price * 0.995, price * 1.008))
            price *= 1.008
        return bars(rows)

    def test_a_longer_horizon_finds_a_larger_move(self) -> None:
        one = expected_excursion(
            self.rising(), direction=PUT, lookback=60, fill_probability=0.5, horizon=1
        )["excursion"]
        two = expected_excursion(
            self.rising(), direction=PUT, lookback=60, fill_probability=0.5, horizon=2
        )["excursion"]

        assert two > one

    def test_horizon_one_is_the_plain_daily_extreme(self) -> None:
        frame = bars([(100.0, 103.0, 98.0, 102.0)])

        assert excursions(frame, direction=PUT, lookback=10, horizon=1).iloc[0] == pytest.approx(0.03)

    def test_the_exit_uses_the_hold_period_and_the_entry_does_not(self) -> None:
        """Pricing a two-day target off a one-day excursion sets it systematically too low."""
        from src.algorithms.options_flip.algorithm import _budget

        frame, sess = self.rising(), session(fraction_remaining=1.0)

        assert _budget(frame, None, CALL, cfg(max_hold_sessions=2), sess, exit_side=True) > \
               _budget(frame, None, CALL, cfg(max_hold_sessions=1), sess, exit_side=True)

    def test_the_exit_measures_the_favourable_tail_not_the_adverse_one(self) -> None:
        """A call's profit target comes from how far the stock *rises*, not how far it falls.

        The two are close on a symmetric name, which is why passing the trade's own direction
        here looked right -- and wrong on anything skewed.
        """
        from src.algorithms.options_flip.algorithm import _budget

        # Falls 3% below the open, rises only 1% above it: the two tails are far apart.
        skewed = bars([(100.0, 101.0, 97.0, 100.5)] * 60)
        sess = session(fraction_remaining=1.0)

        exit_budget = _budget(skewed, None, CALL, cfg(max_hold_sessions=1), sess, exit_side=True)
        entry_budget = _budget(skewed, None, CALL, cfg(), sess, exit_side=False)

        assert exit_budget == pytest.approx(0.01, abs=1e-3)     # the upside tail
        assert entry_budget == pytest.approx(0.03, abs=1e-3)    # the downside tail
