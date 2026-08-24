"""Schwab option chains and the pre-market read.

Both are deliberately outside ``MarketDataProvider`` and outside the bar cache. The pre-market
tests in particular guard a boundary that is easy to erode: extended-hours candles must never
reach ``market_bars``, because that table has no session dimension and RTH and ETH bars for the
same symbol and interval would merge into one corrupted series.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.connectors.market import schwab_options
from src.connectors.market.schwab_options import fetch_option_chain, fetch_premarket_summary
from src.core.interfaces import MARKET_TZ
from src.core.options import CALL, PUT


class Config:
    pass


def chain_payload() -> dict:
    return {
        "callExpDateMap": {
            "2026-03-05:23": {
                "140.0": [{
                    "symbol": "QQQM  260305C00140000", "strikePrice": 140.0,
                    "bid": 3.00, "ask": 3.10, "mark": 3.05, "delta": 0.45,
                    "openInterest": 1500, "totalVolume": 800, "volatility": 22.0,
                }],
                "145.0": [{
                    "symbol": "QQQM  260305C00145000", "strikePrice": 145.0,
                    "bid": 1.10, "ask": 1.20, "mark": 1.15, "delta": 0.28,
                    "openInterest": 900, "totalVolume": 300, "volatility": 23.5,
                }],
            }
        },
        "putExpDateMap": {
            "2026-03-05:23": {
                "140.0": [{
                    "symbol": "QQQM  260305P00140000", "strikePrice": 140.0,
                    "bid": 2.80, "ask": 2.95, "mark": 2.88, "delta": -0.44,
                    "openInterest": 1100, "totalVolume": 450, "volatility": 24.0,
                }]
            }
        },
    }


@pytest.fixture
def captured(monkeypatch):
    """Intercept the HTTP call, returning the canned payload and recording the request."""
    calls: list[tuple[str, dict]] = []

    def fake_request(provider, category, url, params=None, headers=None):
        calls.append((url, dict(params or {})))
        return fake_request.payload

    fake_request.payload = {}
    monkeypatch.setattr(schwab_options, "_request_json", fake_request)
    monkeypatch.setattr(schwab_options, "_schwab_token", lambda config, category: "token")
    return calls, fake_request


class TestOptionChain:
    def test_parses_calls_and_puts_with_greeks(self, captured) -> None:
        calls, fake = captured
        fake.payload = chain_payload()

        contracts = fetch_option_chain(Config(), "QQQM", as_of=date(2026, 2, 10))

        assert len(contracts) == 3
        call = next(c for c in contracts if c.osi_symbol == "QQQM  260305C00140000")
        assert call.option_type == CALL
        assert call.strike == 140.0
        assert call.expiry == date(2026, 3, 5)
        assert call.delta == 0.45
        assert call.open_interest == 1500
        assert call.dte(date(2026, 2, 10)) == 23
        assert any(c.option_type == PUT and c.delta == -0.44 for c in contracts)

    def test_the_dte_window_is_applied_in_the_request(self, captured) -> None:
        calls, fake = captured
        fake.payload = chain_payload()

        fetch_option_chain(Config(), "QQQM", min_dte=10, max_dte=45, as_of=date(2026, 2, 10))

        # Asking for the fortnight the strategy can trade, rather than filtering thousands of
        # rows after the fact.
        _, params = calls[0]
        assert params["fromDate"] == "2026-02-20"
        assert params["toDate"] == "2026-03-27"
        assert params["symbol"] == "QQQM"

    def test_option_type_narrows_the_request(self, captured) -> None:
        calls, fake = captured
        fake.payload = chain_payload()

        fetch_option_chain(Config(), "QQQM", option_type=PUT, as_of=date(2026, 2, 10))

        assert calls[0][1]["contractType"] == "PUT"

    def test_schwabs_own_osi_string_is_preferred_over_rebuilding_one(self, captured) -> None:
        _, fake = captured
        fake.payload = chain_payload()

        contracts = fetch_option_chain(Config(), "QQQM", as_of=date(2026, 2, 10))

        # A contract sent back on an order leg must be byte-identical to the one the venue named.
        assert all(c.osi_symbol.startswith("QQQM ") for c in contracts)

    def test_an_uncomputable_delta_reads_as_zero_not_as_a_crash(self, captured) -> None:
        _, fake = captured
        fake.payload = {"callExpDateMap": {"2026-03-05:23": {"200.0": [{
            "symbol": "QQQM  260305C00200000", "strikePrice": 200.0,
            "bid": 0.01, "ask": 0.05, "delta": "NaN", "openInterest": 3,
        }]}}}

        contracts = fetch_option_chain(Config(), "QQQM", as_of=date(2026, 2, 10))

        # Zero clears no delta band, which is the correct outcome for a contract with no
        # computable delta.
        assert contracts[0].delta == 0.0

    def test_an_empty_chain_is_empty_not_an_error(self, captured) -> None:
        _, fake = captured
        fake.payload = {}

        assert fetch_option_chain(Config(), "NOPE", as_of=date(2026, 2, 10)) == []

    def test_rows_without_a_strike_are_dropped(self, captured) -> None:
        _, fake = captured
        fake.payload = {"callExpDateMap": {"2026-03-05:23": {"0": [{"symbol": "X", "strikePrice": 0}]}}}

        assert fetch_option_chain(Config(), "QQQM", as_of=date(2026, 2, 10)) == []


class TestPremarket:
    def payload(self) -> dict:
        def stamp(hour, minute):
            moment = datetime(2026, 2, 10, hour, minute, tzinfo=MARKET_TZ)
            return int(moment.timestamp() * 1000)

        return {
            "previousClose": 100.0,
            "candles": [
                {"datetime": stamp(7, 30), "close": 100.5, "volume": 1000},
                {"datetime": stamp(9, 15), "close": 101.0, "volume": 2000},
                # After the bell: must not count toward the pre-market reading.
                {"datetime": stamp(9, 45), "close": 103.0, "volume": 50000},
            ],
        }

    def test_reads_the_gap_against_the_prior_close(self, captured) -> None:
        _, fake = captured
        fake.payload = self.payload()

        summary = fetch_premarket_summary(
            Config(), "QQQM", as_of=datetime(2026, 2, 10, 14, 35, tzinfo=timezone.utc)
        )

        assert summary["prior_close"] == 100.0
        assert summary["last"] == 101.0
        assert summary["change_pct"] == pytest.approx(0.01)

    def test_the_regular_session_is_excluded(self, captured) -> None:
        _, fake = captured
        fake.payload = self.payload()

        summary = fetch_premarket_summary(
            Config(), "QQQM", as_of=datetime(2026, 2, 10, 20, 0, tzinfo=timezone.utc)
        )

        # A run at 09:35 and a run at 15:00 must give the same answer: the pre-market session is
        # over and immutable by the bell.
        assert summary["bars"] == 2
        assert summary["volume"] == 3000
        assert summary["last"] == 101.0

    def test_bar_count_is_reported_so_a_thin_gap_can_be_told_from_a_real_one(self, captured) -> None:
        _, fake = captured
        fake.payload = {"previousClose": 100.0, "candles": [
            {"datetime": int(datetime(2026, 2, 10, 8, 0, tzinfo=MARKET_TZ).timestamp() * 1000),
             "close": 102.0, "volume": 100},
        ]}

        summary = fetch_premarket_summary(
            Config(), "QQQM", as_of=datetime(2026, 2, 10, 14, 35, tzinfo=timezone.utc)
        )

        assert summary["bars"] == 1
        assert summary["change_pct"] == pytest.approx(0.02)

    def test_extended_hours_is_requested_but_never_cached(self, captured) -> None:
        calls, fake = captured
        fake.payload = self.payload()

        fetch_premarket_summary(
            Config(), "QQQM", as_of=datetime(2026, 2, 10, 14, 35, tzinfo=timezone.utc)
        )

        _, params = calls[0]
        assert params["needExtendedHoursData"] == "true"
        assert params["frequencyType"] == "minute"
        # This path goes straight to the API. Nothing here may reach ``market_bars``: that table
        # has no session dimension, so an extended-hours bar would merge into the regular-hours
        # series for the same symbol and interval.
        assert not any("write_market_bars" in str(call) for call in calls)

    def test_no_candles_is_a_zero_reading_not_a_crash(self, captured) -> None:
        _, fake = captured
        fake.payload = {"previousClose": 100.0, "candles": []}

        summary = fetch_premarket_summary(
            Config(), "QQQM", as_of=datetime(2026, 2, 10, 14, 35, tzinfo=timezone.utc)
        )

        assert summary["bars"] == 0 and summary["change_pct"] == 0.0
