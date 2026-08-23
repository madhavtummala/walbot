"""Distributions as cash events, kept out of the price cache.

The thing under test is a separation as much as a calculation: ``market_bars`` must stay a
record of what the market printed, while the income still reaches the ledger and the signal.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.core.interfaces import CashDividend
from src.data.bars import TOTAL_RETURN_COLUMN, attach_total_return, signal_price
from src.data.dividends import (
    _collapse,
    dividends_by_symbol,
    read_dividends,
    write_dividends,
)


@pytest.fixture()
def db(tmp_path) -> str:
    return str(tmp_path / "dividends.duckdb")


def _dividend(symbol: str, ex: str, amount: float, **kwargs) -> CashDividend:
    return CashDividend(
        symbol=symbol,
        ex_date=date.fromisoformat(ex),
        amount=amount,
        payable_date=date.fromisoformat(kwargs.get("payable", ex)),
        special=kwargs.get("special", False),
        source="test",
    )


# =========================================================================================
# Storage
# =========================================================================================


def test_a_refetch_does_not_pay_the_holder_twice(db: str) -> None:
    """A published dividend never changes, so writing it again must be a no-op."""
    events = [_dividend("SGOV", "2026-07-01", 0.2958)]

    write_dividends(events, db_path=db)
    write_dividends(events, db_path=db)

    stored = read_dividends(["SGOV"], db_path=db)
    assert len(stored) == 1
    assert float(stored["amount"].iloc[0]) == pytest.approx(0.2958)


def test_one_payment_filed_under_two_ids_is_counted_once() -> None:
    """Alpaca reports GPIX's 2025-01-03 distribution twice at an identical rate.

    yfinance and the 0.35 print both say it was paid once, so keeping both would invent
    income that never arrived.
    """
    collapsed = _collapse([
        _dividend("GPIX", "2025-01-03", 0.34964),
        _dividend("GPIX", "2025-01-03", 0.34964),
    ])

    assert len(collapsed) == 1
    assert collapsed[0].amount == pytest.approx(0.34964)


def test_two_real_payments_on_one_ex_date_are_added_up() -> None:
    """XYLD paid ordinary income and a capital gain on 2021-12-30. Both are cash."""
    collapsed = _collapse([
        _dividend("XYLD", "2021-12-30", 0.343335),
        _dividend("XYLD", "2021-12-30", 0.114365),
    ])

    assert len(collapsed) == 1
    assert collapsed[0].amount == pytest.approx(0.4577)


def test_dividends_by_symbol_returns_an_ex_date_indexed_series(db: str) -> None:
    write_dividends(
        [_dividend("SGOV", "2026-06-01", 0.2995), _dividend("SGOV", "2026-07-01", 0.2958)],
        db_path=db,
    )

    series = dividends_by_symbol(["SGOV"], db_path=db)["SGOV"]

    assert list(series.index) == [pd.Timestamp("2026-06-01"), pd.Timestamp("2026-07-01")]
    assert series.sum() == pytest.approx(0.5953)


# =========================================================================================
# The derived total-return series
# =========================================================================================


def _bars(closes: list[float], start: str = "2026-01-01") -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range(start, periods=len(closes), freq="D", tz="UTC"),
        "close": closes,
    })


def test_a_non_payer_has_a_total_return_equal_to_its_price() -> None:
    """GLD pays nothing, so the two series must not diverge by even a rounding step."""
    bars = attach_total_return(_bars([100.0, 101.0, 102.0]), "GLD", dividends=pd.Series(dtype=float))

    assert bars[TOTAL_RETURN_COLUMN].tolist() == pytest.approx([100.0, 101.0, 102.0])


def test_a_flat_price_that_pays_a_dividend_still_earns_a_return() -> None:
    """The SGOV case in miniature: price returns to where it started, the holder is richer."""
    dividends = pd.Series([1.0], index=[pd.Timestamp("2026-01-03")])
    # Price drops by the payment on the ex-date and recovers -- a sawtooth, as SGOV prints.
    bars = attach_total_return(_bars([100.0, 100.0, 99.0, 100.0]), "SGOV", dividends=dividends)

    total = bars[TOTAL_RETURN_COLUMN]
    # 1/99, not 1/100: the payment is reinvested at the ex-date price it actually drops to,
    # which is the standard total-return convention and what a DRIP would really buy.
    assert total.iloc[-1] / total.iloc[0] - 1 == pytest.approx(1 / 99, abs=1e-6)
    # Raw price says the holder made nothing at all.
    assert bars["close"].iloc[-1] / bars["close"].iloc[0] - 1 == pytest.approx(0.0)


def test_the_total_return_series_is_anchored_to_the_latest_raw_price() -> None:
    """The right-hand edge has to be a real, quotable price rather than a synthetic index."""
    dividends = pd.Series([1.0], index=[pd.Timestamp("2026-01-03")])
    bars = attach_total_return(_bars([100.0, 100.0, 99.0, 100.0]), "SGOV", dividends=dividends)

    assert bars[TOTAL_RETURN_COLUMN].iloc[-1] == pytest.approx(bars["close"].iloc[-1])


def test_signal_price_prefers_the_derived_series_over_a_stored_column() -> None:
    """A number this project derived beats one it may once have written into the cache."""
    bars = _bars([10.0, 11.0])
    bars["adjusted_close"] = [1.0, 2.0]
    bars[TOTAL_RETURN_COLUMN] = [5.0, 6.0]

    assert signal_price(bars).tolist() == [5.0, 6.0]


def test_signal_price_falls_back_to_close_when_nothing_is_derived() -> None:
    assert signal_price(_bars([10.0, 11.0])).tolist() == [10.0, 11.0]


def test_attaching_total_return_never_edits_the_raw_price() -> None:
    """The whole point of the split: prices are evidence, not a place to store conclusions."""
    original = _bars([100.0, 99.0, 100.0])
    dividends = pd.Series([1.0], index=[pd.Timestamp("2026-01-02")])

    attached = attach_total_return(original, "SGOV", dividends=dividends)

    assert attached["close"].tolist() == original["close"].tolist()
    assert TOTAL_RETURN_COLUMN not in original


# =========================================================================================
# The ledger: cash, not price appreciation
# =========================================================================================


def test_the_replay_credits_cash_on_the_payable_date_not_the_ex_date() -> None:
    """The settlement gap is real cash drag, so the backtest waits it out too."""
    from src.execution.replay import _payable_on, dividend_schedule

    trade_dates = [pd.Timestamp(d, tz="UTC") for d in
                   ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-08"]]
    dividend = _dividend("SGOV", "2026-01-02", 0.30, payable="2026-01-07")

    schedule = dividend_schedule(["SGOV"], trade_dates, {"SGOV": [dividend]})
    assert list(schedule) == [pd.Timestamp("2026-01-02", tz="UTC")]

    # 2026-01-07 is not a trade date, so the cash lands on the next one that is.
    assert _payable_on(pd.Timestamp("2026-01-02", tz="UTC"), dividend, trade_dates) == \
        pd.Timestamp("2026-01-08", tz="UTC")


def test_an_ex_date_off_the_trading_grid_still_pays() -> None:
    """A payment must never be dropped just because its ex-date fell on a closed day."""
    from src.execution.replay import dividend_schedule

    trade_dates = [pd.Timestamp(d, tz="UTC") for d in ["2026-01-02", "2026-01-05"]]
    # 2026-01-03 is a Saturday.
    schedule = dividend_schedule(["SGOV"], trade_dates, {"SGOV": [_dividend("SGOV", "2026-01-03", 0.3)]})

    assert list(schedule) == [pd.Timestamp("2026-01-05", tz="UTC")]


def test_a_payment_past_the_end_of_the_replay_is_not_lost() -> None:
    from src.execution.replay import _payable_on

    trade_dates = [pd.Timestamp(d, tz="UTC") for d in ["2026-01-02", "2026-01-05"]]
    dividend = _dividend("SGOV", "2026-01-05", 0.30, payable="2026-02-01")

    assert _payable_on(pd.Timestamp("2026-01-05", tz="UTC"), dividend, trade_dates) == \
        pd.Timestamp("2026-01-05", tz="UTC")


def test_the_paper_book_credits_a_dividend_once(db: str, monkeypatch, tmp_path) -> None:
    """A real account is paid whether or not anything asked, but only once per payment."""
    from src.brokerages.paper import brokerage as paper

    import src.data.dividends as dividends_module

    write_dividends([_dividend("SGOV", "2026-07-01", 0.30)], db_path=db)
    # ``credit_dividends`` imports inside the function body, so the name has to be replaced on
    # the module it is imported *from* rather than on the brokerage module.
    monkeypatch.setattr(
        dividends_module, "read_dividends",
        lambda symbols, start=None, end=None: read_dividends(symbols, start=start, end=end, db_path=db),
    )

    class _Config:
        paper_starting_cash = 1000.0
        account_id = "test"

    with __import__("src.data.state_store", fromlist=["ephemeral_state"]).ephemeral_state():
        broker = paper.PaperBrokerage(_Config())
        broker.state["positions"] = {"SGOV": 100.0}
        broker.state["dividends_credited_through"] = "2026-06-01"
        cash_before = broker.state["cash"]

        first = broker.credit_dividends(as_of=date(2026, 7, 15))
        second = broker.credit_dividends(as_of=date(2026, 7, 15))

    assert first["credited"] == pytest.approx(30.0), "100 shares x $0.30"
    assert second["credited"] == pytest.approx(0.0), "the watermark makes a rerun a no-op"
    assert broker.state["cash"] == pytest.approx(cash_before + 30.0)


def test_a_dividend_that_went_ex_before_the_replay_opened_is_not_paid() -> None:
    """Whoever held the shares then earned it, and that was nobody in this replay.

    Without the guard every historical distribution collapses onto the first trade date. That
    is invisible while the book starts empty and silently inflates day one as soon as a
    replay is seeded with positions.
    """
    from src.execution.replay import dividend_schedule

    trade_dates = [pd.Timestamp(d, tz="UTC") for d in ["2026-01-05", "2026-01-06"]]
    dividends = {"SGOV": [
        _dividend("SGOV", "2025-11-01", 0.30),   # long before the window
        _dividend("SGOV", "2026-01-06", 0.31),   # inside it
    ]}

    schedule = dividend_schedule(["SGOV"], trade_dates, dividends)

    assert list(schedule) == [pd.Timestamp("2026-01-06", tz="UTC")]
    assert schedule[pd.Timestamp("2026-01-06", tz="UTC")][0][1].amount == pytest.approx(0.31)


# =========================================================================================
# Per-account income, through the brokerage interface
# =========================================================================================


def test_a_schwab_account_number_matches_however_it_is_punctuated() -> None:
    """Schwab reports "00000000"; statements and the website write "0000-0000".

    Matching the raw strings 404'd a correctly configured account, and because the dashboard
    falls back to the default account on error, the Schwab tab then showed Alpaca's money.
    """
    from src.brokerages.schwab.client import _digits, account_hash

    assert _digits("0000-0000") == _digits("00000000") == "00000000"

    class _Session:
        def get(self, url, **kwargs):
            return [{"accountNumber": "00000000", "hashValue": "ABC123"}]

    assert account_hash(_Session(), "0000-0000") == "ABC123"
    assert account_hash(_Session(), "00000000") == "ABC123"
    assert account_hash(_Session(), "") == "ABC123", "no number configured takes the first account"


def test_an_unknown_schwab_account_still_raises() -> None:
    from src.brokerages.schwab.client import SchwabAPIError, account_hash

    class _Session:
        def get(self, url, **kwargs):
            return [{"accountNumber": "00000000", "hashValue": "ABC123"}]

    with pytest.raises(SchwabAPIError):
        account_hash(_Session(), "1111-2222")


def test_a_brokerage_reports_no_income_rather_than_failing() -> None:
    """The default keeps an account page working on a broker that cannot report activity."""
    from src.core.interfaces import Brokerage

    class _Minimal(Brokerage):
        def get_account_state(self): return {}
        def get_positions(self): return {}
        def submit_order(self, request): return {}
        def cancel_all_orders(self): return None

    assert _Minimal().get_dividend_activity() == []


def test_the_paper_book_reports_the_dividends_it_booked(db: str, monkeypatch) -> None:
    from src.brokerages.paper import brokerage as paper
    import src.data.dividends as dividends_module

    write_dividends([_dividend("SGOV", "2026-07-01", 0.30)], db_path=db)
    monkeypatch.setattr(
        dividends_module, "read_dividends",
        lambda symbols, start=None, end=None: read_dividends(symbols, start=start, end=end, db_path=db),
    )

    class _Config:
        paper_starting_cash = 1000.0
        account_id = "test"

    from src.data.state_store import ephemeral_state
    with ephemeral_state():
        broker = paper.PaperBrokerage(_Config())
        broker.state["positions"] = {"SGOV": 100.0}
        broker.state["dividends_credited_through"] = "2026-06-01"
        broker.credit_dividends(as_of=date(2026, 7, 15))

        rows = broker.get_dividend_activity()

    assert len(rows) == 1
    assert rows[0]["symbol"] == "SGOV"
    assert rows[0]["amount"] == pytest.approx(30.0)
