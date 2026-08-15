"""The property that makes minute-stated horizons worth having: grid invariance.

Every one of these asks the same question of the same price path sampled at different
resolutions, and requires the same answer. If any of them starts failing, an algorithm's
tuned horizons have quietly become a function of whichever feed answered.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data import bars as bars_module


def _path(freq: str, periods: int, *, start: str = "2026-05-11 13:30", rate: float = 0.0003) -> pd.DataFrame:
    stamps = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    closes = [100.0 * ((1.0 + rate) ** index) for index in range(periods)]
    return pd.DataFrame({"timestamp": stamps, "close": closes})


def test_bar_interval_is_read_from_the_data_not_assumed() -> None:
    assert bars_module.bar_interval_minutes(_path("5min", 20)) == 5
    assert bars_module.bar_interval_minutes(_path("15min", 20)) == 15
    assert bars_module.bar_interval_minutes(_path("1D", 20)) == 1440
    # Nothing to measure falls back to the reference grid rather than to zero.
    assert bars_module.bar_interval_minutes(pd.DataFrame()) == bars_module.REFERENCE_INTERVAL_MINUTES


def test_a_return_over_minutes_is_identical_on_any_grid() -> None:
    fine = _path("5min", 145)
    coarse = fine.iloc[::3].reset_index(drop=True)  # same path, same end, 15-minute bars

    for minutes in (60, 240, 720):
        assert bars_module.return_over_minutes(fine, minutes) == pytest.approx(
            bars_module.return_over_minutes(coarse, minutes), abs=1e-12
        )


def test_a_horizon_past_the_start_of_history_reports_no_signal() -> None:
    """Silently measuring a shorter span would understate some symbols and not others."""
    history = _path("5min", 25)  # two hours

    assert bars_module.return_over_minutes(history, 60) != 0.0
    assert bars_module.return_over_minutes(history, 6000) == 0.0


def test_a_lookup_resolves_against_daily_bars_when_that_is_all_there_is() -> None:
    """"Any nearby value we have available" includes a daily close.

    One daily bar advances the market clock by one session, so five sessions of history is
    1950 minutes -- not the 7200 calendar minutes those five days sit across.
    """
    daily = _path("1D", 30, rate=0.004)

    assert bars_module.return_over_minutes(daily, 5 * bars_module.TRADING_MINUTES_PER_DAY) == pytest.approx(
        float(daily["close"].iloc[-1]) / float(daily["close"].iloc[-6]) - 1.0
    )


def test_horizons_are_market_minutes_so_a_weekend_does_not_consume_them() -> None:
    """A calendar reading would let a Friday-to-Monday gap eat most of a lookback window."""
    # Two sessions of 15-minute bars, Friday then Monday, ~64 hours apart on the clock.
    friday = pd.date_range("2026-05-08 13:30", periods=26, freq="15min", tz="UTC")
    monday = pd.date_range("2026-05-11 13:30", periods=26, freq="15min", tz="UTC")
    closes = [100.0 + index for index in range(52)]
    history = pd.DataFrame({"timestamp": friday.append(monday), "close": closes})

    # One full session back reaches Friday's close, not somewhere inside Monday morning.
    one_session = bars_module.return_over_minutes(history, bars_module.TRADING_MINUTES_PER_DAY)
    assert one_session == pytest.approx(closes[-1] / closes[25] - 1.0)
    # And the frame is reported as spanning two sessions, not the three days it sits across.
    assert bars_module.coverage_minutes(history) == pytest.approx(2 * 390, abs=30)


def test_realized_volatility_is_quoted_per_reference_bar_on_any_grid() -> None:
    """The tuned thresholds (0.06, 0.08) were fitted on 15-minute bars and must keep meaning."""
    import numpy as np

    # A random walk, because sqrt-of-time scaling is a statement about one of those. A
    # deterministic wobble would alias when resampled and prove nothing either way.
    rng = np.random.default_rng(7)
    stamps = pd.date_range("2026-05-11 13:30", periods=901, freq="5min", tz="UTC")
    closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.0009, 901)))
    fine = pd.DataFrame({"timestamp": stamps, "close": closes})
    coarse = fine.iloc[::3].reset_index(drop=True)

    fine_vol = bars_module.realized_volatility(fine, 3000)
    coarse_vol = bars_module.realized_volatility(coarse, 3000)

    assert fine_vol > 0
    # Without the sqrt(reference / interval) rescaling these would differ by about sqrt(3).
    assert fine_vol == pytest.approx(coarse_vol, rel=0.15)


def test_realized_volatility_ignores_the_coarse_tail_of_a_blended_frame() -> None:
    """Differencing across the seam would read one daily step as a single bar return."""
    fine = _path("5min", 60, start="2026-05-12 13:30")
    fine["interval_minutes"] = 5
    daily = _path("1D", 10, start="2026-05-01", rate=0.02)
    daily["interval_minutes"] = 1440
    blended = pd.concat([daily, fine], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    fine_only = bars_module.realized_volatility(fine, 300)

    assert bars_module.realized_volatility(blended, 300) == pytest.approx(fine_only)


def test_the_smoothing_span_holds_its_wall_clock_length_across_grids() -> None:
    """45 minutes is three points at 15m and nine at 5m, and both mean three-quarters of an hour."""
    fine = _path("5min", 145)
    coarse = fine.iloc[::3].reset_index(drop=True)

    fine_path = bars_module.return_path(fine, 60, span_minutes=45)
    coarse_path = bars_module.return_path(coarse, 60, span_minutes=45)

    assert len(fine_path) == 9
    assert len(coarse_path) == 3
    # Different sample counts, same span -- so the endpoints line up.
    assert fine_path[-1] == pytest.approx(coarse_path[-1], abs=1e-12)
    assert fine_path[0] == pytest.approx(coarse_path[0], abs=1e-12)

    assert bars_module.ema(fine_path, 45, 5) == pytest.approx(
        bars_module.ema(coarse_path, 45, 15), rel=0.05
    )


def test_coverage_is_measured_in_minutes_spanned() -> None:
    assert bars_module.coverage_minutes(_path("5min", 13)) == 60
    assert bars_module.coverage_minutes(_path("15min", 5)) == 60
    assert bars_module.coverage_minutes(pd.DataFrame()) == 0


@pytest.mark.parametrize(
    "bars_at_15m, minutes",
    [(4, 60), (10, 150), (16, 240), (13, 195), (26, 390), (78, 1170), (80, 1200), (320, 4800)],
)
def test_the_converted_horizons_pick_exactly_the_bars_they_used_to(bars_at_15m: int, minutes: int) -> None:
    """The conversion is faithful, not merely plausible.

    Every horizon in this project was a bar count on a 15-minute grid, and each one was
    replaced by its wall-clock equivalent. On that same grid the two must select the identical
    observation -- otherwise the tuning behind those numbers no longer applies to anything.
    """
    history = _path("15min", 400, rate=0.0007)
    closes = history["close"]

    positional = float(closes.iloc[-1]) / float(closes.iloc[-bars_at_15m - 1]) - 1.0

    assert bars_module.return_over_minutes(history, minutes) == pytest.approx(positional, abs=1e-12)
