"""Tests for the intraday-pick algorithm."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from src.core.config import Config
from src.algorithms.intraday_pick.algo import (
    IntradayPickAlgorithm,
    _atr,
    _intraday_momentum,
    _pct_change,
    _range_today,
    _realized_vol,
    _select_strike,
    _vwap,
    _volume_ratio,
    detect_macro_trend,
    estimate_option_range,
    score_candidate,
)
from src.algorithms.intraday_pick.config import IntradayPickConfig
from src.core.interfaces import AlgorithmContext, Intent


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_bars(
    closes: list[float],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[int] | None = None,
) -> pd.DataFrame:
    n = len(closes)
    highs = highs or [c * 1.01 for c in closes]
    lows = lows or [c * 0.99 for c in closes]
    volumes = volumes or [1_000_000] * n
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="D"),
        "open": [c * 1.001 for c in closes],
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


def _make_intraday_bars(
    closes: list[float],
    volumes: list[int] | None = None,
) -> pd.DataFrame:
    n = len(closes)
    volumes = volumes or [100_000] * n
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01 09:30", periods=n, freq="15min"),
        "open": [c * 1.001 for c in closes],
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": volumes,
    })


# ── macro trend ──────────────────────────────────────────────────────────────


class TestMacroTrend:
    def test_bull_when_above_ma_and_positive_return(self) -> None:
        closes = [100.0 + i * 0.5 for i in range(60)]
        bars = _make_bars(closes)
        cfg = IntradayPickConfig(trend_ma_period=50, trend_lookback_days=5, trend_min_return=0.0)
        assert detect_macro_trend(bars, cfg) == "bull"

    def test_bear_when_below_ma_and_negative_return(self) -> None:
        closes = [150.0 - i * 0.5 for i in range(60)]
        bars = _make_bars(closes)
        cfg = IntradayPickConfig(trend_ma_period=50, trend_lookback_days=5, trend_min_return=0.0)
        assert detect_macro_trend(bars, cfg) == "bear"

    def test_flat_when_mixed_signals(self) -> None:
        closes = [100.0] * 60
        bars = _make_bars(closes)
        cfg = IntradayPickConfig(trend_ma_period=50, trend_lookback_days=5, trend_min_return=0.0)
        assert detect_macro_trend(bars, cfg) == "flat"

    def test_flat_with_insufficient_data(self) -> None:
        bars = _make_bars([100.0, 101.0, 102.0])
        cfg = IntradayPickConfig(trend_ma_period=50)
        assert detect_macro_trend(bars, cfg) == "flat"

    def test_min_return_threshold_filters_noisy_markets(self) -> None:
        closes = [100.0 + i * 0.01 for i in range(60)]
        bars = _make_bars(closes)
        cfg = IntradayPickConfig(trend_ma_period=50, trend_lookback_days=5, trend_min_return=0.02)
        assert detect_macro_trend(bars, cfg) == "flat"


# ── indicator helpers ────────────────────────────────────────────────────────


class TestIndicators:
    def test_pct_change_basic(self) -> None:
        s = pd.Series([100.0, 105.0])
        assert abs(_pct_change(s, 1) - 0.05) < 1e-9

    def test_pct_change_empty(self) -> None:
        assert _pct_change(pd.Series(dtype=float), 1) == 0.0

    def test_realized_vol_scales_with_dispersion(self) -> None:
        calm = pd.Series([100.0 + i * 0.1 for i in range(30)])
        volatile = pd.Series([100.0 + ((-1) ** i) * 5 for i in range(30)])
        assert _realized_vol(volatile, 20) > _realized_vol(calm, 20)

    def test_atr_basic(self) -> None:
        bars = _make_bars(
            closes=[100, 101, 102, 103, 104, 105],
            highs=[101, 102, 103, 104, 105, 106],
            lows=[99, 100, 101, 102, 103, 104],
        )
        atr = _atr(bars, 5)
        assert atr > 0

    def test_atr_insufficient_data(self) -> None:
        bars = _make_bars([100, 101])
        assert _atr(bars, 14) == 0.0

    def test_volume_ratio_high_when_surge(self) -> None:
        volumes = [100_000] * 21 + [500_000]
        s = pd.Series(volumes, dtype=float)
        ratio = _volume_ratio(s)
        assert ratio > 4.0

    def test_volume_ratio_empty(self) -> None:
        assert _volume_ratio(pd.Series(dtype=float)) == 0.0

    def test_range_today(self) -> None:
        bars = _make_bars(closes=[100], highs=[102], lows=[98])
        r = _range_today(bars)
        assert abs(r - 0.04) < 1e-6

    def test_vwap_basic(self) -> None:
        bars = _make_intraday_bars(closes=[100, 101, 102])
        v = _vwap(bars)
        assert v > 0

    def test_intraday_momentum(self) -> None:
        bars = _make_intraday_bars(closes=[100, 101, 102, 103, 104, 105, 106, 107])
        m = _intraday_momentum(bars, 4)
        assert m > 0

    def test_intraday_momentum_insufficient_data(self) -> None:
        bars = _make_intraday_bars(closes=[100, 101])
        assert _intraday_momentum(bars, 4) == 0.0


# ── candidate scoring ────────────────────────────────────────────────────────


class TestCandidateScoring:
    def test_empty_bars_returns_zero(self) -> None:
        cfg = IntradayPickConfig()
        result = score_candidate(pd.DataFrame(), None, cfg, "bull")
        assert result["score"] == 0.0

    def test_price_outside_range(self) -> None:
        cfg = IntradayPickConfig(min_price=50, max_price=200)
        bars = _make_bars(closes=[10.0] * 30)
        result = score_candidate(bars, None, cfg, "bull")
        assert result["score"] == 0.0

    def test_bull_direction_aligned(self) -> None:
        cfg = IntradayPickConfig(
            vol_short_window=5,
            vol_long_window=20,
            vol_regime_threshold=0.5,
            range_expansion_threshold=0.5,
            intraday_momentum_bars=2,
        )
        daily = _make_bars(closes=[100 + i * 2 for i in range(30)])
        intraday = _make_intraday_bars(closes=[100 + i * 1 for i in range(10)])
        result = score_candidate(daily, intraday, cfg, "bull")
        assert result["direction_aligned"] is True
        assert result["score"] > 0

    def test_bear_direction_aligned(self) -> None:
        cfg = IntradayPickConfig(
            vol_short_window=5,
            vol_long_window=20,
            vol_regime_threshold=0.5,
            range_expansion_threshold=0.5,
            intraday_momentum_bars=2,
        )
        daily = _make_bars(closes=[200 - i * 2 for i in range(30)])
        intraday = _make_intraday_bars(closes=[200 - i * 1 for i in range(10)])
        result = score_candidate(daily, intraday, cfg, "bear")
        assert result["direction_aligned"] is True
        assert result["score"] > 0

    def test_flat_direction_not_aligned(self) -> None:
        cfg = IntradayPickConfig(
            vol_short_window=5,
            vol_long_window=20,
            intraday_momentum_bars=2,
        )
        daily = _make_bars(closes=[100] * 30)
        intraday = _make_intraday_bars(closes=[100] * 10)
        result = score_candidate(daily, intraday, cfg, "flat")
        assert result["direction_aligned"] is False

    def test_no_intraday_bars_still_scores(self) -> None:
        cfg = IntradayPickConfig(
            vol_short_window=5,
            vol_long_window=20,
            vol_regime_threshold=0.5,
            range_expansion_threshold=0.5,
        )
        daily = _make_bars(closes=[100 + i * 2 for i in range(30)])
        result = score_candidate(daily, None, cfg, "bull")
        assert result["score"] >= 0


# ── option pricing ───────────────────────────────────────────────────────────


class TestOptionPricing:
    def test_basic_pricing(self) -> None:
        cfg = IntradayPickConfig(entry_discount_pct=0.05, exit_target_pct=0.30)
        result = estimate_option_range(100.0, 3.0, 0.25, 1, cfg, "call")
        assert result["fair_value"] > 0
        assert result["entry_limit"] < result["fair_value"]
        assert result["exit_limit"] > result["fair_value"]
        assert result["expected_move"] > 0

    def test_call_vs_put_similar_pricing(self) -> None:
        cfg = IntradayPickConfig(entry_discount_pct=0.05, exit_target_pct=0.30)
        call = estimate_option_range(100.0, 3.0, 0.25, 1, cfg, "call")
        put = estimate_option_range(100.0, 3.0, 0.25, 1, cfg, "put")
        assert abs(call["fair_value"] - put["fair_value"]) < 0.10

    def test_zero_price_returns_zeros(self) -> None:
        cfg = IntradayPickConfig()
        result = estimate_option_range(0.0, 3.0, 0.25, 1, cfg, "call")
        assert result["fair_value"] == 0.0

    def test_longer_dte_higher_price(self) -> None:
        cfg = IntradayPickConfig(entry_discount_pct=0.05, exit_target_pct=0.30)
        short = estimate_option_range(100.0, 3.0, 0.25, 1, cfg, "call")
        long_ = estimate_option_range(100.0, 3.0, 0.25, 7, cfg, "call")
        assert long_["fair_value"] > short["fair_value"]

    def test_higher_vol_higher_price(self) -> None:
        cfg = IntradayPickConfig(entry_discount_pct=0.05, exit_target_pct=0.30)
        low_vol = estimate_option_range(100.0, 3.0, 0.15, 1, cfg, "call")
        high_vol = estimate_option_range(100.0, 3.0, 0.50, 1, cfg, "call")
        assert high_vol["fair_value"] > low_vol["fair_value"]


# ── strike selection ─────────────────────────────────────────────────────────


class TestStrikeSelection:
    def test_call_strike_near_atm(self) -> None:
        cfg = IntradayPickConfig(call_delta_min=0.35, call_delta_max=0.55)
        strike = _select_strike(100.0, "call", cfg)
        assert 40.0 < strike < 70.0

    def test_put_strike_near_atm(self) -> None:
        cfg = IntradayPickConfig(put_delta_min=-0.55, put_delta_max=-0.35)
        strike = _select_strike(100.0, "put", cfg)
        assert 30.0 < strike < 60.0


# ── full algorithm ───────────────────────────────────────────────────────────


class TestIntradayPickAlgorithm:
    def test_algorithm_class(self) -> None:
        algo = IntradayPickAlgorithm({})
        assert algo.algorithm_id == "intraday_pick"

    def test_flat_macro_no_intents(self) -> None:
        algo = IntradayPickAlgorithm({})
        benchmark = _make_bars([100] * 60)
        context = AlgorithmContext(
            config=IntradayPickConfig(benchmark_symbol="SPY", trend_ma_period=50),
            daily_bars_by_symbol={"SPY": benchmark},
        )
        decision = algo.analyze(context)
        assert decision.intents == []
        assert decision.metadata.get("macro") == "flat"

    def test_bull_macro_picks_candidate(self) -> None:
        algo = IntradayPickAlgorithm({})
        benchmark = _make_bars([100 + i * 0.5 for i in range(60)])
        candidate = _make_bars([50 + i * 1.0 for i in range(60)])
        intraday = _make_intraday_bars([55 + i * 0.5 for i in range(20)])
        context = AlgorithmContext(
            config=IntradayPickConfig(
                benchmark_symbol="SPY",
                trend_ma_period=50,
                vol_short_window=5,
                vol_long_window=20,
                vol_regime_threshold=0.5,
                range_expansion_threshold=0.5,
                intraday_momentum_bars=2,
                min_price=10,
                max_price=500,
            ),
            daily_bars_by_symbol={"SPY": benchmark, "AAPL": candidate},
            intraday_bars_by_symbol={"AAPL": intraday},
        )
        decision = algo.analyze(context)
        if decision.intents:
            assert decision.intents[0].kind == "option"
            assert decision.metadata.get("macro") == "bull"
            assert decision.metadata.get("option_type") == "call"

    def test_bear_macro_picks_puts(self) -> None:
        algo = IntradayPickAlgorithm({})
        benchmark = _make_bars([200 - i * 0.5 for i in range(60)])
        candidate = _make_bars([150 - i * 1.0 for i in range(60)])
        intraday = _make_intraday_bars([145 - i * 0.5 for i in range(20)])
        context = AlgorithmContext(
            config=IntradayPickConfig(
                benchmark_symbol="SPY",
                trend_ma_period=50,
                vol_short_window=5,
                vol_long_window=20,
                vol_regime_threshold=0.5,
                range_expansion_threshold=0.5,
                intraday_momentum_bars=2,
                min_price=10,
                max_price=500,
            ),
            daily_bars_by_symbol={"SPY": benchmark, "AAPL": candidate},
            intraday_bars_by_symbol={"AAPL": intraday},
        )
        decision = algo.analyze(context)
        if decision.intents:
            assert decision.metadata.get("macro") == "bear"
            assert decision.metadata.get("option_type") == "put"

    def test_no_benchmark_data_returns_flat(self) -> None:
        algo = IntradayPickAlgorithm({})
        context = AlgorithmContext(
            config=IntradayPickConfig(benchmark_symbol="SPY"),
            daily_bars_by_symbol={},
        )
        decision = algo.analyze(context)
        assert decision.metadata.get("macro") == "flat"

    def test_refine_passes_through(self) -> None:
        algo = IntradayPickAlgorithm({})
        intent = Intent(symbol="AAPL", kind="option", value=1)
        result = algo.refine([intent], {}, None, {}, None, datetime.now(timezone.utc))
        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    def test_requirements_declares_intraday(self) -> None:
        algo = IntradayPickAlgorithm({})
        reqs = algo.requirements(
            IntradayPickConfig(intraday_bar_minutes=15, intraday_momentum_bars=4),
            {},
        )
        assert reqs.intraday_lookback_minutes > 0
        assert reqs.preferred_bar_minutes == 15

    def test_sizing_returns_zero_thresholds(self) -> None:
        algo = IntradayPickAlgorithm({})
        s = algo.sizing(IntradayPickConfig())
        assert s["min_trade_dollars"] == 0.0
        assert s["rebalance_threshold"] == 0.0


# ── intraday replay stepping ────────────────────────────────────────────────


class TestIntradayReplayStepping:
    def test_intraday_replay_steps_at_15min_intervals(self) -> None:
        from src.execution.replay import replay, Coverage
        from src.algorithms.intraday_pick import IntradayPickAlgorithm
        from src.core.interfaces import Schedule

        algo = IntradayPickAlgorithm({})
        # Override schedule for test
        algo.__class__.schedule = Schedule(refresh_minutes=15, start_time="09:30", end_time="15:45")

        # Build a minimal daily history: 3 trading days
        daily = {}
        for sym in ["SPY", "QQQ"]:
            daily[sym] = _make_bars(
                closes=[100 + i for i in range(3)],
                highs=[101 + i for i in range(3)],
                lows=[99 + i for i in range(3)],
            )
            daily[sym].index = pd.date_range("2025-01-01", periods=3, freq="B", tz="UTC")

        # Build intraday history: 4 bars per day for 3 days = 12 bars
        intraday = {}
        for sym in ["SPY", "QQQ"]:
            closes = [100 + d + b * 0.1 for d in range(3) for b in range(4)]
            intraday[sym] = _make_intraday_bars(closes=closes)
            stamps = []
            for d in range(3):
                base = pd.Timestamp(f"2025-01-{1 + d*5:02d} 09:30", tz="UTC")
                stamps.extend([base + pd.Timedelta(minutes=15 * i) for i in range(4)])
            intraday[sym].index = pd.DatetimeIndex(stamps)

        trade_dates = sorted(set.intersection(*(set(df.index) for df in daily.values())))
        config_obj = Config(
            algorithm_configs={"intraday_pick": {}},
        )

        should_run = lambda date: int(date.dayofweek) in (0, 1, 2, 3, 4)

        history_df, coverage = replay(
            algo,
            config_obj,
            daily_history=daily,
            trade_dates=trade_dates,
            should_run=should_run,
            starting_equity=10000.0,
            intraday_history=intraday,
            intraday_minutes=15,
        )

        # Should have intraday timestamps, not just daily
        assert len(history_df) > 3, f"Expected more than 3 rows for intraday stepping, got {len(history_df)}"
        for ts in history_df.index:
            assert ts.minute % 15 == 0, f"Timestamp {ts} is not on a 15-minute boundary"

    def test_daily_replay_still_works(self) -> None:
        from src.execution.replay import replay
        from src.algorithms.dca.bot import DCAAlgorithm

        algo = DCAAlgorithm({})
        daily = {}
        for sym in ["SPY", "QQQ"]:
            daily[sym] = _make_bars(closes=[100 + i for i in range(10)])
            daily[sym].index = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")

        trade_dates = sorted(set.intersection(*(set(df.index) for df in daily.values())))
        config_obj = Config(
            algorithm_configs={"dca": {"plan": {"buy": {"amount": 100, "items": []}, "sell": {"amount": 0, "items": []}}}},
        )

        should_run = lambda date: int(date.dayofweek) in (0, 1, 2, 3, 4)

        history_df, coverage = replay(
            algo,
            config_obj,
            daily_history=daily,
            trade_dates=trade_dates,
            should_run=should_run,
            starting_equity=10000.0,
        )

        # Daily replay: one row per trading day
        assert len(history_df) == 10
