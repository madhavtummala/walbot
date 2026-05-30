from __future__ import annotations

from src.config import Config
import pandas as pd

from src.invest_spy import InvestSpyConfig, classify_spy_state, compute_invest_spy_price_features, decide_invest_spy_weights


def test_invest_spy_config_loads_state_knobs() -> None:
    runtime = Config(
        algorithm_configs={
            "invest_spy": {
                "spy_symbol": "SPY",
                "flat_equity_income_exposure": 0.4,
                "max_crisis_hedge_exposure": 0.2,
                "sentiment_negative": -0.15,
            }
        }
    )

    config = InvestSpyConfig.from_runtime_config(runtime)

    assert config.spy_symbol == "SPY"
    assert config.flat_equity_income_exposure == 0.4
    assert config.max_crisis_hedge_exposure == 0.2
    assert config.sentiment_negative == -0.15


def test_invest_spy_price_features_use_intraday_meso_and_daily_macro() -> None:
    config = InvestSpyConfig(
        micro_momentum_lookback_bars=2,
        meso_momentum_lookback_bars=4,
        macro_trend_lookback_days=3,
    )
    intraday = pd.DataFrame({"close": [100, 101, 103, 106, 110, 115]})
    daily = pd.DataFrame({"close": [100, 101, 102, 104, 108]})

    features = compute_invest_spy_price_features("SPY", intraday, daily, config)

    assert features["micro_return"] == 115 / 106 - 1
    assert features["meso_return"] == 115 / 101 - 1
    assert features["macro_return"] == 108 / 101 - 1
    assert features["nano_return"] == features["micro_return"]


def test_spy_state_classifies_growth_pullback_flat_falling_and_crisis() -> None:
    config = InvestSpyConfig()

    assert classify_spy_state({"macro_return": 0.04, "meso_return": 0.01, "micro_return": 0.002}, 0.0, config) == "GROWING"
    assert classify_spy_state({"macro_return": 0.04, "meso_return": 0.01, "micro_return": -0.01}, 0.0, config) == "PULLBACK"
    assert classify_spy_state({"macro_return": 0.01, "meso_return": 0.001, "micro_return": 0.0}, 0.0, config) == "FLAT"
    assert classify_spy_state({"macro_return": 0.01, "meso_return": -0.011, "micro_return": 0.0}, 0.0, config) == "FALLING"
    assert classify_spy_state({"macro_return": 0.01, "meso_return": -0.03, "micro_return": -0.02}, 0.0, config) == "CRISIS"


def test_invest_spy_growth_allocates_to_spy() -> None:
    config = InvestSpyConfig(max_gross_exposure=1.0, max_single_position_weight=1.0)
    scores = {
        "SPY": {"symbol": "SPY", "score": 0.5, "macro_trend_ok": True, "meso_return": 0.01},
        "XYLD": {"symbol": "XYLD", "score": 0.2, "macro_trend_ok": True, "meso_return": 0.0},
    }

    weights = decide_invest_spy_weights(scores, "GROWING", config)

    assert weights["SPY"] == 1.0
    assert weights["XYLD"] == 0.0


def test_invest_spy_flat_uses_income_and_defensive() -> None:
    config = InvestSpyConfig(flat_equity_income_exposure=0.5, max_defensive_positions=1)
    scores = {
        "SPY": {"symbol": "SPY", "score": 0.1, "macro_trend_ok": True, "meso_return": 0.0},
        "XYLD": {"symbol": "XYLD", "score": 0.3, "macro_trend_ok": True, "meso_return": 0.001},
        "BIL": {"symbol": "BIL", "score": 0.1, "macro_trend_ok": True, "meso_return": 0.0},
    }

    weights = decide_invest_spy_weights(scores, "FLAT", config)

    assert weights["SPY"] == 0.0
    assert weights["XYLD"] == 0.5
    assert weights["BIL"] == 0.5


def test_invest_spy_crisis_caps_hedge_and_uses_defensive() -> None:
    config = InvestSpyConfig(max_crisis_hedge_exposure=0.15, max_crisis_hedge_weight=0.10, max_defensive_positions=1)
    scores = {
        "SPY": {"symbol": "SPY", "score": -0.3, "macro_trend_ok": False, "meso_return": -0.03},
        "SH": {"symbol": "SH", "score": 0.6, "macro_trend_ok": True, "meso_return": 0.02},
        "VXX": {"symbol": "VXX", "score": 0.5, "macro_trend_ok": False, "meso_return": 0.03},
        "BIL": {"symbol": "BIL", "score": 0.1, "macro_trend_ok": True, "meso_return": 0.0},
    }

    weights = decide_invest_spy_weights(scores, "CRISIS", config)

    assert weights["SH"] == 0.10
    assert weights["VXX"] == 0.0
    assert weights["BIL"] == 0.9


def test_invest_spy_falling_uses_configured_safe_defensive_subset() -> None:
    config = InvestSpyConfig(
        defensive_universe=["BIL"],
        max_defensive_positions=2,
    )
    scores = {
        "SPY": {"symbol": "SPY", "score": -0.2, "macro_trend_ok": False, "meso_return": -0.02},
        "BIL": {"symbol": "BIL", "score": 0.1, "macro_trend_ok": True, "meso_return": 0.0},
        "TLT": {"symbol": "TLT", "score": 0.8, "macro_trend_ok": True, "meso_return": 0.02},
        "GLD": {"symbol": "GLD", "score": 0.7, "macro_trend_ok": True, "meso_return": 0.02},
    }

    weights = decide_invest_spy_weights(scores, "FALLING", config)

    assert weights["BIL"] == 1.0
