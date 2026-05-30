from __future__ import annotations

import pandas as pd

from src.config import Config
from src.fast_momentum import (
    DefensiveMomentumConfig,
    apply_risk_guards,
    compute_composite_scores,
    compute_price_features,
    decide_target_weights,
    get_sentiment_snapshot,
)


def _intraday(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2026-05-22 14:30", periods=len(prices), freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
            "volume": [1_000_000 for _ in prices],
        }
    )


def _daily(prices: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-04-20", periods=len(prices), tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
            "volume": [1_000_000 for _ in prices],
        }
    )


def test_price_features_use_intraday_and_daily_momentum() -> None:
    config = DefensiveMomentumConfig(
        nano_momentum_lookback_bars=2,
        micro_momentum_lookback_bars=4,
        meso_trend_lookback_days=2,
        macro_trend_lookback_days=3,
    )

    features = compute_price_features("SPY", _intraday([100, 101, 103, 106, 110, 115, 121]), _daily([100, 101, 103, 106, 110]), config)

    assert features["nano_return"] > 0
    assert features["micro_return"] > 0
    assert features["meso_return"] > 0
    assert features["macro_return"] > 0
    assert features["macro_trend_ok"] is True
    assert features["realized_volatility"] >= 0


def test_fast_momentum_config_uses_named_horizons() -> None:
    runtime = Config(
        algorithm_configs={
            "fast_momentum": {
                "nano_momentum_lookback_bars": 3,
                "micro_momentum_lookback_bars": 26,
                "meso_trend_lookback_days": 60,
                "macro_trend_lookback_days": 180,
                "max_positions": 4,
                "min_risk_on_micro_return": 0.0,
                "w_price_nano": 0.25,
                "w_price_micro": 0.35,
                "w_price_meso": 0.20,
                "w_price_macro": 0.10,
            }
        }
    )

    config = DefensiveMomentumConfig.from_runtime_config(runtime)

    assert config.nano_momentum_lookback_bars == 3
    assert config.micro_momentum_lookback_bars == 26
    assert config.meso_trend_lookback_days == 60
    assert config.macro_trend_lookback_days == 180
    assert config.max_positions == 4
    assert config.min_risk_on_micro_return == 0.0
    assert config.w_price_nano == 0.25
    assert config.w_price_micro == 0.35
    assert config.w_price_meso == 0.20
    assert config.w_price_macro == 0.10


def test_weights_select_risk_on_leaders_by_symbol_trend_and_score() -> None:
    config = DefensiveMomentumConfig(
        risk_on_universe=["SPY", "QQQ"],
        defensive_universe=["TLT"],
        max_positions=1,
        max_single_position_weight=0.25,
    )
    features = {
        "SPY": {"micro_return": 0.01, "meso_return": 0.03, "macro_return": 0.03, "macro_trend_ok": True, "realized_volatility": 0.01},
        "QQQ": {"micro_return": 0.03, "meso_return": 0.06, "macro_return": 0.04, "macro_trend_ok": True, "realized_volatility": 0.01},
        "TLT": {"micro_return": 0.00, "meso_return": 0.00, "macro_return": 0.01, "macro_trend_ok": True, "realized_volatility": 0.01},
    }
    scores = compute_composite_scores(features, {"SPY": 0.4, "QQQ": 0.7, "TLT": 0.0}, config)
    weights = decide_target_weights(scores, config)

    assert weights["QQQ"] == 0.25
    assert weights["SPY"] == 0.0
    assert weights["TLT"] == 0.0


def test_pullback_uptrend_bonus_rewards_short_term_dip_in_strong_trend() -> None:
    config = DefensiveMomentumConfig(
        w_price_nano=0.25,
        w_price_micro=0.35,
        w_price_meso=0.20,
        w_price_macro=0.10,
        w_sentiment=0.1,
        w_pullback_uptrend=0.25,
        pullback_meso_z_threshold=1.0,
        pullback_nano_z_threshold=-0.5,
        pullback_nano_z_cap=3.0,
        pullback_min_micro_return=0.0,
    )
    features = {
        "XSD": {"nano_return": -0.0218, "micro_return": 0.0007, "meso_return": 0.8519, "macro_return": 0.5519, "macro_trend_ok": True},
        "AIQ": {"nano_return": 0.0025, "micro_return": 0.0333, "meso_return": 0.3549, "macro_return": 0.2549, "macro_trend_ok": True},
        "SPY": {"nano_return": 0.0017, "micro_return": 0.0055, "meso_return": 0.1114, "macro_return": 0.0914, "macro_trend_ok": True},
        "VXX": {"nano_return": -0.0101, "micro_return": -0.0488, "meso_return": -0.2335, "macro_return": -0.3335, "macro_trend_ok": False},
    }

    scores = compute_composite_scores(features, {}, config)

    assert scores["XSD"]["components"]["pullback_uptrend"] > 0.0
    assert scores["AIQ"]["components"]["pullback_uptrend"] == 0.0
    assert scores["VXX"]["components"]["pullback_uptrend"] == 0.0


def test_weights_fall_back_to_defensive_when_no_risk_on_qualifies() -> None:
    config = DefensiveMomentumConfig(
        risk_on_universe=["QQQ"],
        defensive_universe=["TLT", "SHY"],
        max_single_position_weight=0.25,
        max_positions=1,
    )
    scores = {
        "QQQ": {"symbol": "QQQ", "score": 1.2, "macro_trend_ok": False},
        "TLT": {"symbol": "TLT", "score": 0.5, "macro_trend_ok": True},
        "SHY": {"symbol": "SHY", "score": 0.2, "macro_trend_ok": True},
        "SPY": {"symbol": "SPY", "score": 0.0, "macro_trend_ok": True},
    }

    weights = decide_target_weights(scores, config)

    assert weights["QQQ"] == 0.0
    assert weights["TLT"] == 0.25
    assert weights["SHY"] == 0.0


def test_risk_on_selection_requires_micro_floor() -> None:
    config = DefensiveMomentumConfig(
        risk_on_universe=["QQQ"],
        defensive_universe=["BIL"],
        min_risk_on_micro_return=0.0,
        max_single_position_weight=0.25,
    )
    scores = {
        "SPY": {"symbol": "SPY", "score": 0.2, "macro_trend_ok": True, "macro_return": 0.03, "micro_return": 0.01},
        "QQQ": {"symbol": "QQQ", "score": 1.2, "macro_trend_ok": True, "macro_return": 0.04, "micro_return": -0.001},
        "BIL": {"symbol": "BIL", "score": 0.1, "macro_trend_ok": True, "micro_return": 0.0},
    }

    weights = decide_target_weights(scores, config)

    assert weights["QQQ"] == 0.0
    assert weights["BIL"] == 0.25


def test_dynamic_weights_follow_scores_with_position_caps() -> None:
    config = DefensiveMomentumConfig(
        risk_on_universe=["QQQ", "XSD"],
        defensive_universe=["BIL"],
        max_positions=3,
        max_gross_exposure=0.60,
        max_single_position_weight=0.25,
    )
    scores = {
        "QQQ": {"symbol": "QQQ", "score": 4.0, "macro_trend_ok": True, "meso_return": 0.02},
        "XSD": {"symbol": "XSD", "score": 2.0, "macro_trend_ok": True, "meso_return": 0.02},
        "BIL": {"symbol": "BIL", "score": 1.0, "macro_trend_ok": True, "meso_return": 0.0},
    }

    weights = decide_target_weights(scores, config)

    assert weights["QQQ"] == 0.25
    assert round(weights["XSD"], 6) == 0.233333
    assert round(weights["BIL"], 6) == 0.116667


def test_risk_guards_scale_high_volatility_and_preserve_small_drifts(monkeypatch) -> None:
    monkeypatch.setattr("src.fast_momentum.intraday_kill_switch_triggered", lambda *_args, **_kwargs: False)
    config = DefensiveMomentumConfig(max_intraday_volatility=0.02, high_volatility_weight_scale=0.5, per_trade_value_min=100.0)
    scores = {
        "QQQ": {"realized_volatility": 0.03},
        "TLT": {"realized_volatility": 0.01},
    }

    guarded = apply_risk_guards({"QQQ": 0.2, "TLT": 0.201}, scores, {"TLT": 0.2}, 10_000.0, config)

    assert guarded["QQQ"] == 0.1
    assert guarded["TLT"] == 0.2


def test_risk_guards_do_not_keep_unselected_positions(monkeypatch) -> None:
    monkeypatch.setattr("src.fast_momentum.intraday_kill_switch_triggered", lambda *_args, **_kwargs: False)
    config = DefensiveMomentumConfig(per_trade_value_min=100.0, rebalance_threshold=0.05)

    guarded = apply_risk_guards(
        {"QQQ": 0.25, "TLT": 0.0},
        {"QQQ": {"realized_volatility": 0.0}, "TLT": {"realized_volatility": 0.0}},
        {"QQQ": 0.25, "TLT": 0.01},
        1_000.0,
        config,
    )

    assert guarded["QQQ"] == 0.25
    assert guarded["TLT"] == 0.0


def test_sentiment_snapshot_defaults_missing_records_to_neutral(monkeypatch) -> None:
    monkeypatch.setattr("src.fast_momentum.fetch_latest_news_sentiment", lambda *_args, **_kwargs: [])

    symbol_sentiment, market_sentiment = get_sentiment_snapshot(["SPY", "QQQ"], 60, Config())

    assert symbol_sentiment == {"SPY": 0.0, "QQQ": 0.0}
    assert market_sentiment == 0.0


def test_sentiment_snapshot_combines_configured_providers(monkeypatch) -> None:
    seen_orders = []

    def fake_fetch(_symbols, config):
        provider = config.news_sentiment_provider_order[0]
        seen_orders.append(provider)
        sentiment = 0.6 if provider == "marketaux" else -0.2
        return [{"symbol": "SPY", "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "sentiment": sentiment}]

    monkeypatch.setattr("src.fast_momentum.fetch_latest_news_sentiment", fake_fetch)

    symbol_sentiment, market_sentiment = get_sentiment_snapshot(
        ["SPY"],
        60,
        Config(news_sentiment_provider_order=["marketaux", "stocktwits"]),
    )

    assert seen_orders == ["marketaux", "stocktwits"]
    assert abs(symbol_sentiment["SPY"] - 0.2) < 1e-12
    assert abs(market_sentiment - 0.2) < 1e-12
