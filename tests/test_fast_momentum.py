from __future__ import annotations

import pandas as pd

from src.config import Config
from src.fast_momentum import (
    DefensiveMomentumConfig,
    apply_risk_guards,
    compute_composite_scores,
    compute_market_regime,
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
    config = DefensiveMomentumConfig(daily_abs_momentum_lookback_days=2)

    features = compute_price_features("SPY", _intraday([100, 101, 103, 106, 110, 115, 121]), _daily([100, 101, 103, 106, 110]), config)

    assert features["short_return"] > 0
    assert features["medium_return"] > features["short_return"]
    assert features["daily_return"] > 0
    assert features["daily_trend_ok"] is True
    assert features["realized_volatility"] >= 0


def test_regime_and_weights_select_risk_on_leaders() -> None:
    config = DefensiveMomentumConfig(
        risk_on_universe=["SPY", "QQQ"],
        defensive_universe=["TLT"],
        max_risk_on_positions=1,
        max_single_position_weight=0.25,
    )
    features = {
        "SPY": {"short_return": 0.01, "medium_return": 0.03, "daily_return": 0.03, "daily_trend_ok": True, "realized_volatility": 0.01},
        "QQQ": {"short_return": 0.03, "medium_return": 0.06, "daily_return": 0.04, "daily_trend_ok": True, "realized_volatility": 0.01},
        "TLT": {"short_return": 0.00, "medium_return": 0.00, "daily_return": 0.01, "daily_trend_ok": True, "realized_volatility": 0.01},
    }
    scores = compute_composite_scores(features, {"SPY": 0.4, "QQQ": 0.7, "TLT": 0.0}, config)
    regime, inputs = compute_market_regime(scores["SPY"], 0.4, config)
    weights = decide_target_weights(scores, regime, config)

    assert regime == "RISK_ON"
    assert inputs["market_sentiment"] == 0.4
    assert weights["QQQ"] == 0.25
    assert weights["SPY"] == 0.0
    assert weights["TLT"] == 0.0


def test_cautious_regime_can_mix_strong_risk_on_and_defensive() -> None:
    config = DefensiveMomentumConfig(
        risk_on_universe=["QQQ"],
        defensive_universe=["TLT", "SHY"],
        cautious_min_risk_on_score=1.0,
        cautious_max_risk_on_exposure=0.3,
        max_single_position_weight=0.25,
        max_defensive_positions=1,
    )
    scores = {
        "QQQ": {"symbol": "QQQ", "score": 1.2, "daily_trend_ok": True},
        "TLT": {"symbol": "TLT", "score": 0.5, "daily_trend_ok": True},
        "SHY": {"symbol": "SHY", "score": 0.2, "daily_trend_ok": True},
        "SPY": {"symbol": "SPY", "score": 0.0, "daily_trend_ok": True},
    }

    weights = decide_target_weights(scores, "CAUTIOUS", config)

    assert weights["QQQ"] == 0.25
    assert weights["TLT"] == 0.25
    assert weights["SHY"] == 0.0


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
