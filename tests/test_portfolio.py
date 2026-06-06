from __future__ import annotations

from src.core.portfolio import compute_target_weights


def test_compute_target_weights_uses_scores_caps_and_risk() -> None:
    signals = {
        "AAA": {"signal": 1, "score": 0.80, "realized_vol": 0.20},
        "BBB": {"signal": 1, "score": 0.40, "realized_vol": 0.10},
        "CCC": {"signal": 0, "score": 0.99, "realized_vol": 0.10},
    }

    weights = compute_target_weights(
        signals,
        max_weight_per_symbol=0.40,
        max_portfolio_exposure=0.80,
        max_longs=2,
        target_annual_vol=None,
    )

    assert weights["CCC"] == 0.0
    assert weights["AAA"] <= 0.40
    assert weights["BBB"] <= 0.40
    assert round(sum(weights.values()), 6) == 0.80


def test_compute_target_weights_scales_down_high_estimated_vol() -> None:
    signals = {
        "AAA": {"signal": 1, "score": 0.80, "realized_vol": 0.60},
        "BBB": {"signal": 1, "score": 0.40, "realized_vol": 0.50},
    }

    weights = compute_target_weights(
        signals,
        max_weight_per_symbol=0.60,
        max_portfolio_exposure=1.00,
        target_annual_vol=0.20,
    )

    assert sum(weights.values()) < 1.0
    assert all(weight >= 0.0 for weight in weights.values())
