from __future__ import annotations
from typing import Any

import pandas as pd

from src.execution import live_runner as live_runner
from src.core.config import Config
from src.algorithms.registry import get_algorithm_class
from src.core.interfaces import AlgorithmDecision, OrderRequest


class FakeBrokerage:
    def __init__(self, is_open: bool = True):
        self._is_open = is_open

    def get_account_state(self) -> dict[str, Any]:
        return {"equity": 1_000.0, "is_market_open": self._is_open}

    def get_positions(self) -> dict[str, int]:
        return {}

    def is_market_open(self) -> bool:
        return self._is_open

    def submit_order(self, request: OrderRequest) -> dict[str, Any]:
        return {"order_id": "order-1"}

    def validate_short_sale_feasibility(self, *a, **kw) -> dict[str, Any]:
        return {"shortable": True, "reason": "ok"}


def _bars() -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=280, freq="B", tz="UTC")
    prices = [100 + index * 0.1 for index in range(len(dates))]
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
            "volume": [1_000_000 + index for index in range(len(dates))],
        }
    )


def test_live_runner_sizing_equity_cap_is_optional() -> None:
    assert live_runner._sizing_equity(Config(algorithm_equity_cap=10_000.0), 50_000.0) == 10_000.0
    assert live_runner._sizing_equity(Config(algorithm_equity_cap=0.0), 50_000.0) == 50_000.0


def test_algorithm_registry_returns_plugin_class() -> None:
    algorithm = get_algorithm_class("risk_parity").from_config(Config(symbols=["AAA"]))

    requirements = algorithm.requirements(Config(symbols=["AAA"]), {})

    assert algorithm.algorithm_id == "risk_parity"
    assert requirements.price_symbols == ["AAA"]
    assert requirements.daily_lookback_days == Config().momentum_lookback_days


def test_live_runner_uses_selected_template_strategy(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(live_runner, "configure_logging", lambda *a, **kw: None)
    monkeypatch.setattr(
        live_runner,
        "get_config",
        lambda **kw: Config(
            symbols=["AAA"],
            kill_switch=False,
            max_weight_per_symbol=0.5,
            max_portfolio_exposure=0.8,
            max_longs=1,
            cash_buffer=0.0,
        ),
    )
    monkeypatch.setattr(
        live_runner, "load_controls", lambda: {"algorithm_enabled": True, "active_strategy": "risk_parity"}
    )

    fake_registry = {"alpaca": lambda config: FakeBrokerage(is_open=True)}
    monkeypatch.setattr(live_runner, "BROKERAGE_REGISTRY", fake_registry)
    monkeypatch.setattr(live_runner, "create_data_client", lambda config: object())

    fake_decision = AlgorithmDecision(
        target_weights={"AAA": 0.5},
        signals={"AAA": {"signal": 1, "score": 1.0}},
        metadata={},
    )

    def fake_pipeline(strategy, config, brokerage, data_client, current_positions, equity):
        captured.update({"strategy": strategy, "equity": equity})
        return {
            "decision": fake_decision,
            "latest_prices": {"AAA": 100.0},
            "current_positions": current_positions,
            "planned_orders": [],
            "equity": equity,
            "requirements": type("R", (), {"paper_only": False})(),
        }

    monkeypatch.setattr(live_runner, "_run_pipeline", fake_pipeline)
    monkeypatch.setattr(live_runner, "log_signals", lambda signals, prices: captured.update({"signals": signals}))
    monkeypatch.setattr(live_runner, "log_portfolio", lambda weights, equity: captured.update({"weights": weights}))
    monkeypatch.setattr(live_runner, "log_orders", lambda orders: None)
    monkeypatch.setattr(
        live_runner, "sync_positions_to_targets",
        lambda brokerage, _prices, _positions, weights, *a, **kw: captured.update({"orders_weights": weights}) or [],
    )

    live_runner.main()

    assert captured["signals"]["AAA"]["signal"] == 1
    assert captured["weights"]["AAA"] == 0.5
    assert captured["orders_weights"]["AAA"] == 0.5


def test_live_runner_exits_when_market_clock_is_closed(monkeypatch) -> None:
    called = {"data_client": False}

    monkeypatch.setattr(live_runner, "configure_logging", lambda *a, **kw: None)
    monkeypatch.setattr(live_runner, "get_config", lambda **kw: Config(kill_switch=False))
    monkeypatch.setattr(
        live_runner, "load_controls", lambda: {"algorithm_enabled": True, "active_strategy": "momentum_social"}
    )
    fake_registry = {"alpaca": lambda config: FakeBrokerage(is_open=False)}
    monkeypatch.setattr(live_runner, "BROKERAGE_REGISTRY", fake_registry)
    monkeypatch.setattr(live_runner, "create_data_client", lambda config: called.__setitem__("data_client", True))

    live_runner.run_once()

    assert not called["data_client"]
