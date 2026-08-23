from __future__ import annotations
from typing import Any

import pandas as pd

from src.execution import live_runner as live_runner
from src.core import pipeline as pipeline
from src.core import market_context as market_context
from src.core.config import Config
from src.algorithms.registry import get_algorithm_class
from src.core.interfaces import AlgorithmPlan, OrderRequest, intents_from_weights


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
    assert pipeline.sizing_equity(Config(algorithm_equity_cap=10_000.0), 50_000.0) == 10_000.0
    assert pipeline.sizing_equity(Config(algorithm_equity_cap=0.0), 50_000.0) == 50_000.0


def test_algorithm_registry_returns_the_algorithm_class() -> None:
    config = Config(symbols=["AAA"])
    algorithm = get_algorithm_class("rally_rotation").from_config(config)

    requirements = algorithm.requirements(config, {"HELD": 1})

    assert algorithm.algorithm_id == "rally_rotation"
    # A held symbol has to be priced even when it is not in the algorithm's own universe.
    assert "HELD" in requirements.price_symbols
    assert requirements.daily_lookback_days > 0


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
        live_runner, "load_controls", lambda: {"algorithm_enabled": True, "active_strategy": "rally_rotation"}
    )

    monkeypatch.setattr(pipeline, "get_brokerage_class", lambda _t: lambda _c: FakeBrokerage(is_open=True))
    monkeypatch.setattr(market_context, "create_data_client", lambda config: object())

    def fake_run_algorithm(strategy, config, **kwargs):
        captured["strategy"] = strategy
        return AlgorithmPlan(
            strategy=strategy,
            intents=intents_from_weights({"AAA": 0.5}),
            signals={"AAA": {"signal": 1, "score": 1.0}},
            latest_prices={"AAA": 100.0},
            metadata={"requirements": type("R", (), {"paper_only": False})()},
        )

    def fake_execute_algorithm(plan, config, brokerage, **kwargs):
        captured["orders_weights"] = plan.target_weights
        return {"final_weights": plan.target_weights, "equity": 1_000.0, "order_results": []}

    monkeypatch.setattr(live_runner, "run_algorithm", fake_run_algorithm)
    monkeypatch.setattr(live_runner, "execute_algorithm", fake_execute_algorithm)
    monkeypatch.setattr(live_runner, "log_signals", lambda signals, prices: captured.update({"signals": signals}))
    monkeypatch.setattr(live_runner, "log_portfolio", lambda weights, equity: captured.update({"weights": weights}))
    monkeypatch.setattr(live_runner, "log_orders", lambda orders: None)

    live_runner.main()

    assert captured["strategy"] == "rally_rotation"
    assert captured["signals"]["AAA"]["signal"] == 1
    assert captured["weights"]["AAA"] == 0.5
    assert captured["orders_weights"]["AAA"] == 0.5


def test_live_runner_exits_when_market_clock_is_closed(monkeypatch) -> None:
    called = {"data_client": False}

    monkeypatch.setattr(live_runner, "configure_logging", lambda *a, **kw: None)
    monkeypatch.setattr(live_runner, "get_config", lambda **kw: Config(kill_switch=False))
    monkeypatch.setattr(
        live_runner, "load_controls", lambda: {"algorithm_enabled": True, "active_strategy": "rally_rotation"}
    )
    monkeypatch.setattr(pipeline, "get_brokerage_class", lambda _t: lambda _c: FakeBrokerage(is_open=False))
    monkeypatch.setattr(market_context, "create_data_client", lambda config: called.__setitem__("data_client", True))

    live_runner.run_once()

    assert not called["data_client"]


def test_every_retired_id_still_resolves_to_its_algorithm() -> None:
    """Retired ids still resolve to their current algorithm."""
    from src.algorithms.registry import ALGORITHM_ALIASES, canonical_algorithm_id

    for retired, current in ALGORITHM_ALIASES.items():
        assert canonical_algorithm_id(retired) == current
        assert get_algorithm_class(retired) is get_algorithm_class(current)


def test_renamed_algorithm_still_reads_tuning_saved_under_an_older_id() -> None:
    """The reverse alias map has to keep every retired id so saved tuning does not revert to defaults."""
    from src.algorithms.registry import ALGORITHM_ALIASES, LEGACY_ALGORITHM_IDS, canonical_algorithm_id

    for retired, current in ALGORITHM_ALIASES.items():
        assert canonical_algorithm_id(retired) == current
        assert get_algorithm_class(retired) is get_algorithm_class(current)
        assert retired in LEGACY_ALGORITHM_IDS[current]
