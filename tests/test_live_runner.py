from __future__ import annotations

import pandas as pd

from src.execution import live_runner as live_runner
from src.core.config import Config
from src.algorithms.registry import get_algorithm_class


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
    captured = {}

    monkeypatch.setattr(live_runner, "configure_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        live_runner,
        "get_config",
        lambda **_kwargs: Config(
            symbols=["AAA"],
            kill_switch=False,
            max_weight_per_symbol=0.5,
            max_portfolio_exposure=0.8,
            max_longs=1,
            cash_buffer=0.0,
        ),
    )
    monkeypatch.setattr(
        live_runner,
        "load_controls",
        lambda: {"algorithm_enabled": True, "active_strategy": "risk_parity"},
    )
    monkeypatch.setattr(live_runner, "create_trading_client", lambda config: object())
    monkeypatch.setattr(live_runner, "is_market_open", lambda client: True)
    monkeypatch.setattr(live_runner, "create_data_client", lambda config: object())
    monkeypatch.setattr(live_runner, "get_account_equity", lambda client: 1_000.0)
    monkeypatch.setattr(live_runner, "get_positions", lambda client: {})
    monkeypatch.setattr(live_runner, "fetch_daily_bars", lambda *args, **kwargs: {"AAA": _bars()})
    monkeypatch.setattr(
        live_runner,
        "fetch_latest_market_quotes",
        lambda symbols, config, data_client=None: {symbol: {"price": 100.0} for symbol in symbols},
    )
    monkeypatch.setattr(live_runner, "get_latest_price", lambda symbol, client, data_feed=None: 100.0)
    monkeypatch.setattr(live_runner, "log_signals", lambda signals, prices: captured.update({"signals": signals}))
    monkeypatch.setattr(live_runner, "log_portfolio", lambda weights, equity: captured.update({"weights": weights}))
    monkeypatch.setattr(live_runner, "log_orders", lambda orders: None)
    monkeypatch.setattr(
        live_runner,
        "sync_positions_to_targets",
        lambda _client, _prices, _positions, weights, *_args, **_kwargs: captured.update({"orders_weights": weights}) or [],
    )

    live_runner.main()

    assert captured["signals"]["AAA"]["signal"] == 1
    assert captured["weights"]["AAA"] == 0.5
    assert captured["orders_weights"]["AAA"] == 0.5


def test_live_runner_exits_when_market_clock_is_closed(monkeypatch) -> None:
    called = {"data_client": False}

    monkeypatch.setattr(live_runner, "configure_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(live_runner, "get_config", lambda **_kwargs: Config(kill_switch=False))
    monkeypatch.setattr(
        live_runner,
        "load_controls",
        lambda: {"algorithm_enabled": True, "active_strategy": "momentum_social"},
    )
    monkeypatch.setattr(live_runner, "create_trading_client", lambda config: object())
    monkeypatch.setattr(live_runner, "is_market_open", lambda client: False)
    monkeypatch.setattr(live_runner, "create_data_client", lambda config: called.__setitem__("data_client", True))

    live_runner.run_once()

    assert not called["data_client"]
