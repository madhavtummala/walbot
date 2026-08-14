from __future__ import annotations

import pandas as pd
import pytest

from src.api import api_payloads as api_payloads
from src.core import market_context
from src.algorithms import fast_momentum as fast_momentum
from src.algorithms import dca as dca
from src.algorithms.dca import bot as dca_bot
from src.core.config import Config
from src.core.interfaces import Schedule
from src.execution import replay as replay_module
from src.algorithms.registry import canonical_algorithm_id
from src.api.api_payloads import (
    backtest_payload,
    controls_payload,
    dca_payload,
    status_payload,
    strategy_signals_payload,
    universe_payload,
)


def _strategy_bars(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=len(prices), freq="B", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": prices,
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": [price * 1.002 for price in prices],
            "volume": [1_000_000 + index for index in range(len(prices))],
        }
    )


def _dated_bars(dates: list[str], price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(dates, utc=True),
            "open": [price for _date in dates],
            "high": [price * 1.01 for _date in dates],
            "low": [price * 0.99 for _date in dates],
            "close": [price for _date in dates],
            "volume": [1_000_000 for _date in dates],
        }
    )


def _fresh_bars(periods: int = 280, price: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=periods, tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": [price for _date in dates],
            "high": [price * 1.01 for _date in dates],
            "low": [price * 0.99 for _date in dates],
            "close": [price for _date in dates],
            "volume": [10_000 for _date in dates],
        }
    )


def test_status_payload_redacts_secret_values() -> None:
    payload = status_payload()

    assert isinstance(payload["config"]["alpaca_api_key"], bool)
    assert isinstance(payload["config"]["alpaca_api_secret"], bool)
    assert isinstance(payload["config"]["alpha_vantage_api_key"], bool)


def test_universe_payload_returns_configured_rows() -> None:
    payload = universe_payload()

    assert payload["count"] > 0
    assert {"symbol", "name", "bucket", "tradable", "enabled"} <= set(payload["rows"][0])


def test_apply_universe_payload_writes_approved_subset(tmp_path, monkeypatch) -> None:
    tradables_csv = tmp_path / "tradables.csv"
    universe_yaml = tmp_path / "universe.yaml"
    tradables_csv.write_text(
        "Ticker,Name\nSPY,SPDR S&P 500 ETF Trust\nQQQ,Invesco QQQ Trust\nGLD,SPDR Gold Trust\n",
        encoding="utf-8",
    )
    universe_yaml.write_text(
        f"""
tradable_universe:
  master_list: {tradables_csv}
  symbols:
    - SPY
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_UNIVERSE_FILE", str(universe_yaml))
    monkeypatch.delenv("TRADABLES_CSV", raising=False)

    payload = api_payloads.apply_universe_payload(
        {
            "rows": [
                {"symbol": "SPY", "bucket": "Broad US equity"},
                {"symbol": "QQQ", "bucket": "Large cap growth"},
                {"symbol": "GLD", "bucket": "Gold"},
            ]
        }
    )

    assert payload["saved"] is True
    assert [row["symbol"] for row in payload["universe"]["rows"]] == ["SPY", "QQQ", "GLD"]
    saved_yaml = (tmp_path / "universe.yaml").read_text(encoding="utf-8")
    assert "- SPY" in saved_yaml
    assert "- QQQ" in saved_yaml
    assert "- GLD" in saved_yaml


def test_apply_universe_payload_accepts_symbols_without_master_csv(tmp_path, monkeypatch) -> None:
    universe_yaml = tmp_path / "universe.yaml"
    universe_yaml.write_text(
        """
tradable_universe:
  master_list: missing_tradables.csv
  symbols:
    - SPY
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_UNIVERSE_FILE", str(universe_yaml))
    monkeypatch.delenv("TRADABLES_CSV", raising=False)

    payload = api_payloads.apply_universe_payload({"symbols": ["AAA", "BBB", "CCC"]})

    assert [row["symbol"] for row in payload["universe"]["rows"]] == ["AAA", "BBB", "CCC"]


def test_recommend_universe_payload_scores_fresh_candidates(tmp_path, monkeypatch) -> None:
    tradables_csv = tmp_path / "tradables.csv"
    universe_yaml = tmp_path / "universe.yaml"
    tradables_csv.write_text(
        "Ticker,Name\nSPY,SPDR S&P 500 ETF Trust\nQQQ,Invesco QQQ Trust\nIBIT,iShares Bitcoin Trust\nGLD,SPDR Gold Trust\nTLT,iShares 20+ Year Treasury Bond ETF\n",
        encoding="utf-8",
    )
    universe_yaml.write_text(
        f"""
tradable_universe:
  master_list: {tradables_csv}
  symbols:
    - SPY
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_UNIVERSE_FILE", str(universe_yaml))
    monkeypatch.delenv("TRADABLES_CSV", raising=False)
    monkeypatch.setattr(api_payloads, "create_data_client", lambda config: object())

    def fake_fetch_daily_bars(symbols, **kwargs):
        assert kwargs["force_refresh"] is True
        return {symbol: _fresh_bars(price=100.0 + index) for index, symbol in enumerate(symbols)}

    monkeypatch.setattr(api_payloads, "fetch_daily_bars", fake_fetch_daily_bars)

    payload = api_payloads.recommend_universe_payload({"max_symbols": 5, "refresh": True})

    assert [row["symbol"] for row in payload["rows"]] == ["SPY", "GLD", "TLT", "QQQ", "IBIT"]
    assert payload["eligible_count"] == 5


def test_dca_payload_returns_plan_and_preview_shape() -> None:
    payload = dca_payload()

    assert "plan" in payload
    assert "available" in payload
    assert "preview" in payload
    assert {"max_item_amount", "buy", "sell"} <= set(payload["plan"])
    # Cadence and the on/off switch are not the plan's to carry any more.
    assert not {"enabled", "frequency", "schedule_pattern"} & set(payload["plan"])


def test_controls_payload_returns_persisted_choices() -> None:
    payload = controls_payload()

    assert {"equities", "options", "algorithm_enabled", "options_trading_enabled", "active_strategy", "options_strategy"} <= set(payload["controls"])
    assert {"algorithm", "options"} <= set(payload["bot"])
    assert "dca" not in payload["bot"]


def test_fast_momentum_reason_uses_configured_defensive_symbols() -> None:
    row = {"symbol": "XYLD", "macro_trend_ok": True}
    config = fast_momentum.DefensiveMomentumConfig(defensive_universe=["BIL", "XYLD"])

    reason = fast_momentum._defensive_momentum_reason(row, 0.25, config)

    assert reason == "Top Rank"


def test_fast_momentum_reason_describes_risk_on_rank_cutoff() -> None:
    config = fast_momentum.DefensiveMomentumConfig(
        risk_on_universe=["XSD", "AIQ"],
        defensive_universe=["BIL", "XYLD"],
        max_positions=4,
        min_risk_on_micro_return=0.0,
    )

    assert fast_momentum._defensive_momentum_reason({"symbol": "AIQ", "macro_trend_ok": True}, 0.0, config) == "No rank slot"
    assert (
        fast_momentum._defensive_momentum_reason({"symbol": "XSD", "macro_trend_ok": True, "micro_return": -0.001}, 0.0, config)
        == "Micro too low"
    )


def test_save_controls_payload_does_not_wake_runtimes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TRADING_ALGORITHM_BOT_FILE", str(tmp_path / "algorithm_bot.yaml"))
    monkeypatch.setenv("TRADING_OPTIONS_BOT_FILE", str(tmp_path / "options_bot.yaml"))

    payload = api_payloads.save_controls_payload(
        {
            "controls": {
                "active_strategy": "none",
                "algorithm_enabled": True,
                "options_strategy": "none",
                "options_trading_enabled": True,
            }
        }
    )

    # A saved "none" resolves to DCA but lands off: it used to force the bot idle whatever
    # the enabled flag said, so carrying that flag over would start trading on upgrade.
    assert payload["controls"]["active_strategy"] == "dca"
    assert payload["controls"]["algorithm_enabled"] is False
    assert payload["controls"]["options_trading_enabled"] is False
    assert not hasattr(api_payloads.bot_runtime, "wake_algorithm")
    assert not hasattr(api_payloads.bot_runtime, "wake_options")


def test_cache_only_backtest_does_not_compute_without_cached_rows(monkeypatch) -> None:
    def fail_compute(*_args, **_kwargs):
        raise AssertionError("cache-only request should not compute a backtest")

    monkeypatch.setattr(api_payloads, "universe_payload", lambda: {"rows": []})
    monkeypatch.setattr(
        api_payloads,
        "load_dca_plan",
        lambda rows, **kwargs: {"buy": {"items": []}, "sell": {"items": []}},
    )
    monkeypatch.setattr(api_payloads, "_load_backtest_cache", lambda: {"version": 2, "items": {}})
    monkeypatch.setattr(api_payloads, "_compute_backtest", fail_compute)

    payload = backtest_payload({"strategy": "fast_momentum", "period": "6m", "cache_only": True})

    assert payload["strategy"] == "fast_momentum"
    assert payload["period_label"] == "6M"
    assert payload["cached"] is False
    assert "error" in payload


def test_non_refresh_backtest_does_not_compute_without_cached_rows(monkeypatch) -> None:
    def fail_compute(*_args, **_kwargs):
        raise AssertionError("non-refresh request should not compute a backtest")

    monkeypatch.setattr(api_payloads, "universe_payload", lambda: {"rows": []})
    monkeypatch.setattr(
        api_payloads,
        "load_dca_plan",
        lambda rows, **kwargs: {"buy": {"items": []}, "sell": {"items": []}},
    )
    monkeypatch.setattr(api_payloads, "_load_backtest_cache", lambda: {"version": 4, "items": {}})
    monkeypatch.setattr(api_payloads, "_compute_backtest", fail_compute)

    payload = backtest_payload({"strategy": "fast_momentum", "period": "2m"})

    assert payload["strategy"] == "fast_momentum"
    assert payload["cached"] is False
    assert "error" in payload


def test_refresh_backtest_computes_with_market_data_refresh(monkeypatch) -> None:
    calls = []

    def fake_compute(strategy, period, dca_plan):
        calls.append((strategy, period))
        return {
            "strategy": strategy,
            "period": period,
            "period_label": "6M",
            "cached": False,
            "rows": [{"timestamp": "2026-05-01", "equity": 10_000.0}],
        }

    monkeypatch.setattr(api_payloads, "universe_payload", lambda: {"rows": []})
    monkeypatch.setattr(
        api_payloads,
        "load_dca_plan",
        lambda rows, **kwargs: {"buy": {"items": []}, "sell": {"items": []}},
    )
    monkeypatch.setattr(api_payloads, "_load_backtest_cache", lambda: {"version": 4, "items": {}})
    monkeypatch.setattr(api_payloads, "_save_backtest_cache", lambda cache: None)
    monkeypatch.setattr(api_payloads, "_compute_backtest", fake_compute)

    payload = backtest_payload({"strategy": "trend_following", "period": "6m", "refresh": True})

    assert payload["strategy"] == "trend_following"
    assert calls == [("trend_following", "6m")]


def test_backtest_response_explains_profit_loss_breakdown() -> None:
    history = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "equity": 10_000.0,
                "cash": 9_000.0,
                "invested": 1_000.0,
                "positions": {"SPY": 1_000.0},
            },
            {
                "timestamp": pd.Timestamp("2026-01-02", tz="UTC"),
                "equity": 10_461.0,
                "cash": 1_250.0,
                "invested": 9_211.0,
                "positions": {"SPY": 7_000.0, "QQQ": 2_211.0},
            },
        ]
    ).set_index("timestamp")

    payload = api_payloads._backtest_response(
        history,
        strategy="none",
        label="None / DCA",
        period="6m",
        source="dca",
    )

    assert payload["ending_equity"] == 10_461.0
    assert payload["ending_cash"] == 1_250.0
    assert payload["ending_invested"] == 9_211.0
    assert payload["profit_loss"] == 461.0
    assert payload["sizing"]["cash_account_only"] is True
    assert payload["orders"]["capital_limit"] == 10_000.0
    assert payload["orders"]["max_capital_at_work"] == 9_211.0
    assert payload["rows"][-1]["positions"] == {"SPY": 7000.0, "QQQ": 2211.0}


def test_fast_momentum_backtest_honors_configured_max_positions(monkeypatch) -> None:
    symbols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    configured_symbols = sorted([*symbols, "FFF"])
    bars = {symbol: _strategy_bars([100 + index * 0.2 for index in range(280)]) for symbol in configured_symbols}
    captured_fetch = {}
    config = Config(
        symbols=["IGNORED"],
        cash_buffer=0.0,
        transaction_cost_bps=0.0,
        algorithm_configs={
            "fast_momentum": {
                "risk_on_universe": symbols,
                "defensive_universe": ["FFF"],
                "max_positions": 4,
                "max_single_position_weight": 0.25,
                "max_gross_exposure": 1.0,
            }
        },
    )

    monkeypatch.setattr(api_payloads, "get_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(api_payloads, "create_data_client", lambda config: object())

    def fake_fetch_daily_bars(requested_symbols, **kwargs):
        captured_fetch.update({"symbols": requested_symbols, **kwargs})
        return {symbol: bars[symbol] for symbol in requested_symbols}

    monkeypatch.setattr(api_payloads, "fetch_daily_bars", fake_fetch_daily_bars)
    payload = api_payloads._compute_backtest("fast_momentum", "1m", {})
    history = pd.DataFrame(payload["rows"]).set_index("timestamp")

    position_counts = history["positions"].apply(len)
    assert position_counts.max() == 4
    assert all("EEE" not in positions for positions in history["positions"])
    assert captured_fetch["symbols"] == configured_symbols
    assert captured_fetch["lookback_days"] >= 180
    assert captured_fetch["extra_buffer_days"] >= 22 + 10


def test_fast_momentum_intraday_backtest_reads_configured_provider_order(monkeypatch) -> None:
    calls = []
    config = Config(intraday_market_data_provider_order=["finnhub", "yfinance"])

    def fake_read_market_bars(category, provider, symbol, timeframe, **kwargs):
        calls.append((category, provider, symbol, timeframe, kwargs))
        if provider == "finnhub":
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2026-05-01"], utc=True),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "volume": [1000],
                "adjusted_close": [100.5],
            }
        )

    monkeypatch.setattr(replay_module, "read_market_bars", fake_read_market_bars)

    coverage = replay_module.Coverage()
    bars = replay_module._read_intraday(
        ["SPY"],
        pd.Timestamp("2026-05-01", tz="UTC"),
        providers=api_payloads._configured_intraday_providers(config),
        lookback_bars=79,
        bar_minutes=15,
        coverage=coverage,
    )

    assert not bars["SPY"].empty
    assert [call[1] for call in calls] == ["finnhub", "yfinance"]
    # Falling back to a second provider still counts as covered.
    assert coverage.as_dict()["intraday_ratio"] == 1.0
    assert coverage.missing_symbols == set()


def test_spy_rotation_backtest_uses_spy_state_logic(monkeypatch) -> None:
    symbols = ["SPY", "XYLD", "BIL", "SH"]
    bars = {symbol: _strategy_bars([100 + index * 0.4 for index in range(280)]) for symbol in symbols}
    config = Config(
        symbols=["IGNORED"],
        cash_buffer=0.0,
        transaction_cost_bps=0.0,
        algorithm_configs={
            "invest_spy": {
                "spy_symbol": "SPY",
                "equity_income_universe": ["XYLD"],
                "defensive_universe": ["BIL"],
                "crisis_hedge_universe": ["SH"],
                "macro_trend_lookback_days": 60,
                "micro_momentum_lookback_bars": 3,
                "meso_momentum_lookback_bars": 26,
                "max_gross_exposure": 1.0,
                "max_single_position_weight": 1.0,
            }
        },
    )

    monkeypatch.setattr(api_payloads, "get_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(api_payloads, "create_data_client", lambda config: object())
    monkeypatch.setattr(
        api_payloads,
        "fetch_daily_bars",
        lambda symbols, *args, **kwargs: {symbol: bars[symbol] for symbol in symbols},
    )

    payload = api_payloads._compute_backtest("spy_rotation", "6m", {})
    history = pd.DataFrame(payload["rows"]).set_index("timestamp")

    assert history["order_count"].sum() > 0
    assert history["positions"].iloc[-1]["SPY"] > 0
    assert "IGNORED" not in history["positions"].iloc[-1]


def test_none_strategy_resolves_to_dca(monkeypatch) -> None:
    """The retired "None" card always meant "just run DCA", so its saved id maps there."""
    assert canonical_algorithm_id("none") == "dca"


def test_dca_view_states_its_cadence_and_planned_total(monkeypatch) -> None:
    """The dashboard has no cadence control, so the view is where the schedule is readable."""
    plan = {
        "buy": {"items": [{"symbol": "SPY", "amount": 250.0}]},
        "sell": {"items": []},
    }
    monkeypatch.setattr(api_payloads, "universe_payload", lambda: {"rows": [{"symbol": "SPY"}]})
    monkeypatch.setattr(api_payloads, "load_dca_plan", lambda rows, **kwargs: plan)
    monkeypatch.setattr(dca_bot.DCAAlgorithm, "plan", lambda self, config: plan)
    monkeypatch.setattr(dca, "unknown_plan_symbols", lambda *a, **kw: [])
    monkeypatch.setattr(dca_bot, "unknown_plan_symbols", lambda *a, **kw: [])
    monkeypatch.setattr(market_context, "create_data_client", lambda config: object())
    monkeypatch.setattr(market_context, "load_latest_prices", lambda symbols, config, client: {"SPY": 500.0})

    payload = strategy_signals_payload("dca")

    rows = {row["symbol"]: row for row in payload["leaders"]}
    assert rows["SPY"]["reason"].startswith("Accruing")
    assert payload["summary"][0] == {"label": "Mode", "value": "DCA"}
    assert payload["summary"][1] == {"label": "Schedule", "value": "Mondays at 08:30"}
    # The configured monthly total, not what happens to be deployable this minute.
    assert payload["summary"][2] == {"label": "Planned", "value": "$250/month"}


def _dca_backtest_history(monkeypatch, strategy, plan, *, price=100.0, equity=100_000.0):
    """Drive a DCA backtest through the same unified path the dashboard uses."""
    dates = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=130)
    symbols = [item["symbol"] for item in plan["buy"]["items"]]
    bars = {symbol: _dated_bars([str(d.date()) for d in dates], price=price) for symbol in symbols}
    config = Config(symbols=symbols, cash_buffer=0.0, transaction_cost_bps=0.0,
                    backtest_starting_equity=equity)

    monkeypatch.setattr(api_payloads, "get_config", lambda *a, **kw: config)
    monkeypatch.setattr(api_payloads, "create_data_client", lambda config: object())
    monkeypatch.setattr(api_payloads, "fetch_daily_bars", lambda syms, **kw: {s: bars[s] for s in syms})
    monkeypatch.setattr(dca_bot.DCAAlgorithm, "plan", lambda self, config: plan)
    monkeypatch.setattr(dca_bot, "broker_supports_fractional_shares", lambda account_id: True)

    payload = api_payloads._compute_backtest(strategy, "6m", plan)
    return pd.DataFrame(payload["rows"]).set_index("timestamp")


def test_dca_backtest_spend_rate_does_not_depend_on_cadence(monkeypatch) -> None:
    """Plan amounts are dollars per MONTH. Spending the full amount on every scheduled run
    would model (runs per month) x the plan, and would make a frequent cadence look better
    purely because it deployed more capital -- so the replay accrues like the live algorithm.
    """
    # Well above the $50 min_executable floor, so the two cadences genuinely diverge in how
    # often they can act rather than both being gated by the floor.
    plan = {"buy": {"items": [{"symbol": "AAA", "amount": 3_000.0}]}, "sell": {"items": []}}

    weekly = _dca_backtest_history(monkeypatch, "dca", plan)
    monkeypatch.setattr(dca_bot.DCAAlgorithm, "schedule", Schedule())
    daily = _dca_backtest_history(monkeypatch, "dca", plan)

    # ~6 months x $3,000/month, whichever cadence deployed it.
    assert weekly["dca_contributions"].iloc[-1] == pytest.approx(18_000.0, rel=0.15)
    assert daily["dca_contributions"].iloc[-1] == pytest.approx(18_000.0, rel=0.15)
    # The cadence changes how the same money is split, not how much of it there is.
    assert daily["order_count"].sum() > weekly["order_count"].sum()


def test_dca_backtest_only_trades_on_its_scheduled_weekdays(monkeypatch) -> None:
    """The replay gates on the same Schedule the runtime does, so a backtest cannot model a
    different cadence than the one that will actually trade."""
    plan = {"buy": {"items": [{"symbol": "AAA", "amount": 3_000.0}]}, "sell": {"items": []}}

    history = _dca_backtest_history(monkeypatch, "dca", plan)
    traded = history[history["order_count"] > 0]

    assert not traded.empty
    assert set(pd.DatetimeIndex(traded.index).dayofweek) == {0}


def test_dca_backtest_never_spends_more_cash_than_it_has(monkeypatch) -> None:
    """A budget the account cannot fund is held back, not overdrawn."""
    plan = {"buy": {"items": [{"symbol": "AAA", "amount": 5_000.0}]}, "sell": {"items": []}}

    history = _dca_backtest_history(monkeypatch, "dca", plan, equity=1_000.0)

    assert (history["cash"] >= -0.01).all()
    assert history["dca_contributions"].iloc[-1] <= 1_000.0 + 0.01


def test_defensive_momentum_signals_include_inactive_universe_rows(monkeypatch) -> None:
    monkeypatch.setattr(market_context, "create_data_client", lambda config: object())

    def intraday(symbol: str) -> pd.DataFrame:
        step = 0.8 if symbol == "XSD" else -0.2 if symbol == "VXX" else 0.05
        return _strategy_bars([100 + index * step for index in range(80)])

    def daily(symbol: str) -> pd.DataFrame:
        step = 0.5 if symbol in {"SPY", "XSD"} else -0.2 if symbol == "VXX" else 0.05
        return _strategy_bars([100 + index * step for index in range(220)])

    monkeypatch.setattr(fast_momentum, "get_intraday_bars", lambda symbols, *_args, **_kwargs: {symbol: intraday(symbol) for symbol in symbols})
    monkeypatch.setattr(fast_momentum, "get_daily_bars", lambda symbols, *_args, **_kwargs: {symbol: daily(symbol) for symbol in symbols})
    monkeypatch.setattr(
        fast_momentum,
        "fetch_latest_news_sentiment",
            lambda symbols, config: [
                {"symbol": "SPY", "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "sentiment": 0.5, "provider": "stocktwits"},
                {"symbol": "XSD", "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "sentiment": 0.3, "provider": "stocktwits"},
            ],
    )

    payload = strategy_signals_payload("fast_momentum")

    by_symbol = {row["symbol"]: row for row in payload["leaders"]}
    assert {"XSD", "BIL"} <= set(by_symbol)
    assert by_symbol["XSD"]["signal"] == "LONG"
    assert "score_components" in by_symbol["XSD"]
    assert payload["summary"][0]["value"] == "Dynamic rank"
    assert by_symbol["BIL"]["reason"]


