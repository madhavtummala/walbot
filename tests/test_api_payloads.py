from __future__ import annotations

import pandas as pd

from src import api_payloads
from src.api_payloads import (
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
    assert {"enabled", "frequency", "buy", "sell"} <= set(payload["plan"])


def test_controls_payload_returns_persisted_choices() -> None:
    payload = controls_payload()

    assert {"equities", "options", "algorithm_enabled", "options_trading_enabled", "active_strategy", "options_strategy"} <= set(payload["controls"])
    assert {"algorithm", "options", "dca"} <= set(payload["bot"])


def test_fast_momentum_reason_uses_configured_defensive_symbols() -> None:
    row = {"symbol": "XYLD", "macro_trend_ok": True}
    config = api_payloads.DefensiveMomentumConfig(defensive_universe=["BIL", "XYLD"])

    reason = api_payloads._defensive_momentum_reason(row, 0.25, config)

    assert reason == "Dynamic rank"


def test_fast_momentum_reason_describes_risk_on_rank_cutoff() -> None:
    config = api_payloads.DefensiveMomentumConfig(
        risk_on_universe=["XSD", "AIQ"],
        defensive_universe=["BIL", "XYLD"],
        max_positions=4,
        min_risk_on_micro_return=0.0,
    )

    assert api_payloads._defensive_momentum_reason({"symbol": "AIQ", "macro_trend_ok": True}, 0.0, config) == "Outside top 4 rank"
    assert (
        api_payloads._defensive_momentum_reason({"symbol": "XSD", "macro_trend_ok": True, "micro_return": -0.001}, 0.0, config)
        == "Micro trend below risk-on floor"
    )


def test_save_controls_payload_wakes_options_runtime(tmp_path, monkeypatch) -> None:
    woke = {"algorithm": 0, "options": 0}
    monkeypatch.setenv("TRADING_ALGORITHM_BOT_FILE", str(tmp_path / "algorithm_bot.yaml"))
    monkeypatch.setenv("TRADING_OPTIONS_BOT_FILE", str(tmp_path / "options_bot.yaml"))

    monkeypatch.setattr(api_payloads.bot_runtime, "wake_algorithm", lambda: woke.__setitem__("algorithm", woke["algorithm"] + 1))
    monkeypatch.setattr(api_payloads.bot_runtime, "wake_options", lambda: woke.__setitem__("options", woke["options"] + 1))

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

    assert payload["controls"]["algorithm_enabled"] is False
    assert payload["controls"]["options_trading_enabled"] is False
    assert woke == {"algorithm": 1, "options": 1}


def test_none_backtest_returns_flat_payload_without_market_data(monkeypatch) -> None:
    saved_cache = {}

    monkeypatch.setattr(api_payloads, "universe_payload", lambda: {"rows": []})
    monkeypatch.setattr(
        api_payloads,
        "load_dca_plan",
        lambda rows: {"enabled": False, "frequency": "weekly", "buy": {"items": []}, "sell": {"items": []}},
    )
    monkeypatch.setattr(api_payloads, "_load_backtest_cache", lambda: {"version": 2, "items": {}})
    monkeypatch.setattr(api_payloads, "_save_backtest_cache", lambda cache: saved_cache.update(cache))

    payload = backtest_payload({"strategy": "none", "refresh": True})

    assert payload["strategy"] == "none"
    assert payload["source"] == "flat"
    assert payload["period_label"] == "6M"
    assert len(payload["rows"]) > 1
    assert saved_cache["items"]


def test_cache_only_backtest_does_not_compute_without_cached_rows(monkeypatch) -> None:
    def fail_compute(*_args, **_kwargs):
        raise AssertionError("cache-only request should not compute a backtest")

    monkeypatch.setattr(api_payloads, "universe_payload", lambda: {"rows": []})
    monkeypatch.setattr(
        api_payloads,
        "load_dca_plan",
        lambda rows: {"enabled": False, "frequency": "weekly", "buy": {"items": []}, "sell": {"items": []}},
    )
    monkeypatch.setattr(api_payloads, "_load_backtest_cache", lambda: {"version": 2, "items": {}})
    monkeypatch.setattr(api_payloads, "_compute_backtest", fail_compute)

    payload = backtest_payload({"strategy": "momentum_social", "period": "6m", "cache_only": True})

    assert payload["strategy"] == "momentum_social"
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
        lambda rows: {"enabled": False, "frequency": "weekly", "buy": {"items": []}, "sell": {"items": []}},
    )
    monkeypatch.setattr(api_payloads, "_load_backtest_cache", lambda: {"version": 4, "items": {}})
    monkeypatch.setattr(api_payloads, "_compute_backtest", fail_compute)

    payload = backtest_payload({"strategy": "momentum_social", "period": "2m"})

    assert payload["strategy"] == "momentum_social"
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
        lambda rows: {"enabled": False, "frequency": "weekly", "buy": {"items": []}, "sell": {"items": []}},
    )
    monkeypatch.setattr(api_payloads, "_load_backtest_cache", lambda: {"version": 4, "items": {}})
    monkeypatch.setattr(api_payloads, "_save_backtest_cache", lambda cache: None)
    monkeypatch.setattr(api_payloads, "_compute_backtest", fake_compute)

    payload = backtest_payload({"strategy": "trend_following", "period": "6m", "refresh": True})

    assert payload["strategy"] == "trend_following"
    assert calls == [("trend_following", "6m")]


def test_dca_backtest_uses_cron_schedule_and_reports_skipped_cash(monkeypatch) -> None:
    bars = {"AAA": _dated_bars(["2026-01-05", "2026-01-06", "2026-01-07"], price=100.0)}
    plan = {
        "enabled": True,
        "frequency": "weekly",
        "schedule_pattern": "0 12 * * 1-5",
        "buy": {"items": [{"symbol": "AAA", "amount": 100.0}]},
        "sell": {"items": []},
    }

    monkeypatch.setattr(api_payloads, "create_data_client", lambda config: object())
    monkeypatch.setattr(api_payloads, "fetch_daily_bars", lambda *args, **kwargs: bars)

    history = api_payloads._dca_backtest(plan, "6m", starting_equity=250.0)
    summary = api_payloads._backtest_order_summary(history)

    assert history["scheduled_order_count"].sum() == 3
    assert history["order_count"].sum() == 2
    assert summary["planned_order_value"] == 300.0
    assert summary["total_order_value"] == 200.0
    assert summary["skipped_order_value"] == 100.0
    assert summary["capital_limit"] == 250.0
    assert summary["max_capital_at_work"] == 200.0


def test_backtest_response_explains_profit_loss_breakdown() -> None:
    history = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2026-01-01", tz="UTC"), "equity": 10_000.0, "cash": 9_000.0, "invested": 1_000.0},
            {"timestamp": pd.Timestamp("2026-01-02", tz="UTC"), "equity": 10_461.0, "cash": 1_250.0, "invested": 9_211.0},
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


def test_strategy_backtest_is_cash_account_constrained(monkeypatch) -> None:
    monkeypatch.setenv("SYMBOLS", "AAA")
    monkeypatch.setenv("MAX_WEIGHT_PER_SYMBOL", "1.0")
    monkeypatch.setenv("MAX_PORTFOLIO_EXPOSURE", "1.0")
    monkeypatch.setenv("MAX_LONGS", "1")
    monkeypatch.setenv("CASH_BUFFER", "0")
    monkeypatch.setenv("TRANSACTION_COST_BPS", "0")
    bars = {"AAA": _strategy_bars([100 + index * 0.1 for index in range(280)])}

    monkeypatch.setattr(api_payloads, "create_data_client", lambda config: object())
    monkeypatch.setattr(api_payloads, "fetch_daily_bars", lambda *args, **kwargs: bars)
    monkeypatch.setattr(
        api_payloads,
        "strategy_signal_rows_from_prepared",
        lambda strategy, snapshots: [{"symbol": "AAA", "side": "SHORT", "signal": -1, "score": 1.0}],
    )

    history = api_payloads._strategy_backtest("trend_following", "6m")

    assert history["order_count"].sum() == 0
    assert history["cash"].min() == 10_000.0
    assert history["invested"].max() == 0.0


def test_trend_and_mean_reversion_backtests_can_diverge(monkeypatch) -> None:
    monkeypatch.setenv("SYMBOLS", "AAA")
    monkeypatch.setenv("MAX_WEIGHT_PER_SYMBOL", "1.0")
    monkeypatch.setenv("MAX_PORTFOLIO_EXPOSURE", "1.0")
    monkeypatch.setenv("MAX_LONGS", "1")
    monkeypatch.setenv("CASH_BUFFER", "0")
    monkeypatch.setenv("TRANSACTION_COST_BPS", "0")
    bars = {"AAA": _strategy_bars([100 + index * 0.4 for index in range(280)])}

    monkeypatch.setattr(api_payloads, "create_data_client", lambda config: object())
    monkeypatch.setattr(api_payloads, "fetch_daily_bars", lambda *args, **kwargs: bars)

    trend = api_payloads._strategy_backtest("trend_following", "6m")
    mean = api_payloads._strategy_backtest("mean_reversion", "6m")

    assert trend["order_count"].sum() > 0
    assert trend["order_count"].sum() != mean["order_count"].sum()
    assert trend["equity"].iloc[-1] != mean["equity"].iloc[-1]


def test_none_strategy_signals_summarize_dca_plan(monkeypatch) -> None:
    monkeypatch.setattr(api_payloads, "universe_payload", lambda: {"rows": []})
    monkeypatch.setattr(
        api_payloads,
        "load_dca_plan",
        lambda rows: {
            "enabled": True,
            "schedule_pattern": "0 12 * * 1-5",
            "frequency": "weekly",
            "buy": {"items": [{"symbol": "SPY", "amount": 25}]},
            "sell": {"items": []},
        },
    )

    payload = strategy_signals_payload("none")

    assert payload["strategy"] == "none"
    assert payload["leaders"][0]["symbol"] == "SPY"
    assert payload["summary"][0]["value"] == "DCA"


def test_defensive_momentum_signals_include_inactive_universe_rows(monkeypatch) -> None:
    monkeypatch.setattr(api_payloads, "create_data_client", lambda config: object())

    def intraday(symbol: str) -> pd.DataFrame:
        step = 0.8 if symbol == "XSD" else -0.2 if symbol == "VXX" else 0.05
        return _strategy_bars([100 + index * step for index in range(80)])

    def daily(symbol: str) -> pd.DataFrame:
        step = 0.5 if symbol in {"SPY", "XSD"} else -0.2 if symbol == "VXX" else 0.05
        return _strategy_bars([100 + index * step for index in range(220)])

    monkeypatch.setattr(api_payloads, "get_intraday_bars", lambda symbols, *_args, **_kwargs: {symbol: intraday(symbol) for symbol in symbols})
    monkeypatch.setattr(api_payloads, "get_defensive_daily_bars", lambda symbols, *_args, **_kwargs: {symbol: daily(symbol) for symbol in symbols})
    monkeypatch.setattr(
        api_payloads,
        "fetch_latest_news_sentiment",
            lambda symbols, config: [
                {"symbol": "SPY", "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "sentiment": 0.5, "provider": "stocktwits"},
                {"symbol": "XSD", "timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "sentiment": 0.3, "provider": "stocktwits"},
            ],
    )

    payload = strategy_signals_payload("fast_momentum")

    by_symbol = {row["symbol"]: row for row in payload["leaders"]}
    assert {"XSD", "BIL"} <= set(by_symbol)
    assert by_symbol["XSD"]["side"] == "LONG"
    assert by_symbol["XSD"]["sentiment_providers"] == ["stocktwits"]
    assert by_symbol["XSD"]["sentiment_records"] == 1
    assert "score_components" in by_symbol["XSD"]
    assert by_symbol["XSD"]["sentiment_component"] is not None
    assert by_symbol["XSD"]["pullback_score"] is not None
    assert payload["summary"][0]["value"] == "Dynamic rank"
    assert by_symbol["BIL"]["reason"]


def test_momentum_social_signals_include_all_tracked_universe_rows(monkeypatch) -> None:
    monkeypatch.setenv("SYMBOLS", "VTI,XBI,BIL")
    monkeypatch.setattr(api_payloads, "create_data_client", lambda config: object())

    bars = {
        "VTI": _strategy_bars([100 + index * 0.16 for index in range(320)]),
        "XBI": _strategy_bars([110 - index * 0.03 for index in range(320)]),
        "BIL": _strategy_bars([100 + index * 0.01 for index in range(320)]),
    }
    monkeypatch.setattr(api_payloads, "fetch_daily_bars", lambda symbols, *_args, **_kwargs: {symbol: bars[symbol] for symbol in symbols})
    monkeypatch.setattr(api_payloads, "load_social_trends_csv", lambda csv, symbols: {symbol: 0 for symbol in symbols})

    payload = strategy_signals_payload("momentum_social")

    by_symbol = {row["symbol"]: row for row in payload["leaders"]}
    assert set(by_symbol) == {"VTI", "XBI", "BIL"}
    assert by_symbol["VTI"]["side"] in {"LONG", "FLAT"}
    assert by_symbol["XBI"]["side"] in {"LONG", "FLAT"}
    assert by_symbol["BIL"]["side"] in {"LONG", "FLAT"}
    assert by_symbol["XBI"]["reason"]


def test_dual_momentum_signals_include_all_tracked_universe_rows(monkeypatch) -> None:
    monkeypatch.setenv("SYMBOLS", "VTI,XBI,BIL,QQQ")
    monkeypatch.setattr(api_payloads, "create_data_client", lambda config: object())

    bars = {
        "VTI": _strategy_bars([100 + index * 0.16 for index in range(320)]),
        "XBI": _strategy_bars([110 - index * 0.03 for index in range(320)]),
        "BIL": _strategy_bars([100 + index * 0.01 for index in range(320)]),
        "QQQ": _strategy_bars([150 + index * 0.02 for index in range(320)]),
    }
    monkeypatch.setattr(api_payloads, "fetch_daily_bars", lambda symbols, *_args, **_kwargs: {symbol: bars[symbol] for symbol in symbols})

    payload = strategy_signals_payload("dual_momentum")

    by_symbol = {row["symbol"]: row for row in payload["leaders"]}
    assert set(by_symbol) == {"VTI", "XBI", "BIL", "QQQ"}
    assert by_symbol["QQQ"]["side"] in {"LONG", "SHORT", "FLAT"}
    assert by_symbol["BIL"]["side"] in {"LONG", "SHORT", "FLAT"}
