from __future__ import annotations

import pandas as pd
import pytest

from src.api import api_payloads as api_payloads
from src.core import market_context
from src.algorithms import dca as dca
from src.algorithms.dca import bot as dca_bot
from src.core.config import Config
from src.core.interfaces import Schedule
from src.execution import replay as replay_module
from src.algorithms.registry import canonical_algorithm_id
from src.api.api_payloads import (
    backtest_payload,
    controls_payload,
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



def _patch_payloads(monkeypatch, name, value):
    """Patch ``name`` on every payload module that resolves it.

    ``api_payloads`` is a facade now: the implementations live in ``src/api/payloads/`` and
    each module resolves its imports in its own namespace, so setting an attribute on the
    facade has no effect. Patching wherever the name actually exists keeps these tests stating
    an intent ("this dependency returns X") rather than a location.
    """
    import importlib
    import pkgutil

    import src.api.payloads as payloads_package

    patched = 0
    for info in pkgutil.iter_modules(payloads_package.__path__):
        module = importlib.import_module(f"src.api.payloads.{info.name}")
        if hasattr(module, name):
            monkeypatch.setattr(module, name, value)
            patched += 1
    assert patched, f"no payload module defines {name!r}"


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
    _patch_payloads(monkeypatch, "create_data_client", lambda config: object())

    def fake_fetch_daily_bars(symbols, **kwargs):
        assert kwargs["force_refresh"] is True
        return {symbol: _fresh_bars(price=100.0 + index) for index, symbol in enumerate(symbols)}

    _patch_payloads(monkeypatch, "fetch_daily_bars", fake_fetch_daily_bars)

    payload = api_payloads.recommend_universe_payload({"max_symbols": 5, "refresh": True})

    assert [row["symbol"] for row in payload["rows"]] == ["SPY", "GLD", "TLT", "QQQ", "IBIT"]
    assert payload["eligible_count"] == 5


def test_a_plan_carries_only_what_to_buy_and_how_much() -> None:
    """Cadence lives on the algorithm class and the on/off switch is the binding's, so a plan
    carrying either would be a second source of truth that nothing reads."""
    from src.algorithms.dca import sanitize_dca_plan

    plan = sanitize_dca_plan(
        {"enabled": True, "frequency": "1hr", "schedule_pattern": "0 9 * * 1-5",
         "buy": {"items": [{"symbol": "SPY", "amount": 40}]}, "sell": {"items": []}},
        [{"symbol": "SPY", "enabled": True}],
    )

    assert set(plan) == {"buy", "sell"}


def test_controls_payload_returns_persisted_choices() -> None:
    payload = controls_payload()

    assert {"equities", "options", "algorithm_enabled", "options_trading_enabled", "active_strategy", "options_strategy"} <= set(payload["controls"])
    assert {"algorithm", "options"} <= set(payload["bot"])
    assert "dca" not in payload["bot"]


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
    # Asserted on the module that owns the controls payload: ``api_payloads`` is a facade and
    # forwards only the names it exports, so a runtime handle would not be visible through it.
    from src.api.payloads import controls as controls_payloads

    assert not hasattr(controls_payloads.bot_runtime, "wake_algorithm")
    assert not hasattr(controls_payloads.bot_runtime, "wake_options")


def test_cache_only_backtest_does_not_compute_without_cached_rows(monkeypatch) -> None:
    def fail_compute(*_args, **_kwargs):
        raise AssertionError("cache-only request should not compute a backtest")

    _patch_payloads(monkeypatch, "universe_payload", lambda: {"rows": []})
    _patch_payloads(monkeypatch, "_load_backtest_cache", lambda: {"version": 2, "items": {}})
    _patch_payloads(monkeypatch, "_compute_backtest", fail_compute)

    payload = backtest_payload({"strategy": "rally_rotation", "period": "6m", "cache_only": True})

    assert payload["strategy"] == "rally_rotation"
    assert payload["period_label"] == "6M"
    assert payload["cached"] is False
    assert "error" in payload


def test_non_refresh_backtest_does_not_compute_without_cached_rows(monkeypatch) -> None:
    def fail_compute(*_args, **_kwargs):
        raise AssertionError("non-refresh request should not compute a backtest")

    _patch_payloads(monkeypatch, "universe_payload", lambda: {"rows": []})
    _patch_payloads(monkeypatch, "_load_backtest_cache", lambda: {"version": 4, "items": {}})
    _patch_payloads(monkeypatch, "_compute_backtest", fail_compute)

    payload = backtest_payload({"strategy": "rally_rotation", "period": "2m"})

    assert payload["strategy"] == "rally_rotation"
    assert payload["cached"] is False
    assert "error" in payload


def test_refresh_backtest_computes_with_market_data_refresh(monkeypatch) -> None:
    calls = []

    def fake_compute(strategy, period, account_id=""):
        calls.append((strategy, period, account_id))
        return {
            "strategy": strategy,
            "period": period,
            "period_label": "6M",
            "cached": False,
            "rows": [{"timestamp": "2026-05-01", "equity": 10_000.0}],
        }

    _patch_payloads(monkeypatch, "universe_payload", lambda: {"rows": []})
    _patch_payloads(monkeypatch, "_load_backtest_cache", lambda: {"version": 4, "items": {}})
    _patch_payloads(monkeypatch, "_save_backtest_cache", lambda cache: None)
    _patch_payloads(monkeypatch, "_compute_backtest", fake_compute)

    payload = backtest_payload({"strategy": "trend_following", "period": "6m", "refresh": True})

    assert payload["strategy"] == "trend_following"
    assert calls == [("trend_following", "6m", "")]


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



def test_history_backtest_reads_configured_provider_order(monkeypatch) -> None:
    calls = []
    config = Config(intraday_market_data_provider_order=["finnhub", "yfinance"])
    # Two sessions of 15-minute bars ending at the signal date, which covers the window asked
    # for below and so stops the walk before it reaches the all-providers read.
    covering = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-04-30 13:30", periods=53, freq="15min", tz="UTC"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
            "adjusted_close": 100.5,
        }
    )

    def fake_read_history(symbol, *, lookback_minutes, end, provider=None, **kwargs):
        calls.append((symbol, provider, lookback_minutes, end))
        return pd.DataFrame() if provider == "finnhub" else covering

    monkeypatch.setattr(replay_module, "read_history", fake_read_history)

    coverage = replay_module.Coverage()
    as_of = pd.Timestamp("2026-05-01", tz="UTC")
    history = replay_module.HistoryCache(
        ["SPY"],
        [as_of],
        providers=api_payloads._configured_history_providers(config),
        lookback_minutes=780,
    )
    bars = history.bars_as_of(["SPY"], as_of, coverage)

    assert not bars["SPY"].empty
    # The provider walk happens once for the whole replay rather than once per date, but it
    # still walks the configured order, and a provider with nothing to say falls through to
    # the next. The trailing ``None`` reads across all providers, which is only reached
    # because this fixture cannot cover the whole replay span from one of them.
    assert [call[1] for call in calls] == ["finnhub", "yfinance", None]
    # Falling back to a second provider still counts as covered.
    assert coverage.as_dict()["history_ratio"] == 1.0
    assert coverage.missing_symbols == set()



def test_none_strategy_resolves_to_dca(monkeypatch) -> None:
    """The retired "None" card always meant "just run DCA", so its saved id maps there."""
    assert canonical_algorithm_id("none") == "dca"


def test_dca_view_states_its_planned_total(monkeypatch) -> None:
    """What the plan commits to per month, not what happens to be deployable this minute.

    There is no Schedule row: cadence is set per binding on the dashboard and every algorithm
    runs inside the trading session regardless, so restating it beside the signals only
    repeated the deployment the reader had just configured.
    """
    plan = {
        "buy": {"items": [{"symbol": "SPY", "amount": 250.0}]},
        "sell": {"items": []},
    }
    _patch_payloads(monkeypatch, "universe_payload", lambda: {"rows": [{"symbol": "SPY"}]})
    monkeypatch.setattr(dca_bot.DCAAlgorithm, "plan", lambda self, config: plan)
    monkeypatch.setattr(dca_bot, "unknown_plan_symbols", lambda *a, **kw: [])
    monkeypatch.setattr(market_context, "create_data_client", lambda config: object())
    monkeypatch.setattr(market_context, "load_latest_prices", lambda symbols, config, client: {"SPY": 500.0})

    payload = strategy_signals_payload("dca")

    rows = {row["symbol"]: row for row in payload["leaders"]}
    assert rows["SPY"]["reason"].startswith("Accruing")
    assert payload["summary"][0] == {"label": "Mode", "value": "DCA"}
    assert payload["summary"][1] == {"label": "Planned", "value": "$250/month"}
    assert "Schedule" not in {row["label"] for row in payload["summary"]}


def _binding_controls(strategy: str, account_id: str) -> dict:
    return {
        "bindings": [
            {"id": "b1", "strategy": strategy, "account_id": account_id, "enabled": True, "frequency": "1hr"}
        ]
    }


def test_signal_view_is_computed_for_the_account_the_strategy_is_deployed_on(monkeypatch) -> None:
    """A DCA plan is per account, so which account the view reads is not a detail.

    The signal view used to build its config with no account at all, so it fell back to the
    default account while the dashboard's bubble board wrote the plan of the binding's account.
    The two never showed the same plan, an edit appeared to do nothing, and -- because
    ``analyze`` persists accrual -- the preview wrote a ledger under the wrong account too.
    """
    from src.api import controls as controls_module
    from src.api.payloads import strategy_config

    captured: dict = {}

    def capturing_get_config(account_id=None, strategy_id=None):
        captured["account_id"] = account_id
        return Config(symbols=["SPY"], account_id=str(account_id or ""))

    monkeypatch.setattr(controls_module, "load_controls", lambda *a, **kw: _binding_controls("dca", "local_paper"))
    monkeypatch.setattr(strategy_config, "get_config", capturing_get_config)

    assert strategy_config.config_for_strategy_view("dca").account_id == "local_paper"
    assert captured["account_id"] == "local_paper"

    # An explicit account still wins: the dashboard sends the one its own editor is writing.
    assert strategy_config.config_for_strategy_view("dca", "paper").account_id == "paper"


def test_signal_view_survives_a_binding_naming_a_deleted_account(monkeypatch) -> None:
    """``sanitize_binding`` never checks the account exists, so a binding can outlive one.

    Refusing here would take the whole dashboard down for a stale binding, so the view falls
    back to the default account. It reports which account it used, so the substitution is
    visible rather than silent.
    """
    from src.api import controls as controls_module
    from src.api.payloads import strategy_config
    from src.core.config import UnknownAccountError

    def strict_get_config(account_id=None, strategy_id=None):
        if account_id:
            raise UnknownAccountError(str(account_id), ["paper"])
        return Config(symbols=["SPY"], account_id="paper")

    monkeypatch.setattr(controls_module, "load_controls", lambda *a, **kw: _binding_controls("dca", "deleted"))
    monkeypatch.setattr(strategy_config, "get_config", strict_get_config)

    assert strategy_config.config_for_strategy_view("dca").account_id == "paper"


def test_backtest_cache_key_separates_accounts(monkeypatch) -> None:
    """Two DCA bindings on different accounts have different plans, so different curves.

    Without the account in the basis they collided on one cache entry and whichever ran first
    answered for both -- and editing one account's plan did not invalidate the other's.
    """
    from src.api.payloads import strategy_config
    from src.api.payloads.backtest import _cache_key

    monkeypatch.setattr(
        strategy_config,
        "get_config",
        lambda account_id=None, strategy_id=None: Config(symbols=["SPY"], account_id=str(account_id or "")),
    )

    assert _cache_key("dca", "6m", "paper") != _cache_key("dca", "6m", "local_paper")
    assert _cache_key("dca", "6m", "paper") == _cache_key("dca", "6m", "paper")
    assert _cache_key("dca", "6m", "paper") != _cache_key("dca", "4m", "paper")


def _dca_backtest_history(monkeypatch, strategy, plan, *, price=100.0, equity=100_000.0):
    """Drive a DCA backtest through the same unified path the dashboard uses."""
    dates = pd.bdate_range(end=pd.Timestamp.now(tz="UTC").normalize(), periods=130)
    symbols = [item["symbol"] for item in plan["buy"]["items"]]
    bars = {symbol: _dated_bars([str(d.date()) for d in dates], price=price) for symbol in symbols}
    config = Config(symbols=symbols, cash_buffer=0.0, transaction_cost_bps=0.0,
                    backtest_starting_equity=equity)

    _patch_payloads(monkeypatch, "get_config", lambda *a, **kw: config)
    _patch_payloads(monkeypatch, "create_data_client", lambda config: object())
    _patch_payloads(monkeypatch, "fetch_daily_bars", lambda syms, **kw: {s: bars[s] for s in syms})
    monkeypatch.setattr(dca_bot.DCAAlgorithm, "plan", lambda self, config: plan)
    monkeypatch.setattr(dca_bot, "broker_supports_fractional_shares", lambda account_id: True)

    payload = api_payloads._compute_backtest(strategy, "6m")
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





def test_an_algorithm_declares_what_invalidates_its_cached_backtest() -> None:
    """Editing a DCA plan must invalidate that backtest, and nothing else must know why.

    The plan used to be passed down through three backtest signatures purely to reach the
    cache hash -- an argument the algorithm never received, existing only to change a string.
    """
    from src.algorithms.registry import get_algorithm_class
    from src.api.payloads.backtest import _cache_key
    from src.core.config import get_config

    dca = get_algorithm_class("dca").from_config(get_config(strategy_id="dca"))
    assert "plan" in dca.config_fingerprint(get_config(strategy_id="dca"))

    momentum = get_algorithm_class("rally_rotation").from_config(get_config(strategy_id="rally_rotation"))
    assert "plan" not in momentum.config_fingerprint(get_config(strategy_id="rally_rotation"))

    # Stable for the same inputs, and distinct per strategy and per period.
    assert _cache_key("rally_rotation", "6m") == _cache_key("rally_rotation", "6m")
    assert _cache_key("rally_rotation", "6m") != _cache_key("rally_rotation", "4m")
    assert _cache_key("rally_rotation", "6m") != _cache_key("dca", "6m")


def test_an_unknown_strategy_still_hashes_rather_than_raising() -> None:
    """Reporting a bad strategy id is the compute step's job, not the cache key's."""
    from src.api.payloads.backtest import _cache_key

    assert _cache_key("no_such_algorithm", "6m")
