from __future__ import annotations

from src.core.config import get_config


def test_get_config_reads_yaml_accounts_and_knobs(tmp_path, monkeypatch) -> None:
    algorithm_bot_path = tmp_path / "algorithm_bot.yaml"
    algorithms_path = tmp_path / "algorithms.yaml"
    universe_path = tmp_path / "universe.yaml"
    accounts_path = tmp_path / "accounts.yaml"
    connectors_path = tmp_path / "connectors.yaml"
    algorithm_bot_path.write_text(
        """
runtime:
  kill_switch: true
  backtest_period: 4m
social:
  trends_csv: data/custom_social.csv
""",
        encoding="utf-8",
    )
    algorithms_path.write_text(
        """
algorithms:
  rally_rotation:
    momentum_lookback_days: 42
    max_longs: 3
""",
        encoding="utf-8",
    )
    universe_path.write_text(
        """
tradable_universe:
  master_list: missing_tradables.csv
  symbols:
    - AAA
    - BBB
""",
        encoding="utf-8",
    )
    accounts_path.write_text(
        """
default: paper
accounts:
  items:
    paper:
      label: Paper Desk
      api_key: paper-key
      api_secret: paper-secret
      base_url: https://paper-api.alpaca.markets
      data_feed: iex
    second:
      label: Second Desk
      api_key: second-key
      api_secret: second-secret
      base_url: https://example.invalid
      data_feed: sip
""",
        encoding="utf-8",
    )
    connectors_path.write_text(
        """
data_sources:
  market_data:
    cache_ttl_seconds: 15
    providers:
      finnhub:
      alpaca:
  intraday_market_data:
    cache_ttl_seconds: 9
    providers:
      yfinance:
  news_sentiment:
    cache_ttl_seconds: 20
    providers:
      stocktwits:
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_ALGORITHM_BOT_FILE", str(algorithm_bot_path))
    monkeypatch.setenv("TRADING_ALGORITHMS_FILE", str(algorithms_path))
    monkeypatch.setenv("TRADING_UNIVERSE_FILE", str(universe_path))
    monkeypatch.setenv("TRADING_ACCOUNTS_FILE", str(accounts_path))
    monkeypatch.setenv("TRADING_CONNECTORS_FILE", str(connectors_path))
    monkeypatch.delenv("KILL_SWITCH", raising=False)
    monkeypatch.delenv("ALPHA_VANTAGE_NEWS_CSV", raising=False)

    config = get_config()

    assert config.account_id == "paper"
    assert config.account_label == "Paper Desk"
    assert config.alpaca_api_key == "paper-key"
    assert config.alpaca_api_secret == "paper-secret"
    assert config.symbols == ["AAA", "BBB"]
    assert config.momentum_lookback_days == 42
    assert config.max_longs == 3
    # The kill switch is env-only now: a deployment brake, not a dashboard control, so a
    # runtime: block in the config document no longer turns it on.
    assert config.kill_switch is False
    assert config.backtest_period == "4m"
    assert config.social_trends_csv == "data/custom_social.csv"
    assert config.market_data_provider_order == ["finnhub", "alpaca"]
    assert config.market_data_cache_ttl_seconds == 15
    assert config.intraday_market_data_provider_order == ["yfinance"]
    assert config.intraday_market_data_cache_ttl_seconds == 9
    assert config.eod_market_data_provider_order == ["finnhub", "alpaca"]
    assert config.eod_market_data_cache_ttl_seconds == 15
    assert config.news_sentiment_provider_order == ["stocktwits"]
    assert config.sentiment_data_provider_order == ["stocktwits"]
    assert config.news_sentiment_cache_ttl_seconds == 20
    assert config.sentiment_data_cache_ttl_seconds == 20
    assert "rally_rotation" in config.algorithm_configs
    assert config.account_options == [
        {"id": "paper", "label": "Paper Desk"},
        {"id": "second", "label": "Second Desk"},
    ]


def test_get_config_reads_alpaca_connector_auth_from_connectors(tmp_path, monkeypatch) -> None:
    universe_path = tmp_path / "universe.yaml"
    accounts_path = tmp_path / "accounts.yaml"
    connectors_path = tmp_path / "connectors.yaml"
    universe_path.write_text(
        """
tradable_universe:
  symbols:
    - AAA
""",
        encoding="utf-8",
    )
    accounts_path.write_text(
        """
accounts:
  items:
    paper:
      label: Paper Desk
      api_key: paper-key
      api_secret: paper-secret
      base_url: https://paper-api.alpaca.markets
      data_feed: iex
""",
        encoding="utf-8",
    )
    connectors_path.write_text(
        """
data_sources:
  eod_market_data:
    providers:
      alpaca:
        api_key_env: ALPACA_API_KEY
        api_secret_env: ALPACA_API_SECRET
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_UNIVERSE_FILE", str(universe_path))
    monkeypatch.setenv("TRADING_ACCOUNTS_FILE", str(accounts_path))
    monkeypatch.setenv("TRADING_CONNECTORS_FILE", str(connectors_path))
    monkeypatch.setenv("ALPACA_API_KEY", "connector-key")
    monkeypatch.setenv("ALPACA_API_SECRET", "connector-secret")

    config = get_config()

    assert config.alpaca_api_key == "paper-key"
    assert config.alpaca_api_secret == "paper-secret"
    assert config.alpaca_data_api_key == "connector-key"
    assert config.alpaca_data_api_secret == "connector-secret"


def test_get_config_uses_empty_external_account_and_connector_files(tmp_path, monkeypatch) -> None:
    universe_path = tmp_path / "universe.yaml"
    accounts_path = tmp_path / "accounts.yaml"
    connectors_path = tmp_path / "connectors.yaml"
    universe_path.write_text(
        """
tradable_universe:
  symbols:
    - AAA
""",
        encoding="utf-8",
    )
    accounts_path.write_text("accounts: []\n", encoding="utf-8")
    connectors_path.write_text(
        """
data_sources:
  market_data:
    providers: []
  news_sentiment:
    providers: []
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_UNIVERSE_FILE", str(universe_path))
    monkeypatch.setenv("TRADING_ACCOUNTS_FILE", str(accounts_path))
    monkeypatch.setenv("TRADING_CONNECTORS_FILE", str(connectors_path))
    monkeypatch.delenv("SYMBOLS", raising=False)

    config = get_config()

    assert config.account_options == []
    assert config.market_data_provider_order == []
    assert config.news_sentiment_provider_order == []
    assert config.symbols == ["AAA"]


def test_get_config_deduces_provider_order_from_provider_map(tmp_path, monkeypatch) -> None:
    universe_path = tmp_path / "universe.yaml"
    accounts_path = tmp_path / "accounts.yaml"
    connectors_path = tmp_path / "connectors.yaml"
    universe_path.write_text(
        """
tradable_universe:
  symbols:
    - AAA
""",
        encoding="utf-8",
    )
    accounts_path.write_text("accounts: []\n", encoding="utf-8")
    connectors_path.write_text(
        """
data_sources:
  market_data:
    providers:
      finnhub:
      alpha_vantage:
  news_sentiment:
    providers:
      stocktwits:
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_UNIVERSE_FILE", str(universe_path))
    monkeypatch.setenv("TRADING_ACCOUNTS_FILE", str(accounts_path))
    monkeypatch.setenv("TRADING_CONNECTORS_FILE", str(connectors_path))
    monkeypatch.delenv("SYMBOLS", raising=False)

    config = get_config()

    assert config.market_data_provider_order == ["finnhub", "alpha_vantage"]
    assert config.news_sentiment_provider_order == ["stocktwits"]


def test_get_config_reads_new_data_source_categories(tmp_path, monkeypatch) -> None:
    universe_path = tmp_path / "universe.yaml"
    accounts_path = tmp_path / "accounts.yaml"
    connectors_path = tmp_path / "connectors.yaml"
    universe_path.write_text(
        """
tradable_universe:
  symbols:
    - AAA
""",
        encoding="utf-8",
    )
    accounts_path.write_text("accounts: []\n", encoding="utf-8")
    connectors_path.write_text(
        """
data_sources:
  intraday_market_data:
    cache_ttl_seconds: 7
    providers:
      finnhub:
      yfinance:
  eod_market_data:
    cache_ttl_seconds: 11
    providers:
      yfinance:
      alpaca:
  sentiment_data:
    cache_ttl_seconds: 13
    providers:
      newsapi:
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_UNIVERSE_FILE", str(universe_path))
    monkeypatch.setenv("TRADING_ACCOUNTS_FILE", str(accounts_path))
    monkeypatch.setenv("TRADING_CONNECTORS_FILE", str(connectors_path))
    monkeypatch.delenv("SYMBOLS", raising=False)

    config = get_config()

    assert config.intraday_market_data_provider_order == ["finnhub", "yfinance"]
    assert config.intraday_market_data_cache_ttl_seconds == 7
    assert config.eod_market_data_provider_order == ["yfinance", "alpaca"]
    assert config.market_data_provider_order == ["yfinance", "alpaca"]
    assert config.eod_market_data_cache_ttl_seconds == 11
    assert config.sentiment_data_provider_order == ["newsapi"]
    assert config.news_sentiment_provider_order == ["newsapi"]
    assert config.sentiment_data_cache_ttl_seconds == 13


def test_get_config_reads_split_bot_and_universe_files(tmp_path, monkeypatch) -> None:
    algorithm_bot_path = tmp_path / "algorithm_bot.yaml"
    algorithms_path = tmp_path / "algorithms.yaml"
    dca_bot_path = tmp_path / "dca_bot.yaml"
    universe_path = tmp_path / "universe.yaml"
    accounts_path = tmp_path / "accounts.yaml"
    connectors_path = tmp_path / "connectors.yaml"
    algorithm_bot_path.write_text(
        """
runtime:
  kill_switch: false
algorithm_bot:
  require_trade_approval: true
  trade_approval_timeout_seconds: 120
  trade_approval_poll_seconds: 2
""",
        encoding="utf-8",
    )
    algorithms_path.write_text(
        """
algorithms:
  rally_rotation:
    momentum_lookback_days: 126
    max_longs: 4
""",
        encoding="utf-8",
    )
    dca_bot_path.write_text("dca_bot: {}\n", encoding="utf-8")
    universe_path.write_text(
        """
tradable_universe:
  master_list: missing_tradables.csv
  symbols:
    - SPY
    - QQQ
    - GLD
""",
        encoding="utf-8",
    )
    accounts_path.write_text("accounts: []\n", encoding="utf-8")
    connectors_path.write_text("data_sources: {}\n", encoding="utf-8")
    monkeypatch.setenv("TRADING_ALGORITHM_BOT_FILE", str(algorithm_bot_path))
    monkeypatch.setenv("TRADING_ALGORITHMS_FILE", str(algorithms_path))
    monkeypatch.setenv("TRADING_DCA_BOT_FILE", str(dca_bot_path))
    monkeypatch.setenv("TRADING_UNIVERSE_FILE", str(universe_path))
    monkeypatch.setenv("TRADING_ACCOUNTS_FILE", str(accounts_path))
    monkeypatch.setenv("TRADING_CONNECTORS_FILE", str(connectors_path))
    monkeypatch.delenv("SYMBOLS", raising=False)

    config = get_config(strategy_id="rally_rotation")

    assert config.symbols == ["SPY", "QQQ", "GLD"]
    assert config.momentum_lookback_days == 126
    assert config.max_longs == 4
    assert config.require_trade_approval is True
    assert config.trade_approval_timeout_seconds == 120
    assert config.trade_approval_poll_seconds == 2


def test_the_kill_switch_is_an_environment_brake_not_a_config_key(tmp_path, monkeypatch) -> None:
    """Per-binding switches decide what trades; this stays for an emergency stop on the host."""
    from src.core.config import get_config

    config_path = tmp_path / "walbot.yaml"
    config_path.write_text("runtime:\n  kill_switch: true\naccounts:\n  items:\n    paper: {}\n", encoding="utf-8")
    monkeypatch.setenv("TRADING_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("KILL_SWITCH", raising=False)

    assert get_config().kill_switch is False

    monkeypatch.setenv("KILL_SWITCH", "true")
    assert get_config().kill_switch is True


def test_the_tradable_universe_covers_every_symbol_an_algorithm_can_hold() -> None:
    """It drifted: it still listed names dropped for illiquidity and missed ten being traded.

    This list decides what the Universe view shows and what gets priced when an algorithm
    does not declare its own symbols, so a gap there is a symbol nobody fetches.
    """
    import yaml

    from src.core.config import config_file_path

    document = yaml.safe_load(config_file_path().read_text(encoding="utf-8"))
    algorithms = document["algorithms"]

    held: set[str] = set()
    for key in ("rally_rotation",):
        section = algorithms[key]
        held |= set(section["risk_on_universe"]) | set(section["defensive_universe"])
        held.add(section.get("benchmark"))
    held.discard(None)

    universe = set(document["tradable_universe"]["symbols"])

    assert held <= universe, f"tradable_universe is missing {sorted(held - universe)}"


def test_an_explicitly_empty_universe_is_honoured_not_replaced_by_the_default() -> None:
    """``defensive_universe: []`` must mean "no sleeve", not "the five-name default".

    Absent and empty were the same value here, so a setting written to turn something off
    looked obeyed and restored the default instead.
    """
    from src.common.config_utils import parse_symbols

    assert parse_symbols([], ["BIL", "GLD"]) == []
    assert parse_symbols(None, ["BIL", "GLD"]) == ["BIL", "GLD"]
    # A blank string is still "unset": that is what an empty form field sends.
    assert parse_symbols("", ["BIL", "GLD"]) == ["BIL", "GLD"]
    assert parse_symbols("spy, qqq", ["BIL"]) == ["SPY", "QQQ"]


def test_as_bool_falls_back_rather_than_guessing() -> None:
    """One truth table, and an unrecognised value is not silently true.

    There were five bool coercions and they disagreed: this one reached ``bool(value)`` for
    anything it did not recognise, so ``enabled: mabye`` in a config file read as *on*, while
    the dashboard's own coercer read the same typo as *off*.
    """
    from src.common.config_utils import as_bool

    for truthy in ("true", "1", "yes", "y", "on", "  TRUE  ", True):
        assert as_bool(truthy) is True, truthy
    for falsey in ("false", "0", "no", "n", "off", False):
        assert as_bool(falsey, default=True) is False, falsey

    # The divergence that mattered: unrecognised means "use the default", both ways.
    assert as_bool("mabye", default=False) is False
    assert as_bool("mabye", default=True) is True
    assert as_bool(None, default=True) is True


def test_as_float_rejects_infinity_as_well_as_nan() -> None:
    """``isnan`` alone let ``inf`` through, where it survives arithmetic and only surfaces
    later as an unserialisable weight. The private ``_finite`` copies this replaced were the
    strict ones, so consolidating onto the shared helper had to tighten it, not loosen them."""
    from src.common.config_utils import as_float

    assert as_float(float("inf"), 0.5) == 0.5
    assert as_float(float("-inf"), 0.5) == 0.5
    assert as_float(float("nan"), 0.5) == 0.5
    assert as_float("not a number", 0.5) == 0.5
    assert as_float(None, 0.5) == 0.5
    assert as_float("2.5") == 2.5
