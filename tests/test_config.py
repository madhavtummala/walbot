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
  algorithm_check_seconds: 45
  backtest_period: 4m
social:
  trends_csv: data/custom_social.csv
""",
        encoding="utf-8",
    )
    algorithms_path.write_text(
        """
algorithms:
  momentum_social:
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
    assert config.kill_switch is True
    assert config.backtest_period == "4m"
    assert config.social_trends_csv == "data/custom_social.csv"
    assert config.algorithm_check_seconds == 45
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
    assert "momentum_social" in config.algorithm_configs
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
    options_bot_path = tmp_path / "options_bot.yaml"
    dca_bot_path = tmp_path / "dca_bot.yaml"
    universe_path = tmp_path / "universe.yaml"
    accounts_path = tmp_path / "accounts.yaml"
    connectors_path = tmp_path / "connectors.yaml"
    algorithm_bot_path.write_text(
        """
runtime:
  kill_switch: false
algorithm_bot:
  algorithm_check_seconds: 30
  algorithm_market_data_refresh_minutes: 10
  algorithm_run_jitter_minutes: 3
  trading_start_time: "09:30"
  trading_end_time: "14:00"
  require_trade_approval: true
  trade_approval_timeout_seconds: 120
  trade_approval_poll_seconds: 2
""",
        encoding="utf-8",
    )
    algorithms_path.write_text(
        """
algorithms:
  dual_momentum:
    momentum_lookback_days: 126
    max_longs: 4
""",
        encoding="utf-8",
    )
    options_bot_path.write_text(
        """
options:
  swing_dte_min: 35
""",
        encoding="utf-8",
    )
    dca_bot_path.write_text(
        """
dca_bot:
  dca_check_seconds: 90
""",
        encoding="utf-8",
    )
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
    monkeypatch.setenv("TRADING_OPTIONS_BOT_FILE", str(options_bot_path))
    monkeypatch.setenv("TRADING_DCA_BOT_FILE", str(dca_bot_path))
    monkeypatch.setenv("TRADING_UNIVERSE_FILE", str(universe_path))
    monkeypatch.setenv("TRADING_ACCOUNTS_FILE", str(accounts_path))
    monkeypatch.setenv("TRADING_CONNECTORS_FILE", str(connectors_path))
    monkeypatch.delenv("SYMBOLS", raising=False)

    config = get_config(strategy_id="dual_momentum")

    assert config.symbols == ["SPY", "QQQ", "GLD"]
    assert config.momentum_lookback_days == 126
    assert config.max_longs == 4
    assert config.algorithm_check_seconds == 30
    assert config.algorithm_market_data_refresh_minutes == 10
    assert config.algorithm_run_jitter_minutes == 3
    assert config.trading_start_time == "09:30"
    assert config.trading_end_time == "14:00"
    assert config.require_trade_approval is True
    assert config.trade_approval_timeout_seconds == 120
    assert config.trade_approval_poll_seconds == 2
    assert config.dca_check_seconds == 90
    assert config.options_swing_dte_min == 35
