"""The ``Config`` dataclass and the one function that assembles it.
"""

from __future__ import annotations

from ...algorithms.ids import LEGACY_ALGORITHM_IDS, canonical_algorithm_id

import os
from dataclasses import dataclass, field
from typing import Any

from ...data.universe import load_tradable_names

from ...common.config_utils import direct_or_env
from .coercion import _algorithm_sections, _config_value, _normalize_data_sources, _parse_symbols, _provider_credential, _provider_secret, _section, _str_to_bool, reader
from .defaults import ALGORITHM_EQUITY_CAP, ALPACA_BASE_URL, ALPACA_DATA_FEED, ALPHA_VANTAGE_MAX_SYMBOLS, ALPHA_VANTAGE_NEWS_CSV, ALPHA_VANTAGE_NEWS_LIMIT, ALPHA_VANTAGE_NEWS_LOOKBACK_DAYS, ALPHA_VANTAGE_REQUEST_DELAY_SECONDS, BACKTEST_PERIOD, BACKTEST_STARTING_EQUITY, CASH_BUFFER, CASH_EQUIVALENTS, DEFAULT_STRATEGY_ID, DIVIDEND_PROVIDER_ORDER, EOD_MARKET_DATA_CACHE_TTL_SECONDS, EOD_MARKET_DATA_PROVIDER_ORDER, HISTORY_EXTRA_BUFFER_DAYS, INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS, INTRADAY_MARKET_DATA_PROVIDER_ORDER, KILL_SWITCH, LONG_MA_DAYS, MARKET_DATA_BAR_MINUTES, MARKET_DATA_CACHE_TTL_SECONDS, MARKET_DATA_PROVIDER_ORDER, MAX_LONGS, MAX_PORTFOLIO_EXPOSURE, MAX_WEIGHT_PER_SYMBOL, MIN_COMPOSITE_SCORE, MIN_TRADE_DOLLARS, MOMENTUM_LOOKBACK_DAYS, NEWS_SENTIMENT_CACHE_TTL_SECONDS, NEWS_SENTIMENT_PROVIDER_ORDER, PRICE_MOMENTUM_WEIGHT, REBALANCE_THRESHOLD, REQUIRE_TRADE_APPROVAL, SHORT_MOMENTUM_LOOKBACK_DAYS, SOCIAL_LOOKBACK_DAYS, SOCIAL_MOMENTUM_WEIGHT, SYMBOLS, TARGET_ANNUAL_VOL, TRADABLES_CSV, TRADE_APPROVAL_POLL_SECONDS, TRADE_APPROVAL_TIMEOUT_SECONDS, TRANSACTION_COST_BPS, UNNAMED_ACCOUNT_ID, VOLUME_LOOKBACK_DAYS, VOLUME_MOMENTUM_WEIGHT
from .yaml_io import load_accounts_config, load_algorithm_bot_config, load_algorithms_config, load_connectors_config, load_universe_config

from .accounts import UnknownAccountError, _normalize_accounts_config


@dataclass(frozen=True)
class Config:
    account_id: str = "default"
    account_label: str = "Default"

    algorithm_id: str = DEFAULT_STRATEGY_ID
    symbols: list[str] = field(default_factory=lambda: list(SYMBOLS))
    momentum_lookback_days: int = MOMENTUM_LOOKBACK_DAYS
    short_momentum_lookback_days: int = SHORT_MOMENTUM_LOOKBACK_DAYS
    long_ma_days: int = LONG_MA_DAYS
    volume_lookback_days: int = VOLUME_LOOKBACK_DAYS
    social_lookback_days: int = SOCIAL_LOOKBACK_DAYS
    max_weight_per_symbol: float = MAX_WEIGHT_PER_SYMBOL
    max_portfolio_exposure: float = MAX_PORTFOLIO_EXPOSURE
    max_longs: int = MAX_LONGS
    min_composite_score: float = MIN_COMPOSITE_SCORE
    price_momentum_weight: float = PRICE_MOMENTUM_WEIGHT
    social_momentum_weight: float = SOCIAL_MOMENTUM_WEIGHT
    volume_momentum_weight: float = VOLUME_MOMENTUM_WEIGHT
    target_annual_vol: float = TARGET_ANNUAL_VOL
    cash_buffer: float = CASH_BUFFER
    cash_equivalents: list[str] = field(default_factory=lambda: list(CASH_EQUIVALENTS))
    min_trade_dollars: float = MIN_TRADE_DOLLARS
    rebalance_threshold: float = REBALANCE_THRESHOLD
    transaction_cost_bps: float = TRANSACTION_COST_BPS
    backtest_starting_equity: float = BACKTEST_STARTING_EQUITY
    backtest_period: str = BACKTEST_PERIOD
    algorithm_equity_cap: float = ALGORITHM_EQUITY_CAP
    kill_switch: bool = KILL_SWITCH
    alpaca_api_key: str = ""
    alpaca_api_secret: str = ""
    alpaca_data_api_key: str = ""
    alpaca_data_api_secret: str = ""
    alpaca_base_url: str = ALPACA_BASE_URL
    alpaca_data_feed: str = ALPACA_DATA_FEED
    # Schwab OAuth: app key/secret from the developer portal, plus the refresh token from a
    # one-time browser consent. Left blank until the Schwab app is approved.
    schwab_app_key: str = ""
    schwab_app_secret: str = ""
    schwab_refresh_token: str = ""
    schwab_account_number: str = ""
    # Must byte-for-byte match a callback URL registered on the Schwab developer app.
    schwab_callback_url: str = ""
    paper_starting_cash: float = 100_000.0
    history_extra_buffer_days: int = HISTORY_EXTRA_BUFFER_DAYS
    social_trends_csv: str = ALPHA_VANTAGE_NEWS_CSV
    tradables_csv: str = TRADABLES_CSV
    alpha_vantage_api_key: str = ""
    alpha_vantage_news_csv: str = ALPHA_VANTAGE_NEWS_CSV
    alpha_vantage_news_lookback_days: int = ALPHA_VANTAGE_NEWS_LOOKBACK_DAYS
    alpha_vantage_news_limit: int = ALPHA_VANTAGE_NEWS_LIMIT
    alpha_vantage_max_symbols: int = ALPHA_VANTAGE_MAX_SYMBOLS
    alpha_vantage_request_delay_seconds: float = ALPHA_VANTAGE_REQUEST_DELAY_SECONDS
    account_options: list[dict[str, str]] = field(default_factory=list)
    algorithm_configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    market_data_provider_order: list[str] = field(default_factory=lambda: list(MARKET_DATA_PROVIDER_ORDER))
    intraday_market_data_provider_order: list[str] = field(default_factory=lambda: list(INTRADAY_MARKET_DATA_PROVIDER_ORDER))
    eod_market_data_provider_order: list[str] = field(default_factory=lambda: list(EOD_MARKET_DATA_PROVIDER_ORDER))
    # Streaming is opt-in per deployment: an empty order means no websocket is ever opened
    # and every price path behaves exactly as before.
    streaming_market_data_provider_order: list[str] = field(default_factory=list)
    news_sentiment_provider_order: list[str] = field(default_factory=lambda: list(NEWS_SENTIMENT_PROVIDER_ORDER))
    sentiment_data_provider_order: list[str] = field(default_factory=lambda: list(NEWS_SENTIMENT_PROVIDER_ORDER))
    dividend_provider_order: list[str] = field(default_factory=lambda: list(DIVIDEND_PROVIDER_ORDER))
    market_data_bar_minutes: int = MARKET_DATA_BAR_MINUTES
    market_data_cache_ttl_seconds: int = MARKET_DATA_CACHE_TTL_SECONDS
    intraday_market_data_cache_ttl_seconds: int = INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS
    eod_market_data_cache_ttl_seconds: int = EOD_MARKET_DATA_CACHE_TTL_SECONDS
    news_sentiment_cache_ttl_seconds: int = NEWS_SENTIMENT_CACHE_TTL_SECONDS
    sentiment_data_cache_ttl_seconds: int = NEWS_SENTIMENT_CACHE_TTL_SECONDS
    data_source_configs: dict[str, Any] = field(default_factory=dict)
    require_trade_approval: bool = REQUIRE_TRADE_APPROVAL
    trade_approval_timeout_seconds: int = TRADE_APPROVAL_TIMEOUT_SECONDS
    trade_approval_poll_seconds: int = TRADE_APPROVAL_POLL_SECONDS


def get_config(account_id: str | None = None, strategy_id: str | None = None) -> Config:
    raw_algorithm_bot_config = load_algorithm_bot_config()
    raw_accounts_config = load_accounts_config()
    default_account_id, account_items = _normalize_accounts_config(raw_accounts_config)
    if not default_account_id:
        default_account_id = os.getenv("TRADING_ACCOUNT_ID") or UNNAMED_ACCOUNT_ID
    selected_account_id = str(account_id or os.getenv("TRADING_ACCOUNT_ID") or default_account_id)
    account_config = _section(account_items, selected_account_id)
    if not account_config and account_items:
        # Asking for a *specific* account and getting a different one is never the right
        # answer. This used to substitute the default silently, which meant an account page
        # could show another account's money under the requested name -- and worse, that
        # ``live_runner.run_once(account_id=...)`` would resolve a renamed or deleted binding
        # to the default account and send its orders there. Falling back is only defensible
        # when no account was named at all.
        if account_id and account_id != UNNAMED_ACCOUNT_ID:
            raise UnknownAccountError(str(account_id), sorted(account_items))
        selected_account_id = default_account_id if default_account_id in account_items else next(iter(account_items))
        account_config = _section(account_items, selected_account_id)

    raw_universe_config = load_universe_config()
    universe = _section(raw_universe_config, "tradable_universe")
    raw_algorithms_config = load_algorithms_config()
    algorithm_configs = _algorithm_sections(raw_algorithms_config)

    selected_strategy_id = canonical_algorithm_id(str(strategy_id or DEFAULT_STRATEGY_ID))
    algorithm = _section(algorithm_configs, selected_strategy_id)
    for legacy_id in LEGACY_ALGORITHM_IDS.get(selected_strategy_id, []):
        # A renamed algorithm's tuning is still filed under its old id. Reading every retired
        # id keeps saved tuning working through the rename instead of silently reverting to
        # defaults.
        if algorithm:
            break
        algorithm = _section(algorithm_configs, legacy_id)
    algorithm_bot = _section(raw_algorithm_bot_config, "algorithm_bot")
    runtime = {
        **_section(raw_algorithm_bot_config, "runtime"),
        **{
            key: value
            for key, value in algorithm_bot.items()
            if key in {
                "backtest_period",
                "require_trade_approval",
                "trade_approval_timeout_seconds",
                "trade_approval_poll_seconds",
            }
        },
    }
    social = _section(raw_algorithm_bot_config, "social")
    alpha_vantage = _section(raw_algorithm_bot_config, "alpha_vantage")
    raw_connectors_config = load_connectors_config()
    data_sources = _normalize_data_sources(raw_connectors_config)
    market_sources = _section(data_sources, "market_data")
    intraday_market_sources = _section(data_sources, "intraday_market_data")
    eod_market_sources = _section(data_sources, "eod_market_data")
    streaming_market_sources = _section(data_sources, "streaming_market_data")
    news_sources = _section(data_sources, "news_sentiment")
    dividend_sources = _section(data_sources, "dividends")
    sentiment_sources = _section(data_sources, "sentiment_data")

    # One reader per section: ``read_x("key", DEFAULT)`` casts by the default's type and looks
    # for the key upper-cased in the environment. See ``coercion.reader``.
    read_account_config = reader(account_config)
    read_algorithm = reader(algorithm)
    read_alpha_vantage = reader(alpha_vantage)
    read_dividend_sources = reader(dividend_sources)
    read_eod_market_sources = reader(eod_market_sources)
    read_streaming_sources = reader(streaming_market_sources)
    read_intraday_market_sources = reader(intraday_market_sources)
    read_market_sources = reader(market_sources)
    read_news_sources = reader(news_sources)
    read_runtime = reader(runtime)
    read_sentiment_sources = reader(sentiment_sources)

    # Env only, deliberately. It is a deployment-level brake for an emergency, not a control
    # the dashboard offers: whether an algorithm trades is its binding's own switch.
    kill_switch = _str_to_bool(os.getenv("KILL_SWITCH"), KILL_SWITCH)
    api_key = direct_or_env(account_config, "api_key", "api_key_env", "ALPACA_API_KEY")
    api_secret = direct_or_env(account_config, "api_secret", "api_secret_env", "ALPACA_API_SECRET")
    alpaca_data_api_key = _provider_credential(
        data_sources,
        "eod_market_data",
        "alpaca",
        "api_key",
        "api_key_env",
        "ALPACA_API_KEY",
    ) or api_key
    alpaca_data_api_secret = _provider_credential(
        data_sources,
        "eod_market_data",
        "alpaca",
        "api_secret",
        "api_secret_env",
        "ALPACA_API_SECRET",
    ) or api_secret
    base_url = str(
        _config_value(account_config, "base_url", "ALPACA_BASE_URL", ALPACA_BASE_URL)
    ).strip() or ALPACA_BASE_URL
    symbols_env = os.getenv("SYMBOLS")
    tradables_csv = str(
        _config_value(
            universe,
            "master_list",
            "TRADABLES_CSV",
            universe.get("tradables_csv", TRADABLES_CSV),
        )
    )
    yaml_symbols = universe.get("symbols")
    if symbols_env is not None:
        symbols = _parse_symbols(symbols_env, SYMBOLS)
    elif isinstance(yaml_symbols, list) and yaml_symbols:
        symbols = [str(item).strip().upper() for item in yaml_symbols if str(item).strip()]
    else:
        tradable_names = load_tradable_names(tradables_csv)
        symbols = list(SYMBOLS)
        if tradable_names:
            symbols = [symbol for symbol in symbols if symbol in tradable_names]
    if not symbols:
        raise ValueError("Configured trading universe must include at least one symbol.")
    alpha_vantage_api_key = (
        direct_or_env(alpha_vantage, "api_key", "api_key_env", "ALPHA_VANTAGE_API_KEY")
        or _provider_secret(data_sources, "market_data", "alpha_vantage", "ALPHA_VANTAGE_API_KEY")
        or _provider_secret(data_sources, "news_sentiment", "alpha_vantage", "ALPHA_VANTAGE_API_KEY")
    )
    account_options = [
        {"id": str(key), "label": str(_section(account_items, str(key)).get("label") or key)}
        for key in account_items
    ]
    alpha_vantage_news_csv = str(_config_value(alpha_vantage, "news_csv", "ALPHA_VANTAGE_NEWS_CSV", ALPHA_VANTAGE_NEWS_CSV))
    social_trends_csv = str(social.get("trends_csv") or alpha_vantage_news_csv)

    return Config(
        account_id=selected_account_id,
        account_label=str(account_config.get("label") or selected_account_id),
        algorithm_id=selected_strategy_id,
        symbols=symbols,
        momentum_lookback_days=read_algorithm("momentum_lookback_days", MOMENTUM_LOOKBACK_DAYS),
        short_momentum_lookback_days=read_algorithm("short_momentum_lookback_days", SHORT_MOMENTUM_LOOKBACK_DAYS),
        long_ma_days=read_algorithm("long_ma_days", LONG_MA_DAYS),
        volume_lookback_days=read_algorithm("volume_lookback_days", VOLUME_LOOKBACK_DAYS),
        social_lookback_days=read_algorithm("social_lookback_days", SOCIAL_LOOKBACK_DAYS),
        max_weight_per_symbol=read_algorithm("max_weight_per_symbol", MAX_WEIGHT_PER_SYMBOL),
        max_portfolio_exposure=read_algorithm("max_portfolio_exposure", MAX_PORTFOLIO_EXPOSURE),
        max_longs=read_algorithm("max_longs", MAX_LONGS),
        min_composite_score=read_algorithm("min_composite_score", MIN_COMPOSITE_SCORE),
        price_momentum_weight=read_algorithm("price_momentum_weight", PRICE_MOMENTUM_WEIGHT),
        social_momentum_weight=read_algorithm("social_momentum_weight", SOCIAL_MOMENTUM_WEIGHT),
        volume_momentum_weight=read_algorithm("volume_momentum_weight", VOLUME_MOMENTUM_WEIGHT),
        target_annual_vol=read_algorithm("target_annual_vol", TARGET_ANNUAL_VOL),
        # Account-level, not per-algorithm: both describe how *this account* funds a batch,
        # which is not something a strategy has an opinion about. Exposure is the strategy's
        # call and it already has gross-exposure caps to say so with.
        cash_buffer=read_account_config("cash_buffer", CASH_BUFFER),
        cash_equivalents=[str(s).strip().upper() for s in read_account_config("cash_equivalents", list(CASH_EQUIVALENTS)) if str(s).strip()],
        min_trade_dollars=read_algorithm("min_trade_dollars", MIN_TRADE_DOLLARS),
        rebalance_threshold=read_algorithm("rebalance_threshold", REBALANCE_THRESHOLD),
        transaction_cost_bps=read_algorithm("transaction_cost_bps", TRANSACTION_COST_BPS),
        backtest_starting_equity=read_algorithm("backtest_starting_equity", BACKTEST_STARTING_EQUITY),
        backtest_period=str(_config_value(runtime, "backtest_period", "BACKTEST_PERIOD", BACKTEST_PERIOD)).strip().lower() or BACKTEST_PERIOD,
        algorithm_equity_cap=read_algorithm("algorithm_equity_cap", ALGORITHM_EQUITY_CAP),
        kill_switch=kill_switch,
        alpaca_api_key=api_key,
        alpaca_api_secret=api_secret,
        alpaca_data_api_key=alpaca_data_api_key,
        alpaca_data_api_secret=alpaca_data_api_secret,
        alpaca_base_url=base_url,
        alpaca_data_feed=read_account_config("data_feed", ALPACA_DATA_FEED, env="ALPACA_DATA_FEED"),
        schwab_app_key=read_account_config("schwab_app_key", ""),
        schwab_app_secret=read_account_config("schwab_app_secret", ""),
        schwab_refresh_token=read_account_config("schwab_refresh_token", ""),
        schwab_account_number=read_account_config("schwab_account_number", ""),
        schwab_callback_url=read_account_config("schwab_callback_url", ""),
        history_extra_buffer_days=read_algorithm("history_extra_buffer_days", HISTORY_EXTRA_BUFFER_DAYS),
        social_trends_csv=social_trends_csv,
        tradables_csv=tradables_csv,
        alpha_vantage_api_key=alpha_vantage_api_key,
        alpha_vantage_news_csv=alpha_vantage_news_csv,
        alpha_vantage_news_lookback_days=read_alpha_vantage("news_lookback_days", ALPHA_VANTAGE_NEWS_LOOKBACK_DAYS, env="ALPHA_VANTAGE_NEWS_LOOKBACK_DAYS"),
        alpha_vantage_news_limit=read_alpha_vantage("news_limit", ALPHA_VANTAGE_NEWS_LIMIT, env="ALPHA_VANTAGE_NEWS_LIMIT"),
        alpha_vantage_max_symbols=read_alpha_vantage("max_symbols", ALPHA_VANTAGE_MAX_SYMBOLS, env="ALPHA_VANTAGE_MAX_SYMBOLS"),
        alpha_vantage_request_delay_seconds=read_alpha_vantage("request_delay_seconds", ALPHA_VANTAGE_REQUEST_DELAY_SECONDS, env="ALPHA_VANTAGE_REQUEST_DELAY_SECONDS"),
        account_options=account_options,
        algorithm_configs=algorithm_configs,
        market_data_provider_order=read_market_sources("provider_order", MARKET_DATA_PROVIDER_ORDER, env="MARKET_DATA_PROVIDER_ORDER"),
        intraday_market_data_provider_order=read_intraday_market_sources("provider_order", INTRADAY_MARKET_DATA_PROVIDER_ORDER, env="INTRADAY_MARKET_DATA_PROVIDER_ORDER"),
        eod_market_data_provider_order=read_eod_market_sources("provider_order", EOD_MARKET_DATA_PROVIDER_ORDER, env="EOD_MARKET_DATA_PROVIDER_ORDER"),
        streaming_market_data_provider_order=read_streaming_sources("provider_order", [], env="STREAMING_MARKET_DATA_PROVIDER_ORDER"),
        news_sentiment_provider_order=read_news_sources("provider_order", NEWS_SENTIMENT_PROVIDER_ORDER, env="NEWS_SENTIMENT_PROVIDER_ORDER"),
        sentiment_data_provider_order=read_sentiment_sources("provider_order", NEWS_SENTIMENT_PROVIDER_ORDER, env="SENTIMENT_DATA_PROVIDER_ORDER"),
        dividend_provider_order=read_dividend_sources("provider_order", DIVIDEND_PROVIDER_ORDER, env="DIVIDEND_PROVIDER_ORDER"),
        market_data_bar_minutes=read_intraday_market_sources("bar_minutes", MARKET_DATA_BAR_MINUTES, env="MARKET_DATA_BAR_MINUTES"),
        market_data_cache_ttl_seconds=read_market_sources("cache_ttl_seconds", MARKET_DATA_CACHE_TTL_SECONDS, env="MARKET_DATA_CACHE_TTL_SECONDS"),
        intraday_market_data_cache_ttl_seconds=read_intraday_market_sources("cache_ttl_seconds", INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS, env="INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS"),
        eod_market_data_cache_ttl_seconds=read_eod_market_sources("cache_ttl_seconds", EOD_MARKET_DATA_CACHE_TTL_SECONDS, env="EOD_MARKET_DATA_CACHE_TTL_SECONDS"),
        news_sentiment_cache_ttl_seconds=read_news_sources("cache_ttl_seconds", NEWS_SENTIMENT_CACHE_TTL_SECONDS, env="NEWS_SENTIMENT_CACHE_TTL_SECONDS"),
        sentiment_data_cache_ttl_seconds=read_sentiment_sources("cache_ttl_seconds", NEWS_SENTIMENT_CACHE_TTL_SECONDS, env="SENTIMENT_DATA_CACHE_TTL_SECONDS"),
        data_source_configs=data_sources,
        require_trade_approval=read_runtime("require_trade_approval", REQUIRE_TRADE_APPROVAL),
        trade_approval_timeout_seconds=read_runtime("trade_approval_timeout_seconds", TRADE_APPROVAL_TIMEOUT_SECONDS),
        trade_approval_poll_seconds=read_runtime("trade_approval_poll_seconds", TRADE_APPROVAL_POLL_SECONDS),
    )
