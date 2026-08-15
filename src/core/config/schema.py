"""The ``Config`` dataclass and the one function that assembles it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ...data.universe import load_tradable_names

from .coercion import _algorithm_sections, _as_float, _as_int, _as_list, _config_value, _direct_or_env, _normalize_data_sources, _parse_symbols, _provider_credential, _provider_secret, _section, _str_to_bool
from .defaults import ALGORITHM_EQUITY_CAP, ALPACA_BASE_URL, ALPACA_DATA_FEED, ALPHA_VANTAGE_MAX_SYMBOLS, ALPHA_VANTAGE_NEWS_CSV, ALPHA_VANTAGE_NEWS_LIMIT, ALPHA_VANTAGE_NEWS_LOOKBACK_DAYS, ALPHA_VANTAGE_REQUEST_DELAY_SECONDS, BACKTEST_PERIOD, BACKTEST_STARTING_EQUITY, CASH_BUFFER, DEFAULT_STRATEGY_ID, DIVIDEND_PROVIDER_ORDER, EOD_MARKET_DATA_CACHE_TTL_SECONDS, EOD_MARKET_DATA_PROVIDER_ORDER, HISTORY_EXTRA_BUFFER_DAYS, INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS, INTRADAY_MARKET_DATA_PROVIDER_ORDER, KILL_SWITCH, LONG_MA_DAYS, MARKET_DATA_BAR_MINUTES, MARKET_DATA_CACHE_TTL_SECONDS, MARKET_DATA_PROVIDER_ORDER, MAX_LONGS, MAX_PORTFOLIO_EXPOSURE, MAX_WEIGHT_PER_SYMBOL, MIN_COMPOSITE_SCORE, MIN_TRADE_DOLLARS, MOMENTUM_LOOKBACK_DAYS, NEWS_SENTIMENT_CACHE_TTL_SECONDS, NEWS_SENTIMENT_PROVIDER_ORDER, OPTIONS_SWING_DTE_MAX, OPTIONS_SWING_DTE_MIN, OPTIONS_SWING_MAX_CONTRACTS, OPTIONS_SWING_MAX_DELTA, OPTIONS_SWING_MAX_PREMIUM, OPTIONS_SWING_MAX_SPREAD_PCT, OPTIONS_SWING_MIN_DELTA, OPTIONS_SWING_MIN_OPEN_INTEREST, OPTIONS_SWING_STRIKE_RANGE_PCT, PRICE_MOMENTUM_WEIGHT, REBALANCE_THRESHOLD, REQUIRE_TRADE_APPROVAL, SHORT_MOMENTUM_LOOKBACK_DAYS, SOCIAL_LOOKBACK_DAYS, SOCIAL_MOMENTUM_WEIGHT, SYMBOLS, TARGET_ANNUAL_VOL, TRADABLES_CSV, TRADE_APPROVAL_POLL_SECONDS, TRADE_APPROVAL_TIMEOUT_SECONDS, TRANSACTION_COST_BPS, UNNAMED_ACCOUNT_ID, VOLUME_LOOKBACK_DAYS, VOLUME_MOMENTUM_WEIGHT
from .yaml_io import load_accounts_config, load_algorithm_bot_config, load_algorithms_config, load_connectors_config, load_options_bot_config, load_universe_config

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
    options_swing_dte_min: int = OPTIONS_SWING_DTE_MIN
    options_swing_dte_max: int = OPTIONS_SWING_DTE_MAX
    options_swing_min_delta: float = OPTIONS_SWING_MIN_DELTA
    options_swing_max_delta: float = OPTIONS_SWING_MAX_DELTA
    options_swing_max_contracts: int = OPTIONS_SWING_MAX_CONTRACTS
    options_swing_max_premium: float = OPTIONS_SWING_MAX_PREMIUM
    options_swing_min_open_interest: int = OPTIONS_SWING_MIN_OPEN_INTEREST
    options_swing_max_spread_pct: float = OPTIONS_SWING_MAX_SPREAD_PCT
    options_swing_strike_range_pct: float = OPTIONS_SWING_STRIKE_RANGE_PCT


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
    # Imported here rather than at module scope: the algorithm registry reaches back into
    # config through the algorithms it loads.
    from ...algorithms.registry import LEGACY_ALGORITHM_IDS, canonical_algorithm_id

    selected_strategy_id = canonical_algorithm_id(str(strategy_id or DEFAULT_STRATEGY_ID))
    algorithm = _section(algorithm_configs, selected_strategy_id)
    for legacy_id in LEGACY_ALGORITHM_IDS.get(selected_strategy_id, []):
        # A renamed algorithm's tuning is still filed under its old id. Reading every retired
        # id keeps saved tuning working through the rename instead of silently reverting to
        # defaults -- ``spy_rotation`` has two, having been renamed twice.
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
    raw_options_bot_config = load_options_bot_config()
    options = _section(raw_options_bot_config, "options")
    social = _section(raw_algorithm_bot_config, "social")
    alpha_vantage = _section(raw_algorithm_bot_config, "alpha_vantage")
    raw_connectors_config = load_connectors_config()
    data_sources = _normalize_data_sources(raw_connectors_config)
    market_sources = _section(data_sources, "market_data")
    intraday_market_sources = _section(data_sources, "intraday_market_data")
    eod_market_sources = _section(data_sources, "eod_market_data")
    news_sources = _section(data_sources, "news_sentiment")
    dividend_sources = _section(data_sources, "dividends")
    sentiment_sources = _section(data_sources, "sentiment_data")

    # Env only, deliberately. It is a deployment-level brake for an emergency, not a control
    # the dashboard offers: whether an algorithm trades is its binding's own switch.
    kill_switch = _str_to_bool(os.getenv("KILL_SWITCH"), KILL_SWITCH)
    api_key = _direct_or_env(account_config, "api_key", "api_key_env", "ALPACA_API_KEY")
    api_secret = _direct_or_env(account_config, "api_secret", "api_secret_env", "ALPACA_API_SECRET")
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
        _direct_or_env(alpha_vantage, "api_key", "api_key_env", "ALPHA_VANTAGE_API_KEY")
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
        momentum_lookback_days=_as_int(_config_value(algorithm, "momentum_lookback_days", "MOMENTUM_LOOKBACK_DAYS", MOMENTUM_LOOKBACK_DAYS), MOMENTUM_LOOKBACK_DAYS),
        short_momentum_lookback_days=_as_int(_config_value(algorithm, "short_momentum_lookback_days", "SHORT_MOMENTUM_LOOKBACK_DAYS", SHORT_MOMENTUM_LOOKBACK_DAYS), SHORT_MOMENTUM_LOOKBACK_DAYS),
        long_ma_days=_as_int(_config_value(algorithm, "long_ma_days", "LONG_MA_DAYS", LONG_MA_DAYS), LONG_MA_DAYS),
        volume_lookback_days=_as_int(_config_value(algorithm, "volume_lookback_days", "VOLUME_LOOKBACK_DAYS", VOLUME_LOOKBACK_DAYS), VOLUME_LOOKBACK_DAYS),
        social_lookback_days=_as_int(_config_value(algorithm, "social_lookback_days", "SOCIAL_LOOKBACK_DAYS", SOCIAL_LOOKBACK_DAYS), SOCIAL_LOOKBACK_DAYS),
        max_weight_per_symbol=_as_float(_config_value(algorithm, "max_weight_per_symbol", "MAX_WEIGHT_PER_SYMBOL", MAX_WEIGHT_PER_SYMBOL), MAX_WEIGHT_PER_SYMBOL),
        max_portfolio_exposure=_as_float(_config_value(algorithm, "max_portfolio_exposure", "MAX_PORTFOLIO_EXPOSURE", MAX_PORTFOLIO_EXPOSURE), MAX_PORTFOLIO_EXPOSURE),
        max_longs=_as_int(_config_value(algorithm, "max_longs", "MAX_LONGS", MAX_LONGS), MAX_LONGS),
        min_composite_score=_as_float(_config_value(algorithm, "min_composite_score", "MIN_COMPOSITE_SCORE", MIN_COMPOSITE_SCORE), MIN_COMPOSITE_SCORE),
        price_momentum_weight=_as_float(_config_value(algorithm, "price_momentum_weight", "PRICE_MOMENTUM_WEIGHT", PRICE_MOMENTUM_WEIGHT), PRICE_MOMENTUM_WEIGHT),
        social_momentum_weight=_as_float(_config_value(algorithm, "social_momentum_weight", "SOCIAL_MOMENTUM_WEIGHT", SOCIAL_MOMENTUM_WEIGHT), SOCIAL_MOMENTUM_WEIGHT),
        volume_momentum_weight=_as_float(_config_value(algorithm, "volume_momentum_weight", "VOLUME_MOMENTUM_WEIGHT", VOLUME_MOMENTUM_WEIGHT), VOLUME_MOMENTUM_WEIGHT),
        target_annual_vol=_as_float(_config_value(algorithm, "target_annual_vol", "TARGET_ANNUAL_VOL", TARGET_ANNUAL_VOL), TARGET_ANNUAL_VOL),
        cash_buffer=_as_float(_config_value(algorithm, "cash_buffer", "CASH_BUFFER", CASH_BUFFER), CASH_BUFFER),
        min_trade_dollars=_as_float(_config_value(algorithm, "min_trade_dollars", "MIN_TRADE_DOLLARS", MIN_TRADE_DOLLARS), MIN_TRADE_DOLLARS),
        rebalance_threshold=_as_float(_config_value(algorithm, "rebalance_threshold", "REBALANCE_THRESHOLD", REBALANCE_THRESHOLD), REBALANCE_THRESHOLD),
        transaction_cost_bps=_as_float(_config_value(algorithm, "transaction_cost_bps", "TRANSACTION_COST_BPS", TRANSACTION_COST_BPS), TRANSACTION_COST_BPS),
        backtest_starting_equity=_as_float(_config_value(algorithm, "backtest_starting_equity", "BACKTEST_STARTING_EQUITY", BACKTEST_STARTING_EQUITY), BACKTEST_STARTING_EQUITY),
        backtest_period=str(_config_value(runtime, "backtest_period", "BACKTEST_PERIOD", BACKTEST_PERIOD)).strip().lower() or BACKTEST_PERIOD,
        algorithm_equity_cap=_as_float(_config_value(algorithm, "algorithm_equity_cap", "ALGORITHM_EQUITY_CAP", ALGORITHM_EQUITY_CAP), ALGORITHM_EQUITY_CAP),
        kill_switch=kill_switch,
        alpaca_api_key=api_key,
        alpaca_api_secret=api_secret,
        alpaca_data_api_key=alpaca_data_api_key,
        alpaca_data_api_secret=alpaca_data_api_secret,
        alpaca_base_url=base_url,
        alpaca_data_feed=str(_config_value(account_config, "data_feed", "ALPACA_DATA_FEED", ALPACA_DATA_FEED)),
        schwab_app_key=str(_config_value(account_config, "schwab_app_key", "SCHWAB_APP_KEY", "") or ""),
        schwab_app_secret=str(_config_value(account_config, "schwab_app_secret", "SCHWAB_APP_SECRET", "") or ""),
        schwab_refresh_token=str(_config_value(account_config, "schwab_refresh_token", "SCHWAB_REFRESH_TOKEN", "") or ""),
        schwab_account_number=str(_config_value(account_config, "schwab_account_number", "SCHWAB_ACCOUNT_NUMBER", "") or ""),
        schwab_callback_url=str(_config_value(account_config, "schwab_callback_url", "SCHWAB_CALLBACK_URL", "") or ""),
        history_extra_buffer_days=_as_int(_config_value(algorithm, "history_extra_buffer_days", "HISTORY_EXTRA_BUFFER_DAYS", HISTORY_EXTRA_BUFFER_DAYS), HISTORY_EXTRA_BUFFER_DAYS),
        social_trends_csv=social_trends_csv,
        tradables_csv=tradables_csv,
        alpha_vantage_api_key=alpha_vantage_api_key,
        alpha_vantage_news_csv=alpha_vantage_news_csv,
        alpha_vantage_news_lookback_days=_as_int(_config_value(alpha_vantage, "news_lookback_days", "ALPHA_VANTAGE_NEWS_LOOKBACK_DAYS", ALPHA_VANTAGE_NEWS_LOOKBACK_DAYS), ALPHA_VANTAGE_NEWS_LOOKBACK_DAYS),
        alpha_vantage_news_limit=_as_int(_config_value(alpha_vantage, "news_limit", "ALPHA_VANTAGE_NEWS_LIMIT", ALPHA_VANTAGE_NEWS_LIMIT), ALPHA_VANTAGE_NEWS_LIMIT),
        alpha_vantage_max_symbols=_as_int(_config_value(alpha_vantage, "max_symbols", "ALPHA_VANTAGE_MAX_SYMBOLS", ALPHA_VANTAGE_MAX_SYMBOLS), ALPHA_VANTAGE_MAX_SYMBOLS),
        alpha_vantage_request_delay_seconds=_as_float(_config_value(alpha_vantage, "request_delay_seconds", "ALPHA_VANTAGE_REQUEST_DELAY_SECONDS", ALPHA_VANTAGE_REQUEST_DELAY_SECONDS), ALPHA_VANTAGE_REQUEST_DELAY_SECONDS),
        account_options=account_options,
        algorithm_configs=algorithm_configs,
        market_data_provider_order=_as_list(
            _config_value(market_sources, "provider_order", "MARKET_DATA_PROVIDER_ORDER", MARKET_DATA_PROVIDER_ORDER),
            MARKET_DATA_PROVIDER_ORDER,
        ),
        intraday_market_data_provider_order=_as_list(
            _config_value(
                intraday_market_sources,
                "provider_order",
                "INTRADAY_MARKET_DATA_PROVIDER_ORDER",
                INTRADAY_MARKET_DATA_PROVIDER_ORDER,
            ),
            INTRADAY_MARKET_DATA_PROVIDER_ORDER,
        ),
        eod_market_data_provider_order=_as_list(
            _config_value(
                eod_market_sources,
                "provider_order",
                "EOD_MARKET_DATA_PROVIDER_ORDER",
                EOD_MARKET_DATA_PROVIDER_ORDER,
            ),
            EOD_MARKET_DATA_PROVIDER_ORDER,
        ),
        news_sentiment_provider_order=_as_list(
            _config_value(news_sources, "provider_order", "NEWS_SENTIMENT_PROVIDER_ORDER", NEWS_SENTIMENT_PROVIDER_ORDER),
            NEWS_SENTIMENT_PROVIDER_ORDER,
        ),
        sentiment_data_provider_order=_as_list(
            _config_value(sentiment_sources, "provider_order", "SENTIMENT_DATA_PROVIDER_ORDER", NEWS_SENTIMENT_PROVIDER_ORDER),
            NEWS_SENTIMENT_PROVIDER_ORDER,
        ),
        dividend_provider_order=_as_list(
            _config_value(dividend_sources, "provider_order", "DIVIDEND_PROVIDER_ORDER", DIVIDEND_PROVIDER_ORDER),
            DIVIDEND_PROVIDER_ORDER,
        ),
        market_data_bar_minutes=_as_int(
            _config_value(
                intraday_market_sources, "bar_minutes", "MARKET_DATA_BAR_MINUTES", MARKET_DATA_BAR_MINUTES
            ),
            MARKET_DATA_BAR_MINUTES,
        ),
        market_data_cache_ttl_seconds=_as_int(
            _config_value(market_sources, "cache_ttl_seconds", "MARKET_DATA_CACHE_TTL_SECONDS", MARKET_DATA_CACHE_TTL_SECONDS),
            MARKET_DATA_CACHE_TTL_SECONDS,
        ),
        intraday_market_data_cache_ttl_seconds=_as_int(
            _config_value(
                intraday_market_sources,
                "cache_ttl_seconds",
                "INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS",
                INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS,
            ),
            INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS,
        ),
        eod_market_data_cache_ttl_seconds=_as_int(
            _config_value(
                eod_market_sources,
                "cache_ttl_seconds",
                "EOD_MARKET_DATA_CACHE_TTL_SECONDS",
                EOD_MARKET_DATA_CACHE_TTL_SECONDS,
            ),
            EOD_MARKET_DATA_CACHE_TTL_SECONDS,
        ),
        news_sentiment_cache_ttl_seconds=_as_int(
            _config_value(news_sources, "cache_ttl_seconds", "NEWS_SENTIMENT_CACHE_TTL_SECONDS", NEWS_SENTIMENT_CACHE_TTL_SECONDS),
            NEWS_SENTIMENT_CACHE_TTL_SECONDS,
        ),
        sentiment_data_cache_ttl_seconds=_as_int(
            _config_value(sentiment_sources, "cache_ttl_seconds", "SENTIMENT_DATA_CACHE_TTL_SECONDS", NEWS_SENTIMENT_CACHE_TTL_SECONDS),
            NEWS_SENTIMENT_CACHE_TTL_SECONDS,
        ),
        data_source_configs=data_sources,
        require_trade_approval=_str_to_bool(
            str(_config_value(runtime, "require_trade_approval", "REQUIRE_TRADE_APPROVAL", REQUIRE_TRADE_APPROVAL)),
            REQUIRE_TRADE_APPROVAL,
        ),
        trade_approval_timeout_seconds=_as_int(
            _config_value(runtime, "trade_approval_timeout_seconds", "TRADE_APPROVAL_TIMEOUT_SECONDS", TRADE_APPROVAL_TIMEOUT_SECONDS),
            TRADE_APPROVAL_TIMEOUT_SECONDS,
        ),
        trade_approval_poll_seconds=_as_int(
            _config_value(runtime, "trade_approval_poll_seconds", "TRADE_APPROVAL_POLL_SECONDS", TRADE_APPROVAL_POLL_SECONDS),
            TRADE_APPROVAL_POLL_SECONDS,
        ),
        options_swing_dte_min=_as_int(_config_value(options, "swing_dte_min", "OPTIONS_SWING_DTE_MIN", OPTIONS_SWING_DTE_MIN), OPTIONS_SWING_DTE_MIN),
        options_swing_dte_max=_as_int(_config_value(options, "swing_dte_max", "OPTIONS_SWING_DTE_MAX", OPTIONS_SWING_DTE_MAX), OPTIONS_SWING_DTE_MAX),
        options_swing_min_delta=_as_float(_config_value(options, "swing_min_delta", "OPTIONS_SWING_MIN_DELTA", OPTIONS_SWING_MIN_DELTA), OPTIONS_SWING_MIN_DELTA),
        options_swing_max_delta=_as_float(_config_value(options, "swing_max_delta", "OPTIONS_SWING_MAX_DELTA", OPTIONS_SWING_MAX_DELTA), OPTIONS_SWING_MAX_DELTA),
        options_swing_max_contracts=_as_int(_config_value(options, "swing_max_contracts", "OPTIONS_SWING_MAX_CONTRACTS", OPTIONS_SWING_MAX_CONTRACTS), OPTIONS_SWING_MAX_CONTRACTS),
        options_swing_max_premium=_as_float(_config_value(options, "swing_max_premium", "OPTIONS_SWING_MAX_PREMIUM", OPTIONS_SWING_MAX_PREMIUM), OPTIONS_SWING_MAX_PREMIUM),
        options_swing_min_open_interest=_as_int(_config_value(options, "swing_min_open_interest", "OPTIONS_SWING_MIN_OPEN_INTEREST", OPTIONS_SWING_MIN_OPEN_INTEREST), OPTIONS_SWING_MIN_OPEN_INTEREST),
        options_swing_max_spread_pct=_as_float(_config_value(options, "swing_max_spread_pct", "OPTIONS_SWING_MAX_SPREAD_PCT", OPTIONS_SWING_MAX_SPREAD_PCT), OPTIONS_SWING_MAX_SPREAD_PCT),
        options_swing_strike_range_pct=_as_float(_config_value(options, "swing_strike_range_pct", "OPTIONS_SWING_STRIKE_RANGE_PCT", OPTIONS_SWING_STRIKE_RANGE_PCT), OPTIONS_SWING_STRIKE_RANGE_PCT),
    )
