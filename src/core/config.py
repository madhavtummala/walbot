from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..data.universe import load_tradable_names

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import yaml
except ImportError:  # pragma: no cover - exercised when PyYAML is not installed.
    yaml = None


SYMBOLS: list[str] = [
    "SPY",
    "QQQ",
    "IBIT",
    "GLD",
    "TLT",
]
MOMENTUM_LOOKBACK_DAYS = 63
SHORT_MOMENTUM_LOOKBACK_DAYS = 21
LONG_MA_DAYS = 200
VOLUME_LOOKBACK_DAYS = 20
SOCIAL_LOOKBACK_DAYS = 30
MAX_WEIGHT_PER_SYMBOL = 0.25
MAX_PORTFOLIO_EXPOSURE = 0.95
MAX_LONGS = 5
MIN_COMPOSITE_SCORE = 0.05
PRICE_MOMENTUM_WEIGHT = 0.55
SOCIAL_MOMENTUM_WEIGHT = 0.30
VOLUME_MOMENTUM_WEIGHT = 0.15
TARGET_ANNUAL_VOL = 0.18
CASH_BUFFER = 0.02
MIN_TRADE_DOLLARS = 50.0
REBALANCE_THRESHOLD = 0.02
TRANSACTION_COST_BPS = 1.0
BACKTEST_STARTING_EQUITY = 10_000.0
BACKTEST_PERIOD = "4m"
ALGORITHM_EQUITY_CAP = 0.0
KILL_SWITCH = False
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
HISTORY_EXTRA_BUFFER_DAYS = 250
TRADABLES_CSV = "data/tradable_etfs.csv"
ALPACA_DATA_FEED = "iex"
ALPHA_VANTAGE_NEWS_CSV = "data/social_trends.csv"
ALPHA_VANTAGE_NEWS_LOOKBACK_DAYS = 30
ALPHA_VANTAGE_NEWS_LIMIT = 50
ALPHA_VANTAGE_MAX_SYMBOLS = 20
ALPHA_VANTAGE_REQUEST_DELAY_SECONDS = 0.0
ACCOUNTS_FILE = "config/accounts.yaml"
CONNECTORS_FILE = "config/connectors.yaml"
ALGORITHMS_FILE = "config/algorithms.yaml"
ALGORITHM_BOT_FILE = "config/algorithm_bot.yaml"
OPTIONS_BOT_FILE = "config/options_bot.yaml"
DCA_BOT_FILE = "config/dca_bot.yaml"
UNIVERSE_FILE = "config/universe.yaml"
MARKET_DATA_PROVIDER_ORDER: list[str] = []
INTRADAY_MARKET_DATA_PROVIDER_ORDER: list[str] = ["yfinance"]
EOD_MARKET_DATA_PROVIDER_ORDER: list[str] = []
NEWS_SENTIMENT_PROVIDER_ORDER: list[str] = []
MARKET_DATA_CACHE_TTL_SECONDS = 1800
INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS = 900
EOD_MARKET_DATA_CACHE_TTL_SECONDS = 1800
NEWS_SENTIMENT_CACHE_TTL_SECONDS = 1800
ALGORITHM_CHECK_SECONDS = 60
DCA_CHECK_SECONDS = 300
ALGORITHM_MARKET_DATA_REFRESH_MINUTES = 30
ALGORITHM_RUN_JITTER_MINUTES = 0
TRADING_START_TIME = "08:30"
TRADING_END_TIME = "15:00"
REQUIRE_TRADE_APPROVAL = False
TRADE_APPROVAL_TIMEOUT_SECONDS = 300
TRADE_APPROVAL_POLL_SECONDS = 5
OPTIONS_SWING_DTE_MIN = 30
OPTIONS_SWING_DTE_MAX = 60
OPTIONS_SWING_MIN_DELTA = 0.35
OPTIONS_SWING_MAX_DELTA = 0.65
OPTIONS_SWING_MAX_CONTRACTS = 1
OPTIONS_SWING_MAX_PREMIUM = 500.0
OPTIONS_SWING_MIN_OPEN_INTEREST = 100
OPTIONS_SWING_MAX_SPREAD_PCT = 0.20
OPTIONS_SWING_STRIKE_RANGE_PCT = 0.15
ALGORITHM_IDS = {
    "momentum_social",
    "trend_following",
    "mean_reversion",
    "breakout",
    "risk_parity",
    "dual_momentum",
    "fast_momentum",
    "invest_spy",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | None, default: str = ALGORITHM_BOT_FILE) -> Path:
    if not path:
        return _project_root() / default
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _project_root() / candidate


def _load_yaml_file(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {}
    content = config_path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(content) if yaml is not None else _parse_simple_yaml(content)
    loaded = loaded or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML config must contain a mapping at {config_path}")
    return loaded


def accounts_file_path(path: str | None = None) -> Path:
    return _resolve_path(path or os.getenv("TRADING_ACCOUNTS_FILE", ACCOUNTS_FILE), ACCOUNTS_FILE)


def connectors_file_path(path: str | None = None) -> Path:
    return _resolve_path(path or os.getenv("TRADING_CONNECTORS_FILE", CONNECTORS_FILE), CONNECTORS_FILE)


def _sibling_config_path(path: str | None, env_name: str, default: str) -> Path:
    if path or os.getenv(env_name):
        return _resolve_path(path or os.getenv(env_name), default)
    return _resolve_path(default)


def algorithms_file_path(path: str | None = None) -> Path:
    return _sibling_config_path(path, "TRADING_ALGORITHMS_FILE", ALGORITHMS_FILE)


def algorithm_bot_file_path(path: str | None = None) -> Path:
    return _sibling_config_path(path, "TRADING_ALGORITHM_BOT_FILE", ALGORITHM_BOT_FILE)


def options_bot_file_path(path: str | None = None) -> Path:
    return _sibling_config_path(path, "TRADING_OPTIONS_BOT_FILE", OPTIONS_BOT_FILE)


def dca_config_file_path(path: str | None = None) -> Path:
    return _sibling_config_path(path, "TRADING_DCA_BOT_FILE", DCA_BOT_FILE)


def universe_file_path(path: str | None = None) -> Path:
    return _sibling_config_path(path, "TRADING_UNIVERSE_FILE", UNIVERSE_FILE)


def load_accounts_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(accounts_file_path(path))


def load_connectors_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(connectors_file_path(path))


def load_algorithms_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(algorithms_file_path(path))


def load_algorithm_bot_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(algorithm_bot_file_path(path))


def load_options_bot_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(options_bot_file_path(path))


def load_dca_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(dca_config_file_path(path))


def load_universe_config(path: str | None = None) -> dict[str, Any]:
    return _load_yaml_file(universe_file_path(path))


def _save_yaml_config(config: dict[str, Any], config_path: Path) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        content = yaml.safe_dump(config, sort_keys=False)
    else:
        content = _dump_simple_yaml(config)
    config_path.write_text(content, encoding="utf-8")
    return config_path


def save_dca_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, dca_config_file_path(path))


def save_algorithm_bot_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, algorithm_bot_file_path(path))


def save_algorithms_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, algorithms_file_path(path))


def save_options_bot_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, options_bot_file_path(path))


def save_universe_config(config: dict[str, Any], path: str | None = None) -> Path:
    return _save_yaml_config(config, universe_file_path(path))


def save_universe_symbols(symbols: list[str], path: str | None = None) -> Path:
    raw_config = load_universe_config(path)
    universe = raw_config.setdefault("tradable_universe", {})
    if not isinstance(universe, dict):
        universe = {}
        raw_config["tradable_universe"] = universe
    universe["symbols"] = [str(symbol).strip().upper() for symbol in symbols if str(symbol).strip()]
    return save_universe_config(raw_config, path)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "''", '""'}:
        return ""
    if value == "[]":
        return []
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _parse_simple_yaml(content: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[dict[str, Any]] = [{"indent": -1, "container": root, "parent": None, "key": None}]
    for raw_line in content.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        is_list_item = stripped.startswith("-")
        while stack and (indent < stack[-1]["indent"] or (indent == stack[-1]["indent"] and not is_list_item)):
            stack.pop()
        if stripped.startswith("- "):
            entry = stack[-1]
            container = entry["container"]
            if isinstance(container, dict) and not container and entry["parent"] is not None:
                container = []
                entry["parent"][entry["key"]] = container
                entry["container"] = container
            if isinstance(container, list):
                rest = stripped[2:].strip()
                if ":" in rest:
                    key, value = rest.split(":", 1)
                    item: dict[str, Any] = {}
                    if value.strip():
                        item[key.strip()] = _parse_scalar(value)
                    else:
                        child: dict[str, Any] = {}
                        item[key.strip()] = child
                        stack.append({"indent": indent + 1, "container": child, "parent": item, "key": key.strip()})
                    container.append(item)
                    stack.append({"indent": indent, "container": item, "parent": container, "key": None})
                else:
                    container.append(_parse_scalar(rest))
            continue
        if stripped == "-":
            entry = stack[-1]
            container = entry["container"]
            if isinstance(container, dict) and not container and entry["parent"] is not None:
                container = []
                entry["parent"][entry["key"]] = container
                entry["container"] = container
            if isinstance(container, list):
                item = {}
                container.append(item)
                stack.append({"indent": indent, "container": item, "parent": container, "key": None})
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        parent = stack[-1]["container"]
        if not isinstance(parent, dict):
            continue
        if value.strip():
            parent[key.strip()] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append({"indent": indent, "container": child, "parent": parent, "key": key.strip()})
    return root


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value == "":
        return '""'
    text = str(value)
    if any(char in text for char in [":", "#", "{", "}", "[", "]"]) or text.strip() != text:
        return f'"{text}"'
    return text


def _dump_simple_yaml(value: Any, indent: int = 0) -> str:
    lines: list[str] = []
    prefix = " " * indent
    if isinstance(value, dict):
        for key, item in value.items():
            if item == []:
                lines.append(f"{prefix}{key}: []")
                continue
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_dump_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {_format_scalar(item)}")
    elif isinstance(value, list):
        if not value:
            lines.append(f"{prefix}[]")
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(_dump_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {_format_scalar(item)}")
    return "\n".join(line for line in lines if line != "") + ("\n" if indent == 0 else "")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, dict) else {}


def _normalize_keyed_items(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        return {str(key): item if isinstance(item, dict) else {} for key, item in value.items()}
    if isinstance(value, list):
        items: dict[str, dict[str, Any]] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or item.get("name") or item.get("provider") or "").strip()
            if not item_id:
                continue
            items[item_id] = {key: val for key, val in item.items() if key not in {"id", "name"}}
        return items
    return {}


def _normalize_accounts_config(raw: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    if not raw:
        return "", {}
    if isinstance(raw.get("accounts"), list):
        return str(raw.get("default") or ""), _normalize_keyed_items(raw.get("accounts"))
    accounts = raw.get("accounts") if isinstance(raw.get("accounts"), dict) else raw
    if not isinstance(accounts, dict):
        return "", {}
    items = accounts.get("items", accounts.get("accounts", []))
    return str(accounts.get("default") or raw.get("default") or ""), _normalize_keyed_items(items)


def get_account_broker_type(account_id: str) -> str:
    """Resolve the broker type for a given account ID from accounts config."""
    raw = load_accounts_config()
    _, items = _normalize_accounts_config(raw)
    account = items.get(account_id, {})
    if not account and items:
        default_id = raw.get("default", "")
        if isinstance(default_id, str) and default_id in items:
            account = items[default_id]
        elif items:
            account = next(iter(items.values()), {})
    return str(account.get("broker", "alpaca")).strip().lower()


def _normalize_data_sources(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return {}
    sources = raw.get("data_sources", raw.get("connectors", raw))
    if not isinstance(sources, dict):
        return {}
    normalized: dict[str, Any] = {}
    for category in ("market_data", "intraday_market_data", "eod_market_data", "interday_market_data", "news_sentiment", "sentiment_data"):
        section = sources.get(category, {})
        if not isinstance(section, dict):
            normalized[category] = {"provider_order": [], "providers": {}}
            continue
        providers = _normalize_keyed_items(section.get("providers", []))
        normalized[category] = {
            **section,
            "provider_order": list(providers),
            "providers": providers,
        }
    if not normalized["eod_market_data"]["providers"] and normalized["market_data"]["providers"]:
        normalized["eod_market_data"] = dict(normalized["market_data"])
    if not normalized["market_data"]["providers"] and normalized["eod_market_data"]["providers"]:
        normalized["market_data"] = dict(normalized["eod_market_data"])
    if not normalized["sentiment_data"]["providers"] and normalized["news_sentiment"]["providers"]:
        normalized["sentiment_data"] = dict(normalized["news_sentiment"])
    if not normalized["news_sentiment"]["providers"] and normalized["sentiment_data"]["providers"]:
        normalized["news_sentiment"] = dict(normalized["sentiment_data"])
    return normalized


def _direct_or_env(section: dict[str, Any], key: str, env_key: str, fallback_env: str = "") -> str:
    if section.get(key):
        return _env_ref(section.get(key), "")
    env_name = str(section.get(env_key) or "").strip()
    if env_name:
        return os.getenv(env_name, "")
    return os.getenv(fallback_env, "") if fallback_env else ""


def _provider_credential(
    data_sources: dict[str, Any],
    category: str,
    provider: str,
    key: str,
    env_key: str,
    fallback_env: str = "",
) -> str:
    section = _section(data_sources, category)
    providers = _section(section, "providers")
    provider_config = _section(providers, provider)
    if provider_config.get(key):
        return _env_ref(provider_config.get(key), "")
    env_name = str(provider_config.get(env_key) or "").strip()
    if env_name:
        return os.getenv(env_name, "")
    return os.getenv(fallback_env, "") if fallback_env else ""


def _provider_secret(data_sources: dict[str, Any], category: str, provider: str, fallback_env: str = "") -> str:
    return _provider_credential(data_sources, category, provider, "api_key", "api_key_env", fallback_env)


def _env_ref(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.getenv(value[2:-1], default)
    return str(value)


def _config_value(section: dict[str, Any], key: str, env_name: str, default: Any) -> Any:
    if env_name in os.environ:
        return os.getenv(env_name)
    return section.get(key, default)


def _str_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_symbols(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    symbols = [item.strip().upper() for item in value.split(",") if item.strip()]
    return symbols or list(default)


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_list(value: Any, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        parsed = [item.strip() for item in value.split(",") if item.strip()]
        return parsed or list(default)
    if isinstance(value, list):
        parsed = [str(item).strip() for item in value if str(item).strip()]
        return parsed or list(default)
    return list(default)


def _algorithm_sections(raw_algorithms_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    if raw_algorithms_config:
        external = _section(raw_algorithms_config, "algorithms") or raw_algorithms_config
        sections.update(
            {
                str(key): value
                for key, value in external.items()
                if isinstance(value, dict) and key not in {"algorithm_bot", "runtime"}
            }
        )
    return sections


@dataclass(frozen=True)
class Config:
    account_id: str = "default"
    account_label: str = "Default"

    algorithm_id: str = "momentum_social"
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
    market_data_cache_ttl_seconds: int = MARKET_DATA_CACHE_TTL_SECONDS
    intraday_market_data_cache_ttl_seconds: int = INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS
    eod_market_data_cache_ttl_seconds: int = EOD_MARKET_DATA_CACHE_TTL_SECONDS
    news_sentiment_cache_ttl_seconds: int = NEWS_SENTIMENT_CACHE_TTL_SECONDS
    sentiment_data_cache_ttl_seconds: int = NEWS_SENTIMENT_CACHE_TTL_SECONDS
    data_source_configs: dict[str, Any] = field(default_factory=dict)
    algorithm_check_seconds: int = ALGORITHM_CHECK_SECONDS
    dca_check_seconds: int = DCA_CHECK_SECONDS
    algorithm_market_data_refresh_minutes: int = ALGORITHM_MARKET_DATA_REFRESH_MINUTES
    algorithm_run_jitter_minutes: int = ALGORITHM_RUN_JITTER_MINUTES
    trading_start_time: str = TRADING_START_TIME
    trading_end_time: str = TRADING_END_TIME
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
        default_account_id = os.getenv("TRADING_ACCOUNT_ID") or "default"
    selected_account_id = str(account_id or os.getenv("TRADING_ACCOUNT_ID") or default_account_id)
    account_config = _section(account_items, selected_account_id)
    if not account_config and account_items:
        selected_account_id = default_account_id if default_account_id in account_items else next(iter(account_items))
        account_config = _section(account_items, selected_account_id)

    raw_universe_config = load_universe_config()
    universe = _section(raw_universe_config, "tradable_universe")
    raw_algorithms_config = load_algorithms_config()
    algorithm_configs = _algorithm_sections(raw_algorithms_config)
    selected_strategy_id = str(strategy_id or "momentum_social").strip().lower()
    algorithm = _section(algorithm_configs, selected_strategy_id)
    raw_dca_bot_config = load_dca_config()
    algorithm_bot = _section(raw_algorithm_bot_config, "algorithm_bot")
    dca_bot = _section(raw_dca_bot_config, "dca_bot")
    runtime = {
        **_section(raw_algorithm_bot_config, "runtime"),
        **{
            key: value
            for key, value in algorithm_bot.items()
            if key in {
                "algorithm_check_seconds",
                "algorithm_market_data_refresh_minutes",
                "algorithm_run_jitter_minutes",
                "backtest_period",
                "trading_start_time",
                "trading_end_time",
                "require_trade_approval",
                "trade_approval_timeout_seconds",
                "trade_approval_poll_seconds",
            }
        },
        **{key: value for key, value in dca_bot.items() if key in {"dca_check_seconds"}},
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
    sentiment_sources = _section(data_sources, "sentiment_data")

    kill_switch = _str_to_bool(str(_config_value(runtime, "kill_switch", "KILL_SWITCH", KILL_SWITCH)), KILL_SWITCH)
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
        algorithm_check_seconds=_as_int(
            _config_value(runtime, "algorithm_check_seconds", "ALGORITHM_CHECK_SECONDS", ALGORITHM_CHECK_SECONDS),
            ALGORITHM_CHECK_SECONDS,
        ),
        dca_check_seconds=_as_int(
            _config_value(runtime, "dca_check_seconds", "DCA_CHECK_SECONDS", DCA_CHECK_SECONDS),
            DCA_CHECK_SECONDS,
        ),
        algorithm_market_data_refresh_minutes=_as_int(
            _config_value(
                runtime,
                "algorithm_market_data_refresh_minutes",
                "ALGORITHM_MARKET_DATA_REFRESH_MINUTES",
                ALGORITHM_MARKET_DATA_REFRESH_MINUTES,
            ),
            ALGORITHM_MARKET_DATA_REFRESH_MINUTES,
        ),
        algorithm_run_jitter_minutes=_as_int(
            _config_value(
                runtime,
                "algorithm_run_jitter_minutes",
                "ALGORITHM_RUN_JITTER_MINUTES",
                ALGORITHM_RUN_JITTER_MINUTES,
            ),
            ALGORITHM_RUN_JITTER_MINUTES,
        ),
        trading_start_time=str(_config_value(runtime, "trading_start_time", "TRADING_START_TIME", TRADING_START_TIME)),
        trading_end_time=str(_config_value(runtime, "trading_end_time", "TRADING_END_TIME", TRADING_END_TIME)),
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
