"""Default values and file locations.

Every tunable the application ships with, in one place, so a default can be found without
reading the loader that consumes it.
"""

from __future__ import annotations


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
#: Cash held back from the buying power an order batch is allowed to spend. An account-level
#: floor rather than a haircut on target weights: how much of the book a strategy wants
#: deployed is the strategy's decision, expressed through its own gross-exposure cap.
CASH_BUFFER = 0.02
#: Holdings that are cash in all but name. A rebalance short of buying power may sell these to
#: fund its buys, so parking idle cash in T-bills no longer blocks the next batch.
CASH_EQUIVALENTS = ["SGOV", "BIL"]
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
#: One file holds every section. The per-section constants below are the pre-unification
#: paths, kept because their env overrides still work and because an existing deployment is
#: migrated from them on first start.
CONFIG_FILE = "config/walbot.yaml"
ACCOUNTS_FILE = "config/accounts.yaml"
CONNECTORS_FILE = "config/connectors.yaml"
ALGORITHMS_FILE = "config/algorithms.yaml"
ALGORITHM_BOT_FILE = "config/algorithm_bot.yaml"
DCA_BOT_FILE = "config/dca_bot.yaml"
UNIVERSE_FILE = "config/universe.yaml"
MARKET_DATA_PROVIDER_ORDER: list[str] = []
INTRADAY_MARKET_DATA_PROVIDER_ORDER: list[str] = ["yfinance"]
EOD_MARKET_DATA_PROVIDER_ORDER: list[str] = []
NEWS_SENTIMENT_PROVIDER_ORDER: list[str] = []
#: Alpaca first because it carries ``payable_date`` and flags special distributions; yfinance
#: is the credential-free fallback. Both were measured to report identical events.
DIVIDEND_PROVIDER_ORDER: list[str] = ["alpaca", "yfinance"]
MARKET_DATA_CACHE_TTL_SECONDS = 1800
INTRADAY_MARKET_DATA_CACHE_TTL_SECONDS = 900
EOD_MARKET_DATA_CACHE_TTL_SECONDS = 1800
#: Preferred resolution for fine-grained bars, in minutes. Five rather than the old fifteen
#: because the grid used to be dictated by yfinance's floor, and Schwab -- now the primary
#: feed -- serves 1/5/10/15/30 in real time. Algorithm horizons are stated in minutes, so
#: this only sets fidelity: a finer grid resolves them more precisely, a coarser one still
#: answers them. Providers that cannot serve it fall back to their nearest coarser grid.
MARKET_DATA_BAR_MINUTES = 5
NEWS_SENTIMENT_CACHE_TTL_SECONDS = 1800
REQUIRE_TRADE_APPROVAL = False
TRADE_APPROVAL_TIMEOUT_SECONDS = 300
TRADE_APPROVAL_POLL_SECONDS = 5
ALGORITHM_IDS = {
    "bursty_dca",
    "rally_rotation",
    "intraday_pick",
}

#: Used wherever no strategy was selected, and as the fallback for a retired id.
DEFAULT_STRATEGY_ID = "rally_rotation"


#: Stands for "no account was named" -- the value ``Config.account_id`` carries before any
#: accounts config is read, and the id used when none is configured. Distinct from a real
#: account id, so asking for it is not the same as asking for an account that does not exist.
UNNAMED_ACCOUNT_ID = "default"
