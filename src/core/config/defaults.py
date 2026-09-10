"""Default values and file locations."""

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
MAX_WEIGHT_PER_SYMBOL = 0.25
MAX_PORTFOLIO_EXPOSURE = 0.95
#: Cash held back from the buying power an order batch is allowed to spend.
CASH_BUFFER = 0.02
#: Holdings treated as cash; may be sold to fund a batch short of buying power.
#:
#: **Order is liquidation preference** -- ``get_cash_equivalents`` walks this list front to back
#: and stops once the shortfall is covered, so the first entry is sold first.
#:
#: GUMI sits last deliberately. The other two are T-bill funds, which is what makes selling them
#: to raise cash a non-decision: they barely move, so the price you get is the price you saw.
#: GUMI is filed as ``Other / Equity`` in ``data/tradable_etfs.csv`` rather than
#: ``Fixed Income / Treasury``, so listing it here makes a security that can actually move
#: something the funding ladder may sell at market to pay for an unrelated buy. Last in the
#: list means that only happens once the real bills are exhausted.
CASH_EQUIVALENTS = ["SGOV", "BIL", "GUMI"]
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
#: One file holds every section; the per-section constants below are the pre-unification
#: paths, kept because their env overrides still work and existing deployments migrate off them.
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
DIVIDEND_PROVIDER_ORDER: list[str] = ["alpaca", "yfinance"]
MARKET_DATA_CACHE_TTL_SECONDS = 1800
#: Preferred resolution for fine-grained bars, in minutes. Providers that cannot serve it
#: fall back to their nearest coarser grid.
MARKET_DATA_BAR_MINUTES = 5
NEWS_SENTIMENT_CACHE_TTL_SECONDS = 1800
ALGORITHM_IDS = {
    "bursty_dca",
    "rally_rotation",
    "options_flip",
}

#: Used wherever no strategy was selected, and as the fallback for a retired id.
DEFAULT_STRATEGY_ID = "rally_rotation"


#: Stands for "no account was named" -- distinct from a real account id, so asking for it is
#: not the same as asking for an account that does not exist.
UNNAMED_ACCOUNT_ID = "default"
