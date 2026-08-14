# Walbot - Pluggable Trading Platform

Walbot is a plugin-oriented trading automation platform with optional agent integration. It supports modular market data connectors, sentiment analysis, extensible trading algorithms, MCP-facing tools for agent runtimes, and automatic DuckDB caching.

## Architecture

The project follows a **Plugin-Oriented Architecture**:

- **Core (`src/core/`)**: Config, interfaces, orders, portfolio logic, and bot runtime.
- **Connectors (`src/connectors/`)**: Modular market-data and sentiment providers.
- **Data (`src/data/`)**: State, cache, DuckDB storage, universe, and provider fallback logic.
- **Algorithms (`src/algorithms/`)**: Strategy plugins for equities, options, and DCA.
- **Notifications (`src/notifications/`)**: Notification provider connectors such as Telegram.
- **MCP (`src/mcp_server.py`)**: Agent-facing tools for runtimes such as OpenClaw.

## Features

- **Standardized Interfaces**: Easily add new connectors or algorithms.
- **Automatic Caching**: `BaseConnector` automatically caches all requests to DuckDB.
- **Waterfall Fallback**: Configurable provider order ensures data availability.
- **Shared Signals**: Reusable indicator library for all strategies.
- **DuckDB Integration**: Fast, columnar storage for market and sentiment data.

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Configuration

Edit files under `config/` to set provider order, runtime behavior, universes, and account wiring. Keep secrets in environment variables referenced by `*_env` fields, not directly in YAML.

### Running the Dashboard Bot

```bash
python -m src.container_entrypoint --bot
```

Open the dashboard at:

```text
http://localhost:8000
```

`--bot` uses port `8000` by default. Override it with `--port` or the `PORT` environment variable:

```bash
python -m src.container_entrypoint --bot --port 8010
```

## Local Development

Install dependencies:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
```

Create `.env` in the project root. For local non-Docker development, paths may stay relative:

```bash
TRADING_ACCOUNTS_FILE=config/accounts.yaml
TRADING_CONNECTORS_FILE=config/connectors.yaml
TRADING_ALGORITHM_BOT_FILE=config/algorithm_bot.yaml
TRADING_ALGORITHMS_FILE=config/algorithms.yaml
TRADING_OPTIONS_BOT_FILE=config/options_bot.yaml
TRADING_DCA_BOT_FILE=config/dca_bot.yaml
TRADING_UNIVERSE_FILE=config/universe.yaml
STATE_DUCKDB_PATH=data/walbot.duckdb
ALPHA_VANTAGE_NEWS_CSV=data/social_trends.csv
```

After setup, run the dashboard bot with the command in [Running the Dashboard Bot](#running-the-dashboard-bot).

Runtime modes:

- `--bot`: dashboard/API plus the built-in non-AI bot scheduler.
- `--mcp`: dashboard/API on port `8000` with the internal scheduler disabled, plus an MCP tool server on port `8001` for an external agent such as OpenClaw.

Built-in bot scheduling and approval behavior is configured in `algorithm_bot.yaml`:

```yaml
algorithm_bot:
  backtest_period: 4m
  trading_start_time: "08:30"
  trading_end_time: "15:00"
  require_trade_approval: false
  trade_approval_timeout_seconds: 300
  trade_approval_poll_seconds: 5
```

When `require_trade_approval` is true, planned orders are sent to Telegram for approval before submission. You can approve or deny with the inline buttons or by replying `/approve <approval_id>` / `/deny <approval_id>`.

In `--mcp` mode, an external agent such as OpenClaw should use the `request_trade_approval` MCP tool for the same Telegram approval flow. That keeps approval behavior consistent between agent-driven runs and the built-in bot runtime.

Current config files:

- `config/accounts.yaml` - brokerage account wiring; use `api_key_env` and `api_secret_env`.
- `config/connectors.yaml` - market/sentiment/notification providers and provider order.
- `config/algorithm_bot.yaml` - built-in equity bot runtime, trading window, approval, and kill switch.
- `config/algorithms.yaml` - per-algorithm strategy knobs and universes.
- `config/options_bot.yaml` - options bot runtime and options strategy knobs.
- `config/dca_bot.yaml` - DCA scheduler and DCA plan.
- `config/universe.yaml` - dashboard/tradable universe and master ETF list path.

## Schwab Authorization

Schwab uses three-legged OAuth. The dashboard drives the whole consent flow, so no manual
copy-paste of authorization codes is needed.

Set these alongside the other environment variables:

```bash
SCHWAB_APP_KEY=...
SCHWAB_APP_SECRET=...
SCHWAB_CALLBACK_URL=https://your-host/schwab/callback
SCHWAB_ACCOUNT_NUMBER=...
```

`SCHWAB_CALLBACK_URL` must match a callback URL registered on the Schwab developer app
byte-for-byte, and Schwab requires HTTPS. Behind a TLS reverse proxy, point it at this app's
`/schwab/callback` route.

Once set, a status pill appears at the top of the dashboard:

- **green** - authorized, more than 36 hours of refresh-token life left
- **amber** - expires within 36 hours
- **red** - expired, or never authorized

Click the pill to authorize. A popup carries you through Schwab login and consent, the
callback route exchanges the code for a refresh token, and the pill turns green.

Schwab refresh tokens expire **7 days** after the consent that minted them. Refreshing an
access token does not extend that window, so this is a weekly click. Leave
`SCHWAB_REFRESH_TOKEN` unset — the token lives in the state store under `schwab_oauth_token`
along with its issue time, and a stale env value would override it.

Warm or refresh the local YFinance market-data cache:

```bash
python -m src.data.cache_warmup
```

By default this refreshes the configured algorithm universe with enough daily EOD bars for the configured backtest period and recent 15-minute intraday bars. Existing cached rows are preserved unless `--clear` is supplied.

Warm only the `fast_momentum` symbols for one America/Chicago market date:

```bash
python -m src.data.cache_warmup --algorithm fast_momentum --start-date 2026-06-03 --end-date 2026-06-03
```

Use `--eod` or `--intraday` to warm only that bar category. Use `--symbols SPY QQQM` instead of `--algorithm` to warm an explicit symbol list.

Docker defaults to `--bot`. To run agent-facing tool mode:

```bash
docker run --rm -p 8000:8000 -p 8001:8001 ghcr.io/madhavtummala/walbot:latest --mcp
```

With compose:

```bash
WALBOT_MODE=--mcp docker compose up
```

Run tests:

```bash
pytest
```

## Project Layout

- `src/api/` - FastAPI API, payload builders, and static web app server
- `src/algorithms/` - algorithm plugins and strategy implementations
- `src/connectors/` - market-data and news/sentiment provider connectors
- `src/core/` - config, interfaces, orders, portfolio logic, and bot runtime
- `src/data/` - state, cache, social data, universe, and storage helpers
- `src/execution/` - live runner and backtest execution
- `src/notifications/` - notification provider connectors
- `src/mcp_server.py` - MCP tools for external agent runtimes
- `src/container_entrypoint.py` - Docker/local runtime mode entrypoint
- `web/` - dashboard HTML, CSS, and JavaScript
- `config/` - committed non-secret runtime config
- `Dockerfile` - production image
- `docker-compose.yml` - local container run helper
- `.github/workflows/docker-publish.yml` - GHCR publisher

## Safety Notes

- `ALPACA_BASE_URL=https://paper-api.alpaca.markets` is the default and recommended starting point.
- `KILL_SWITCH=true` prevents order submission.
- Treat generated signals and backtests as decision support, not instructions.
- Backtests are approximate and do not guarantee live execution quality.
