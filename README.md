# Walbot - Pluggable Trading Platform

Walbot is a plugin-oriented trading automation platform with optional agent integration. It supports modular market data connectors, sentiment analysis, extensible trading algorithms, MCP-facing tools for agent runtimes, and automatic DuckDB caching.

## Architecture

The project follows a **Plugin-Oriented Architecture**:

- **Core (`src/core/`)**: Config, interfaces, orders, portfolio logic, and bot runtime.
- **Connectors (`src/connectors/`)**: Modular market-data and sentiment providers.
- **Data (`src/data/`)**: State, cache, DuckDB storage, universe, and provider fallback logic.
- **Algorithms (`src/algorithms/`)**: Strategy plugins for equities, options, and DCA.
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

### Running the Dashboard

```bash
python -m src.container_entrypoint
```

This starts the dashboard/API on port `8000` and the MCP tool server for external agents on
port `8001`. The built-in bot scheduler also runs, but nothing fires unless a binding is
switched on in the dashboard (see below).

Open the dashboard at:

```text
http://localhost:8000
```

Override the ports with `--port` / `--mcp-port` or the `PORT` / `MCP_PORT` environment
variables:

```bash
python -m src.container_entrypoint --port 8010
```

Logging defaults to `WARNING`. Raise it with `TRADING_LOG_LEVEL=INFO` or lower it with
`DEBUG` (which also turns Uvicorn's per-request access logs back on).

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

After setup, run the dashboard with the command in [Running the Dashboard](#running-the-dashboard).

There are no `--bot` / `--mcp` process-wide modes. The dashboard, the bot scheduler, and the
MCP tool server all start together; whether an algorithm is driven by the clock or by an
agent is a property of its *binding* in the dashboard. Each binding declares a `frequency`
for the scheduler -- `15m`, `30m`, `1hr`, `2hr`, `1d` -- or `mcp` to park it, switched on but
waiting for an external agent to drive it. A `--mcp` process mode could only contradict the
binding's own frequency, and it did: the dashboard reported "MCP mode" with every algorithm
switched off.

Built-in bot scheduling is configured in `algorithm_bot.yaml`:

```yaml
algorithm_bot:
  backtest_period: 4m
  trading_start_time: "08:30"
  trading_end_time: "15:00"
```

There is no out-of-band trade approval. Review happens through the MCP flow instead:
`get_algorithm_plan` runs the algorithm and places nothing, and an external agent such as
OpenClaw -- driving a binding parked on `mcp` through the tool server on port `8001` -- edits
the plan's intents and calls `place_orders` only when it is satisfied.

Current config files:

- `config/accounts.yaml` - brokerage account wiring; use `api_key_env` and `api_secret_env`.
- `config/connectors.yaml` - market and sentiment providers, and provider order.
- `config/algorithm_bot.yaml` - built-in equity bot runtime, trading window, and kill switch.
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

Long backtests (e.g. 12M) fetch a lot of history and replay thousands of cached reads. The
replay runs on read-only DuckDB connections, so other *processes* -- a cache warmup, CLI tools
-- can keep reading while it runs. The network fetch itself holds no database lock at all.

DuckDB allows one read-write process per database file, and while a process holds it
read-write no other process can open the file at all -- not even read-only. So the dashboard,
the scheduler and the MCP tool server all share a single process rather than contending for
`data/walbot.duckdb`; inside that process DuckDB's own MVCC interleaves readers and writers.
Running a CLI tool against the same file while the dashboard is up is still the one case that
can collide, and it retries briefly before failing (`LOCK_WAIT_SECONDS` in
`src/data/duckdb_store.py`).

Docker starts the dashboard, the scheduler, and the MCP tool server together (there is no mode
flag -- see [Running the Dashboard](#running-the-dashboard)):

```bash
docker run --rm -p 8000:8000 -p 8001:8001 ghcr.io/madhavtummala/walbot:latest
```

With compose:

```bash
docker compose up
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
- `src/mcp_server.py` - MCP tools for external agent runtimes
- `src/container_entrypoint.py` - Docker/local runtime entrypoint: dashboard, scheduler, and MCP server
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
