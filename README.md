# Social + Momentum Trading Bot

A Docker-ready FastAPI dashboard and trading bot for Alpaca. It supports DCA planning, strategy signal inspection, cached 6-month backtests, and a trading runner built around a configurable tradable universe.

This is research infrastructure, not a profit guarantee. Point each account in `accounts.yaml` at the Alpaca endpoint you intend to use, review generated orders, and use `KILL_SWITCH=true` whenever you want the app to be read-only.

## What Runs In The Container

The default container command starts the web app:

```text
uvicorn src.api_app:app --host 0.0.0.0 --port 8000
```

The image intentionally does not include your secrets, SQLite state, or generated social data. Provide those at runtime:

- `/config` - mounted writable YAML config: `algorithm_bot.yaml`, `algorithms.yaml`, `accounts.yaml`, `connectors.yaml`, `options_bot.yaml`, `dca_bot.yaml`, and `universe.yaml`
- `/data` - mounted writable app state, SQLite DB, cached backtests, and optional social trend CSV
- environment variables or `.env` - runtime paths and non-secret toggles

Logs go to stdout/stderr by default, so use `docker compose logs -f trading-bot`, `docker logs -f trading-bot`, or Portainer's container logs.

## Runtime Files

Create this layout on the machine that will run the app:

```text
runtime/
  config/
    algorithm_bot.yaml
    algorithms.yaml
    accounts.yaml
    connectors.yaml
    options_bot.yaml
    dca_bot.yaml
    universe.yaml
  data/
.env
```

Starter examples are in `deploy/examples/`:

```bash
mkdir -p runtime/config runtime/data
cp deploy/examples/env.example .env
cp deploy/examples/algorithm_bot.yaml runtime/config/algorithm_bot.yaml
cp deploy/examples/algorithms.yaml runtime/config/algorithms.yaml
cp deploy/examples/accounts.yaml runtime/config/accounts.yaml
cp deploy/examples/connectors.yaml runtime/config/connectors.yaml
cp deploy/examples/options_bot.yaml runtime/config/options_bot.yaml
cp deploy/examples/dca_bot.yaml runtime/config/dca_bot.yaml
cp deploy/examples/universe.yaml runtime/config/universe.yaml
```

Edit `runtime/config/accounts.yaml` and `runtime/config/connectors.yaml` with your account and provider API keys. The empty examples are safe defaults; `deploy/examples/accounts.with-keys.example.yaml` and `deploy/examples/connectors.with-keys.example.yaml` show the direct-key schema.

Edit `.env` for runtime toggles:

```bash
KILL_SWITCH=false
```

The provided Docker setup uses these container paths:

```bash
STATE_DB_PATH=/data/trading_bot.sqlite
TRADING_ACCOUNTS_FILE=/config/accounts.yaml
TRADING_CONNECTORS_FILE=/config/connectors.yaml
TRADING_ALGORITHM_BOT_FILE=/config/algorithm_bot.yaml
TRADING_ALGORITHMS_FILE=/config/algorithms.yaml
TRADING_OPTIONS_BOT_FILE=/config/options_bot.yaml
TRADING_DCA_BOT_FILE=/config/dca_bot.yaml
TRADING_UNIVERSE_FILE=/config/universe.yaml
TRADABLES_CSV=/app/data/tradable_etfs.csv
ALPHA_VANTAGE_NEWS_CSV=/data/social_trends.csv
CORS_ALLOW_ORIGINS=*
```

Do not commit `.env`, `runtime/`, `config/`, `data/*.sqlite`, or generated social/backtest files. They are ignored by `.gitignore`.

The master tradables file is bundled inside the image at `/app/data/tradable_etfs.csv`. The smaller deployment universe lives in `universe.yaml` under `tradable_universe.symbols`, and dashboard universe refreshes write back to that YAML file. Algorithm bot controls live in `algorithm_bot.yaml`; per-strategy algorithm knobs live in `algorithms.yaml`; options controls and strategy knobs live in `options_bot.yaml`; DCA scheduling and planning lives in `dca_bot.yaml`; account keys live in `accounts.yaml`; connector keys live in `connectors.yaml`. Social trends are app-managed/generated data in `/data/social_trends.csv`; users are not expected to provide them as ground truth.

## Run With Docker Compose

Start the published image:

```bash
docker compose up -d
```

Open:

```text
http://127.0.0.1:8000
```

View logs:

```bash
docker compose logs -f trading-bot
```

Stop:

```bash
docker compose down
```

## Test A Local Image

Build the image:

```bash
docker build -t trading-bot:local .
```

Run that local image with the same Compose file:

```bash
TRADING_BOT_IMAGE=trading-bot:local TRADING_BOT_PULL_POLICY=never docker compose up -d
```

## Publish To GitHub Container Registry

This repo includes `.github/workflows/docker-publish.yml`. When pushed to `main`, GitHub Actions builds and pushes:

```text
ghcr.io/<owner>/<repo>:latest
ghcr.io/<owner>/<repo>:main
ghcr.io/<owner>/<repo>:sha-<commit>
```

Tagged releases like `v1.0.0` also publish a matching tag.

Before relying on GHCR:

1. Push the repository to GitHub.
2. Make sure GitHub Actions is enabled.
3. Push to `main`.
4. In GitHub, open `Packages` and confirm the image exists.
5. If your repo/package is private, log in on the deployment host:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
```

The token needs permission to read packages.

## Deploy From GHCR On A Server

On your server:

```bash
mkdir -p trading-bot/runtime/config trading-bot/runtime/data
cd trading-bot
```

Create `.env` using `deploy/examples/env.example` as a template, or paste the same values manually. Then edit the YAML config:

```bash
nano .env
nano runtime/config/algorithm_bot.yaml
nano runtime/config/algorithms.yaml
```

Copy `docker-compose.yml` to the server or paste it into an OMV/Portainer stack. Then start the published image:

```bash
docker compose up -d
```

Upgrade later:

```bash
docker compose pull
docker compose up -d
```

Your app state survives upgrades because it lives in `runtime/data`, not inside the image.

## Run One-Shot Jobs From The Same Image

Run a backtest:

```bash
docker run --rm \
  --env-file .env \
  -v "$PWD/runtime/config:/config" \
  -v "$PWD/runtime/data:/data" \
  ghcr.io/<owner>/<repo>:latest \
  python -m src.backtest
```

Run the trading worker once for manual testing:

```bash
docker run --rm \
  --env-file .env \
  -v "$PWD/runtime/config:/config" \
  -v "$PWD/runtime/data:/data" \
  ghcr.io/<owner>/<repo>:latest \
  python -m src.live_runner
```

For normal use, leave the dashboard running and use the DCA, Algorithm, or Options tab power buttons to control the backend bot loops. Keep `KILL_SWITCH=true` until you are ready for order submission.

## Configuration Reference

Common runtime overrides:

```bash
KILL_SWITCH=false
CORS_ALLOW_ORIGINS=*
```

`KILL_SWITCH=true` keeps the app able to show data/backtests/signals while preventing the trading runner from submitting orders.

For a public or reverse-proxied deployment, set `CORS_ALLOW_ORIGINS` to the exact dashboard origin, for example `https://trading.example.com`. Leave it as `*` only for trusted local/LAN access.

To receive portfolio-change notifications when submitted orders change positions, add a Telegram provider to `connectors.yaml`:

```yaml
notifications:
  providers:
    telegram:
      enabled: true
      bot_token: replace_me
      chat_id: replace_me
      api_root: https://api.telegram.org
      timeout_seconds: 5
```

If you use an OpenClaw Telegram bot, use that bot's token and chat ID; `api_root` can point at a Telegram-compatible API root if your OpenClaw setup routes Bot API traffic through a proxy.

YAML config:

`algorithm_bot.yaml` contains global runtime settings such as `kill_switch` plus the algorithm bot power state, selected equity strategy, trading account id, refresh cadence, and optional cadence jitter. `algorithms.yaml` contains only the nested knobs for each server-side algorithm. `options_bot.yaml` contains the options bot power state, selected options strategy, options account id, and options strategy knobs. `dca_bot.yaml` contains DCA scheduling and the DCA plan. `universe.yaml` contains `tradable_universe.symbols` and optional `master_list`. `accounts.yaml` contains brokerage accounts with direct `api_key` / `api_secret` values. `connectors.yaml` contains market/news providers, notification providers, and direct connector API keys; fallback order follows the order of entries under each market/news `providers` mapping unless you add an explicit `provider_order`. `TRADABLES_CSV` still overrides the master-list path from the environment if set.

By default, `accounts.yaml` and `connectors.yaml` are empty arrays, so the container expects you to mount filled versions. Raw provider responses, short-lived cache entries, and provider limit state are stored in the SQLite DB under `STATE_DB_PATH`.

Risk controls:

```bash
MAX_WEIGHT_PER_SYMBOL=0.25
MAX_PORTFOLIO_EXPOSURE=0.95
MAX_LONGS=5
TARGET_ANNUAL_VOL=0.18
CASH_BUFFER=0.02
REBALANCE_THRESHOLD=0.02
MIN_TRADE_DOLLARS=50
BACKTEST_STARTING_EQUITY=10000
ALGORITHM_EQUITY_CAP=0
```

`BACKTEST_STARTING_EQUITY` is the simulated cash account size used by backtests. `ALGORITHM_EQUITY_CAP` is optional for live/paper order sizing; leave it at `0` to use the full account equity, or set it to a dollar amount such as `10000` to cap algorithm sizing to that amount.

Optional Alpha Vantage settings for app-managed social data:

```bash
ALPHA_VANTAGE_NEWS_CSV=/data/social_trends.csv
ALPHA_VANTAGE_NEWS_LOOKBACK_DAYS=30
ALPHA_VANTAGE_NEWS_LIMIT=50
ALPHA_VANTAGE_MAX_SYMBOLS=20
```

Persistence:

```bash
STATE_DB_PATH=/data/trading_bot.sqlite
```

## Non-Docker Development

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
STATE_DB_PATH=data/trading_bot.sqlite
ALPHA_VANTAGE_NEWS_CSV=data/social_trends.csv
```

Run the dashboard:

```bash
uvicorn src.api_app:app --host 0.0.0.0 --port 8000
```

Run tests:

```bash
pytest
```

## Project Layout

- `src/api_app.py` - FastAPI API and static web app server
- `src/api_payloads.py` - API payload builders, cached backtests, signals, and config updates
- `src/config.py` - YAML-first runtime config
- `src/connectors.py` - market-data and news/sentiment provider connectors with fallback
- `src/provider_cache.py` - SQLite cache and provider limit state
- `src/dual_momentum_optimizer.py` - dual-momentum config and universe experiment runner
- `src/state_store.py` - SQLite state persistence
- `src/dca.py` - DCA plan validation and allocation preview
- `src/signals.py` - momentum/social signal engine
- `src/portfolio.py` - target weight calculation
- `src/live_runner.py` - one-shot trading worker
- `web/` - dashboard HTML, CSS, and JavaScript
- `deploy/examples/` - safe example runtime files
- `Dockerfile` - production image
- `docker-compose.yml` - local container run helper
- `.github/workflows/docker-publish.yml` - GHCR publisher

## Safety Notes

- `ALPACA_BASE_URL=https://paper-api.alpaca.markets` is the default and recommended starting point.
- `KILL_SWITCH=true` prevents order submission.
- Treat generated signals and backtests as decision support, not instructions.
- Backtests are approximate and do not guarantee live execution quality.
