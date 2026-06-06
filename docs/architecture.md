# Architecture

## Data Sources

Data sources are grouped by the shape of data they return:

- `intraday_market_data`: minute-style OHLCV bars.
- `eod_market_data`: daily OHLCV bars.
- `sentiment_data`: news, social, and sentiment records.

Provider waterfall order is the order of keys under each category's `providers:` mapping in `config/connectors.yaml`. There is no separate `provider_order` key in the desired connector contract.

Connector contracts are abstract base classes in `src/core/interfaces.py` and re-exported through `src/connectors/base.py`. Concrete implementations live under provider-specific connector packages and are registered in `src/connectors/registry.py`.

## Market Bars

Intraday and EOD data share the `market_bars` DuckDB table because both are OHLCV bars:

- `category` separates `intraday_market_data` from `eod_market_data`.
- `timeframe` separates `1m`, `15m`, `30m`, `1d`, etc.
- `(category, provider, symbol, timeframe, timestamp)` is the natural uniqueness key.

This keeps reads simple for algorithms: ask for symbol + timeframe + category, and receive the same `timestamp/open/high/low/close/volume` frame regardless of provider.

## Algorithms

Concrete algorithm implementations live under `src/algorithms/`.

The common algorithm contract is the `AlgorithmPlugin` abstract base class in `src/core/interfaces.py`, with the reusable base implementation in `src/algorithms/base.py`. Runtime plugins expose:

- `requirements(config, current_positions)`: symbols, shared data needs, and runtime restrictions.
- `decide(context)`: normalized target weights, signal logs, metadata, and order-sync settings.

Plugins are registered in `src/algorithms/registry.py`. The live runner resolves the selected strategy through that registry instead of branching on individual strategy ids.

DCA and options strategies live in the same hierarchy as equity algorithms:

- `src/algorithms/dca/`
- `src/algorithms/options/swing.py`

They can still be rendered on separate frontend pages, but backend-wise they are algorithms that produce normalized signal/order-intent dictionaries from an `AlgorithmContext`.

## Notifications

Notification delivery is modeled as provider connectors under `src/notifications/`.

- `NotificationConnector` and `NotificationMessage` define the provider contract.
- Concrete providers live under `src/notifications/providers/`.
- Providers are registered in `src/notifications/registry.py`.
- `src/notifications/service.py` formats portfolio-change messages and fans them out through configured providers.

Telegram is the only provider today. It is configured under `notifications.providers.telegram` in `config/connectors.yaml`, with environment fallback for bot token, chat id, API root, timeout, and the global enabled flag.

`src/common/notifications.py` is a compatibility facade for older imports; new code should use `src.notifications.service`.

## Agent/MCP Direction

The clean MCP boundary is around deterministic components, not around the whole bot loop:

- data tools: fetch bars, quotes, provider status, sentiment snapshots.
- algorithm tools: describe requirements, generate decision, explain signals.
- execution tools: preview orders, submit approved orders, cancel pending orders.
- notification tools: format notification, send notification, request Telegram approval, record approval summary.
- state tools: read/write pending approvals and run snapshots.

An agent runtime such as OpenClaw can then run the scheduled workflow externally:

1. call a market/sentiment data tool.
2. call an algorithm decision tool.
3. call an order preview tool.
4. summarize the proposed position changes.
5. call the `request_trade_approval` MCP tool so approval uses the same Telegram flow as `--bot` mode.
6. call `submit_approved_orders` only after approval.
7. call notification delivery with the final submitted-order summary.

This keeps strategy math, approval mechanics, and brokerage execution deterministic inside this repo, while the agent handles orchestration, external search/sentiment tools, and natural-language summaries.

## Runtime Modes

The container entrypoint supports two modes:

- `--bot`: starts the dashboard/API and the built-in deterministic bot scheduler.
- `--mcp`: starts the dashboard/API with the internal scheduler disabled and launches the MCP tool server for an external agent runtime.

Both modes keep the same dashboard API. The difference is who owns scheduling and execution orchestration: this repo in `--bot`, or an external agent such as OpenClaw in `--mcp`.

In `--bot` mode, `algorithm_bot.yaml` can narrow the scheduler window with `trading_start_time` and `trading_end_time`. Setting `require_trade_approval: true` makes the built-in bot send planned orders to Telegram and wait for an approve/deny response before submitting.
