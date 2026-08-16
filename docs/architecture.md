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

Three modules hold what more than one algorithm needs, so a rule cannot be fixed in one place and left broken in another:

- `src/algorithms/allocation.py`: ranking a scored universe, spreading an exposure budget by score under a per-name cap, and holding a weight set inside a gross limit.
- `src/algorithms/risk.py`: the session drawdown breaker, keyed on the timestamp the algorithm saw rather than on the wall clock so a replay does not latch it for the whole backtest.
- `src/common/config_utils.py`: `load_tuning(cls, section)` builds a tuning dataclass from saved config, coercing each field by its declared type. A new knob is a dataclass field and nothing else -- there is no parser to update alongside it.

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
5. call the `request_trade_approval` MCP tool so approval uses the same Telegram flow as the built-in scheduler.
6. call `submit_approved_orders` only after approval.
7. call notification delivery with the final submitted-order summary.

This keeps strategy math, approval mechanics, and brokerage execution deterministic inside this repo, while the agent handles orchestration, external search/sentiment tools, and natural-language summaries.

## Runtime

The container entrypoint starts the dashboard/API, the built-in bot scheduler, and the MCP
tool server together, in one process -- the MCP server runs on its own port from a thread
(`src.mcp_server.serve_in_thread`). One process, because DuckDB permits a single read-write
process per database file and locks every other opener out entirely; a separate MCP process
contended with the dashboard for `data/walbot.duckdb` and one of the two would fail with
"Conflicting lock is held". There is no `--bot` / `--mcp` process-wide mode: whether an algorithm is
driven by the clock or by an external agent is a property of its *binding*, not of the
process. Each binding declares a `frequency` for the scheduler -- `15m`, `30m`, `1hr`, `2hr`,
`1d` -- or `mcp` to park it, switched on but waiting for an agent-driven request. A process
mode could only contradict the binding's own frequency, so the dashboard reports per-binding
scheduler state instead.

`algorithm_bot.yaml` can narrow the scheduler window with `trading_start_time` and
`trading_end_time`. Setting `require_trade_approval: true` makes the built-in bot send planned
orders to Telegram and wait for an approve/deny response before submitting. The same approval
flow is available to agent-driven bindings through the `request_trade_approval` MCP tool.

Logging defaults to `WARNING`. `TRADING_LOG_LEVEL` (INFO, DEBUG, ...) overrides it, and DEBUG
also re-enables Uvicorn's per-request access logs.
