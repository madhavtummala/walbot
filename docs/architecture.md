# Architecture

## Data Sources

Data sources are grouped by the shape of data they return:

- `intraday_market_data`: minute-style OHLCV bars.
- `eod_market_data`: daily OHLCV bars.
- `sentiment_data`: news, social, and sentiment records.

Provider waterfall order is the order of keys under each category's `providers:` mapping in `config/walbot.yaml`'s `data_sources`. There is no separate `provider_order` key in the desired connector contract.

A provider is one module of fetcher functions -- `market/<name>.py`, `news/<name>.py` -- registered by path in `src/connectors/registry.py` and imported on first use. There is deliberately no provider base class: `connectors/base.py` held three ABCs that nothing implemented and nothing referenced, the second attempt at an abstraction over these fetchers, and it went the way of the first.

The shared plumbing is split by what it does rather than gathered into one module called `support`:

| module | holds |
| --- | --- |
| `grid.py` | Which bar resolutions each provider serves, and how many bars a market-minutes window spans. |
| `sources.py` | Which provider to try next, whether one is usable, and its credentials. |
| `http.py` | Making the call, and telling a rate limit from an outage. |
| `cache.py` | The DuckDB bar store, and the payload cache in front of it. |
| `frames.py` | Shaping whatever came back into the one frame everything else reads. |

`service.py` is the dispatcher on top: provider fallback, and the public fetch functions. It used to re-export all forty names from `support` so callers could reach the toolkit through the dispatcher -- twenty-eight of them unused there, which made the wall a second, drifting description of the toolkit.

## Market Bars

Intraday and EOD data share the `market_bars` DuckDB table because both are OHLCV bars:

- `category` separates `intraday_market_data` from `eod_market_data`.
- `timeframe` separates `1m`, `15m`, `30m`, `1d`, etc.
- `(category, provider, symbol, timeframe, timestamp)` is the natural uniqueness key.

This keeps reads simple for algorithms: ask for symbol + timeframe + category, and receive the same `timestamp/open/high/low/close/volume` frame regardless of provider.

## Algorithms

Concrete algorithm implementations live under `src/algorithms/`.

The common algorithm contract is `BaseAlgorithm` in `src/algorithms/base.py`; the dataclasses it speaks in live in `src/core/interfaces.py`. An algorithm exposes:

- `requirements(config, current_positions)`: symbols, shared data needs, whether it carries state, and runtime restrictions. Declaring the need rather than fetching is what makes an algorithm replayable — the backtester satisfies the same declaration from cached history.
- `plan(context)`: **pure**. Everything it reads arrives on the `AlgorithmContext` — bars, holdings, equity, and the memory of its previous runs. No fetching, no clock (`context.timestamp` is the moment the run describes), no brokerage, no state store. It returns an `AlgorithmPlan`: intents, signals, metadata, and the state it wants remembered.
- `execute(plan, config, brokerage)`: the half that touches the world. Places the plan's orders through `place_orders` and persists `state_after(plan, outcome)` — so state is committed only when orders actually go out.
- `signal_view(plan)`: renders a plan for the dashboard. A rendering, never a second run.
- `tuning(config)`: the algorithm's own knobs, built from the dataclass it names in `tuning_class`. Loading is framework work — each algorithm used to write its own loader, and one of the three ended up bypassing `load_tuning` entirely.

The split is what makes looking free of consequence: the dashboard, a reviewing MCP agent and the scheduler all make the same `run_algorithm` call, and none of them can move money or shift the next run's memory by doing so.

### Every algorithm has the same shape

A package with three modules, so that knowing one is knowing all of them:

| module | holds |
| --- | --- |
| `config.py` | The frozen tuning dataclass, and nothing else. |
| `algorithm.py` | The subclass and the market logic only it uses. |
| `signals.py` | The per-symbol rows `plan` publishes, and the `signal_view` that renders them. |

`__init__.py` stays a docstring — a facade there re-exporting internals makes the package a second, silently drifting description of what the algorithm is made of, and importing `algorithms.ids` from `core.config` would drag the whole runtime in behind it. Further private modules are for an `algorithm.py` that would otherwise be unreadable, not a default; `rally_rotation` is the only one that qualifies, splitting into `scoring.py` (features and the cross-sectional score), `gates.py` (every test a name must pass), `sizing.py` (weights, and how much of the move to make today) and `memory.py` (what it remembers between runs).

### Decisions are auditable, not narrated

`plan` publishes per-symbol rows, and each one carries a structured account of its own outcome rather than a sentence about it:

- `action` — one of `enter`, `hold`, `exit`, `blocked`, `idle`. Five outcomes cover every algorithm, which is what lets one dashboard render all of them without knowing which it is looking at.
- `checks` — every gate the run applied, passed and failed alike, each a `Check(label, ok, value, limit, blocking)` carrying **what was measured beside the threshold it had to clear**. Passed gates matter: they distinguish "cleared everything but the vol ceiling" from "failed at the first hurdle".
- `blocking` marks the one gate that actually decided the outcome. Several can fail at once, and they are usually consequences of the same fact.

A verdict is `all(check.ok for check in checks)` — the test and the message about the test are one object, so they cannot drift apart. Gates a *selection* imposes (a settling period, a slot contest, a re-rank throttle) are recorded by the pass that imposes them and appended to the same list; re-deriving them from the market gates produced confident nonsense like "Rank 1, outside the top 5".

Two module-level functions in `src/core/runner.py` drive it: `run_algorithm(strategy, config)` builds the context and returns the plan; `execute_algorithm(plan, config, brokerage)` acts on it. The scheduler chains them; the MCP flow pauses in between so an agent can review the plan first (see *Agent/MCP Direction*).

`src/core/pipeline.py` is the order-placing layer underneath allocation strategies, and knows nothing about algorithms — it takes intents, prices and two sizing knobs, and reports what the broker did. That ignorance is load-bearing: it is why `BaseAlgorithm` can import it outright without closing an import cycle back through the registry. A lifecycle algorithm whose output is a book of resting orders rather than a target portfolio -- Options Flip is the one deployed example -- bypasses this layer entirely: its `plan()` fills `AlgorithmPlan.desired_orders` instead of `intents`, and its own `execute()` override goes straight to `src/algorithms/reconcile.py`'s `reconcile_orders`, which diffs the wanted set against the broker's actual working orders (cancel what dropped out, submit or re-price the rest) rather than sizing a portfolio at all.

Algorithms are registered in `src/algorithms/registry.py`. The live runner resolves the selected strategy through that registry instead of branching on individual strategy ids.

What more than one algorithm needs lives outside them, so a rule cannot be fixed in one place and left broken in another:

- `src/common/config_utils.py`: `load_tuning(cls, section)` builds a tuning dataclass from saved config, coercing each field by its declared type. A new knob is a dataclass field and nothing else -- there is no parser to update alongside it.
- `src/algorithms/reconcile.py`: `reconcile_orders`, for algorithms whose output is a book of resting orders.
- `src/algorithms/risk.py`: the session drawdown breaker. **Currently wired to nothing**, and deliberately so -- it measures the drop from the equity it first saw *this session*, so at a daily cadence every run rebases the reference and the drawdown it reports is identically zero. `test_a_session_breaker_cannot_fire_at_this_algorithms_cadence` is the record of why the knob was removed rather than switched off, and the module is kept for an intraday algorithm that could use it honestly.

DCA and options strategies live in the same hierarchy as equity algorithms -- `src/algorithms/bursty_dca/` and `src/algorithms/options_flip/`. They can be rendered on separate frontend pages, but backend-wise they are algorithms producing an `AlgorithmPlan` from an `AlgorithmContext` like any other.

## Brokerages

One package per venue, each with the same two roles — the same rule the algorithms follow.

| module | holds |
| --- | --- |
| `brokerage.py` | The `BaseBrokerage` subclass: positions, account state, orders. What the order path talks to. |
| `client.py` | The authenticated session and the calls built on it. Shared, because "place an order" and "fetch a bar" hit the same vendor with the same credentials — `connectors/market/` reads this module too. |

`src/brokerages/{alpaca,schwab,paper}/`. Schwab adds `auth.py` for the one-time OAuth consent the dashboard drives; `paper` has no `client.py` because there is nothing to talk to, which is the point — an algorithm can be run, sized and "traded" before any real broker is configured.

`BaseBrokerage.supports_fractional_shares` is part of the contract rather than an attribute each provider happens to define. It is the most consequential brokerage fact in the system: it sets the smallest trade that can reach the market, which is what makes a $100/month DCA budget against a $600 share accrue for six months before it can trade. It used to be read with `getattr(broker, ..., False)`, so a provider that forgot it silently became whole-share — the safe-looking default that is wrong for two of the three venues.

`registry.py` maps an id to a `"module:Class"` string resolved on first use. Importing the classes eagerly meant touching *any* brokerage loaded the Alpaca SDK — on a Schwab-only deployment, for a class never constructed — and put `core.config` at the end of an import chain starting in `core.pipeline`, which is how that cycle formed.

## Agent/MCP Direction

The clean MCP boundary is around deterministic components, not around the whole bot loop.
`src/mcp_server.py` exposes six tools: `list_bindings`, `get_algorithm_plan`, `list_accounts`,
`get_account`, `get_account_orders`, `place_orders`. The account is the unit for reading —
`list_accounts` then one call per row, so a slow broker costs that account and not the answer.

Proposing and submitting are two calls with a gap for judgement in between:

1. `get_algorithm_plan` runs the algorithm and returns the proposal plus a `plan_token`.
2. The agent checks the named symbols against news the algorithm could not see.
3. `place_orders(plan_token)` submits it. Declining needs no call at all.

**The plan never travels back.** It is held in `src/core/plan_cache.py` — in-process,
single-use, ~90s TTL — so what executes is what was reviewed, rather than a plan rebuilt from
whatever the agent returned. An agent cannot forge a proposal, revise `latest_prices`, or edit
the opaque `state` an algorithm commits, because it is never asked for any of them. The TTL is
what keeps the reviewed plan and the live market close enough to be the same thing: a plan
commits the prices it was built with, so age is staleness, and expiry is a normal outcome.

Within that, `place_orders` takes `edits` — `{"op": "skip"|"exit", "symbol": ...}`, applied
server-side by `src/core/plan_edits.py`. An edit names an action, never an amount: the agent
judges whether a plan survives contact with today's news, and sizing stays the algorithm's.
`skip` is spelled per mode because absence means opposite things in the two — under
`MODE_TARGET` the intent list *is* the portfolio, so a row dropped as a "veto" would sell the
position, and skip therefore pins to the shares currently held. Order-book plans (Options Flip)
refuse edits outright: reconciliation cancels whatever is not in `desired_orders`, so removing
a leg cancels a resting stop rather than declining an action.

This keeps strategy math and brokerage execution deterministic inside this repo, while the agent handles orchestration, external search/sentiment tools, natural-language summaries -- and the review itself, which is the only approval step there is.

## Runtime

The container entrypoint starts the dashboard/API, the built-in bot scheduler, and the MCP
tool server together, in one process -- the MCP server runs on its own port from a thread
(`src.mcp_server.serve_in_thread`). One process, because DuckDB permits a single read-write
process per database file and locks every other opener out entirely; a separate MCP process
contended with the dashboard for `data/walbot.duckdb` and one of the two would fail with
"Conflicting lock is held". There is no `--bot` / `--mcp` process-wide mode: whether an algorithm is
driven by the clock or by an external agent is a property of its *binding*, not of the
process. Each binding carries a **cron expression** in market time -- `0 11 * * 1-5` -- or an
empty one, which parks it: switched on, waiting for an agent. A process mode could only
contradict the binding's own schedule, so the dashboard reports per-binding scheduler state
instead. `src/core/cron.py` holds the evaluator and the two deliberate departures from
crontab(5): day-of-month and day-of-week are ANDed rather than ORed, and there is no implicit
catch-up for missed slots.

There is no out-of-band trade approval. It existed as a Telegram approve/deny round trip and
went when Telegram did: an approval gate whose only transport is removed does not fail safe, it
silently skips every order. Review before submission is the MCP flow's job -- `get_algorithm_plan`
returns without placing anything, and nothing is sent until `place_orders` is called.

Logging defaults to `WARNING`. `TRADING_LOG_LEVEL` (INFO, DEBUG, ...) overrides it, and DEBUG
also re-enables Uvicorn's per-request access logs.
