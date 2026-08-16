# Consolidation pass 2

The first pass (recorded below under "Pass 1, completed") unified the backtest paths and
extracted `common/config_utils.py`. This second pass came out of a survey looking for the same
failure mode elsewhere: **two implementations of one concept, where only one of them got
fixed.** That is what produced the separate DCA backtest, and it is still the dominant pattern
in this codebase.

Two live bugs found during the survey are the clearest examples, so they lead.

> **Status: all eight passes implemented.** Full suite: **437 passed** (from a 437 baseline;
> six tests deleted with the code they covered, six added). Both bugs were reproduced against
> the described scenario before and after the fix -- see "Verification" at the end.

## What the drawdown breaker actually does

Worth stating plainly, because the name suggests something narrower than the behaviour.

The trigger has nothing to do with order size against equity. It is
`equity / session_opening_equity - 1 <= intraday_drawdown_limit` -- a fall in *account equity*
within one session, measured from the equity at the session's first run (-2% for Fast Momentum,
-3% for SPY Rotation, -1.5% for Dual Momentum).

When it trips, `apply_risk_guards` returns every proposed weight as zero. In `target` mode that
is not "stop buying" -- `resolve_target_shares` seeds every *currently held* symbol at zero
first, so a zero-weight book is an instruction to **sell everything**, including holdings the
proposal did not mention:

```
weights after the breaker trips: {'QQQ': 0.0, 'TLT': 0.0}
held before:                     {'QQQ': 50.0, 'TLT': 30.0, 'GLD': 20.0}
=> orders:                       {'QQQ': -50.0, 'TLT': -30.0, 'GLD': -20.0}
```

So: it liquidates to cash, and stays flat for the rest of that session however the market
moves. There is no "buys blocked but sells allowed" state, and no partial de-risking -- the
volatility scaler is the graduated response; this one is the stop. The next session re-anchors
on whatever equity it opens with and the algorithm trades normally again, which is why it is a
session breaker rather than a kill switch in the permanent sense. (Dual Momentum's breaker is
gentler: it parks the book in the defensive sleeve rather than in cash.)

One consequence worth knowing when reading backtest numbers: **the breaker never fires in a
daily-stepped replay.** The replay runs the algorithm once per trade date, so every call opens
a new session and the drawdown is measured from that same call's equity -- always zero. That is
the honest answer given daily closes, since intraday drawdown is not observable in them, but it
does mean a backtest omits a stop that live trading applies.

---

## Pass A -- the two bugs, which are both duplication bugs

### A1. Fast Momentum's kill-switch latches for an entire backtest

Observed as thousands of `Fast Momentum kill-switch active; target weights set to zero`
warnings from a single backtest run.

`fast_momentum.intraday_kill_switch_triggered` keys its session on `date.today()` and keeps its
own state through `load_state`/`save_state`. Inside a replay every simulated date is "today", so
the first simulated day that is 2% below the *first* day of the backtest latches `halted = True`
and every subsequent date proposes all-zero weights. The backtest is measuring nothing after
that point, and it says so once per date.

`dual_momentum.stateful.intraday_drawdown_breached` already solves exactly this, by taking a
`session` string derived from the algorithm's own timestamp rather than the wall clock. Its
docstring even describes the failure Fast Momentum still has. One concept, two implementations,
one of them fixed.

**Change.** `as_of` becomes a required argument of the step-2 contract, and the only clock any
shared code reads.

- `AlgorithmPlugin.refine(..., as_of: datetime)`. It is required rather than defaulted on
  purpose: a default is reached by forgetting, so it puts the wall clock back one call site at
  a time.
- `AlgorithmResult.as_of` is set from `context.timestamp` in both `run_algorithm` and `replay`,
  instead of defaulting to `datetime.now()`. It was wall-clock in a backtest before, which was
  the same bug one layer up and had simply not bitten yet.
- One `session_drawdown_breached(state, equity, limit, as_of)` in `src/algorithms/risk.py`,
  used by both algorithms, with no fallback of any kind.
- Dual momentum stops denormalising `as_of` onto every signal row and parsing it back out --
  that was a workaround for the argument that now exists. Its cooldown, selection throttle and
  breaker all read the parameter.
- Bursty DCA's `refine` measured its value-averaging path from `datetime.now()`; it uses
  `as_of`. DCA's `analyze` replaced a naive `context.timestamp` with `now`; it localises it
  instead.

`tests/test_architecture.py::test_an_algorithm_never_reads_the_clock` walks the AST of every
module under `src/algorithms` and fails on `datetime.now()` or `date.today()`, so the class of
bug cannot come back quietly. The only exemption is the options runner, which has no replay
path.

### A2. `Can't open a connection to same database file with a different configuration`

Observed as a 500 from `/api/schwab-auth` while a backtest was running: `load_state` ->
`_connect` -> `duckdb.connect(path)` raised `ConnectionException`.

`pooled_connections(read_only=True)` holds a read-only DuckDB connection for the length of the
replay. The pool bookkeeping is **thread-local** (`_POOLING`, `_READ_POOLING`, and pools keyed by
`(thread_ident, path)`), but DuckDB's constraint is **per process**: once one connection to a
file is open read-only, no other connection in that process may open it read-write. Any other
request thread touching the state store during a backtest therefore dies -- including pure
reads, which only fail because `_connect` opens read-write by default.

The thread-local design also means the same process opens N connections to one file, which is
what makes the modes able to disagree in the first place.

**Change.** One process-wide registry, one connection per database path:

- `_Handle` per resolved path holding the connection, its mode, a scope refcount and a count of
  outstanding cursors, all under one lock.
- `pooled_connections` refcounts scopes process-wide rather than per thread; the outermost exit
  closes the connection so other *processes* get the file back.
- `_connect` hands out `connection.cursor()` -- DuckDB's supported way to use one database from
  several threads -- wrapped so `with` closes the cursor and not the connection.
- A write attempted on a read-only handle upgrades it: wait under the lock until no cursors are
  outstanding, reopen read-write, retry once. Detected by catching DuckDB's read-only error in
  the wrapper's `execute`, so no call site has to declare intent.

This deletes `_CONNECTIONS`, `_READ_CONNECTIONS`, `_POOLING`, `_READ_POOLING`, the composite
thread keys and the duplicated read-only/read-write branches of `_connect`, and replaces them
with one path that cannot get into the state that produced the crash.

---

## Pass A3 -- "fetch aborted" on a 12M backtest

Not an error at all: the server completed the backtest successfully in **97.5s**, and the
dashboard's request timeout is 60s, so the browser aborted the fetch and reported its own
abort as the failure.

The cost was the same asymmetry as everywhere else in this document. Daily bars are loaded once
for the whole replay and sliced per date -- `_slice_daily`, with a comment saying exactly why --
but the *intraday* window was re-read from DuckDB on every trade date, for every symbol, for
every configured provider. Measured: 84 connections per date, 14 symbols x 2 provider attempts,
so **21,000 database round trips** for a twelve-month window. 97 of the 99 seconds; the daily
fetch was 1.8s of it.

**Change.** `HistoryCache` reads each symbol once for the whole replay span and slices per
date, reproducing `read_history`'s own window (`[end - calendar_days_for(lookback), end]`) so
each date sees exactly the rows it saw before. The provider walk still runs in configured
order, now once rather than once per date, and stops early on a provider that covers the whole
span.

A second, smaller one fell out of the profile: the deepest-wins comparison called
`coverage_minutes` three times per provider per symbol per date, and measuring a window rebuilds
its frame and market-minute axis. Measuring once and carrying the number made the bookkeeping
cheaper than the scoring it feeds, where it had been more expensive. The old per-date code had
the same redundancy.

| window | before | after |
|---|---|---|
| 4M | 34.7s | **14.5s** |
| 12M | 97.5s | **37.1s** |
| 24M | -- | **68.3s** (coverage 1.0) |

Equity curve, metrics, order count, turnover and coverage are identical at every step -- checked
by running the old per-date read and the new sliced read against the same cache and comparing
the full curve, not just the summary.

Two frontend changes go with it: the backtest request gets a 300s timeout (a fresh 24M replay
covers roughly five times the trade dates of a 4M one, and the cache probe stays on a short
one), and an aborted fetch now reports *"Timed out after Ns. The server may still be working;
try again in a moment."* rather than whichever wording the browser happens to use for an
`AbortError`.

## Pass A4 -- a new backtest period showed nothing until something else moved focus

Selecting 12M in the dropdown fetched the cached 12M backtest, put it in state, and never
painted it. Only clicking "Run backtest" -- which recomputes -- appeared to work.

`render()` declines to repaint while a `<select>`, `<input>` or `<textarea>` inside `#content`
has focus, so that rebuilding the body cannot close an open dropdown or discard half-typed
input. But a `<select>` still holds focus *after* its own `change` event, and the skipped
repaint was dropped rather than deferred. So the state was right and the screen was stale.

**Change.** The change handler renders with `{ force: true }` -- a change event means the
interaction is over, so there is nothing left to disturb. And a render skipped by the focus
guard now sets `renderDeferred` and is flushed on `focusout`, so no repaint is silently lost:
the same hole would have swallowed any background update that landed while a control was
focused.

## Pass A5 -- an empty universe silently meant the default one

Found while building the sweep harness. `parse_symbols` returned the default whenever its
parsed result was empty, which conflated "unset" with "explicitly empty" -- so
`defensive_universe: []`, written to turn a sleeve off, restored the five-name default instead.
A setting that looks obeyed and is not.

An empty *list* is now honoured. Only `None` or a blank string -- which is what an empty form
field sends -- still means unset.

## Pass B -- one way to load an algorithm's tuning

`from_runtime_config` is implemented four times as field-by-field coercion:

| | lines |
|---|---|
| `DualMomentumConfig.from_runtime_config` | 80 |
| `InvestSpyConfig.from_runtime_config` | 70 |
| `DefensiveMomentumConfig.from_runtime_config` | 48 |
| `BurstyConfig.from_runtime_config` | 12 (already generic -- the pattern to generalise) |

Every line is `field=as_float(raw.get("field"), defaults.field)`. Adding a knob means editing
the dataclass *and* the loader, and forgetting the second half silently ignores the knob.

**Change.** `load_tuning(cls, raw, *, fallbacks=None)` and `tuning_section(config, *ids)` in
`common/config_utils.py`. Coercion is inferred from the declared field type:

- `list[str]` -> `parse_symbols`
- `bool` -> `as_bool`, `float` -> `as_float`
- `int` -> `as_int`, except a name ending `_minutes`, which goes through `minutes_knob`
- `str` -> stripped text, or an upper-cased ticker when the field declares
  `metadata={"coerce": "symbol"}` (only `benchmark` and `spy_symbol`)

`fallbacks` supplies the two defaults that come from the account config rather than the
dataclass (`per_trade_value_min` <- `min_trade_dollars`, `rebalance_threshold`). The
`selection_horizon_*` fields declare their non-standard legacy key in metadata.

Each `from_runtime_config` becomes two or three lines, and a new knob is a dataclass field and
nothing else.

## Pass C -- one copy of each small helper

Found by name across `src/`:

- `_return_over(closes, periods)` -- identical in `fast_momentum`, `invest_spy`,
  `dual_momentum/scoring` -> `data.bars.return_over_periods`.
- `_closes(bars)` -- `dual_momentum/scoring`, `brokerages/schwab_datacheck`, plus the same
  expression inline in both momentum algorithms -> `data.bars.closes`.
- `json_number` / `_finite` -- four copies (`common/config_utils`, `core/strategy_models`,
  `connectors/support`, `connectors/utils`) -> one, in `common/config_utils`.
- `_parse_time` / `_parse_timestamp` -- `dual_momentum/stateful`, `dca/accrual` ->
  `common/timeutils.parse_iso_utc`.
- `dca/accrual._as_float` -> `config_utils.as_float`.
- `fast_momentum.weights_from_positions` -> `PortfolioSnapshot.weights` (same arithmetic).
- `connectors/support.EXTERNAL_AUTH_PROVIDERS` is defined twice in one file.

## Pass D -- one ranking and allocation primitive

`invest_spy._ranked` and the `ranked` closure inside `fast_momentum.decide_target_weights` are
the same filter-and-sort differing only in which feature is the secondary gate. `_allocate_dynamic`
and the allocation loop in `decide_target_weights` are two spellings of score-proportional
allocation with a per-name cap. "Rescale if gross exceeds the maximum" appears four times.

**Change.** `src/algorithms/allocation.py`:

- `rank_by_score(scores, symbols, *, min_score, gate_key=None, min_gate=None, require_trend=True)`
- `allocate_by_score(candidates, exposure, max_positions, max_weight)`
- `scale_to_gross(weights, max_gross)`

Unified on the redistributing allocator (Fast Momentum's), which spreads the residual when a
name hits its cap instead of dropping it. Under SPY Rotation's current settings the cap can
never bind on more than one candidate, so the numbers are unchanged; where it would bind, the
redistributing version is the correct one.

## Pass E -- delete the third dual momentum

`src/algorithms/equities/dual_momentum_optimizer.py` is a complete third implementation of dual
momentum -- its own `DualMomentumConfig` (colliding by name with the real one), its own scoring,
its own simulation loop. Nothing imports it except its own test: not the registry, the API, the
dashboard, the MCP server, or `bot_runtime`. Delete the module, the `equities` package and
`tests/test_dual_momentum_optimizer.py`.

## Pass F -- strip the parallel scoring model to what is actually reached

`core/strategy_models.strategy_row_from_prepared` implements six strategies. Production reaches
exactly one: `"dual_momentum"`, from `options/swing.py`. The `trend_following`,
`mean_reversion`, `breakout`, `risk_parity` and `fast_momentum` branches, plus
`_rank_defensive_momentum_rows`, `_defensive_momentum_flat_reason` and the
`DEFENSIVE_MOMENTUM_*` constants (whose only other user is the module Pass E deletes), are
reachable only from their own tests. Remove them and prune the tests to the surviving path.

*Deliberately not done:* pointing `options/swing.py` at the real `dual_momentum` algorithm
instead of this daily 126/252-day rule. It would delete the rest of the module, but it changes
which underlyings the options runner trades, which is a strategy decision rather than a
refactor.

## Pass G -- package hygiene in `dual_momentum`

The five modules of the package each redeclare `STATE_KEY`, `EPSILON` and `TRADING_DAYS`
(leftovers from the file split), and `__init__` re-exports 25 names of which 12 are private.
Import the constants from `config.py`; trim `__all__` to the public surface plus what the tests
actually import.

## Pass H -- the config schema reader

`core/config/schema.py` ends with ~120 lines of
`_as_int(_config_value(section, "key", "KEY", DEFAULT), DEFAULT)`, in which the default is
written twice and the environment name is always the key upper-cased. A `reader(section)`
helper returning `read(key, default, env=None)` -- casting from the default's type -- removes
the repetition and the two-defaults-out-of-sync failure. `coercion._parse_symbols` and
`_str_to_bool` fold into `config_utils`.

---

## Verification

One run at the end, not between passes:

```
.verify_venv/bin/python -m pytest -q      # 435 passed (baseline 437)
```

Passes E and F delete production code, so six of their tests went with it; four were added
(the two connection behaviours, and the kill-switch session reset). No test covering surviving
behaviour was removed or weakened. `tests/test_architecture.py` guards the import-cycle and
layering invariants and still passes.

Both bugs were also reproduced directly, since a passing suite is not evidence that the
reported symptom is gone.

**A1.** A 60-day Fast Momentum replay over a steadily falling universe, counting the
kill-switch warnings:

| | warnings | dates that traded |
|---|---|---|
| session keyed on the wall clock (the old behaviour) | 34 | 21 of 60 |
| session keyed on the algorithm's timestamp | **0** | **46 of 60** |

The old path stopped trading a third of the way in and logged the rest; that is the warning
storm in the report, and the equity curve after the latch was measuring nothing.

**A2.** A second thread reading and writing the state store while a read-only pool is held on
the same file -- the exact `load_state` path from the traceback -- now reads
`{'access_token': 'abc'}` and completes its write, where it previously raised
`ConnectionException: Can't open a connection to same database file with a different
configuration`.

---

# Pass 1, completed

Unified the backtest paths and extracted the coercion helpers.

1. **Divergent backtesting paths**: an ad-hoc simulation loop in
   `algorithms/equities/dual_momentum_optimizer.py` alongside the canonical point-in-time
   replay in `execution/replay.py`. (Pass E above finishes this by deleting the module.)
2. **Duplicated type coercion**: `_as_float`, `number`, `integer`, `symbols`, `flag`,
   `json_number`, `minutes_knob` copied across seven modules -> `common/config_utils.py`.
3. **Legacy algorithm helpers**: `get_history_bars`, `get_daily_bars`,
   `get_sentiment_snapshot` removed from `fast_momentum.py`.

Validation: full suite passed (437).

Follow-up fix from that pass: every `db_path` default was bound to `DUCKDB_STATE_PATH` at import
time, so the test fixture's runtime rebinding produced a path mismatch inside the read-only
pool. All signatures now resolve the path at call time (`db_path: str | None = None`).
