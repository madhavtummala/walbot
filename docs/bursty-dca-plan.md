# Implementation plan: Bursty DCA, unified intents, and algorithm pruning

Handoff from the session that produced `c95f646` (branch `refactor/two-step-pipeline`).
Everything below is a settled decision, not an open question.

> **Status: implemented.** Sections 1-6 are built and tested. Deviations and decisions taken
> along the way are recorded in "Implementation notes" at the end of this document; the
> sections themselves are left as written so the reasoning behind each choice stays readable.

## Context you already have

`c95f646` split algorithm runs into two stages that both the scheduled runner and the MCP
agent share:

- **Step 1** `pipeline.run_algorithm()` — algorithm + data sources. No brokerage.
- **Step 2** `pipeline.place_orders()` — algorithm + brokerage. No data fetching; prices ride
  along on the step-1 result.

Algorithms expose `requirements` / `analyze` / `refine` / `sizing`. Position-aware logic takes
a `PortfolioSnapshot` (positions + equity), never a `Brokerage`, so it also works for
backtests and for accounts with no broker configured (`PaperBrokerage`).

---

## 1. Unified intent model

The blocker for folding DCA in: algorithms speak *target portfolio* ("hold 44% BBC") while
DCA speaks *increment* ("buy $200 of VTI"). One concept absorbs both.

Step 1 emits **intents**; step 2 resolves them against the portfolio.

```python
Intent = {"symbol": str, "kind": "weight" | "notional" | "shares", "value": float}

AlgorithmResult:
    intents: list[Intent]
    mode: "target" | "incremental"
    signals, latest_prices, metadata, as_of   # unchanged
```

`mode` carries the only genuinely different semantic:

- **`target`** — the intent list is the complete portfolio; a held symbol absent from it is
  exited. Used by Fast Momentum and Regime Rotation.
- **`incremental`** — only listed symbols are touched; everything else is left alone. Used by
  DCA and Bursty DCA.

Step 2 becomes uniform:

1. read `PortfolioSnapshot`
2. `algorithm.refine(intents, signals, snapshot, prices, config)` → final intents
3. resolve to target shares
   - `target`: `weight × equity ÷ price`; unlisted holdings → 0
   - `incremental`: `current_shares + notional ÷ price`
4. `plan_position_orders` → `submit_planned_orders` (both unchanged)

**Gotcha:** `rebalance_threshold` is a target-drift concept and would wrongly suppress a small
DCA buy. Apply sizing thresholds in `target` mode only — the `sizing()` hook already gives the
per-algorithm seam.

Payoff: DCA stops being a special case and becomes a normal algorithm, which deletes the
separate submission path in `core/bot_runtime.py`.

---

## 2. Monthly budgets and wall-clock accrual

The number in each DCA bubble means **dollars per month, per symbol**. Same meaning in both
DCA and Bursty DCA, so switching algorithms never silently changes spend rate.

**Do not divide the budget by run count.** At hourly cadence that is ~141 runs/month, so a
$100 budget yields $0.71 per run — below every broker minimum, and the divisor would change
whenever the cron is edited.

Instead accrue on elapsed wall-clock time:

```
accrued += B * (hours_since_last_run / hours_in_month)
```

Cadence then controls only *opportunity to act*, never spend rate. Missed runs self-correct,
because the next run observes a longer interval.

Both variants accrue identically and differ in one predicate:

| | Executes when | Character |
|---|---|---|
| **DCA** | `accrued >= min_executable` | steady |
| **Bursty DCA** | `accrued >= min_executable` **and** signal fires | bursty |

`min_executable = max(min_trade_dollars, broker minimum, price of one share when the
brokerage has no fractional support)`.

State per symbol in the state store (`load_state` / `save_state`, same mechanism
`PaperBrokerage` uses): `{accrued, last_run_at, deployed_this_month}`.

### Suggested ranges (for UI validation)

- floor: `B` such that a trade can ever clear `min_executable`
- ceiling: ~5% of equity per symbol per month
- typical: 0.5–2% of equity per symbol per month
- show a derived line: `"$100/month ~ $4.60/trading day ~ 1.1% of equity"`

---

## 3. Bursty DCA signals

Use established rules, not a bespoke percentile.

- **Regime gate:** only buy when price is above its 200-day MA. This is the component that
  stops the strategy accumulating into a genuine decline — the main failure mode.
- **Timing:** Bollinger %B below the lower band (2 sigma, 20-day), or Connors RSI(2) < 5–10.
  `compute_rsi` already exists in `data/signals/signals.py` and is currently dead code.
- **Sizing:** value averaging (Edleson). Target value path `t * B`; trade the gap between path
  and actual position value.

Sell side is the mirror: sell the excess when position value runs above the path at a peak.

**Guards (both required):**
- clamp any single trade to ~3x `B` — expressed against the **monthly budget**, not the
  per-run increment. Getting this wrong makes the position fall permanently behind the path
  while erroring nowhere.
- cap cumulative deployment per symbol per month.

---

## 4. Rename and prune

| ID | Label |
|---|---|
| `dca` | DCA |
| `bursty_dca` | Bursty DCA |
| `fast_momentum` | Fast Momentum |
| `regime_rotation` | Regime Rotation (was `invest_spy`) |

`regime_rotation` classifies SPY into GROWING / FLAT / FALLING / CRISIS and rotates between
growth, covered-call income, cash, and hedges — regime switching, hence the name.

Remove the six `generic:*` strategies (momentum_social, trend_following, mean_reversion,
breakout, risk_parity, dual_momentum).

Targets:
- `src/algorithms/registry.py` — drop the six `generic:*` entries
- `src/core/strategy_models.py:10` — rewrite `STRATEGY_LABELS`; afterwards check whether
  `strategy_signal_rows` / `weights_from_strategy_rows` still have callers besides
  `execution/backtest.py`
- **delete `src/algorithms/generic.py`** (182 lines)
- `src/api/api_payloads.py` — the `strategy == "none"` branch and `_dca_signal_payload` become
  the real `dca` algorithm
- `web/static/app.js` — strategy dropdown and hardcoded ids
- tests referencing momentum_social / risk_parity / dual_momentum need rewriting

**Decide first:** renaming `invest_spy` changes its key under `algorithm_configs`, so saved
tuning silently falls back to defaults. Either read both keys for a release, or keep the id
and change only the display label (zero risk).

---

## 5. Signal views

Verified working: Fast Momentum shows all 16 of its universe; Regime Rotation shows all 6.
(An earlier claim that Regime Rotation was dropping VXX/XYLD/GPIX was wrong — its universe is
exactly those 6.)

**DCA's view renders nothing** when the plan is disabled, even with symbols configured.
`allocation_preview()` returns `[]` for a disabled plan. Change it to always render the
configured buckets, with `enabled` controlling only whether orders are placed:

- a row per buy/sell item regardless of `enabled`
- `signal` = LONG/FLAT by bucket
- `reason` = `Disabled` / `Accruing ($32 of $50)` / `Ready`
- summary `Planned` shows the configured monthly total, not `$0`

Bursty DCA then gets its view for free — same rows, `reason` carrying trigger state
(`Waiting for valley`, `Below 200-day MA`, `Ready`). Surface the whole-share warning here too:
*"VOO: $100/month accrues ~5 months before it can trade on this brokerage."*

Also: validate bucket symbols against the tradable universe — a typo currently produces a
silently missing row.

---

## 6. Tests to write (each fails silently otherwise)

1. **budget conservation** — a simulated month of hourly runs deploys ~`B`, within one
   `min_executable`
2. **cadence independence** — hourly vs daily cadence deploy the same monthly total
3. **clamp vs accrual** — 60 sessions with 3 triggers; cumulative deployment tracks the path
   within the clamp (catches the per-run-vs-monthly clamp bug)
4. **whole-share brokers** — Schwab + a $500 ETF + $100/month accrues ~5 months before
   trading; assert it eventually trades rather than never

---

## Known issues worth fixing alongside

- **The backtest cannot measure any of this.** `execution/backtest.py:182` calls
  `compute_signals_for_universe` regardless of the selected strategy, so it backtests daily
  momentum+social no matter what. Fast Momentum's nano/micro horizons (60% of its score) only
  exist intraday and cannot be reproduced from daily bars at all. Bursty DCA *can* be built and
  tested on daily bars — nine months are cached — but Fast Momentum cannot.
- **Intraday history is capped** at yfinance's ~60-day window (only ~8 days currently cached);
  daily EOD has ~9 months. `fetch_schwab_intraday_bars` already exists with duckdb caching —
  put `schwab` first in `intraday_market_data_provider_order` once the app is approved.
- **Schwab is entirely unrun.** Balance field names (`liquidationValue` / `cashBalance`) are the
  least certain part.
- **`web_app.py:47`** passes `"src.web_app:app"` to uvicorn but the module is `src.api.web_app`.
- **Two functions in `fast_momentum.py` are reconstructions**, not the originals:
  `sentiment_scores_from_records` and `_defensive_momentum_reason`. They satisfy the tests, but
  the `metadata` dict in the first is unconstrained guesswork.
- `tests/test_api_payloads.py::test_dca_backtest_uses_cron_schedule_and_reports_skipped_cash`
  fails because it pins bars to Jan 2026 against a "6m" window computed from today. It aged out
  of its own window; unrelated to this work.

---

## Implementation notes

Where the build departed from, or had to decide beyond, the plan above.

**A `settle` hook was added to `AlgorithmPlugin`.** The plan puts `deployed_this_month` in the
state store but never says when it is written. Deducting at proposal time would spend budget on
a run whose orders were denied at approval or rejected by the broker, so step 2 calls
`algorithm.settle(config, order_results, intents)` after submission and DCA draws down by the
*filled* notional. The remainder that rounded away below one share stays accrued, which is what
makes a whole-share brokerage eventually trade rather than never.

**Accrual is persisted in `analyze`, not step 2.** Accrual is a pure function of elapsed time,
so running it twice in quick succession accrues the same total as running it once. That makes a
dashboard preview safe: it cannot inflate or lose budget.

**Catch-up is capped at one month** (`DEFAULT_MAX_CATCHUP_MONTHS`). The plan wants missed runs
to self-correct, but with no ceiling a bot that was off for half a year would come back and try
to deploy half a year of budget at once.

**`min_executable` has a $1 floor** (`MIN_TRADE_DOLLARS_FLOOR`). On a fractional brokerage with
no configured minimum the floor would otherwise be zero, and a "ready" symbol would order a
quantity that rounds to no shares at all -- accruing forever while looking like it was trading.

**Value averaging is measured from the run's timestamp, not the wall clock.** `refine` has no
timestamp of its own, so it reads `last_run_at` from the state `analyze` just wrote. Using
`datetime.now()` put the path wherever the machine clock happened to be -- wrong by months in a
replayed or backtested run. Caught by the clamp-vs-accrual test.

**`invest_spy` kept its id on disk.** Per section 4's "decide first": the algorithm is
`regime_rotation` everywhere in code and UI, `ALGORITHM_ALIASES` resolves the old id, and both
`get_config` and `InvestSpyConfig.from_runtime_config` read either key. `config/algorithms.yaml`
still says `invest_spy:`, so saved tuning survives with zero risk.

**The backtest lost its generic rule-set path.** With the six `generic:*` strategies gone, the
`strategy_signal_rows_from_prepared` fallback in `_compute_backtest` was reachable only for
unregistered names, so it now raises instead. That left `strategy_signal_rows_from_prepared`
and `weights_from_strategy_rows` with no callers and they were deleted; `strategy_signal_rows`
stays, because `algorithms/options/swing.py` still uses the `dual_momentum` *rule set* (which is
not the same thing as the deleted `dual_momentum` strategy). Three tests that backtested deleted
strategies were removed rather than retargeted.

**Fixed alongside, as the plan suggested:** `web_app.py`'s uvicorn module path, and the
`test_dca_backtest_...` date pinning (bars are now dated relative to today, so it cannot age out
of its own window again).

**Found by the new validation:** this repo's own `config/dca_bot.yaml` lists `QQQM`, which is not
in the configured universe. It was being silently dropped by `sanitize_dca_plan` -- exactly the
failure section 5 predicted. `unknown_plan_symbols()` now surfaces it in the signal view.

### Still open

- The two backtest limitations in "Known issues" are untouched: `execution/backtest.py:182`
  still calls `compute_signals_for_universe` regardless of strategy, and Bursty DCA has no
  backtest path of its own even though nine months of daily bars would support one.
- Schwab remains unrun, so the whole-share path is covered by tests and not by a live account.

---

## Follow-up: per-algorithm schedules, and DCA on the shared loop

The plan above left cadence in config, where a single set of knobs
(`algorithm_market_data_refresh_minutes`, `algorithm_run_jitter_minutes`,
`trading_start_time`/`trading_end_time`, `algorithm_check_seconds`, `dca_check_seconds`) had to
serve every algorithm at once, and DCA had a cron of its own on top. Cadence is a property of
the strategy, not of the deployment, so it now lives on the algorithm class:

```python
@dataclass(frozen=True)
class Schedule:
    refresh_minutes: int = 60
    jitter_minutes: int = 5
    start_time: str = "08:30"
    end_time: str = "15:00"
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    check_seconds: int = 60
```

| Algorithm | Schedule | Why |
|---|---|---|
| `fast_momentum` | `Schedule()` — hourly | nano/micro momentum come from intraday bars |
| `regime_rotation` | `DAILY_AT_OPEN` | every input is a daily bar |
| `dca` | `DAILY_AT_OPEN`, `weekdays=(0,)` | fewer, larger trades clear a broker minimum sooner |
| `bursty_dca` | `DAILY_AT_OPEN` | a dip can open and close inside a week |

`weekdays` is the field that made the cron removable: `_algorithm_bucket_key` buckets *inside* a
session, so `refresh_minutes` alone tops out at daily granularity and cannot express "weekly".

**No config override**, deliberately. Two algorithms wanting different cadences cannot share one
knob, and a per-deployment override would let a backtest and the live runtime model different
schedules. `_dca_runs_on_trade_date` now reads the same `Schedule` the runtime does.

**Three runtime loops became two.** DCA runs on the equities loop like any other algorithm;
`_run_dca`, `_dca_run_key`, `_dca_check_seconds` and `src/core/cron.py` are gone, and `run_once`
already did everything `_run_dca` did (kill switch, market-open check, the two-step pipeline).
Keeping the third loop would have allowed two schedulers to drive one accrual state.

**One switch.** `dca_plan.enabled` is removed: DCA is live when it is the selected
`active_strategy` and the algorithm bot is on. Before this, DCA could be selected in the deck
*and* enabled in its own plan, and both loops would have run it at different cadences. The cost
is that DCA can no longer run alongside a second algorithm.

The plan is now only what to buy and how much — `max_item_amount`, `buy`, `sell`. `enabled`,
`algorithm`, `frequency`, `schedule_pattern` and `next_run_date` are gone from the schema, the
sanitizer, the API payload and the dashboard. `strategy_signals_payload("none")` returns an
empty view instead of falling back to the DCA plan, since DCA is selectable in its own right.

**Budgets were still on the old daily scale.** The dashboard clamped every bubble to `$50` and
wrote that clamp back into `max_item_amount`, overriding the backend's `$5,000`. `MAX_AMOUNT` is
now `$2,000/month` per symbol with `WHEEL_STEP` at `25`, and the bubble radius is normalized
(`sqrt(amount / MAX_AMOUNT)`) so the largest budget still fits the board.

### Still open after this pass

- Per-symbol budgets in `config/dca_bot.yaml` are unchanged and still read as the old daily
  numbers ($120/month total). On Schwab (whole-share) a symbol needs roughly one share price
  per month to trade monthly; on a fractional broker the floor is `config.min_trade_dollars`
  ($50), because `analyze` reads it directly while the algorithm's `sizing()` override of `0.0`
  applies only in step 2.
- Options keep the base `Schedule()`: `options_swing` is not in the equities registry, so it has
  no schedule of its own to read.

### Follow-up: SPY Rotation rename, and the DCA backtest spend rate

`regime_rotation` is now `spy_rotation` / "SPY Rotation". It has been renamed twice, which broke
an assumption in the alias machinery: `LEGACY_ALGORITHM_IDS` was a plain reverse dict, so two
retired ids mapping to one current id kept only the newest (`regime_rotation`) and dropped
`invest_spy` — which is the key actually in `config/algorithms.yaml`. Saved tuning would have
reverted to defaults with no error anywhere, exactly the failure the original comment warned
about. It is now a list per id, and both `get_config` and `InvestSpyConfig.from_runtime_config`
try every retired id in turn.

**The DCA backtest was modelling the wrong spend rate.** It deployed each symbol's full plan
amount on *every* scheduled run, but amounts have meant dollars per month since the accrual
rewrite. Against a $300/month plan over six months it contributed $7,800 on a weekly cadence and
$38,700 on a daily one — so a more frequent schedule looked better purely because it deployed
more capital, which the per-algorithm `Schedule` work would have made much more visible.
`_dca_backtest` now accrues against elapsed wall-clock time and gates on `min_executable`, reusing
`accrual.HOURS_IN_MONTH` and `accrual.min_executable`. Both cadences now converge on ~$1,750 over
the same window and differ only in trade count (21 vs 32), which is the real difference between
them.

Verified operational for every registered algorithm — backtest and signal view both:

| Algorithm | Backtest | Signal view |
|---|---|---|
| `dca` | `source=dca` | 3 leaders |
| `bursty_dca` | `source=dca` | 3 leaders |
| `fast_momentum` | `source=algorithm` | 16 leaders |
| `spy_rotation` | `source=algorithm` | 6 leaders |
| `none` | `source=flat` | empty by design |

`_strategy_backtest` still branches explicitly on `spy_rotation` and `fast_momentum` and raises
for anything else, so a newly registered algorithm gets a signal view for free but needs a
backtest branch added by hand. `bursty_dca` still backtests through the plain DCA path, so its
trigger is not modelled — only its schedule is.

---

## Follow-up: one replay harness instead of two hand-written backtests

The backtest used to be a parallel implementation of each strategy. `_strategy_backtest`
branched on `spy_rotation` and `fast_momentum` and called `compute_price_features` +
`decide_target_weights` directly against hand-sliced snapshots, because it could not call
`analyze` — `analyze` ignored `context.bars_by_symbol` and refetched live data through
`score_universe(config, data_client, ...)`. Nothing kept the two implementations in sync, and
anything not in the two branches raised.

**`analyze` is now a pure function of its context.** `score_universe` in both momentum
algorithms takes the `AlgorithmContext` and reads `bars_by_symbol` / `intraday_bars_by_symbol` /
`sentiment_scores` instead of fetching. What they need is declared in `requirements()`, which
grew `intraday_lookback_bars`, `intraday_bar_minutes`, and `needs_sentiment`. `get_daily_bars`,
`get_intraday_bars`, and `get_sentiment_snapshot` are gone from the algorithm modules.

**One place builds a live context.** `market_context.build_algorithm_context` returns a ready
`AlgorithmContext`, replacing three near-identical copies in `pipeline.run_algorithm`,
`BaseAlgorithm.signal_view`, and `DCAAlgorithm.signal_view`. `sentiment_by_symbol` was loaded,
threaded through all three, and read by nothing — it is now `sentiment_scores` plus
`market_sentiment`, which is what the algorithms actually consume.

**`src/execution/replay.py` backtests anything.** It steps trade dates, builds the context from
a point-in-time slice, and runs `analyze → refine → simulate fills → settle` — the live loop
with the brokerage replaced by a fill simulator. Signals come from the previous bar and execute
at the current close, so no decision can see the price it trades at. `_compute_backtest` has no
per-strategy branch left; a new algorithm is backtestable as soon as it declares
`requirements()` and a `Schedule`.

**State is isolated per replay.** `state_store.ephemeral_state()` redirects every read and write
to a throwaway dict via a `ContextVar`, so no algorithm needs to know it is being replayed. A
replay starts from a clean slate rather than inheriting live accrual — reproducible, and it
cannot write back over the live account's state.

**Intraday coverage is reported, not assumed.** The cache is write-through on every live
intraday fetch (`_write_duckdb_bars`) and the PK is per-bar, so it accumulates and deepens over
time; the TTL is a read-freshness gate, not eviction. A window the cache does not reach used to
score every symbol near zero and read as a poor strategy. The replay now returns a `Coverage`
record — requested vs supplied, and which symbols were missing — surfaced on the payload as
`coverage`.

Net effect: `api_payloads.py` lost 505 lines and gained 66; `replay.py` is 273. Bursty DCA is
now backtested through its *real* trigger for the first time — previously it ran the plain DCA
path and its regime gate was never modelled, which is why it withholds on flat synthetic bars
where the old backtest happily bought.

Verified for all four algorithms — backtest and signal view, every one now `source=algorithm`:

| Algorithm | Backtest | Signal view |
|---|---|---|
| `dca` | OK, 129 rows | 3 leaders |
| `bursty_dca` | OK, 129 rows | 3 leaders |
| `fast_momentum` | OK, 129 rows | 16 leaders |
| `spy_rotation` | OK, 129 rows | 6 leaders |
| `none` | flat, by design | empty, by design |

### Adding an algorithm now

1. Subclass `BaseAlgorithm`, set `algorithm_id` and `schedule`.
2. Declare data needs in `requirements()`.
3. Implement `analyze(context)` reading only from the context; `refine`/`settle` if it is
   position-aware or stateful.
4. Register it in `ALGORITHM_REGISTRY` and `ALGORITHM_IDS`, add a `STRATEGY_LABELS` entry and a
   deck card in `app.js`.

Signal view and backtest both work with no further wiring.
