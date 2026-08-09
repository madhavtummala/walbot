# Implementation plan: DCA Bursty, unified intents, and algorithm pruning

Handoff from the session that produced `c95f646` (branch `refactor/two-step-pipeline`).
Everything below is a settled decision, not an open question.

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
  DCA and DCA Bursty.

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
DCA and DCA Bursty, so switching algorithms never silently changes spend rate.

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
| **DCA Bursty** | `accrued >= min_executable` **and** signal fires | bursty |

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

## 3. DCA Bursty signals

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
| `dca_bursty` | DCA Bursty |
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

DCA Bursty then gets its view for free — same rows, `reason` carrying trigger state
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
  exist intraday and cannot be reproduced from daily bars at all. DCA Bursty *can* be built and
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
