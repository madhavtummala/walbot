---
name: walbot-mcp-agent
description: Use this skill when an external agent needs to operate Walbot through its MCP tools — review the algorithm's proposed portfolio, validate it against outside research, and submit the reviewed set of target weights.
---

## What this is

Walbot runs in two steps. Step 1 proposes a portfolio from market data. Step 2 turns a
reviewed proposal into orders. You sit in between: the scheduled runner chains the two
directly, and the only thing you add is validation. Research is done with your own tools
(web search, etc.) — the trading bot does not provide it.

## MCP Tools (walbot, from `src/mcp_server.py`)

### `get_algorithm_result(algorithm="fast_momentum")`
Step 1 — algorithm plus market data. Touches no brokerage, so it works even with no account
configured. Nothing is submitted.

- `target_weights` — `{symbol: fraction_of_equity}`, the proposed portfolio.
- `signals` — per symbol: `score`, `reason`, `signal`, `score_components`, `realized_volatility`.
- `latest_prices` — the prices the proposal was built from.
- `as_of` — when it ran. Step 2 refuses results older than 15 minutes.

**Pass the whole payload back to `place_orders` unchanged.** It carries the prices step 2
uses to size shares; step 2 does no market-data fetching of its own.

### `get_current_positions()`
Live holdings from the brokerage — `symbol`, `shares` — plus `equity`, `cash`, `buying_power`.
Use as the source of truth for what is actually held.

### `place_orders(algorithm_result, target_weights=None)`
Step 2 — brokerage plus algorithm. Submits immediately.

- `algorithm_result` is the step-1 payload, passed back as-is.
- `target_weights` is **optional**. Omit it to submit the algorithm's own proposal. Supply it
  only to override — and then it is the complete intended portfolio, so any held symbol left
  out is sold to zero. Weights must be non-negative and sum to ≤ 1.0.
- You cannot introduce a symbol step 1 never priced; it is rejected by name rather than
  silently skipped.
- The algorithm still applies its own **stickiness** and risk guards on top of whatever you
  submit, so a held name you dropped may be retained if no challenger beats it by the
  configured score delta. This is intentional churn control, not the tool ignoring you.
- Returns `diff` — per symbol `current_weight`, `final_weight`, `change`, `action`
  (`add`/`trim`/`hold`), largest change first. This is what to summarise.
- Returns `status` and `order_results`; each order carries `submitted` (with `order_id`) or
  `rejected` (with `reason`).
- Top-level `status`: `submitted`, `partial` (some legs rejected — see `rejected`),
  `rejected`, `no_orders`, `skipped` (kill switch), `error`. A rejected leg does not abort the
  rest; a common cause is shares still held for an earlier unfilled order.

## Workflow

1. `get_algorithm_result()` — read the proposal and the `score`/`reason` behind each symbol.
2. `get_current_positions()` — confirm what is actually held.
3. Validate with your own research tools. Check each symbol the algorithm wants to add or drop
   against recent news, earnings, and pending corporate actions. This step is yours.
4. Decide. If the proposal stands, call step 2 with just the payload. If research rejects a
   name, pass `target_weights` with it removed or resized — remembering that omitting a held
   symbol liquidates it.
5. `place_orders(algorithm_result=<step 1 payload>, target_weights=<only if overriding>)`.
6. Report using `diff` and `order_results`, tying each change back to the algorithm's `reason`
   and to what your research found.

Do not invent symbols the algorithm did not surface, and do not treat a backtest as approval.
If step 1 returns a stale-result error at step 2, re-run step 1 rather than retrying.

## Fast Momentum Algorithm

Composite score from z-scored nano (10 bars), micro (78), meso (60d), macro (180d) returns +
sentiment + pullback bonus. Risk-on symbols need `macro_trend_ok` and a minimum micro return.
The top `max_positions` are selected; weights are proportional to score, capped at
`max_single_position_weight`.

### Reason codes (`signals[symbol].reason`)
`Top Rank` = selected. Otherwise: `Macro negative`, `Score too low`, `Micro too low`,
`No rank slot`.

## Fractional shares

Order sizing follows the brokerage's `supports_fractional_shares` capability — true for
Alpaca and the local paper brokerage, false for brokerages that require whole shares.

- Fractional quantities are kept to **2 decimal places**.
- Sizing always truncates rather than rounds up, so a filled position never exceeds its
  target dollar amount. A target weight may therefore be slightly under-filled — more so on
  whole-share brokerages and high-priced symbols.
- Short targets are always sized in whole shares, since fractional quantities cannot be
  shorted.
