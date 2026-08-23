---
name: walbot-mcp-agent
description: Use this skill when an external agent needs to operate Walbot through its MCP tools — review the plan an algorithm proposes, validate it against outside research, and submit the reviewed plan.
---

## What this is

Walbot separates *deciding* from *acting*. `get_algorithm_plan` runs the algorithm and returns
what it wants to do; `place_orders` submits it. You sit in between: the scheduled runner chains
the two directly, and the only thing you add is validation. Research is done with your own tools
(web search, etc.) — the trading bot does not provide it.

## MCP Tools (walbot, from `src/mcp_server.py`)

### `get_algorithm_plan(algorithm="rally_rotation", binding_id="")`
Runs the algorithm against market data and the bound account's book. Nothing is submitted and
nothing is remembered, so calling it costs nothing and changes nothing.

- `intents` — what the algorithm wants, each `{symbol, kind, value}`. `kind` is `weight`
  (fraction of equity), `notional` (dollars), `shares`, or `option`.
- `mode` — `target` means the intent list is the **complete** portfolio, so a held symbol
  absent from it is sold to zero. `incremental` means only the listed symbols are touched.
- `signals` — per symbol: `score`, `reason`, `signal`, `score_components`,
  `realized_volatility`.
- `latest_prices` — the prices the plan was built from.
- `state` — the algorithm's private memory (an accrued budget, a cooldown). Opaque to you.
- `can_place_orders` / `reason` — whether acting on this plan will be accepted.

**Pass the whole payload back to `place_orders`.** It carries the prices used to size shares
and the state that gets committed; `place_orders` fetches no market data of its own.

### `list_bindings()`
Which algorithm is bound to which account, and what drives each one. Only bindings with
`can_place_orders: true` — switched on, `frequency: "mcp"` — will accept orders; the rest are
off or are the scheduler's to run. Name a `binding_id` when one algorithm is bound to more than
one account.

### `get_current_positions(binding_id="")`
Live holdings from the brokerage — `symbol`, `shares` — plus `equity`, `cash`, `buying_power`.
Use as the source of truth for what is actually held.

### `place_orders(algorithm_plan, binding_id="")`
Submits immediately.

- `algorithm_plan` is the `get_algorithm_plan` payload. Return it unchanged to submit the
  proposal as-is, **or edit its `intents`** — that list is the whole instruction, so under
  `mode: "target"` dropping a held symbol liquidates it.
- Everything else in the payload must come back untouched. `latest_prices` sizes the shares;
  `state` is what the algorithm commits once orders go out.
- You cannot introduce a symbol the plan never priced; it is rejected by name rather than
  silently skipped.
- Returns `diff` — per symbol `current_weight`, `final_weight`, `change`, `action`
  (`add`/`trim`/`hold`), largest change first. This is what to summarise.
- Returns `order_results`; each order carries `submitted` (with `order_id`), `rejected` (with
  `reason`), or `unfunded`.
- Returns `funding` — how the batch was paid for: buying power, the reserve held back, sale
  proceeds, and any cash-equivalent holdings liquidated to cover a shortfall.

Top-level `status`, and only one of these is worth reacting to:

- `submitted` — every leg went out at the size asked for.
- `submitted_reduced` — the batch was deliberately trimmed to what the account can pay for.
  **This is a success. Do not resubmit**: the legs in `funding.reduced` were shrunk on purpose.
- `unfunded` / `partial` / `rejected` — something did not reach the market. `unfunded` lists
  legs no amount of shrinking could fund, `rejected` lists legs the broker itself refused; each
  row carries the reason. A rejected leg does not abort the rest; a common cause is shares
  still held for an earlier unfilled order.
- `refused` (not your binding), `skipped` (kill switch), `error`.

## Workflow

1. `list_bindings()` — find a binding you are allowed to drive.
2. `get_algorithm_plan()` — read the intents and the `score`/`reason` behind each symbol.
3. `get_current_positions()` — confirm what is actually held.
4. Validate with your own research tools. Check each symbol the algorithm wants to add or drop
   against recent news, earnings, and pending corporate actions. This step is yours.
5. Decide. If the plan stands, submit it unchanged. If research rejects a name, remove or
   resize its intent — remembering what `mode` says about omission.
6. `place_orders(algorithm_plan=<the payload, edited or not>)`.
7. Report using `diff` and `order_results`, tying each change back to the algorithm's `reason`
   and to what your research found.

Do not invent symbols the algorithm did not surface, and do not treat a backtest as approval.
The algorithm has already applied its own stickiness and risk guards inside `plan`, so what you
receive is the final intent rather than a draft it will revise.

## Rally Rotation

Composite score from z-scored nano (10 bars), micro (78), meso (60d), macro (180d) returns +
sentiment + pullback bonus. Risk-on symbols need `macro_trend_ok` and a minimum micro return.
The top `max_positions` are selected; weights are proportional to score, capped at
`max_single_position_weight`.

Entry and exit are deliberately asymmetric: a name below the score floor is not bought, but an
eligible name already held is kept. A challenger must beat an incumbent by the configured
replacement margin to displace it — that is churn control, not indecision.

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
