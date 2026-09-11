# Walbot MCP tools

Reference only — the workflows live in `skills/`. Read this before your first tool call.

## What this is

Walbot separates *deciding* from *acting*. `get_algorithm_plan` runs the algorithm and returns
what it wants to do; `place_orders` submits it. You sit in between: the scheduled runner chains
the two directly, and the only thing you add is validation. Research is done with your own tools
(web search, etc.) — the trading bot does not provide it.

**A plan takes one of two shapes, and reading the wrong one for a given algorithm looks like an
empty plan rather than an error.** Bursty DCA and Rally Rotation are allocation strategies: they
propose a *portfolio* in `intents`. Options Flip is an order-book strategy: it proposes resting
orders in `desired_orders` instead, and `intents` is always empty for it. Check which algorithm
you are looking at before deciding which field means "the plan."

## MCP Tools (walbot, from `src/mcp_server.py`)

### `get_algorithm_plan(algorithm="rally_rotation", binding_id="")`
Runs the algorithm against market data and the bound account's book. Nothing is submitted and
nothing is remembered, so calling it costs nothing and changes nothing.

- `intents` — allocation strategies only. What the algorithm wants, each `{symbol, kind,
  value}`. `kind` is `weight` (fraction of equity), `notional` (dollars), or `shares`. Empty
  for Options Flip.
- `mode` — `target` means `intents` is the **complete** portfolio, so a held symbol absent
  from it is sold to zero. `incremental` means only the listed symbols are touched.
- `desired_orders` — Options Flip only. Empty for allocation strategies. Each entry is one
  order that should be resting at the broker right now: `{key, request, replace_tolerance}`,
  where `key` names the *role* (`SYMBOL:entry` / `SYMBOL:target` / `SYMBOL:stop`) and `request`
  carries `symbol` (the option contract's OSI string, not the underlying), `action`,
  `order_type`, `limit_price`/`stop_price`, `time_in_force`.
- `signals` — per symbol, passed through whole rather than trimmed to a fixed field list,
  because the two plan shapes report completely different things:
  - Allocation strategies: `score`, `reason`, `signal`, `score_components`,
    `realized_volatility`.
  - Options Flip: `state` (`flat`/`bidding`/`held`), `headline` (one-line summary),
    `checks` (every gate this run evaluated — `{label, ok, value, limit, blocking}`),
    `estimate` (the band prediction and greeks-priced profit), `contract` (the OSI symbol
    chosen, if any).
- `latest_prices` — the prices the plan was built from.
- `state` — the algorithm's private memory (an accrued budget, a held position's fill price
  and hold count). Opaque to you.
- `can_place_orders` / `reason` — whether acting on this plan will be accepted.

**Pass the whole payload back to `place_orders`.** It carries the prices used to size shares
and the state that gets committed; `place_orders` fetches no market data of its own.

### `list_bindings()`
Which algorithm is bound to which account, and what drives each one. Only bindings with
`can_place_orders: true` will accept orders from you: switched on, and with an **empty
schedule**. An empty cron means no clock drives the binding, so an agent does; a binding
carrying a cron expression belongs to Walbot's own scheduler and will refuse your orders. Name a `binding_id` when one algorithm is bound to more than
one account.

### `get_current_positions(binding_id="")`
Live holdings from the brokerage — `symbol`, `shares` — plus `equity`, `cash`, `buying_power`.
Use as the source of truth for what is actually held. For an Options Flip binding, a held
contract appears here too, keyed by its OSI symbol (e.g. `GLD   260918C00380000`) with
`shares` as the contract count -- the field name is literal ("shares"), not filtered by asset
type. Cross-check it against `signals[underlying].state == "held"` from `get_algorithm_plan`,
which names the *underlying* rather than the OSI string.

### `place_orders(algorithm_plan, binding_id="")`
Submits immediately. What counts as "editing the plan" and what the response contains both
depend on the plan's shape.

**Allocation strategies (Bursty DCA, Rally Rotation) — edit `intents`.**
- `intents` is the whole instruction under `mode: "target"` — dropping a held symbol liquidates
  it. Everything else in the payload must come back untouched: `latest_prices` sizes the
  shares, `state` is what the algorithm commits.
- You cannot introduce a symbol the plan never priced; it is rejected by name rather than
  silently skipped.
- Returns `diff` — per symbol `current_weight`, `final_weight`, `change`, `action`
  (`add`/`trim`/`hold`), largest change first. This is what to summarise.
- Returns `order_results` (each order carries `submitted`, `rejected`, or `unfunded`) and
  `funding` (how the batch was paid for: buying power, reserve, sale proceeds, any liquidated
  cash-equivalents).
- `status`: `submitted` (every leg went out as asked), `submitted_reduced` (deliberately
  trimmed to what the account can pay for — **a success, do not resubmit**), `unfunded` /
  `partial` / `rejected` (something did not reach the market — see each leg's own reason).

**Options Flip — edit `desired_orders`, never `intents` (it has none).**
- Each key is independent: `SYMBOL:entry` while flat, or `SYMBOL:target` and `SYMBOL:stop`
  while held. **A key missing from the list you send is not "leave it alone" — it is
  cancelled.** The reconciler cancels any order at the broker whose key is not in what you
  submit. Sending back the payload with one key edited and another silently dropped cancels
  the dropped one, stop included, on a position that may still be open.
- If you are not deliberately closing a position, return every `desired_orders` entry exactly
  as given. There is little reason to edit this strategy's orders at all: the levels, ratchets
  and stop are already priced from the gates in `signals[symbol].checks` — second-guessing one
  number without re-deriving the rest is more likely to break the pricing than improve it.
- Returns no `diff`/`funding`. Read `order_results` (each entry `submitted` / `replaced` /
  `cancelled` / `unchanged` / `rejected`, with the order id and reason where relevant) and
  `working_orders` (what is now actually resting at the broker, for confirming nothing you
  did not intend to touch got cancelled).
- `status`: `ok` once reconciliation ran (`skipped` if the kill switch caught it first).

Refusal statuses common to both shapes: `refused` (not your binding), `skipped` (kill switch),
`error`.

## Workflow

**Allocation strategies (Bursty DCA, Rally Rotation):**

1. `list_bindings()` — find a binding you are allowed to drive.
2. `get_algorithm_plan()` — read `intents` and the `score`/`reason` behind each symbol.
3. `get_current_positions()` — confirm what is actually held.
4. Validate with your own research tools. Check each symbol the algorithm wants to add or drop
   against recent news, earnings, and pending corporate actions. This step is yours.
5. Decide. If the plan stands, submit it unchanged. If research rejects a name, remove or
   resize its intent — remembering what `mode` says about omission.
6. `place_orders(algorithm_plan=<the payload, edited or not>)`.
7. Report using `diff` and `order_results`, tying each change back to the algorithm's `reason`
   and to what your research found.

**Options Flip:**

1. `list_bindings()` — find a binding you are allowed to drive.
2. `get_algorithm_plan(algorithm="options_flip")` — read `signals[symbol].state` first (is it
   flat, bidding, or already held), then `.headline` for the one-line summary, then `.checks`
   for which gates passed or blocked and why. Read `desired_orders` for the actual resting
   order(s) this would place.
3. `get_current_positions()` — confirm the account's actual holdings agree with `state`.
4. Validate the *underlying* symbol with your own research tools if you want to override the
   gates for a name it is bidding on or holding. There is no research step for the option
   contract itself — its selection (strike, delta, expiry) is mechanical, not a judgment call.
5. Decide. This strategy's own gates (trend, VWAP, gap, reachability, profit floor) already
   encode most of what research would check; the more useful override is refusing a *new*
   entry on a name with fresh bad news the gates cannot see, not re-pricing an order.
6. `place_orders(algorithm_plan=<the payload, unedited unless deliberately closing a
   position>)`.
7. Report using `order_results` and `working_orders`, tying each order back to the `checks`
   that justified it (e.g. "held VWAP, entry reachable at 68%, priced the target at $X per the
   band estimate").

Do not invent symbols the algorithm did not surface, and do not treat a backtest as approval.
The algorithm has already applied its own stickiness and risk guards inside `plan`, so what you
receive is the final intent rather than a draft it will revise.

## The algorithms

### Bursty DCA
Dollar-cost-averages a fixed monthly budget per symbol, but sizes each purchase by how far
below its moving average the price sits and how far ahead of or behind its own accrual plan
that symbol already is — cheap names buy more, rich names buy less or sell, and a symbol that
overspent resists spending again until it catches up. `intents` are `notional` amounts. Read
`signals[symbol]` for `valuation`, `buying`, `size`, and how much of the monthly budget is
already spent.

### Rally Rotation
Ranks a fixed universe by a cross-sectional momentum score (blended across four horizons) and
holds only the names that also clear their own absolute floors (trend, minimum return,
volatility ceiling) — a name can score well and still not be held if it fails those floors.
Entry and exit are asymmetric: a challenger must beat a held name by a replacement margin to
displace it, so the book does not churn on noise. Too few qualifying names parks the unused
book in T-bills. `intents` are `weight` fractions under `mode: "target"`. Read
`signals[symbol].reason` — `Top Rank` (selected), or why not: `Macro negative`, `Score too
low`, `Micro too low`, `No rank slot`.

### Options Flip
A bull-regime, pullback-entry, rebound-exit long-call strategy, one contract per symbol. Three
gates decide whether a symbol trades today: the bull thesis must be intact (trend, VWAP, no
gap down), the pullback entry and rebound target must both be statistically reachable given
comparable past sessions, and the trade must clear a minimum modelled profit through the full
greeks. On a pass it rests a limit buy that walks toward the market as the session runs out and
is abandoned unfilled at the close; on a fill it rests an independent profit-target sell and a
stop, both `gtc`, ratcheting the target down as the hold session count runs toward
`max_hold_sessions`. No `intents` — see `desired_orders` and `signals[symbol].checks` above.

## Fractional shares

Order sizing follows the brokerage's `supports_fractional_shares` capability — true for
Alpaca and the local paper brokerage, false for brokerages that require whole shares. Applies
to Bursty DCA and Rally Rotation; Options Flip trades whole option contracts always.

- Fractional quantities are kept to **2 decimal places**.
- Sizing always truncates rather than rounds up, so a filled position never exceeds its
  target dollar amount. A target weight may therefore be slightly under-filled — more so on
  whole-share brokerages and high-priced symbols.
- Short targets are always sized in whole shares, since fractional quantities cannot be
  shorted.

---

## Summary tools

Read-only, and account-wide rather than per binding. Used by `skills/daily-summary`.

### `get_portfolio_summary()`
Every configured account in one call: `label`, `broker`, `equity`, `cash`, `day_pl`,
`day_pl_percent`, `total_pl`, `dividend_pl`, and `positions[]` with `symbol`, `qty`,
`avg_entry_price`, `market_value`, `unrealized_pl`.

`day_pl` is `null` where the broker cannot say what the session opened at. Null means
*unknown*, never zero — do not report it as flat.

### `get_recent_orders(since_hours=24, account_id="", strategy="")`
What was submitted, filled, replaced or rejected, merged from two sources that answer different
questions: the bot's own journal, which knows **which algorithm** placed an order, and the
broker's history, which knows what actually **happened** to it. Rows carry `symbol`, `side`,
`quantity`, `order_type`, `limit_price`/`stop_price`, `filled_avg_price`, `status`, `reason`,
`strategy`, `submitted_at`.

A `rejected` row always carries the broker's own `reason`. Quote it verbatim — it is the
difference between "the strategy did nothing" and "the strategy tried and was refused".

### `get_working_orders(account_id="")`
Orders resting at the broker right now, which is tomorrow's exposure: `symbol`, `side`,
`quantity`, `order_type`, `limit_price`/`stop_price`, `status`, `account_id`. An option order's
`symbol` is the contract's OSI string, not the underlying.
