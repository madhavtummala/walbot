---
name: run-algorithm
description: Run one Walbot algorithm over MCP, validate its plan against outside information, submit or decline it, and report the result to Telegram. Use when a cron prompt names an algorithm to run.
---

# Run an algorithm

The prompt names the algorithm. Everything below is the same whichever one it is.

## 1. Check it is yours to run

`list_bindings()`. Find the binding for this algorithm.

- `can_place_orders: false` → report the `reason` and stop. Do not call `place_orders`.
- More than one binding for the algorithm → the prompt must name a `binding_id`. If it does
  not, report the ambiguity and stop rather than guessing which account to trade.

A binding is yours only when its schedule is empty — an empty cron means no clock drives it, so
you do. A binding with a cron belongs to Walbot's own scheduler.

## 2. Get the plan

`get_algorithm_plan(algorithm, binding_id)`. Read-only: it places nothing and remembers
nothing, so a run that ends here has changed no state.

**Read the right field.** Allocation strategies (Bursty DCA, Rally Rotation) propose a
portfolio in `intents`. Options Flip proposes resting orders in `desired_orders` and its
`intents` is always empty. Reading the wrong one looks like an empty plan rather than an error.

If the plan proposes nothing, say so and stop. That is a normal outcome, not a failure — and
`signals[].checks` says which gate refused, which is what makes the message worth sending.

## 3. Validate against what the bot cannot see

Only now, and only for the symbols the plan actually touches. Search for each:

- **Company or fund news in the last 48 hours** — earnings, guidance, an SEC action, a halt,
  an index change, a fund closure.
- **A scheduled event inside the holding horizon** — earnings date, CPI/FOMC, an expiry.
- **Whether today's move has a cause.** This is the one that matters most. The bot sees a
  price series; it cannot tell a trend from a one-day reaction. A symbol up 6% on a buyout
  rumour is not a symbol in an uptrend.

Then decide, per symbol, one of three things:

| | |
|---|---|
| **Confirms** | the news supports what the algorithm wants |
| **Silent** | nothing found — the common case, and not a reason to block |
| **Contradicts** | a fact the bot could not see makes this trade unwise |

**Silence is not a veto.** Most trades will have no news at all, and refusing them all would
reduce the strategy to "trades only what is in the headlines".

## 4. Submit, or decline

- No contradictions → `place_orders(plan, binding_id)` with the payload **passed back whole**.
  It carries the prices the sizing used and the state that gets committed. Read that tool's own
  description before the first call of a session — for Options Flip in particular, a key
  dropped from `desired_orders` cancels that order at the broker rather than leaving it alone.
- Contradictions → you may still submit. Ask whether the fact is one the strategy already
  prices in. A momentum strategy buying a name that has run hard is not a contradiction; that
  *is* the strategy. A momentum strategy buying a name that ran hard on a rumour that has since
  been denied is.
- If you decline, say so explicitly and name the fact. Do not silently skip.

You cannot decline part of a plan — `place_orders` takes it whole or not at all. If one leg is
contradicted and the rest is fine, submit and flag the leg, or decline the lot and say which
leg cost it. Choose deliberately and say which you chose.

## 5. Report

```
🟢 Rally Rotation · Alpaca Paper · 14:32
Submitted · 3 orders · $2,140

WHAT IT WANTS
• BUY QQQM 12 sh ($4,180) — rank 1, score +1.8σ
• SELL IWM 20 sh ($1,940) — slipped to rank 8, past the exit rank

WHY IT HOLDS UP
• Semis leading on NVDA's raised guidance (Reuters, today)
• Nothing scheduled inside the 5-session horizon

WHERE IT DOESN'T
• IWM exits the day before CPI; small caps often snap back after
  → submitted anyway — rank is the strategy's call, not a fact it missed

NOT SUBMITTED
• (nothing)
```

Rules for the message:

- First line: status emoji, algorithm, account, time. 🟢 submitted · 🟡 nothing to do ·
  🔴 declined or error.
- Second line: one-line outcome with the money involved.
- **`WHERE IT DOESN'T` is never omitted.** If you found nothing against the plan, write
  `• Nothing found against it` — an absent section reads as "not checked".
- Quantities in shares or contracts *and* dollars. A share count alone means nothing on a
  phone.
- For Options Flip, describe the order, not a position: `bid $16.84 for the GLD 385C, expires
  at the close` — it is a resting order that may never fill.
- Keep it under ~15 lines. If a section would run longer, cut the weakest bullet.
