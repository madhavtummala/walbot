---
name: run-algorithm
description: Run one Walbot algorithm over MCP, check its plan against outside information, submit or decline it, and report. Use when a cron prompt names an algorithm to run.
---

# Run an algorithm

The prompt names the algorithm. Same steps for all of them.

## 1. Check it is yours

`list_bindings()`. Find this algorithm's binding.

- `can_place_orders: false` → report the `reason`, stop.
- Two bindings, no `binding_id` in the prompt → report the ambiguity, stop. Never guess the
  account.

## 2. Get the plan

`get_algorithm_plan(algorithm, binding_id)`. Places nothing. Returns the proposal and a
`plan_token`.

Read the tool's description for which field holds the proposal. It differs per algorithm, and
reading the wrong one looks like an empty plan.

Nothing proposed → say so and stop. Quote the blocking gate from `signals[].checks`.

## 3. Check what the bot cannot see

Search only the symbols in the plan:

- news in the last 48h — earnings, guidance, SEC action, halt, index change
- a scheduled event inside the holding horizon
- whether today's move has a cause

Mark each symbol **confirms**, **silent**, or **contradicts**.

Silence is not a veto. Most trades have no news.

## 4. Submit or decline

- Nothing contradicts → `place_orders(plan_token)`.
- One symbol contradicts → `place_orders(plan_token, [{"op":"skip","symbol":"IWM"}])`.
- Decline everything → call nothing. Report what you declined.

Options Flip refuses `edits`. Submit it whole or decline it whole.

A contradiction does not force a decline. Ask whether the strategy already prices it in: a
momentum strategy buying a name that ran hard is working as designed.

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
• IWM exits the day before CPI; small caps often snap back
  → submitted anyway — rank is the strategy's call, not a fact it missed
```

- Line 1: 🟢 submitted · 🟡 nothing to do · 🔴 declined or error. Then algorithm, account, time.
- Line 2: outcome and the money.
- Never drop `WHERE IT DOESN'T`. Nothing found → `• Nothing found against it`.
- Quantities in shares or contracts **and** dollars.
- A resting order is an order, not a position: `bid $16.84 for the GLD 385C`.
- Under 15 lines.
