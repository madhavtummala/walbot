---
name: run-algorithm
description: Run one Walbot algorithm over MCP, check its plan against outside information, submit or decline it, and report. Use when a cron prompt names an algorithm to run.
---

# Run an algorithm

The prompt names the algorithm. The steps are the same whichever it is.

## 1. Check it is yours

`list_bindings()`. Find this algorithm's binding.

- `can_place_orders: false` → report the `reason` and stop.
- Two bindings and no `binding_id` in the prompt → report the ambiguity and stop. Do not guess
  which account to trade.

A binding is yours only when its `cron` is empty. A cron means the scheduler owns it.

## 2. Get the plan

`get_algorithm_plan(algorithm, binding_id)` — read-only, places nothing, commits nothing.

Read its description for which field holds the proposal; it differs by algorithm and reading
the wrong one looks like an empty plan rather than an error.

Nothing proposed → say so and stop. That is a normal outcome, and `signals[].checks` names the
gate that refused, which is what makes the message worth sending.

## 3. Check against what the bot cannot see

Only for the symbols the plan touches. Search each for:

- news in the last 48h — earnings, guidance, an SEC action, a halt, an index change
- a scheduled event inside the holding horizon
- **whether today's move has a cause** — the bot sees a price series and cannot tell a trend
  from a one-day reaction

Then per symbol: **confirms**, **silent**, or **contradicts**.

**Silence is not a veto.** Most trades have no news, and refusing those reduces the strategy to
"trades only what is in the headlines".

## 4. Submit or decline

- No contradictions → `place_orders(plan, binding_id)`, payload **passed back whole**. Read
  that tool's description first: for order-book algorithms a dropped key cancels a resting
  order rather than leaving it alone.
- Contradictions → you may still submit. Ask whether the strategy already prices the fact in. A
  momentum strategy buying a name that ran hard is the strategy working; buying one that ran on
  a rumour since denied is not.
- Declining is all-or-nothing — `place_orders` takes the plan whole. Submit and flag the leg,
  or decline the lot and name the leg that cost it. Say which you chose.

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

- First line: 🟢 submitted · 🟡 nothing to do · 🔴 declined or error, then algorithm, account,
  time. Second line: the outcome with the money involved.
- **`WHERE IT DOESN'T` is never omitted.** Nothing found → `• Nothing found against it`. An
  absent section reads as "not checked".
- Quantities in shares or contracts *and* dollars.
- For a resting order, describe the order, not a position: `bid $16.84 for the GLD 385C` — it
  may never fill.
- Under ~15 lines.
