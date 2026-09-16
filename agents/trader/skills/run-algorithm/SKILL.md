---
name: run-algorithm
description: Run one Walbot algorithm — check it is yours to drive, delegate the plan-and-submit decision to algo-bot, and report the outcome. Use when a cron prompt names an algorithm to run.
---

# Run an algorithm

The prompt names the algorithm. Same steps for all of them.

You place nothing. `place_orders` is algo-bot's and is not in your toolset —
whatever the plan looks like, the submission is not yours to make.

## 1. Check it is yours

`list_algorithms()`. Find this algorithm's row.

- `deployed: false` → it names no account and trades nothing. Report that, stop.
- `can_place_orders: false` → report the row's `reason` and stop. This is the
  normal state of a scheduled algorithm: the scheduler owns it, and an agent
  running it anyway would double-submit the day's orders.

Both are ordinary outcomes, not failures. Report them plainly and end.

## 2. Delegate

`sessions_spawn`: `agentId: "algo-bot"`, `context: "isolated"`,
`label: "<algorithm>-run"`, `runTimeoutSeconds: 900`.

The prompt names **the algorithm and its account id, and nothing else.** Its
standing instructions — how to read a plan, when a veto is warranted, the token
rules — live in its own `AGENTS.md`. Do not restate them; that is how two copies
drift apart.

Then `sessions_yield`.

If algo-bot errors, spawn it once more. Still failing, report the error. **Never
guess at what it submitted** — you cannot see the account, and a report that
invents an outcome is worse than one that admits the run is unknown.

## 3. Report

algo-bot returns the brief already shaped. Relay it — do not re-editorialise,
re-rank the reasoning, or soften a decline. It saw the plan and you did not.

Apply only the delivery rules, which are yours:

- Plain text for Telegram. No markdown, no tables, no code fences — asterisks
  and underscores render as formatting or as literal junk depending on the
  client. Emoji and `•` are safe.
- Account labels, never ids.
- Under 15 lines.

The shape it returns:

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

If `WHERE IT DOESN'T` is missing from what it returns, say so rather than
quietly dropping the block. A plan reported with no counter-case reads as
examined when it was not.
