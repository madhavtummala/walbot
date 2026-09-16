---
name: daily-summary
description: After-hours Telegram brief on every Walbot account — what they are worth, what traded, the news behind it, what is coming, and what of ours is exposed. Use when a cron prompt asks for the daily brief.
---

# Daily Brief

After the close. **The unit is the account, not the algorithm.**

## 1. Gather

`list_accounts()` and `list_algorithms()`. That is all you call yourself.

`list_accounts` carries each account's headline: `equity`, `day_pl` (and
percent), `total_pl` (open P/L), `realized_pl`, `positions`, `orders_today`.
**STATUS is written from this alone.** Every figure is current — it forces the
realized figure rather than serving a stale one — so it takes seconds to come
back. That is time, not context.

**Then decide who portfolio-bot needs to visit.** An account with
`orders_today: 0` and a `day_pl` of zero did nothing: no fill to report, nothing
resting, nothing rejected. Pass only the accounts with a non-zero `day_pl` **or**
a non-zero `orders_today`. On a typical day that is two of five.

- `error` on a row means the broker could not be reached — *not* a quiet
  account. Name it in CONCERNS and do not send portfolio-bot after it.
- Every account quiet → skip step 2 entirely; STATUS still gets written and
  INSIGHTS becomes `• No orders today.`
- `list_algorithms` is the source for the `nothing armed` line in CONCERNS —
  what is switched on.

## 2. Portfolio

Skip when step 1 left nobody to visit.

`sessions_spawn`: `agentId: "portfolio-bot"`, `context: "isolated"`,
`label: "portfolio-snapshot"`, `runTimeoutSeconds: 600`, and a prompt naming
only the account ids it should read. Then `sessions_yield`.

Fills and drift become INSIGHTS; resting orders and rejections become CONCERNS.

## 3. News and headwinds

`sessions_spawn`: `agentId: "web-bot"`, `context: "isolated"`,
`label: "news-research"`, `runTimeoutSeconds: 600`, and a prompt naming only the
symbols the portfolio actually holds or has resting. A symbol the book has no
exposure to is not news, it is reading. Then `sessions_yield`.

Its **cause** bullets become NEWS; its **ahead** bullets become HEADWINDS.

If web-bot errors or returns nothing usable, spawn it once more. Still failing,
omit NEWS and HEADWINDS and write `NEWS unavailable (web-bot failed).`

## 4. Report

Plain text for Telegram. No markdown, no tables, no code fences — asterisks and
underscores render as formatting or as literal junk depending on the client, so
use neither. Emoji and `•` are safe.

Five blocks, always in this order: **STATUS, INSIGHTS, NEWS, HEADWINDS,
CONCERNS.** The first four report; the last one is the only place that asks for
anything. HEADWINDS is the world — what is coming. CONCERNS is our book — what
of ours is exposed to it.

```
📊 Daily Brief · Wed 10 Sep

STATUS
• Total $204,514 — up $257 (+0.13%) today
• Alpaca Paper $10,525 (+$560) · Schwab Main $82,300 (−$367)
• Schwab Small $9,812 (−$55) · Local Paper $101,676 (flat)
• 21 positions open · realized $4,806 YTD

INSIGHTS
• Alpaca Paper — sold the USO 142C at $14.10, +59% on the $8.85 fill
• Schwab Main — bought 2 SPYM at $89.74 ($179)
• Schwab Roth — −$55 on no orders. Drift, not activity.

NEWS
• USO $155.49 — crude fell 3.1% after OPEC+ signalled a quota increase
  (Reuters, 4h). The 142C we sold into it was the right side of that.
• SPYM $89.74 — no cause found for the −0.4% drift. Unexplained.

HEADWINDS
• CPI lands 08:30 ET tomorrow; Schwab Main holds IWM into the print.
• The OPEC+ quota is signalled, not ratified — the crude move could reverse.

⚠️ CONCERNS
• Schwab Main carries IWM into tomorrow's CPI print with no stop under it.
• Alpaca Paper — 2 sells rejected on USO260916C00142000. Alpaca reports no
  reason. Both legs were naked; the account may lack options permission.
• Local Paper — stop resting at $86.10 on SPYM, 4% below spot. Protected.
• All three deployed algorithms are switched off. Nothing will trade.
```

**STATUS** — what the accounts are worth now. Total first, then the split, then
one line of totals. Labels, never ids. An unreachable account is named as
unreachable, not folded into the total as if it were zero.

**INSIGHTS** — what today's orders did. Fills with the price and the outcome. An
account that moved on no orders is drift and says so; that is an insight, not
filler. Nothing traded anywhere → `• No orders today.`

**NEWS** — the reason behind an INSIGHTS line. Price from web-bot's `get_price`,
cause from its search, source named in the line. Only symbols held or resting.
No cause found → `Unexplained.` Never a line that explains nothing.

**HEADWINDS** — what could change it, out in the world. Scheduled events,
unresolved decisions, expiries, a print landing tomorrow. Facts about the
market, not about us. Nothing pending → `• Nothing scheduled.`

**⚠️ CONCERNS** — what of ours is exposed, and the only block that asks the
reader for anything. Last, so it is where the message ends. Four things, in this
order:

1. **A resting order a headwind touches.** This is the block's reason for
   existing and the one line that needs you rather than a tool: cross every
   resting order against every headwind and say where they meet. A stop below a
   gap tomorrow's print could open, an entry bidding into an earnings date, an
   option expiring the week of a decision. Say whether it protects or intends.
   **A position carried into a scheduled event with no stop under it is a concern
   even though nothing is resting** — the absence is the exposure.
2. **Rejections.** Quote the broker's reason. Where the broker gave none —
   Alpaca never does — say that plainly rather than leaving it blank.
3. **An account the broker could not reach.** Its figures are missing, not zero,
   and a reader who is not told will read the gap as calm.
4. **`Nothing armed`, if so** — always the final bullet, so a switched-off bot is
   the last thing read. Judge it on `enabled`, **not** `can_place_orders`:
   `can_place_orders: false` is the normal state of every scheduled algorithm and
   means the scheduler owns it, not that it is off. Keying on the wrong field
   fires this every single day, and a warning that always fires is one nobody
   reads.

Resting orders no headwind touches still get a line each — short, with the level
— so the reader knows what is out there. Nothing resting, nothing rejected,
everything armed and reachable → `• Nothing resting, nothing rejected.`

Rules:

- Under 20 lines. A quiet day is five short blocks, not padding.
- Labels, never ids. Dollars before percent.
- No block is ever dropped; an empty one gets its `• Nothing` line.
