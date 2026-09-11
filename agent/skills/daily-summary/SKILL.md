---
name: daily-summary
description: After-hours wrap across every Walbot account and algorithm — what it is worth, what moved today, what is resting overnight, and what tomorrow brings. Use when a cron prompt asks for the daily summary.
---

# Daily wrap

Runs after the close. Covers **every account**, not one binding.

## 1. Gather

| | |
|---|---|
| `get_portfolio_summary()` | every account: equity, cash, day P/L, total P/L, positions |
| `get_recent_orders(since_hours=24)` | what was submitted, filled, replaced or rejected today |
| `get_working_orders()` | what is still resting, per account |
| `list_bindings()` | which algorithms are armed, and on what schedule |

If a tool is missing, say which, and write the summary from what you do have. A partial wrap is
worth sending; a silent evening is not.

## 2. Read the day before writing it

- **Reconcile the P/L against the moves.** If an account is down and placed no orders, the
  move is market drift, and saying so is more useful than listing the number again.
- **A rejected order is the most important line in the file.** It means the strategy tried to
  act and could not. Lead with it, quote the broker's reason verbatim, and say what it implies
  for tomorrow.
- **Resting orders are tomorrow's exposure.** A resting stop is protection; a resting entry is
  an intention. Say which each one is and what it would cost or make if it fills.
- **An armed binding that did nothing is normal** — most sessions refuse most trades. Only
  mention it if something is unusual: refused every session this week, or armed but pointed at
  an account with no money.

## 3. Add what the numbers cannot say

One or two searches, no more. Worth doing only for positions actually held or orders actually
resting:

- Anything scheduled tomorrow that touches a holding — earnings, CPI, FOMC, an expiry.
- Whether today's move in a held name had a cause worth knowing.

Skip this entirely on a flat, quiet day. A wrap that pads itself with market commentary
nobody asked for is worse than a short one.

## 4. Report

```
📊 Daily wrap · Wed 10 Sep

MONEY
• Total $204,514 — up $257 (+0.13%) today
• Alpaca Paper $10,525 (+$560) · Schwab Main $82,300 (−$367)
• Schwab Small $9,812 (−$55) · Local Paper $101,676 (flat)

WHAT MOVED
• Options Flip sold the USO 142C at $14.10 — +59% on the $8.85 fill
• Bursty DCA bought 2 SPYM at $89.74 ($179)

⚠️ REJECTED
• Options Flip, 2 sell orders on USO260916C00142000
  "account not eligible to trade uncovered option contracts"
  The position had already closed, so both legs were naked. Nothing is
  at risk, but the account may lack the options permission level.

RESTING OVERNIGHT
• Nothing. No stops, no entries.

TOMORROW
• CPI 08:30 ET. Rally Rotation reranks at 10:00 and may exit IWM.
• All three bindings are currently switched off — nothing will trade.
```

Rules for the message:

- `MONEY` first, always, in dollars with the percentage second.
- Sections with nothing to say are written as one line (`• Nothing.`), never dropped —
  "resting overnight: nothing" is information, an absent section is not.
- `⚠️ REJECTED` appears only when something was rejected, and goes above `RESTING` when it does.
- Name accounts by their label, not their id.
- **If nothing is armed, say so in `TOMORROW`.** A wrap that reads normally while the whole bot
  is switched off is the most misleading message you can send.
- Under ~20 lines.
