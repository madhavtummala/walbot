---
name: daily-summary
description: After-hours wrap of every Walbot account — worth, today's move, what is resting overnight, what tomorrow brings. Use when a cron prompt asks for the daily summary.
---

# Daily wrap

After the close. **The unit is the account, not the algorithm.**

## 1. Gather

`list_accounts()`, then `get_account(id)` and `get_account_orders(id)` per account.
`list_bindings()` once.

Orders come back in every state — filled, replaced, rejected, resting — in one list.

If an account errors, name it and carry on.

## 2. Read it

- P/L with no orders behind it is market drift. Say so.
- A rejection is the most important line. Quote the broker's reason verbatim and say what it
  means for tomorrow.
- Resting orders are tomorrow's exposure. A stop protects, an entry intends. Say which.
- Mention bindings only when they change the outlook: nothing armed, or armed on an empty
  account. An armed binding that traded nothing is a normal day.

## 3. Outside information

One or two searches, only for symbols held or resting: what is scheduled tomorrow (earnings,
CPI, FOMC, expiry), or the cause of a big move. Skip on a quiet day.

## 4. Report

```
📊 Daily wrap · Wed 10 Sep

MONEY
• Total $204,514 — up $257 (+0.13%) today
• Alpaca Paper $10,525 (+$560) · Schwab Main $82,300 (−$367)
• Schwab Small $9,812 (−$55) · Local Paper $101,676 (flat)

WHAT MOVED
• Alpaca Paper — sold the USO 142C at $14.10, +59% on the $8.85 fill
• Local Paper — bought 2 SPYM at $89.74 ($179)

⚠️ REJECTED
• Alpaca Paper — 2 sells on USO260916C00142000
  "account not eligible to trade uncovered option contracts"
  The position had already closed, so both legs were naked. Nothing at
  risk, but the account may lack the options permission level.

RESTING OVERNIGHT
• Nothing. No stops, no entries.

TOMORROW
• CPI 08:30 ET. Schwab Main holds IWM into it.
• All three bindings are switched off — nothing will trade.
```

- `MONEY` first: total, then the per-account split.
- Group by account. Use labels, not ids.
- Empty sections get `• Nothing.` Never drop one.
- `⚠️ REJECTED` only when something was, and above `RESTING`.
- **If nothing is armed, say so in `TOMORROW`.** A wrap that reads normally while the bot is
  switched off is the worst message you can send.
- Under 20 lines.
