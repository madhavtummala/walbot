---
name: daily-summary
description: After-hours wrap of every Walbot account — what each is worth, what moved today, what is resting overnight, and what tomorrow brings. Use when a cron prompt asks for the daily summary.
---

# Daily wrap

Runs after the close. **The unit is the account, not the algorithm.** The user owns accounts;
algorithms are how those accounts get traded, and they matter here only where they explain a
number or threaten tomorrow's.

## 1. Gather

| | |
|---|---|
| `get_accounts()` | every account: equity, cash, day P/L, total P/L, holdings |
| `get_account_orders(account_id)` | that account's recent orders — resting and filled alike |
| `list_bindings()` | which algorithms are armed on which account, and on what schedule |

One orders call per account. It returns the broker's own recent history, so filled, replaced,
rejected and still-resting orders all arrive together — there is no separate "working orders"
question to ask.

If a tool errors for one account, say which account and carry on with the rest. A wrap missing
one account is worth sending; a silent evening is not.

## 2. Read the day before writing it

- **Reconcile each account's P/L against its orders.** An account down $400 that placed no
  orders drifted with the market, and saying so is more useful than repeating the number.
- **A rejected order is the most important line in the file.** It means something tried to act
  and could not. Lead with it, quote the broker's reason verbatim, and say what it implies for
  tomorrow.
- **Resting orders are tomorrow's exposure.** A resting stop is protection; a resting entry is
  an intention. Say which each one is and what it would cost or make if it fills.
- **Bindings only appear when they change the outlook** — armed but pointed at an empty
  account, or nothing armed at all. An armed binding that traded nothing is a normal day.

## 3. Add what the numbers cannot say

One or two searches, no more, and only for symbols actually held or actually resting:

- Anything scheduled tomorrow that touches a holding — earnings, CPI, FOMC, an expiry.
- Whether today's move in a held name had a cause worth knowing.

Skip this entirely on a flat, quiet day. A wrap padded with market commentary nobody asked for
is worse than a short one.

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
• Alpaca Paper — 2 sell orders on USO260916C00142000
  "account not eligible to trade uncovered option contracts"
  The position had already closed, so both legs were naked. Nothing is
  at risk, but the account may lack the options permission level.

RESTING OVERNIGHT
• Nothing. No stops, no entries.

TOMORROW
• CPI 08:30 ET. Schwab Main holds IWM into it.
• All three bindings are currently switched off — nothing will trade.
```

Rules for the message:

- `MONEY` first, always, with the portfolio total before the per-account split.
- Group every section by account, and name accounts by their label, not their id.
- Sections with nothing to say are written as one line (`• Nothing.`), never dropped —
  "resting overnight: nothing" is information, an absent section is not.
- `⚠️ REJECTED` appears only when something was rejected, and sits above `RESTING` when it does.
- **If nothing is armed, say so in `TOMORROW`.** A wrap that reads normally while the whole bot
  is switched off is the most misleading message you can send.
- Under ~20 lines.
