---
name: daily-summary
description: After-hours brief on every Walbot account — worth, today's move, what is resting overnight, and the news behind it. Use when a cron prompt asks for the daily summary.
---

# Daily Brief

After the close. **The unit is the account, not the algorithm.**

## 1. Gather

`list_accounts()`, then `get_account_positions(id)` and `get_account_orders(id)` per account.
`list_algorithms()` once.

Orders come back in every state — filled, replaced, rejected, resting — in one list.

If an account errors, name it and carry on.

## 2. Read it

- P/L with no orders behind it is market drift. Say so.
- A rejection is the most important line. Quote the broker's reason verbatim and say what it
  means for the next session.
- Resting orders are overnight exposure. A stop protects, an entry intends. Say which.
- Mention deployments only when they change the outlook: nothing armed, or armed on an empty
  account. An armed deployment that traded nothing is a normal day.

## 3. News

`web_search`, two or three searches, **only for symbols actually held or resting.** A symbol the
portfolio has no exposure to is not news, it is reading.

Search for the *cause*, not the calendar. The question is "why did this move, or what is about to
move it" — an earnings print, a guidance cut, a Fed decision, a sector downgrade, an expiry that
concentrates open interest. "Markets were mixed" is not an answer; omit the line instead.

Attribute every claim to its source in the line itself, and only report what a source actually
says. A plausible-sounding reason you inferred is worse than no line, because it reads exactly
like one you found. If a move has no visible explanation, that is the finding: say it is
unexplained.

Skip the section entirely on a quiet day with nothing held. Never pad it.

## 4. Report

```
📊 Daily Brief · Wed 10 Sep

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

NEWS
• USO — crude fell 3.1% after OPEC+ signalled a quota increase (Reuters).
  The 142C we sold into it was the right side of that.
• IWM — CPI lands 08:30 ET tomorrow; Schwab Main holds it into the print.
• SPYM — no explanation found for the −0.4% drift. Unexplained.

⚠️ NOTHING ARMED
• All three deployed algorithms are switched off. Nothing will trade.
```

- `MONEY` first: total, then the per-account split.
- Group by account. Use labels, not ids.
- Empty sections get `• Nothing.` Never drop one — except `NEWS`, which is omitted whole when
  there is nothing held to search for.
- `⚠️ REJECTED` only when something was, and above `RESTING`.
- **`⚠️ NOTHING ARMED` whenever no deployed algorithm is switched on**, as the last block so it
  is the line the reader ends on. A brief that reads normally while the bot is switched off is
  the worst message you can send.
- Under 20 lines.
