---
name: walbot-mcp-agent
description: Use this skill when an external agent such as OpenClaw needs to operate Walbot through its MCP tools, review strategy signals, explain proposed portfolio changes from algorithm scores and target weights, request trade approval, and only submit orders through an explicitly available approved execution path.
---

# Walbot MCP Agent

Use this skill to orchestrate Walbot from an external agent runtime without bypassing the repo's deterministic strategy, approval, and execution boundaries.

## Core Rule

Do not invent trades. Use Walbot MCP payloads and project-generated order previews/results as the source of truth. Explain what the algorithm is doing; do not replace the algorithm with agent judgment.

Orders may be submitted only after a positive approval response and only through a tool or project entry point that explicitly submits approved orders. The current MCP server exposes approval and notification tools, but not a standalone `submit_approved_orders` tool. If no approved execution tool is available, stop after approval and report that submission requires the repo's live runner/API/brokerage execution path.

## MCP Tools

The project MCP server is `walbot` from `src/mcp_server.py`.

Available tools:

- `get_status()`: dashboard status, bot state, risk config, and redacted runtime config.
- `get_controls()`: dashboard controls such as selected strategy and enable flags.
- `get_universe()`: configured trading universe.
- `get_dca_plan()`: DCA plan and allocation preview.
- `generate_strategy_signals(strategy)`: current strategy signal rows for review.
- `run_backtest(strategy, period, refresh)`: backtest payload.
- `request_trade_approval(planned_orders, approval_id, timeout_seconds, poll_seconds)`: Telegram approve/deny flow.
- `format_portfolio_notification(order_results)`: format submitted order results.
- `send_text_notification(text, subject)`: send a notification through configured providers.

## Workflow

1. Read state:
   - Call `get_controls()`.
   - Call `get_status()`.
   - Determine the active strategy from controls. If strategy is `none`, do not propose algorithm orders.
   - Confirm trading is enabled, kill switch is not active, and account/runtime status does not block trading.

2. Generate explanation inputs:
   - Call `generate_strategy_signals(active_strategy)`.
   - Optionally call `run_backtest(active_strategy, period="", refresh=false)` for context, not as the primary trade trigger.
   - Use the returned signal rows, scores, sides, reasons, target weights, and current/target position data exposed by the project payloads.

3. Build or receive planned orders:
   - Prefer a deterministic project order preview if the runtime exposes one.
   - If only a planned-order list is provided by the host runtime, verify each order has at least `symbol`, `action`, `quantity`, and either `target_weight` or `target_shares`.
   - Do not hand-calculate brokerage orders from raw signals unless the host explicitly provides the same order-planning function used by the project.

4. Explain before approval:
   - For each planned order, state the portfolio transition: current shares, target shares, target weight, action, quantity, and estimated dollars when present.
   - Tie the change to the algorithm's signal fields: `signal`, `side`, `score`, `reason`, returns, moving averages, volatility, social/sentiment score, or risk guard metadata when present.
   - For negative target weights or `sell_short`, mention that this is a short exposure and that execution depends on account and Alpaca asset short-sale eligibility.
   - Make the explanation factual and bounded to provided data. Use "the signal payload indicates..." when inferring from MCP results.

5. Request approval:
   - Call `request_trade_approval(planned_orders, approval_id="", timeout_seconds=<runtime_default>, poll_seconds=<runtime_default>)`.
   - If approval returns `approved: false`, do not submit. Report denial/timeout and the approval ID.
   - If approval returns `requested: false`, do not submit. Report the reason.

6. Submit only through an approved execution path:
   - If the MCP host exposes `submit_approved_orders` or equivalent, call it only after approval and pass the exact approved planned orders plus approval ID.
   - If using this repo directly, the deterministic live path is `src.execution.live_runner.run_once`, which generates decisions and uses `sync_positions_to_targets`; do not duplicate it from the agent.
   - If no submit tool/path is available to the agent, stop and report: "Approval was received, but this MCP server does not expose order submission."

7. Notify and summarize:
   - If order results are available, call `format_portfolio_notification(order_results)`.
   - If the formatted message has changes, call `send_text_notification(text, subject="Portfolio changes submitted")`.
   - Summarize submitted, skipped, and failed orders separately.

## Explanation Patterns

Use this structure for each position change:

```text
<SYMBOL>: <ACTION> <QUANTITY> shares, moving from <current_shares> to <target_shares> shares (<target_weight>% target).
Reason: <strategy> signal is <LONG/SHORT/FLAT> with score <score>. <reason field or concise interpretation of available fields>.
Execution note: <approval/short-sale/risk-guard note when relevant>.
```

Examples:

- Long increase: "SPY is moving toward a 25.00% target because the strategy row is LONG, score is positive, and the reason says positive absolute and relative momentum."
- Full exit: "TLT is being sold to zero because its current target weight is 0.00%; the signal is FLAT or outside the current selection."
- Short sale: "XBI is being sold short to a -10.00% target because the strategy row is SHORT with negative absolute momentum. The order must pass Alpaca account and asset shortability checks before execution."
- Cover short: "XBI is being bought to cover because the current position is short and the target is less short or flat."

## Strategy Logic Cheat Sheet

- `momentum_social`: Uses composite price momentum, social momentum, volume momentum, risk-adjusted sizing, per-symbol caps, max exposure, max longs, and optional target-vol scaling.
- `dual_momentum`: Ranks relative and absolute momentum; LONG for positive momentum candidates, SHORT for weak negative momentum candidates. Social sentiment can tilt the score.
- `trend_following`: LONG when price is above 50/200 SMA with positive trend; SHORT when price is below 50/200 SMA with negative trend.
- `mean_reversion`: LONG oversold symbols above long trend; SHORT overbought bounces inside weak long trend.
- `breakout`: LONG near 55-day highs with volume confirmation; SHORT near 55-day lows with expanding volume.
- `risk_parity`: Selects lower-volatility long sleeves.
- `fast_momentum`: Defensive momentum strategy. Risk-on assets compete against a cash hurdle; if risk-on fails, it rotates into configured defensive symbols instead of shorting.
- `invest_spy`: SPY-centered dynamic allocation with defensive and crisis-hedge sleeves; risk guards can reduce or zero target weights.

## Safety Boundaries

- Never submit without explicit approval for the exact planned orders.
- Never trade when kill switch, disabled controls, closed market, blocked account, or missing prices are reported.
- Never treat backtest results as approval to trade.
- Never modify target weights because external news or the agent disagrees with the algorithm.
- For shorts, require the project/brokerage path to check account shorting, asset tradability, asset shortability, borrow availability, marginability, and buying power.
- Preserve order metadata such as `approval_id`, `position_intent`, `target_weight`, `target_shares`, `current_shares`, and `trade_dollars` in summaries.
