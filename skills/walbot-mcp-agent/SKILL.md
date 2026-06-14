---
name: walbot-mcp-agent
description: Use this skill when an external agent needs to operate Walbot through its MCP tools — review signals, explain portfolio changes, and request trade approval.
---

## MCP Tools (walbot, from `src/mcp_server.py`)

- `get_live_signals(algorithm)` — active positions with `score`, `score_components` (price_macro/meso/micro/nano + sentiment + pullback_uptrend), `target_weight`, and `reason` per symbol.
- `get_portfolio_preview(algorithm)` — per-symbol `positions` with `current_weight`, `target_weight`, `status` (retained/dropped/new/none), and `reason`.
- `get_planned_orders(algorithm)` — deterministic order plan: `symbol`, `action`, `quantity`, `current_shares`, `target_shares`, `trade_dollars`.
- `place_orders(algorithm)` — runs the full algorithm pipeline (decide → plan → submit) against the configured brokerage via the generic brokerage interface. Kill switch is honoured. No additional approval step.
- `request_trade_approval(planned_orders, ...)` — Telegram approve/deny flow.

## Fast Momentum Algorithm

Composite score from z-scored nano (10 bars), micro (78), meso (60d), macro (180d) returns + sentiment + pullback bonus. Risk-on symbols need `macro_trend_ok` + min micro return. Top `max_positions` selected (sort key = score + `min_score_delta_to_replace` for incumbents — stickiness). Weights are proportional to score, capped at `max_single_position_weight`.

### Reason codes
**Live signals:** `Top Rank` = selected; `Macro negative` / `Score too low` / `Micro too low` / `No rank slot`
**Portfolio preview:** `Sticky` (incumbent kept by bonus), `Retained`, `New`, `Dropped`, `None`

## Workflow

1. Read state via `get_controls()` + `get_status()`. Only proceed if trading enabled and strategy is not `none`.
2. For fast momentum: call `get_portfolio_preview()` + `get_live_signals()`. Call `get_planned_orders()` for the order plan.
3. Explain each change: current→target shares/weight, action, reason tied to algorithm signal fields. Use payload data as source of truth.
4. When approval has been granted by the third-party agent, call `place_orders(algorithm)` to submit the rebalance directly to the brokerage. Alternatively use `request_trade_approval()` with the planned orders for the Telegram approve/deny flow.
5. Do not invent trades, modify weights, or treat backtests as approval.
