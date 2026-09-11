"""The changes an agent may make to a plan, applied on this side of the wire.

The agent's job is to judge a proposal against things the algorithm could not see -- news, a
headline, a scheduled event -- and that judgement has to be able to change the outcome or it is
decoration. What it must *not* become is a second opinion on sizing: ``SOUL.md`` draws the fence
at "you do not choose strategy or size positions", and an agent that can rewrite a weight from
0.20 to 0.35 has crossed it.

So edits are named operations on symbols rather than a patch of values. The agent says *which*
symbol and *what to do about it*; it never says *how much*. That fence is held by the shape of
the request instead of by validating a diff and hoping the allowed set was drawn correctly.

Each operation is spelled differently per :data:`~src.core.interfaces.MODE_TARGET` /
:data:`~src.core.interfaces.MODE_INCREMENTAL`, because absence means opposite things in the two.
Under ``target`` the intent list *is* the portfolio, so a symbol dropped from it is sold to zero
(:func:`~src.core.orders.resolve_target_shares` seeds every held symbol at zero before applying
intents) -- deleting a row there is an instruction to sell, not a refusal to act. Under
``incremental`` each intent is a delta and anything unlisted is genuinely left alone. Writing
both spellings out here is the whole point of the module: it is the one place the difference has
to be right.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from .interfaces import MODE_INCREMENTAL, MODE_TARGET, AlgorithmPlan, Intent

logger = logging.getLogger(__name__)

#: Leave the symbol exactly as it is. A veto: the algorithm wanted to act and the agent has a
#: reason it should not, so the position neither grows nor shrinks today.
OP_SKIP = "skip"
#: Close the position. The one *active* thing a news check should be able to do -- "this holding
#: has a problem, get out" -- and deliberately a named operation rather than something that
#: happens as a side effect of dropping a row.
OP_EXIT = "exit"

EDIT_OPS = (OP_SKIP, OP_EXIT)


class PlanEditRefused(Exception):
    """An edit this plan cannot take. Carries ``reason``, like the other refusals."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _shares(symbol: str, value: float, template: Intent | None) -> Intent:
    """An absolute share count for ``symbol``, keeping whatever the algorithm attached.

    ``shares`` is the one intent kind :func:`~src.core.orders.resolve_target_shares` reads
    without consulting a price, so an edit expressed this way cannot be re-sized by a market
    that moved between the plan and the claim.
    """
    return Intent(
        symbol=symbol,
        kind="shares",
        value=float(value),
        extra=dict(template.extra) if template is not None else {},
    )


def apply_edits(
    plan: AlgorithmPlan,
    edits: list[dict[str, Any]] | None,
    *,
    positions: dict[str, float],
) -> tuple[AlgorithmPlan, list[dict[str, Any]]]:
    """Return ``plan`` with ``edits`` applied, and a record of what each one did.

    The record is for the response and the journal: an edited plan that reports only its final
    state makes "the agent vetoed three symbols" indistinguishable from "the algorithm proposed
    nothing", and those want very different follow-ups.
    """
    if not edits:
        return plan, []

    # An order-book plan is reconciled against the broker by absence: whatever is not in
    # ``desired_orders`` gets cancelled. That makes every edit here backwards -- removing a
    # held position's ``:stop`` key would not decline an action, it would cancel a live stop
    # and leave the position unprotected. There is no safe partial edit of this shape, so the
    # only veto it gets is the one that needs no mechanism: don't call place_orders.
    if plan.desired_orders:
        raise PlanEditRefused(
            f"{plan.strategy} proposes an order book, which cannot be partially edited: "
            "dropping a leg cancels a resting order rather than declining it. Submit this plan "
            "whole, or decline it by not calling place_orders."
        )

    if plan.mode not in (MODE_TARGET, MODE_INCREMENTAL):
        raise PlanEditRefused(f"Unknown plan mode {plan.mode!r}; refusing to edit it.")

    by_symbol = {intent.symbol: intent for intent in plan.intents}
    held = {str(symbol).upper(): float(qty) for symbol, qty in (positions or {}).items()}

    requested: dict[str, str] = {}
    for edit in edits:
        if not isinstance(edit, dict):
            raise PlanEditRefused(f"Each edit must be an object with 'op' and 'symbol'; got {edit!r}.")
        op = str(edit.get("op") or "").strip().lower()
        symbol = str(edit.get("symbol") or "").strip().upper()
        if op not in EDIT_OPS:
            raise PlanEditRefused(f"Unknown edit op {op!r}; expected one of {', '.join(EDIT_OPS)}.")
        if not symbol:
            raise PlanEditRefused(f"Edit {op!r} names no symbol.")
        # Neither proposed nor held means the agent is editing something this plan has no
        # opinion about -- far more likely a mistaken ticker than a deliberate no-op, and a
        # silently ignored veto is the kind that gets noticed after the order fills.
        if symbol not in by_symbol and symbol not in held:
            raise PlanEditRefused(
                f"{symbol} is neither proposed by this plan nor held in the account; nothing to {op}."
            )
        if symbol in requested:
            raise PlanEditRefused(f"Conflicting edits for {symbol}: {requested[symbol]!r} and {op!r}.")
        requested[symbol] = op

    edited: list[Intent] = []
    applied: list[dict[str, Any]] = []

    for intent in plan.intents:
        op = requested.get(intent.symbol)
        if op is None:
            edited.append(intent)
            continue
        current = held.get(intent.symbol, 0.0)
        if op == OP_SKIP:
            if plan.mode == MODE_TARGET:
                # Pinned to what is already there, not deleted: under target mode a missing
                # row targets zero, which would sell the position the agent meant to leave be.
                edited.append(_shares(intent.symbol, current, intent))
                effect = f"held at {current:g} shares"
            else:
                # Incremental: the intent was a delta, so dropping it applies no delta.
                effect = "increment dropped"
        else:  # OP_EXIT
            target = 0.0 if plan.mode == MODE_TARGET else -current
            edited.append(_shares(intent.symbol, target, intent))
            effect = "closed to zero" if current else "not held; nothing to close"
        applied.append({"op": op, "symbol": intent.symbol, "effect": effect})

    # A held symbol the plan never mentioned can still be exited: under target mode it was
    # already on its way to zero, but saying so explicitly keeps the report honest, and under
    # incremental mode it is the only way to reach it at all.
    for symbol, op in requested.items():
        if symbol in by_symbol:
            continue
        current = held.get(symbol, 0.0)
        if op == OP_EXIT:
            edited.append(_shares(symbol, 0.0 if plan.mode == MODE_TARGET else -current, None))
            applied.append({"op": op, "symbol": symbol, "effect": "closed to zero"})
        elif plan.mode == MODE_TARGET:
            # Held, unmentioned, and target mode: the plan was going to sell it. Skip means
            # keep it, which needs a row saying so.
            edited.append(_shares(symbol, current, None))
            applied.append({"op": op, "symbol": symbol, "effect": f"held at {current:g} shares"})
        else:
            applied.append({"op": op, "symbol": symbol, "effect": "not in this plan; already untouched"})

    logger.info("Applied %d agent edit(s) to the %s plan: %s", len(applied), plan.strategy, applied)
    return replace(plan, intents=edited), applied
