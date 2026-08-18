from __future__ import annotations
import logging
from math import ceil, floor
from typing import Any
from uuid import uuid4

from src.core.interfaces import MODE_INCREMENTAL, MODE_TARGET, Brokerage, Intent, OrderRequest
from src.notifications.service import request_trade_approval

logger = logging.getLogger(__name__)


#: Decimal places kept when a brokerage accepts fractional quantities.
FRACTIONAL_SHARE_PRECISION = 2

#: Share counts closer than this to zero are treated as no position / no trade.
SHARE_EPSILON = 1e-9

#: Dollar slack when comparing a batch's cost against its budget, so float noise on a sum of
#: prices does not read as a shortfall and trim a batch that actually fits.
FUNDING_EPSILON = 0.01

#: Liquidated slightly above the shortfall, because the batch is priced at the last print and
#: market orders fill worse than that. Freeing a little too much leaves harmless idle cash;
#: freeing a little too little re-rejects the leg this exists to save.
LIQUIDATION_PAD = 0.005

#: Fit every buy to the same fraction of its target. Keeps the *shape* of the intended
#: portfolio when the whole of it cannot be afforded, which is what matters to an allocation
#: strategy -- ending at 96% of eight names beats 100% of seven and nothing of the eighth.
FUNDING_PRO_RATA = "pro_rata"
#: Fund each buy in full, largest first, until the money runs out. Right for a batch of
#: independent commitments rather than one portfolio: shaving every DCA leg on a whole-share
#: brokerage can push all of them under one share and deploy nothing at all, where filling
#: the largest few deploys what it can and leaves the rest accrued for next time.
FUNDING_GREEDY = "greedy"


def round_shares(raw_shares: float, supports_fractional_shares: bool = False) -> float:
    """Round a raw share count toward zero, to whole shares unless fractional is allowed."""
    if supports_fractional_shares:
        # Truncate rather than round so sizing never overshoots the target dollar amount.
        scale = 10**FRACTIONAL_SHARE_PRECISION
        scaled = raw_shares * scale
        return (floor(scaled) if scaled >= 0 else ceil(scaled)) / scale
    return floor(raw_shares) if raw_shares >= 0 else ceil(raw_shares)


def _shares_for_target_weight(
    investable_equity: float,
    target_weight: float,
    price: float,
    supports_fractional_shares: bool = False,
) -> float:
    """Size a target weight into shares, rounding down to whole shares unless fractional is allowed."""
    return round_shares(
        (investable_equity * target_weight) / price, supports_fractional_shares=supports_fractional_shares
    )


def resolve_target_shares(
    intents: list[Intent],
    mode: str,
    current_positions: dict[str, float],
    latest_prices: dict[str, float],
    investable_equity: float,
    supports_fractional_shares: bool = False,
) -> dict[str, float]:
    """Turn intents into absolute target share counts for the whole portfolio.

    ``target`` mode reads the intent list as the complete portfolio, so anything held but not
    listed is targeted at zero. ``incremental`` mode reads each intent as a change to apply on
    top of what is already held and leaves every other holding alone -- the difference that
    lets a DCA buy coexist with positions another strategy owns.
    """
    if mode not in (MODE_TARGET, MODE_INCREMENTAL):
        raise ValueError(f"Unknown intent mode: {mode!r}")

    target_shares: dict[str, float] = {symbol: 0.0 for symbol in current_positions} if mode == MODE_TARGET else {}

    for intent in intents:
        price = float(latest_prices.get(intent.symbol, 0.0) or 0.0)
        if intent.kind == "shares":
            raw_shares = float(intent.value)
        elif price <= 0:
            logger.warning("Skipping %s intent for %s: no usable price", intent.kind, intent.symbol)
            continue
        elif intent.kind == "weight":
            raw_shares = (investable_equity * float(intent.value)) / price
        else:  # notional
            raw_shares = float(intent.value) / price

        fractional_ok = supports_fractional_shares and raw_shares >= 0
        shares = round_shares(raw_shares, supports_fractional_shares=fractional_ok)
        if mode == MODE_INCREMENTAL:
            # Round the increment, not the resulting position: a whole-share broker must be
            # able to reject a sub-share buy without disturbing what is already held.
            target_shares[intent.symbol] = float(current_positions.get(intent.symbol, 0.0)) + shares
        else:
            target_shares[intent.symbol] = shares

    return target_shares


def plan_position_orders(
    latest_prices: dict[str, float],
    current_positions: dict[str, float],
    target_weights: dict[str, float],
    equity: float,
    min_trade_dollars: float = 50.0,
    rebalance_threshold: float = 0.02,
    supports_fractional_shares: bool = False,
) -> list[dict[str, str | int | float]]:
    """Build ordered market orders needed to move current positions toward target weights.

    Quantities are whole shares unless ``supports_fractional_shares`` is set, which brokerages
    declare via ``Brokerage.supports_fractional_shares``. Short targets are always sized in whole
    shares because fractional quantities cannot be shorted.

    Sizes against the whole book. Whether the account can pay for the result is a separate
    question, answered by :func:`fund_planned_orders` against buying power.
    """
    target_shares = {
        symbol: _shares_for_target_weight(
            equity,
            target_weights.get(symbol, 0.0),
            latest_prices[symbol],
            supports_fractional_shares=supports_fractional_shares and target_weights.get(symbol, 0.0) >= 0,
        )
        for symbol in sorted(set(target_weights) | set(current_positions))
        if float(latest_prices.get(symbol) or 0.0) > 0
    }
    return plan_share_orders(
        latest_prices,
        current_positions,
        target_shares,
        equity,
        min_trade_dollars=min_trade_dollars,
        rebalance_threshold=rebalance_threshold,
        supports_fractional_shares=supports_fractional_shares,
        target_weights=target_weights,
    )


def plan_share_orders(
    latest_prices: dict[str, float],
    current_positions: dict[str, float],
    target_shares: dict[str, float],
    equity: float,
    min_trade_dollars: float = 50.0,
    rebalance_threshold: float = 0.0,
    supports_fractional_shares: bool = False,
    target_weights: dict[str, float] | None = None,
) -> list[dict[str, str | int | float]]:
    """Build the market orders that move current positions to already-resolved share counts.

    The share-denominated half of :func:`plan_position_orders`, shared with intent resolution
    so weight-mode and incremental-mode runs go through one sizing and threshold path.
    ``target_weights`` is carried through onto the rows for reporting only.
    """
    planned_orders: list[dict[str, str | int | float]] = []
    target_weights = target_weights or {}
    min_rebalance_dollars = max(min_trade_dollars, equity * max(rebalance_threshold, 0.0))

    for symbol in sorted(set(target_shares) | set(current_positions)):
        price = latest_prices.get(symbol)
        if price is None or price <= 0:
            logger.warning("Skipping %s because latest price is invalid: %s", symbol, price)
            continue

        target_weight = target_weights.get(symbol, 0.0)
        current_shares = current_positions.get(symbol, 0)
        symbol_target_shares = target_shares.get(symbol, current_shares)
        diff = symbol_target_shares - current_shares
        if supports_fractional_shares and symbol_target_shares >= 0:
            diff = round(diff, FRACTIONAL_SHARE_PRECISION)
        trade_dollars = abs(diff) * price

        # Closing a position is not a rebalance, so ``rebalance_threshold`` does not gate it:
        # a holding smaller than the threshold can never move far enough to clear it, and
        # would be held forever by a rule meant to suppress small adjustments. The absolute
        # ``min_trade_dollars`` floor still applies, so this cannot spray sub-minimum orders.
        closing = abs(symbol_target_shares) <= SHARE_EPSILON and abs(current_shares) > SHARE_EPSILON
        floor = min_trade_dollars if closing else min_rebalance_dollars

        if abs(diff) <= SHARE_EPSILON or trade_dollars < floor:
            logger.info(
                "No trade required for %s: target_shares=%s current_shares=%s drift_dollars=%.2f",
                symbol,
                symbol_target_shares,
                current_shares,
                trade_dollars,
            )
            continue

        side = "buy" if diff > 0 else "sell"
        quantity = abs(diff)
        opens_or_increases_short = side == "sell" and symbol_target_shares < 0
        planned_orders.append(
            {
                "symbol": symbol,
                "action": side,
                "quantity": quantity,
                "target_weight": target_weight,
                "target_shares": symbol_target_shares,
                "current_shares": current_shares,
                "trade_dollars": trade_dollars,
                "latest_price": price,
                "position_intent": (
                    "sell_short"
                    if opens_or_increases_short
                    else "buy_to_cover"
                    if side == "buy" and current_shares < 0
                    else "sell_to_close"
                    if side == "sell"
                    else "buy_to_open"
                ),
                "opens_short": opens_or_increases_short,
                "short_shares_after": abs(symbol_target_shares) if symbol_target_shares < 0 else 0,
            }
        )

    return sorted(planned_orders, key=lambda order: 0 if order["action"] == "sell" else 1)


def _round_up_shares(raw_shares: float, supports_fractional_shares: bool = False) -> float:
    """Round away from zero -- a liquidation must free at least the shortfall, never less."""
    if supports_fractional_shares:
        scale = 10**FRACTIONAL_SHARE_PRECISION
        return ceil(raw_shares * scale) / scale
    return float(ceil(raw_shares))


def _liquidation_legs(
    shortfall: float,
    cash_equivalents: dict[str, dict[str, float]],
    planned_orders: list[dict[str, str | int | float]],
    supports_fractional_shares: bool = False,
) -> list[dict[str, str | int | float]]:
    """Sell cash-like holdings to cover what buying power cannot.

    Symbols the batch already trades are skipped. That is arithmetic rather than strategy:
    their proceeds are counted already if the plan sells them, and liquidating something the
    plan is *buying* in the same batch would only undo itself.
    """
    already_traded = {str(order["symbol"]) for order in planned_orders}
    needed = max(shortfall, 0.0) * (1.0 + LIQUIDATION_PAD)
    legs: list[dict[str, str | int | float]] = []

    for symbol, holding in cash_equivalents.items():
        if needed <= FUNDING_EPSILON:
            break
        if symbol in already_traded:
            continue
        price = float(holding.get("price") or 0.0)
        held = float(holding.get("shares") or 0.0)
        if price <= 0 or held <= SHARE_EPSILON:
            continue

        quantity = min(held, _round_up_shares(needed / price, supports_fractional_shares))
        if quantity <= SHARE_EPSILON:
            continue
        proceeds = quantity * price
        legs.append(
            {
                "symbol": symbol,
                "action": "sell",
                "quantity": quantity,
                "target_weight": 0.0,
                "target_shares": held - quantity,
                "current_shares": held,
                "trade_dollars": proceeds,
                "latest_price": price,
                "position_intent": "sell_to_close",
                "opens_short": False,
                "short_shares_after": 0,
                "requested_quantity": quantity,
                "funding_status": "full",
                "funding_source": "cash_equivalent",
                "reason": f"Liquidating {symbol} to fund the batch: frees ${proceeds:.2f} against a ${shortfall:.2f} shortfall",
            }
        )
        needed -= proceeds

    return legs


def _fit_buys_to_budget(
    buys: list[dict[str, str | int | float]],
    budget: float,
    policy: str,
    min_trade_dollars: float,
    supports_fractional_shares: bool,
) -> tuple[list[dict[str, str | int | float]], list[dict[str, str | int | float]]]:
    """Shrink or drop buy legs until the batch fits ``budget``. Returns ``(funded, unfunded)``."""
    requested = sum(float(order["trade_dollars"]) for order in buys)
    if policy == FUNDING_GREEDY:
        # Largest first: on a whole-share brokerage that is the leg most likely to survive
        # rounding, and the one whose budget has waited longest to deploy.
        ordered = sorted(buys, key=lambda order: (-float(order["trade_dollars"]), str(order["symbol"])))
        scale = 1.0
    else:
        ordered = list(buys)
        scale = (budget / requested) if requested > 0 else 0.0

    funded: list[dict[str, str | int | float]] = []
    unfunded: list[dict[str, str | int | float]] = []
    remaining = budget

    for order in ordered:
        price = float(order["latest_price"])
        requested_quantity = float(order["quantity"])
        # Rounding is settled here, at the end, rather than being re-applied to an
        # already-rounded quantity: this is the number that reaches the broker.
        quantity = round_shares(requested_quantity * scale, supports_fractional_shares)
        if quantity * price > remaining:
            quantity = round_shares(remaining / price, supports_fractional_shares)
        cost = quantity * price

        if quantity <= SHARE_EPSILON or cost < min_trade_dollars:
            unfunded.append(
                {
                    **order,
                    "requested_quantity": requested_quantity,
                    "funding_status": "unfunded",
                    "reason": (
                        f"Insufficient funds for {order['symbol']}: "
                        f"${float(order['trade_dollars']):.2f} required, ${max(remaining, 0.0):.2f} available"
                    ),
                }
            )
            continue

        remaining -= cost
        reduced = quantity < requested_quantity - SHARE_EPSILON
        funded.append(
            {
                **order,
                "quantity": quantity,
                "trade_dollars": cost,
                "requested_quantity": requested_quantity,
                "funding_status": "reduced" if reduced else "full",
                **(
                    {
                        "reason": (
                            f"Reduced {order['symbol']} from {requested_quantity} to {quantity} shares "
                            f"to fit available funds"
                        )
                    }
                    if reduced
                    else {}
                ),
            }
        )

    return funded, unfunded


def fund_planned_orders(
    planned_orders: list[dict[str, str | int | float]],
    *,
    buying_power: float,
    reserve: float = 0.0,
    cash_equivalents: dict[str, dict[str, float]] | None = None,
    min_trade_dollars: float = 0.0,
    supports_fractional_shares: bool = False,
    policy: str = FUNDING_PRO_RATA,
) -> tuple[list[dict[str, str | int | float]], list[dict[str, str | int | float]], dict[str, Any]]:
    """Fit a planned batch to the money that will actually be there to pay for it.

    Planning sizes a *portfolio* against equity; this sizes a *batch* against buying power,
    and the two are not the same sum. Without it the tail legs of a batch are submitted at
    their full target and refused one at a time by the broker, leaving the book half-rebalanced
    and the next session re-attempting the same impossible state.

    Pure, and deliberately so: every input is a number the caller already read, which is what
    lets the funding rules be tested against a real rejection log without a brokerage.

    Returns ``(fundable, unfunded, report)``. ``fundable`` is submit-ready and still ordered
    sells-first; ``unfunded`` are legs no amount of shrinking could clear, each carrying the
    reason why.
    """
    if not planned_orders:
        return [], [], {}

    sells = [order for order in planned_orders if order["action"] == "sell"]
    buys = [order for order in planned_orders if order["action"] == "buy"]
    requested = sum(float(order["trade_dollars"]) for order in buys)

    # A short sale brings a margin requirement rather than spendable cash, so its notional is
    # deliberately not counted as proceeds.
    proceeds = sum(float(order["trade_dollars"]) for order in sells if not order.get("opens_short"))
    budget = max(float(buying_power) - max(float(reserve), 0.0), 0.0) + proceeds

    liquidation: list[dict[str, str | int | float]] = []
    if requested > budget + FUNDING_EPSILON and cash_equivalents:
        liquidation = _liquidation_legs(
            requested - budget, cash_equivalents, planned_orders, supports_fractional_shares
        )
        for leg in liquidation:
            logger.info(
                "Funding the batch by selling %s %s to free $%.2f",
                leg["quantity"], leg["symbol"], float(leg["trade_dollars"]),
            )
        budget += sum(float(leg["trade_dollars"]) for leg in liquidation)

    if requested <= budget + FUNDING_EPSILON:
        funded = [{**order, "requested_quantity": float(order["quantity"]), "funding_status": "full"} for order in buys]
        unfunded: list[dict[str, str | int | float]] = []
    else:
        if liquidation:
            logger.info(
                "Batch needs $%.2f of buys against $%.2f available (after liquidation); fitting by %s",
                requested, budget, policy,
            )
        else:
            logger.warning(
                "Batch needs $%.2f of buys against $%.2f available; fitting by %s", requested, budget, policy
            )
        funded, unfunded = _fit_buys_to_budget(
            buys, budget, policy, min_trade_dollars, supports_fractional_shares
        )

    priced_sells = [{**order, "requested_quantity": float(order["quantity"]), "funding_status": "full"} for order in sells]
    funded_notional = sum(float(order["trade_dollars"]) for order in funded)
    report = {
        "buying_power": float(buying_power),
        "reserve": max(float(reserve), 0.0),
        "sale_proceeds": proceeds,
        "cash_equivalents_liquidated": sum(float(leg["trade_dollars"]) for leg in liquidation),
        "budget": budget,
        "requested_notional": requested,
        "funded_notional": funded_notional,
        "shortfall": max(requested - funded_notional, 0.0),
        "policy": policy,
        "reduced": [str(order["symbol"]) for order in funded if order.get("funding_status") == "reduced"],
        "unfunded": [str(order["symbol"]) for order in unfunded],
    }
    return liquidation + priced_sells + funded, unfunded, report


def _order_quantity(quantity: Any) -> float | int:
    """Return a submit-ready quantity, keeping whole amounts as ints for brokerages that expect them."""
    value = float(quantity)
    return int(value) if value.is_integer() else round(value, FRACTIONAL_SHARE_PRECISION)


def _approval_skips(planned_orders: list[dict[str, str | int | float]], approval_id: str) -> list[dict[str, str | int | float]]:
    return [
        {
            **planned_order,
            "action": "skip",
            "quantity": 0,
            "approval_id": approval_id,
            "approval_status": "not_approved",
        }
        for planned_order in planned_orders
    ]


def submit_planned_orders(
    brokerage: Brokerage,
    planned_orders: list[dict[str, str | int | float]],
    *,
    require_approval: bool = False,
    approval_id: str | None = None,
    approval_timeout_seconds: int = 300,
    approval_poll_seconds: int = 5,
) -> list[dict[str, str | int | float]]:
    """Submit already-planned market orders, optionally gated on trade approval.

    Skips (and records) short sales the brokerage reports as infeasible. Shared by the
    algorithm rebalancer and the DCA runtime so there is a single order-submission path.
    """
    if not planned_orders:
        return []

    if require_approval:
        approval_id = approval_id or uuid4().hex[:10]
        approved = request_trade_approval(
            planned_orders,
            approval_id=approval_id,
            timeout_seconds=approval_timeout_seconds,
            poll_seconds=approval_poll_seconds,
        )
        if not approved:
            logger.warning("Trade approval %s was denied or timed out; skipping %s planned order(s)", approval_id, len(planned_orders))
            return _approval_skips(planned_orders, approval_id)

    order_results: list[dict[str, str | int | float]] = []
    for desired_order in planned_orders:
        symbol = str(desired_order["symbol"])
        side = str(desired_order["action"])
        quantity = _order_quantity(desired_order["quantity"])
        if bool(desired_order.get("opens_short")):
            feasibility = brokerage.validate_short_sale_feasibility(
                symbol,
                quantity=quantity,
                target_shares=int(desired_order["target_shares"]),
                latest_price=float(desired_order["latest_price"]),
            )
            if not feasibility["shortable"]:
                logger.warning("Skipping short sale for %s: %s", symbol, feasibility["reason"])
                order_results.append(
                    {
                        **desired_order,
                        "action": "skip",
                        "quantity": 0,
                        "approval_status": "short_sale_not_feasible",
                        "reason": feasibility["reason"],
                    }
                )
                continue
        logger.info("Submitting %s order for %s qty=%s", side, symbol, quantity)
        try:
            result = brokerage.submit_order(
                OrderRequest(
                    symbol=symbol,
                    action=side,
                    quantity=quantity,
                    # Carried so a simulated brokerage can fill without its own price source.
                    # Absent on DCA orders, which are sized in notional rather than from a price.
                    extra={"latest_price": float(desired_order.get("latest_price") or 0.0)},
                )
            )
        except Exception as exc:
            # One rejected leg (shares held for an open order, buying power, a halt) must not
            # abandon the rest of the batch, which would leave the account half-rebalanced.
            logger.warning("Order rejected for %s %s qty=%s: %s", side, symbol, quantity, exc)
            order_results.append(
                {
                    **desired_order,
                    "symbol": symbol,
                    "action": side,
                    "quantity": quantity,
                    "status": "rejected",
                    "reason": str(exc),
                }
            )
            continue
        order_results.append(
            {
                **desired_order,
                "symbol": symbol,
                "action": side,
                "quantity": quantity,
                "status": "submitted",
                "order_id": result.get("order_id", "unknown"),
            }
        )

    return order_results
