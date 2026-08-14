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


def _round_shares(raw_shares: float, supports_fractional_shares: bool = False) -> float:
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
    return _round_shares(
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
        shares = _round_shares(raw_shares, supports_fractional_shares=fractional_ok)
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
    cash_buffer: float = 0.02,
    min_trade_dollars: float = 50.0,
    rebalance_threshold: float = 0.02,
    supports_fractional_shares: bool = False,
) -> list[dict[str, str | int | float]]:
    """Build ordered market orders needed to move current positions toward target weights.

    Quantities are whole shares unless ``supports_fractional_shares`` is set, which brokerages
    declare via ``Brokerage.supports_fractional_shares``. Short targets are always sized in whole
    shares because fractional quantities cannot be shorted.
    """
    investable_equity = equity * max(0.0, min(1.0, 1.0 - cash_buffer))
    target_shares = {
        symbol: _shares_for_target_weight(
            investable_equity,
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

        if abs(diff) <= SHARE_EPSILON or trade_dollars < min_rebalance_dollars:
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
