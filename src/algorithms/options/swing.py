from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from math import floor
from typing import Any

from src.brokerages.alpaca_client import (
    create_data_client,
    create_option_data_client,
    create_trading_client,
    get_account_equity,
    get_historical_daily_bars,
    get_latest_price,
    get_option_contracts,
    get_option_latest_quotes,
    get_positions,
    is_market_open,
    submit_option_limit_order,
)
from src.core.config import Config, get_config
from src.api.controls import load_controls
from src.common.logging_utils import log_position_changes
from src.core.strategy_models import strategy_signal_rows

logger = logging.getLogger(__name__)

CONTRACT_MULTIPLIER = 100
SUPPORTED_OPTIONS_STRATEGIES = {"options_swing_dual_momentum", "dual_momentum"}


@dataclass(frozen=True)
class OptionCandidate:
    underlying: str
    side: str
    contract_symbol: str
    contract_type: str
    expiration_date: str
    strike_price: float
    delta: float | None
    bid: float
    ask: float
    limit_price: float
    quantity: int
    estimated_premium: float
    score: float


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _contract_symbol(contract: Any) -> str:
    return str(_attr(contract, "symbol", "id", default="")).strip()


def _contract_type(contract: Any) -> str:
    value = _attr(contract, "type", "contract_type", default="")
    value = getattr(value, "value", value)
    return str(value).lower()


def _contract_expiration(contract: Any) -> str:
    value = _attr(contract, "expiration_date", "expiration", default="")
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _quote_for_symbol(quotes: Any, symbol: str) -> Any:
    if isinstance(quotes, dict):
        return quotes.get(symbol) or quotes.get(symbol.upper())
    return None


def _quote_prices(quote: Any) -> tuple[float, float]:
    bid = _as_float(_attr(quote, "bid_price", "bp", "bid", default=0.0))
    ask = _as_float(_attr(quote, "ask_price", "ap", "ask", default=0.0))
    return bid, ask


def _contract_delta(contract: Any, quote: Any = None) -> float | None:
    for source in (contract, quote):
        value = _attr(source, "delta", default=None)
        if value is not None:
            return _as_float(value)
        greeks = _attr(source, "greeks", default=None)
        value = _attr(greeks, "delta", default=None)
        if value is not None:
            return _as_float(value)
    return None


def _contract_open_interest(contract: Any) -> int:
    return int(_as_float(_attr(contract, "open_interest", default=0.0)))


def _option_type_for_signal(side: str) -> str:
    return "put" if side.upper() == "SHORT" else "call"


def _strike_bounds(price: float, config: Config) -> tuple[float, float]:
    width = max(float(config.options_swing_strike_range_pct), 0.01)
    return price * (1 - width), price * (1 + width)


def _limit_price(bid: float, ask: float) -> float:
    if bid > 0 and ask > 0:
        return round((bid + ask) / 2, 2)
    return round(max(bid, ask), 2)


def _spread_ok(bid: float, ask: float, max_spread_pct: float) -> bool:
    if bid <= 0 or ask <= 0 or ask < bid:
        return False
    mid = (bid + ask) / 2
    return ((ask - bid) / mid) <= max(max_spread_pct, 0.01)


def _delta_ok(delta: float | None, config: Config) -> bool:
    if delta is None:
        return True
    abs_delta = abs(delta)
    return config.options_swing_min_delta <= abs_delta <= config.options_swing_max_delta


def _rank_contract(contract: Any, quote: Any, target_delta: float = 0.5) -> tuple[float, float, float]:
    bid, ask = _quote_prices(quote)
    delta = _contract_delta(contract, quote)
    delta_distance = abs(abs(delta) - target_delta) if delta is not None else 0.25
    open_interest = _contract_open_interest(contract)
    spread = ask - bid if ask and bid else 99.0
    return (delta_distance, spread, -open_interest)


def dual_momentum_option_signals(config: Config, data_client=None) -> list[dict[str, Any]]:
    data_client = data_client or create_data_client(config)
    bars_by_symbol = get_historical_daily_bars(
        config.symbols,
        lookback_days=max(config.momentum_lookback_days, 252),
        extra_buffer_days=config.history_extra_buffer_days,
        data_client=data_client,
        data_feed=config.alpaca_data_feed,
    )
    rows = strategy_signal_rows("dual_momentum", bars_by_symbol)
    active = [row for row in rows if str(row.get("side")) in {"LONG", "SHORT"}]
    active.sort(key=lambda row: abs(_as_float(row.get("score"))), reverse=True)
    return active[: max(int(config.options_swing_max_contracts), 1)]


def select_option_candidate(
    config: Config,
    trading_client,
    option_data_client,
    underlying: str,
    side: str,
    score: float = 0.0,
    underlying_price: float | None = None,
) -> OptionCandidate | None:
    underlying_price = underlying_price or get_latest_price(underlying, create_data_client(config), data_feed=config.alpaca_data_feed)
    strike_gte, strike_lte = _strike_bounds(underlying_price, config)
    today = date.today()
    contract_type = _option_type_for_signal(side)
    contracts = list(
        get_option_contracts(
            trading_client,
            underlying,
            contract_type,
            today + timedelta(days=max(int(config.options_swing_dte_min), 1)),
            today + timedelta(days=max(int(config.options_swing_dte_max), int(config.options_swing_dte_min))),
            strike_price_gte=strike_gte,
            strike_price_lte=strike_lte,
            limit=100,
        )
        or []
    )
    if not contracts:
        logger.info("No option contracts found for %s %s", underlying, contract_type)
        return None

    symbols = [_contract_symbol(contract) for contract in contracts if _contract_symbol(contract)]
    quotes = get_option_latest_quotes(option_data_client, symbols)
    ranked: list[tuple[tuple[float, float, float], Any, Any]] = []
    for contract in contracts:
        symbol = _contract_symbol(contract)
        quote = _quote_for_symbol(quotes, symbol)
        if not symbol or quote is None:
            continue
        if _contract_open_interest(contract) < int(config.options_swing_min_open_interest):
            continue
        bid, ask = _quote_prices(quote)
        if not _spread_ok(bid, ask, float(config.options_swing_max_spread_pct)):
            continue
        if not _delta_ok(_contract_delta(contract, quote), config):
            continue
        ranked.append((_rank_contract(contract, quote), contract, quote))

    if not ranked:
        logger.info("No liquid option candidate passed filters for %s", underlying)
        return None

    _rank, contract, quote = sorted(ranked, key=lambda item: item[0])[0]
    bid, ask = _quote_prices(quote)
    limit_price = _limit_price(bid, ask)
    max_premium = max(float(config.options_swing_max_premium), 0.0)
    quantity = min(
        max(int(config.options_swing_max_contracts), 1),
        floor(max_premium / max(limit_price * CONTRACT_MULTIPLIER, 0.01)) if max_premium else 0,
    )
    if quantity <= 0:
        logger.info("Skipping %s because premium %.2f exceeds cap %.2f", _contract_symbol(contract), limit_price, max_premium)
        return None
    return OptionCandidate(
        underlying=underlying,
        side=side,
        contract_symbol=_contract_symbol(contract),
        contract_type=_contract_type(contract) or contract_type,
        expiration_date=_contract_expiration(contract),
        strike_price=_as_float(_attr(contract, "strike_price", default=0.0)),
        delta=_contract_delta(contract, quote),
        bid=bid,
        ask=ask,
        limit_price=limit_price,
        quantity=quantity,
        estimated_premium=limit_price * quantity * CONTRACT_MULTIPLIER,
        score=score,
    )


def run_options_once(account_id: str | None = None) -> list[dict[str, Any]]:
    controls = load_controls()
    strategy = str(controls.get("options_strategy") or "none")
    if strategy not in SUPPORTED_OPTIONS_STRATEGIES:
        logger.warning("Options strategy %s is not executable yet. Exiting without orders.", strategy)
        return []
    if not controls.get("options_trading_enabled"):
        logger.warning("Options trading is disabled in dashboard controls. Exiting without orders.")
        return []

    selected_account = account_id or str(controls.get("options_trading_account_id") or controls.get("trading_account_id") or "") or None
    config = get_config(account_id=selected_account, strategy_id="dual_momentum")
    if config.kill_switch:
        logger.warning("Kill switch is enabled. Exiting options runner without orders.")
        return []

    trading_client = create_trading_client(config)
    if not is_market_open(trading_client):
        logger.warning("Market is closed according to Alpaca clock. Exiting options runner without orders.")
        return []

    data_client = create_data_client(config)
    option_data_client = create_option_data_client(config)
    account_equity = get_account_equity(trading_client)
    current_positions = get_positions(trading_client)
    signals = dual_momentum_option_signals(config, data_client=data_client)
    results: list[dict[str, Any]] = []

    for signal in signals:
        underlying = str(signal["symbol"])
        side = str(signal.get("side") or "FLAT")
        underlying_price = get_latest_price(underlying, data_client, data_feed=config.alpaca_data_feed)
        candidate = select_option_candidate(
            config,
            trading_client,
            option_data_client,
            underlying,
            side,
            score=_as_float(signal.get("score")),
            underlying_price=underlying_price,
        )
        if candidate is None:
            continue
        if current_positions.get(candidate.contract_symbol, 0) > 0:
            logger.info("Holding existing option position %s; skipping duplicate entry", candidate.contract_symbol)
            results.append({"symbol": candidate.contract_symbol, "action": "hold", "underlying": underlying})
            continue
        if candidate.estimated_premium > account_equity:
            logger.info("Skipping %s because estimated premium exceeds account equity", candidate.contract_symbol)
            continue
        order = submit_option_limit_order(
            trading_client,
            candidate.contract_symbol,
            "buy",
            candidate.quantity,
            candidate.limit_price,
            position_intent="buy_to_open",
        )
        results.append(
            {
                "symbol": candidate.contract_symbol,
                "underlying": underlying,
                "action": "buy_to_open",
                "quantity": candidate.quantity,
                "limit_price": candidate.limit_price,
                "estimated_premium": candidate.estimated_premium,
                "order_id": getattr(order, "id", "unknown"),
            }
        )

    log_position_changes(results)
    return results
