from __future__ import annotations
import logging
import math
from datetime import datetime, timedelta, timezone

from ..common.config_utils import as_bool

try:
    import pandas as pd
except ImportError:  # type: ignore
    pd = None

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType, OrderSide, OrderType, PositionIntent, TimeInForce
from alpaca.trading.requests import GetOptionContractsRequest, LimitOrderRequest, MarketOrderRequest
from alpaca.data.enums import DataFeed
from alpaca.data.historical import OptionHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.core.config import Config


logger = logging.getLogger(__name__)


def create_trading_client(config: Config) -> TradingClient:
    return TradingClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_api_secret,
        paper=False,
        url_override=config.alpaca_base_url or None,
    )


def create_data_client(config: Config) -> StockHistoricalDataClient:
    return StockHistoricalDataClient(
        api_key=config.alpaca_data_api_key or config.alpaca_api_key,
        secret_key=config.alpaca_data_api_secret or config.alpaca_api_secret,
    )


def create_option_data_client(config: Config) -> OptionHistoricalDataClient:
    return OptionHistoricalDataClient(
        api_key=config.alpaca_data_api_key or config.alpaca_api_key,
        secret_key=config.alpaca_data_api_secret or config.alpaca_api_secret,
    )


def get_account_equity(trading_client: TradingClient) -> float:
    account = trading_client.get_account()
    return float(account.equity)


def get_positions(trading_client: TradingClient) -> dict[str, float]:
    """Return held quantities by symbol, preserving fractional shares as reported."""
    positions = trading_client.get_all_positions()
    parsed: dict[str, float] = {}
    for position in positions:
        try:
            quantity = float(position.qty)
        except (TypeError, ValueError):
            quantity = 0.0
        # Whole holdings stay ints so downstream formatting and equality are unchanged.
        parsed[position.symbol] = int(quantity) if quantity.is_integer() else quantity
    return parsed


def get_position_marks(trading_client: TradingClient) -> dict[str, float]:
    """Last price per held symbol, as Alpaca marks it on the position itself.

    Falls back to market value over quantity, which is the same number by a different route
    and survives ``current_price`` being absent on a stale position payload.
    """
    marks: dict[str, float] = {}
    for position in trading_client.get_all_positions():
        price = _float_attr(position, "current_price")
        if price <= 0:
            quantity = _float_attr(position, "qty")
            price = (_float_attr(position, "market_value") / quantity) if quantity else 0.0
        if price > 0:
            marks[position.symbol] = price
    return marks


def _bool_attr(obj, name: str, default: bool = False) -> bool:
    return as_bool(getattr(obj, name, default), default)


def _float_attr(obj, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(obj, name, default))
    except (TypeError, ValueError):
        return default


def validate_short_sale_feasibility(
    trading_client: TradingClient,
    symbol: str,
    quantity: int,
    target_shares: int,
    latest_price: float,
) -> dict[str, object]:
    """Check Alpaca account and asset constraints before opening or increasing a short."""
    if quantity <= 0:
        return {"shortable": False, "reason": "quantity must be positive"}
    if target_shares >= 0:
        return {"shortable": True, "reason": "order does not leave a short position"}
    if latest_price <= 0:
        return {"shortable": False, "reason": "latest price must be positive"}

    try:
        account = trading_client.get_account()
    except Exception as exc:
        return {"shortable": False, "reason": f"unable to verify account shorting permissions: {exc}"}

    if not _bool_attr(account, "shorting_enabled"):
        return {"shortable": False, "reason": "account shorting is not enabled"}
    if _bool_attr(account, "trading_blocked") or _bool_attr(account, "account_blocked"):
        return {"shortable": False, "reason": "account trading is blocked"}

    try:
        asset = trading_client.get_asset(symbol)
    except Exception as exc:
        return {"shortable": False, "reason": f"unable to verify asset shortability: {exc}"}

    checks = {
        "tradable": _bool_attr(asset, "tradable"),
        "shortable": _bool_attr(asset, "shortable"),
        "easy_to_borrow": _bool_attr(asset, "easy_to_borrow"),
        "marginable": _bool_attr(asset, "marginable", True),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return {"shortable": False, "reason": f"{symbol} failed Alpaca asset checks: {', '.join(failed)}", "checks": checks}

    short_notional_after = abs(target_shares) * latest_price
    buying_power = _float_attr(account, "buying_power")
    if buying_power > 0 and short_notional_after > buying_power:
        return {
            "shortable": False,
            "reason": f"target short notional ${short_notional_after:.2f} exceeds buying power ${buying_power:.2f}",
            "checks": checks,
        }

    return {
        "shortable": True,
        "reason": "account and asset are short-sale eligible",
        "checks": checks,
        "short_notional_after": short_notional_after,
    }


def is_market_open(trading_client: TradingClient) -> bool:
    try:
        clock = trading_client.get_clock()
    except Exception as exc:
        logger.warning("Unable to confirm market clock; treating market as closed: %s", exc)
        return False
    return bool(getattr(clock, "is_open", False))


def _iter_bars(bars, symbol: str | None = None):
    if hasattr(bars, "data"):
        if symbol is not None:
            return bars.data.get(symbol, [])
        records = []
        for symbol_bars in bars.data.values():
            records.extend(symbol_bars)
        return records
    if isinstance(bars, dict):
        if symbol is not None:
            return bars.get(symbol, [])
        records = []
        for symbol_bars in bars.values():
            records.extend(symbol_bars)
        return records
    return bars


def _resolve_data_feed(data_feed: str | DataFeed | None) -> DataFeed | None:
    if data_feed is None:
        return None
    if isinstance(data_feed, DataFeed):
        return data_feed
    return DataFeed(data_feed.lower())


def _timestamp_to_utc(timestamp) -> "pd.Timestamp":
    parsed = pd.to_datetime(timestamp)
    if parsed.tzinfo is None:
        return parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _parse_bars_to_df(bars, symbol: str | None = None) -> "pd.DataFrame":
    if pd is None:
        raise ImportError("pandas is required to parse Alpaca bar data")

    records = []
    for bar in _iter_bars(bars, symbol):
        records.append(
            {
                "timestamp": _timestamp_to_utc(bar.timestamp),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            }
        )
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return df
    return df.sort_values("timestamp").reset_index(drop=True)


def get_latest_price(
    symbol: str,
    data_client: StockHistoricalDataClient,
    end_date: datetime | None = None,
    data_feed: str | DataFeed | None = "iex",
) -> float:
    end = end_date or datetime.now(timezone.utc)
    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        end=end,
        limit=1,
        feed=_resolve_data_feed(data_feed),
    )
    bars = data_client.get_stock_bars(request)
    df = _parse_bars_to_df(bars, symbol)
    if df.empty:
        raise RuntimeError(f"No latest bar found for {symbol}")
    return float(df["close"].iloc[-1])


def get_historical_daily_bars(
    symbols: list[str],
    lookback_days: int,
    extra_buffer_days: int = 250,
    data_client: StockHistoricalDataClient | None = None,
    end_date: datetime | None = None,
    start_date: datetime | None = None,
    data_feed: str | DataFeed | None = "iex",
) -> dict[str, "pd.DataFrame"]:
    if data_client is None:
        raise ValueError("data_client is required")

    end = end_date or datetime.now(timezone.utc)
    if start_date is None:
        total_days = lookback_days + extra_buffer_days
        start = end - timedelta(days=total_days * 2)
    else:
        start = start_date

    from alpaca.data.enums import Adjustment

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=1000,
            feed=_resolve_data_feed(data_feed),
            adjustment=Adjustment.ALL,
        )
        bars = data_client.get_stock_bars(request)
        df = _parse_bars_to_df(bars, symbol)
        bars_by_symbol[symbol] = df
        logger.debug("Fetched %s bars for %s", len(df), symbol)

    return bars_by_symbol


def get_historical_intraday_bars(
    symbols: list[str],
    lookback_bars: int,
    bar_minutes: int = 30,
    data_client: StockHistoricalDataClient | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    data_feed: str | DataFeed | None = "iex",
) -> dict[str, "pd.DataFrame"]:
    """Fetch recent intraday OHLCV bars for each symbol."""
    if data_client is None:
        raise ValueError("data_client is required")
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    if bar_minutes <= 0:
        raise ValueError("bar_minutes must be positive")

    end = end_date or datetime.now(timezone.utc)
    trading_minutes_per_day = 390
    needed_sessions = max(1, math.ceil((bar_minutes * lookback_bars) / trading_minutes_per_day))
    calendar_days = max(7, (needed_sessions * 3) + 2)
    start = start_date or end - timedelta(days=calendar_days)
    timeframe = TimeFrame(bar_minutes, TimeFrameUnit.Minute)

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=start,
                end=end,
                limit=max(lookback_bars + 5, 100),
                feed=_resolve_data_feed(data_feed),
            )
            bars = data_client.get_stock_bars(request)
            df = _parse_bars_to_df(bars, symbol)
        except Exception as exc:
            logger.warning("Skipping intraday bars for %s after provider error: %s", symbol, exc)
            df = pd.DataFrame()
        if not df.empty:
            df = df.tail(lookback_bars).reset_index(drop=True)
        bars_by_symbol[symbol] = df
        logger.debug("Fetched %s intraday bars for %s", len(df), symbol)

    return bars_by_symbol


def submit_market_order(trading_client: TradingClient, symbol: str, side: str, qty: float):
    side = side.lower()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if qty <= 0:
        raise ValueError("qty must be a positive quantity")
    order = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
    )
    return trading_client.submit_order(order_data=order)


def get_option_contracts(
    trading_client: TradingClient,
    underlying_symbol: str,
    contract_type: str,
    expiration_date_gte,
    expiration_date_lte,
    strike_price_gte: float | None = None,
    strike_price_lte: float | None = None,
    limit: int = 100,
):
    request = GetOptionContractsRequest(
        underlying_symbols=[underlying_symbol],
        type=ContractType.CALL if contract_type.lower() == "call" else ContractType.PUT,
        expiration_date_gte=expiration_date_gte,
        expiration_date_lte=expiration_date_lte,
        strike_price_gte=str(strike_price_gte) if strike_price_gte is not None else None,
        strike_price_lte=str(strike_price_lte) if strike_price_lte is not None else None,
        limit=limit,
    )
    response = trading_client.get_option_contracts(request)
    if hasattr(response, "option_contracts"):
        return response.option_contracts
    if hasattr(response, "data"):
        return response.data
    return response


def get_option_latest_quotes(option_data_client: OptionHistoricalDataClient, symbols: list[str]):
    if not symbols:
        return {}
    request = OptionLatestQuoteRequest(symbol_or_symbols=symbols)
    response = option_data_client.get_option_latest_quote(request)
    return response.data if hasattr(response, "data") else response


def submit_option_limit_order(
    trading_client: TradingClient,
    symbol: str,
    side: str,
    qty: int,
    limit_price: float,
    position_intent: str = "buy_to_open",
    time_in_force: str = "day",
):
    if side.lower() not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if qty <= 0:
        raise ValueError("qty must be a positive integer")
    if limit_price <= 0:
        raise ValueError("limit_price must be positive")
    intent = PositionIntent(position_intent)
    tif = TimeInForce.GTC if time_in_force.lower() == "gtc" else TimeInForce.DAY
    order = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        type=OrderType.LIMIT,
        time_in_force=tif,
        limit_price=round(float(limit_price), 2),
        position_intent=intent,
    )
    return trading_client.submit_order(order_data=order)


#: Alpaca's activity types for cash distributions. ``DIV`` is the ordinary case; the rest cover
#: capital gains, return of capital, and the withholding variants, all of which are still cash
#: moving into or out of the account and belong in an income figure.
DIVIDEND_ACTIVITY_TYPES = ("DIV", "DIVCGL", "DIVCGS", "DIVNRA", "DIVROC", "DIVTXEX", "DIVWH")


def get_account_activities(
    config: Config,
    activity_types: "tuple[str, ...] | None" = None,
    *,
    page_size: int = 100,
    after: "datetime | None" = None,
    timeout_seconds: float = 20.0,
) -> list[dict]:
    """Raw account activities for this Alpaca account.

    Lives here rather than at the call site because ``alpaca-py`` 0.43 exposes no wrapper for
    ``/v2/account/activities`` -- ``TradingClient`` has ``get_account`` and nothing for
    activities. Keeping the request in the client module means the rest of the codebase still
    talks to a brokerage, not to an endpoint.
    """
    import requests

    base = (config.alpaca_base_url or "").rstrip("/")
    if not base or not config.alpaca_api_key:
        return []
    params: dict = {"page_size": int(page_size)}
    if activity_types:
        params["activity_types"] = ",".join(activity_types)
    if after is not None:
        params["after"] = after.isoformat()
    response = requests.get(
        f"{base}/v2/account/activities",
        headers={
            "APCA-API-KEY-ID": config.alpaca_api_key,
            "APCA-API-SECRET-KEY": config.alpaca_api_secret,
        },
        params=params,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else []
