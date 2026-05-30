from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone

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

from .config import Config


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
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_api_secret,
    )


def create_option_data_client(config: Config) -> OptionHistoricalDataClient:
    return OptionHistoricalDataClient(
        api_key=config.alpaca_api_key,
        secret_key=config.alpaca_api_secret,
    )


def get_account_equity(trading_client: TradingClient) -> float:
    account = trading_client.get_account()
    return float(account.equity)


def get_positions(trading_client: TradingClient) -> dict[str, int]:
    positions = trading_client.get_all_positions()
    parsed: dict[str, int] = {}
    for position in positions:
        try:
            parsed[position.symbol] = int(float(position.qty))
        except (TypeError, ValueError):
            parsed[position.symbol] = 0
    return parsed


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

    bars_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=1000,
            feed=_resolve_data_feed(data_feed),
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
    start = end - timedelta(minutes=bar_minutes * lookback_bars * 3)
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


def submit_market_order(trading_client: TradingClient, symbol: str, side: str, qty: int):
    side = side.lower()
    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if qty <= 0:
        raise ValueError("qty must be a positive integer")
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
):
    if side.lower() not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if qty <= 0:
        raise ValueError("qty must be a positive integer")
    if limit_price <= 0:
        raise ValueError("limit_price must be positive")
    intent = PositionIntent(position_intent)
    order = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
        type=OrderType.LIMIT,
        time_in_force=TimeInForce.DAY,
        limit_price=round(float(limit_price), 2),
        position_intent=intent,
    )
    return trading_client.submit_order(order_data=order)
