from __future__ import annotations
import pandas as pd


BAR_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_LONG_MA_DAYS = 200


def fetch_daily_bars(
    symbols: list[str],
    lookback_days: int,
    ma_days: int = DEFAULT_LONG_MA_DAYS,
    extra_buffer_days: int = 250,
    data_client=None,
    force_refresh: bool = False,
    include_latest: bool = False,
    config=None,
    start_date=None,
    end_date=None,
) -> dict[str, pd.DataFrame]:
    """Fetch enough daily bars for a momentum signal and moving average calculation.

    One path, through the shared bar store. There used to be three: this one, a JSON cache in
    ``app_state`` keyed by data feed, and a direct provider call. Every caller passed
    ``config``, so only this branch ever ran -- the other two were a second cache
    implementation and a bypass, both maintained and neither reachable.
    """
    from ..connectors import fetch_eod_market_bars

    bars_by_symbol = fetch_eod_market_bars(
        symbols=symbols,
        config=config,
        lookback_bars=lookback_days + ma_days + extra_buffer_days,
        force_refresh=force_refresh,
        data_client=data_client,
        start_date=start_date,
        end_date=end_date,
    )

    normalized: dict[str, pd.DataFrame] = {}
    for symbol, df in bars_by_symbol.items():
        if not df.empty:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.sort_values("timestamp").reset_index(drop=True)
        normalized[symbol] = df
    if include_latest:
        from src.core.config import get_config
        from ..connectors import append_latest_quotes_to_bars, load_latest_prices

        runtime_config = config or get_config()
        quotes = load_latest_prices(
            symbols,
            runtime_config,
            data_client=data_client,
            force_refresh=force_refresh,
        )
        normalized = append_latest_quotes_to_bars(normalized, quotes)

    # Last, so the appended live quote is inside the compounding rather than stranded after
    # it. ``close`` stays exactly what the market printed -- this only adds a derived column
    # for the signal layer, and execution keeps sizing and filling against the raw price.
    from ..data.bars import attach_total_return

    return {
        symbol: attach_total_return(frame, symbol) for symbol, frame in normalized.items()
    }
