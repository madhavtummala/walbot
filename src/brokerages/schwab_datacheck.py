"""Decide whether Schwab can replace Alpaca/yfinance as the market data feed.

Answers the two questions that block the switch:

1. Are Schwab's daily closes dividend-adjusted? Alpaca's are (Adjustment.ALL), which is why
   SGOV shows its ~3.8% yield instead of a flat sawtooth. If Schwab returns raw closes,
   switching the EOD provider would silently delete every dividend from every backtest.
2. How far back does minute-level intraday actually go? yfinance caps at 60 days, which is what
   limits the backtest windows today. If Schwab reaches further, that is the real prize.

Run it where the Schwab refresh token already lives, so only one machine ever touches the
credential:

    docker exec walbot python -m src.brokerages.schwab_datacheck
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.connectors.service import ProviderUnavailable, fetch_schwab_eod_bars, fetch_schwab_intraday_bars
from src.core.config import get_config
from src.data import fetch_daily_bars

DIVIDEND_PAYERS = ["SGOV", "HYG", "SCHD"]
INTRADAY_PROBES = ["QQQM", "XSD"]


def _closes(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame.get("close", pd.Series(dtype=float)), errors="coerce").dropna()


def check_dividend_adjustment(config) -> None:
    print("=== 1. dividend adjustment (Schwab vs Alpaca, ~1 year change)")
    schwab = fetch_schwab_eod_bars(DIVIDEND_PAYERS, config, lookback_bars=260, force_refresh=True)
    alpaca = fetch_daily_bars(
        DIVIDEND_PAYERS, lookback_days=260, ma_days=0, extra_buffer_days=5,
        include_latest=True, config=config,
    )
    print(f"{'sym':<6}{'schwab':>10}{'alpaca':>10}{'gap':>9}   verdict")
    for symbol in DIVIDEND_PAYERS:
        s, a = _closes(schwab.get(symbol, pd.DataFrame())), _closes(alpaca.get(symbol, pd.DataFrame()))
        if len(s) < 200 or len(a) < 200:
            print(f"{symbol:<6}  not enough history to compare (schwab={len(s)}, alpaca={len(a)})")
            continue
        s_ret, a_ret = s.iloc[-1] / s.iloc[-252] - 1, a.iloc[-1] / a.iloc[-252] - 1
        gap = a_ret - s_ret
        verdict = "ADJUSTED (matches)" if abs(gap) < 0.01 else "RAW -- dividends missing"
        print(f"{symbol:<6}{s_ret:>9.2%}{a_ret:>10.2%}{gap:>9.2%}   {verdict}")


def check_intraday_depth(config) -> None:
    print("\n=== 2. intraday depth at 5m (yfinance caps at ~60 days, and at 15m)")
    for symbol in INTRADAY_PROBES:
        # Roughly 60, 190, and 460 sessions, stated the way algorithms state horizons.
        for requested in (24000, 75000, 180000):
            try:
                bars = fetch_schwab_intraday_bars(
                    [symbol], config, lookback_minutes=requested, bar_minutes=5,
                    start_date=datetime.now(timezone.utc) - timedelta(days=math.ceil(requested / 390) + 10),
                    force_refresh=True,
                )
            except Exception as error:  # noqa: BLE001 - the failure mode is the answer
                print(f"  {symbol} @ {requested:>7}m -> {type(error).__name__}: {str(error)[:90]}")
                continue
            frame = bars.get(symbol, pd.DataFrame())
            if frame.empty:
                print(f"  {symbol} @ {requested:>7}m -> empty")
                continue
            stamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
            span = (stamps.max() - stamps.min()).days
            print(f"  {symbol} @ {requested:>7}m -> {len(frame):>5} rows spanning {span:>4} days "
                  f"({stamps.min().date()} to {stamps.max().date()})")


def main() -> int:
    config = get_config()
    try:
        check_dividend_adjustment(config)
        check_intraday_depth(config)
    except ProviderUnavailable as error:
        # Run this where the refresh token already lives. Copying the credential to a second
        # machine risks a rotation on one instance retiring the other's token.
        print(f"Schwab is not authorized on this machine: {error}")
        print("Run it where consent was completed, e.g. on the deployment host:")
        print("  docker exec walbot python -m src.brokerages.schwab_datacheck")
        return 2
    print("\nIf section 1 says ADJUSTED and section 2 beats 60 days, Schwab should lead both provider orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
