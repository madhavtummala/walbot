"""Replay today's Options Flip decisions for one held symbol, from real Schwab data.

The walk-forward harness (``options_flip_walk_forward.py``) replays a whole month from cached
history because Schwab serves no historical option chains -- but *today* is not history yet.
Every fetch this tool makes is the same one production makes, at the same 5-minute cadence, and
because the day it is asking about is still in progress, ``fetch_option_price_history`` returns
the contract's own real intraday bars rather than a cache -- no synthetic chain, no Black-Scholes
proxy, the actual thing that happened.

This drives ``_held()``/``_refresh_held()``/``_sell_ok()``/``_option_band_for()`` directly rather
than the whole ``plan()`` -- a held position never re-selects its contract, so nothing here needs
the chain, only the underlying's bars and the contract's own price history, both real.

Run, from the fill onward for a position opened today:

    STATE_DUCKDB_PATH=data/walbot.duckdb python -m tools.options_flip_today_replay \
        --symbol USO --osi "USO   260916C00142000" --fill-price 8.85 --fill-time "11:23"
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime

import pandas as pd

from src.algorithms.options_flip import algorithm as alg
from src.algorithms.options_flip import lifecycle as lc
from src.algorithms.options_flip.excursion import option_price_for
from src.core.config import get_config
from src.core.options import CALL

logger = logging.getLogger("optflip_today_replay")

TICKS = list(range(10 * 60, 16 * 60 + 1, 5))
MARKET_TZ = "America/New_York"


def _fmt_minute(minute: int) -> str:
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _bar_at_or_before(bars: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    if bars is None or bars.empty:
        return None
    upto = bars[bars["timestamp"] <= ts]
    return upto.iloc[-1] if not upto.empty else None


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="Underlying, e.g. USO")
    parser.add_argument("--osi", required=True, help="Option contract's OSI symbol, as Schwab spells it")
    parser.add_argument("--fill-price", type=float, required=True)
    parser.add_argument("--fill-time", required=True, help="HH:MM, market time, when the entry filled")
    parser.add_argument("--delta", type=float, default=None, help="Defaults to the live chain's own delta")
    args = parser.parse_args()

    config = get_config()
    cfg = alg.OptionsFlipAlgorithm(config).tuning(config)
    symbol = args.symbol.upper()
    osi = args.osi.upper()

    from src.connectors.market.schwab_options import fetch_option_chain, fetch_option_price_history
    from src.connectors.service import fetch_market_history
    from src.data import fetch_daily_bars

    today = date.today()
    daily = fetch_daily_bars([symbol], cfg.required_daily_bars, ma_days=cfg.regime_slow_ma_days,
                              config=config).get(symbol)
    intraday_all = fetch_market_history(
        [symbol], config, lookback_minutes=cfg.required_intraday_minutes, bar_minutes=5,
    ).get(symbol)
    opt_bars = fetch_option_price_history(config, osi, interval_minutes=5, lookback_days=5)
    print(f"fetched: {len(daily) if daily is not None else 0} daily bars, "
          f"{len(intraday_all) if intraday_all is not None else 0} intraday bars, "
          f"{len(opt_bars)} option bars")

    delta = args.delta
    if delta is None:
        try:
            chain = fetch_option_chain(config, symbol, as_of=today)
            match = next((c for c in chain if c.osi_symbol.replace(" ", "") == osi.replace(" ", "")), None)
            delta = float(match.delta) if match else float(cfg.target_delta)
        except Exception as exc:
            logger.warning("Could not fetch live chain for delta, using target_delta: %s", exc)
            delta = float(cfg.target_delta)
    print(f"seeded delta={delta:.3f}")

    fill_hour, fill_minute = (int(x) for x in args.fill_time.split(":"))
    fill_minute_of_day = fill_hour * 60 + fill_minute

    intraday_all = intraday_all.copy()
    intraday_all["ts"] = pd.to_datetime(intraday_all["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)
    opt_bars = opt_bars.copy()
    if not opt_bars.empty:
        opt_bars["ts"] = pd.to_datetime(opt_bars["timestamp"], utc=True).dt.tz_convert(MARKET_TZ)

    memory: dict = {
        "state": lc.HELD, "direction": CALL, "contracts": 1, "contract": osi,
        "fill_price": args.fill_price, "bid": args.fill_price, "delta": delta,
        "filled_day": today.isoformat(),
    }

    session_dict = {"market_day": today.isoformat()}
    prev_target: float | None = None
    prev_sell_ok: bool | None = None
    for minute in TICKS:
        if minute < fill_minute_of_day:
            continue
        ts = pd.Timestamp.combine(today, datetime.min.time()).tz_localize(MARKET_TZ) \
            + pd.Timedelta(hours=minute // 60, minutes=minute % 60)
        if ts > pd.Timestamp.now(tz=MARKET_TZ):
            break

        today_bars = intraday_all[intraday_all["ts"] <= ts]
        if today_bars.empty:
            continue
        underlying_now = float(today_bars.iloc[-1]["close"])

        opt_seen = opt_bars[opt_bars["ts"] <= ts] if not opt_bars.empty else opt_bars
        mark_row = _bar_at_or_before(opt_seen, ts)
        mark = float(mark_row["close"]) if mark_row is not None else memory.get("mark", args.fill_price)

        session = {**session_dict, "fraction_remaining": max(0.0, (16 * 60 - minute) / (16 * 60 - 10 * 60))}

        # -- refresh, exactly as production does --
        fake_context = type("_Ctx", (), {"latest_prices": {osi: mark}})()
        memory = alg._refresh_held(memory, osi, fake_context, session, cfg)

        _hist, today_only = alg._split_sessions(
            pd.concat([intraday_all[intraday_all["ts"].dt.date < today], today_bars]), today.isoformat()
        )
        sell_ok = alg._sell_ok(daily, today_only, underlying_now, cfg)

        seen_all = pd.concat([intraday_all[intraday_all["ts"].dt.date < today], today_bars])
        exit_level = alg._held_exit_target(seen_all, session, cfg, underlying_now, daily)
        under_translation = (
            {"entry": 0.0, "target": option_price_for(exit_level, underlying_now=underlying_now,
                                                            option_mark=mark, delta=delta)}
            if (mark > 0 and exit_level > 0) else None
        )

        class _FakeContext:
            extra = {"option_history": lambda o, _bars=opt_seen: _bars.drop(columns=["ts"], errors="ignore")}

        band = alg._option_band_for(osi, _FakeContext(), cfg, session, under_now=underlying_now,
                                     delta=delta, mark=mark, under_translation=under_translation)

        outcome = lc._held(
            symbol, memory, osi, 1, underlying_now, exit_level, [], cfg, session,
            target_premium=float(band.get("target", 0.0)) or None, sell_ok=sell_ok,
        )
        memory = {**memory, **outcome.memory}

        target_order = next((o for o in outcome.orders if o.key.endswith(":target")), None)
        target_price = float(target_order.request.limit_price) if target_order else None

        changed = (target_price != prev_target) or (sell_ok != prev_sell_ok)
        if changed or minute == fill_minute_of_day:
            print(f"{_fmt_minute(minute)}  underlying={underlying_now:.2f}  mark={mark:.2f}  "
                  f"sell_ok={sell_ok!s:5s}  band_source={band.get('source')}  "
                  f"band_target={band.get('target', 0.0):.2f}  -> target_order={target_price}")
        prev_target = target_price
        prev_sell_ok = sell_ok

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
