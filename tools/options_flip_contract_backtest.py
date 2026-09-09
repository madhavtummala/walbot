"""Backtest the Options Flip *contract* gates on real Sep-18 option price history.

This is the contract half of the gate-strictness work. The underlying half was a one-off
gate-necessity study (regime/level/trend gates, independently, across 1,872 opportunities) whose
findings are now in ``docs/options-flip.md``'s "Gate strictness" table rather than in a script
still carried here. This one answers the question those gates could not: of the
contract-selection and economics gates the signal view shows, which actually earn their place,
measured against **real option prices** rather than a live chain.

The historical-options problem: Schwab serves no historical chains, so a contract's delta, open
interest, spread and greeks can never be re-read for a past session. But it *does* serve an
option's own intraday **price history** by OSI symbol. That is the honest piece available, so the
methodology is:

* pin the expiry to Sep-18 2026 (a live, on-the-other-side-of-the-window expiry, so it exists
  every session and is what a >1-month contract choice would almost always land on);
* select the strike **inside the delta band** the same way the algorithm does (``target_delta``
  +- ``DELTA_TOLERANCE``, nearest strike, then liquidity), but drive the conflict by expiry only
  -- strike/delta selection is done per session, not pinned to one fixed contract;
* because deltas are not historical, they are computed from Black-Scholes on the underlying's
  spot and realised volatility as of each session -- exactly the codebase's own
  ``fill_missing_deltas`` fallback (``black_scholes_delta``);
* open interest is *not* historical. The current-day OI snapshot is carried as a "would have
  cleared the floor" proxy, and every in-band Sep-18 strike we tested clears it (they are the
  liquid, high-OI strikes), so the OI floor does not bind here. This is stated, not hidden;
* realized outcome is measured on the option price history: if the underlying pulls back to the
  entry level the option is bought at that bar's actual option price, and the position is marked
  through the option's real prices to its target hit or its hold deadline.

Run (data must be cached first by ``tools/_optcache/fetch.py`` + ``fetch_options.py``):

    STATE_DUCKDB_PATH=data/walbot.duckdb python -m tools.options_flip_contract_backtest
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import date

import pandas as pd

from src.algorithms.options_flip.algorithm import (
    OptionsFlipAlgorithm,
)
from src.algorithms.options_flip.candidates import scoring_parameters, trend_strength
from src.algorithms.options_flip.config import DELTA_TOLERANCE
from src.algorithms.options_flip.excursion import option_price_for
from src.algorithms.options_flip.indicators import average_true_range, directional_volume
from src.algorithms.options_flip.levels import conditional_levels
from src.algorithms.options_flip.lifecycle import HELD, plan_symbol
from src.algorithms.options_flip.option_band import choose_band, prepare_option_bars
from src.algorithms.options_flip.regime import bull_regime
from src.core.options import CALL, black_scholes_delta
from src.data.bars import read_history

logger = logging.getLogger("optflip_contract")

CACHE = os.path.join(os.path.dirname(__file__), "_optcache")
SYMBOLS = ["IBIT", "GLD", "SMH"]
EXPIRY = date(2026, 9, 18)
REGULAR_MINUTES = 390
DECISION_MINUTE = 10 * 60  # 10:00, first fire
TARGET_DELTA = 0.62


# ── data loading ──────────────────────────────────────────────────────────────
def _load_daily(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(CACHE, f"{symbol}_daily.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_intraday(symbol: str) -> pd.DataFrame:
    """Underlying 5m bars: the fetched Aug window merged with the store's cached history.

    The store holds ~30 prior sessions for symbols the bot has run (IBIT/GLD); without them the
    first sessions of August cannot build a ``conditional_levels`` sample (its lookback is 80
    sessions). SMH has no cached history, so it contributes only the fetched window.
    """
    frames = []
    try:
        cached = read_history(symbol, lookback_minutes=390 * 45, provider="schwab")
        if "interval_minutes" in cached and not cached.empty:
            cached = cached[cached["interval_minutes"] <= 5]
        if not cached.empty:
            frames.append(cached[["timestamp", "open", "high", "low", "close", "volume"]])
    except Exception:
        pass
    win = pd.read_csv(os.path.join(CACHE, f"{symbol}_intraday5.csv"))
    if not win.empty:
        frames.append(win[["timestamp", "open", "high", "low", "close", "volume"]])
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], utc=True)
    merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return merged.copy()


def _load_option_strikes(symbol: str) -> pd.DataFrame:
    st = pd.read_csv(os.path.join(CACHE, f"{symbol}_optstrikes.csv"))
    maps = []
    for _, row in st.iterrows():
        path = os.path.join(CACHE, row["file"])
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        if d.empty:
            continue
        d["strike"] = float(row["strike"])
        d["oi"] = int(row["oi"])
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        maps.append(d)
    if not maps:
        return pd.DataFrame()
    return pd.concat(maps, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def _market_daily_through(daily: pd.DataFrame, day: date, today_intraday: pd.DataFrame) -> pd.DataFrame:
    """Daily bars ending at ``day``'s partial bar, so gap/ATR/trend read as of that session."""
    past = daily[pd.to_datetime(daily["timestamp"], utc=True).dt.date < day].copy()
    if today_intraday.empty:
        return past
    stamps = pd.to_datetime(today_intraday["timestamp"], utc=True).dt.tz_convert("America/New_York")
    tod = today_intraday[stamps.dt.date == day]
    if tod.empty:
        return past
    row = {
        "open": float(tod["open"].astype(float).iloc[0]),
        "high": float(tod["high"].astype(float).max()),
        "low": float(tod["low"].astype(float).min()),
        "close": float(tod["close"].astype(float).iloc[-1]),
        "timestamp": pd.Timestamp(pd.Timestamp(day).tz_localize("America/New_York")),
    }
    return pd.concat([past, pd.DataFrame([row])], ignore_index=True)


# ── BS delta on the underlying's own vol (the fill_missing_deltas fallback) ────
def _realized_ann_vol(daily: pd.DataFrame, asof: date) -> float:
    sub = daily[pd.to_datetime(daily["timestamp"], utc=True).dt.date <= asof]
    r = sub["close"].astype(float).pct_change().dropna().tail(20)
    return float(r.std() * (252 ** 0.5)) if len(r) >= 2 else 0.0


def _bs_delta(spot: float, strike: float, asof: date, vol: float) -> float:
    years = max((EXPIRY - asof).days, 0) / 365.0
    return black_scholes_delta(spot, strike, years, vol, CALL)


# ── the economic "Worth trading" gate on the option's own historical price ────
# The production gate runs ``pricing.scenarios`` + ``option_change`` (a second-order Taylor
# expansion in delta/gamma/vega/theta) over the underlying entry->target move and asks whether the
# base case clears ``min_profit_per_contract``. Those greeks are not served historically, so they
# are derived from Black-Scholes at the *option's own* implied volatility, backed out of its real
# historical mark -- the same way ``fill_missing_deltas`` uses BS when the provider supplies no
# greeks. That IV is the only piece real option data supplies that a generic realized-vol delta
# cannot, and it is exactly what makes a 1-month contract "worse" than a near-dated one worth
# measuring.
def _bs_price_iv(spot: float, strike: float, years: float, price: float) -> float:
    from math import log, sqrt, erf, exp

    def _norm(x):
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def _bs_call(iv):
        sd = sqrt(max(years, 1e-6))
        d1 = (log(spot / strike) + (0.04 + 0.5 * iv * iv) * years) / (iv * sd)
        d2 = d1 - iv * sd
        return spot * _norm(d1) - strike * exp(-0.04 * years) * _norm(d2)

    lo, hi = 0.05, 4.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _bs_call(mid) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _bs_greeks(spot: float, strike: float, years: float, iv: float, price: float) -> dict:
    from math import log, sqrt, exp, erf, pi

    def _norm_pdf(x):
        return exp(-0.5 * x * x) / sqrt(2 * pi)

    def _norm(x):
        return 0.5 * (1 + erf(x / sqrt(2.0)))

    r = 0.04
    sd = sqrt(max(years, 1e-6))
    if iv <= 0:
        return {}
    d1 = (log(spot / strike) + (r + 0.5 * iv * iv) * years) / (iv * sd)
    d2 = d1 - iv * sd
    delta = _norm(d1)
    gamma = _norm_pdf(d1) / (spot * iv * sd)
    vega = spot * _norm_pdf(d1) * sqrt(years) / 100.0  # per IV point (chain convention)
    theta = -(spot * _norm_pdf(d1) * iv) / (2 * sd) - r * strike * exp(-r * years) * _norm(d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta / 365.0, "implied_vol": iv}


def _worth_trading_gate(
    spot: float, strike: float, asof: date, opt_mark: float,
    entry_underlying: float, target_underlying: float, cfg,
) -> tuple[bool, float]:
    """Run the production worth-trading logic and return (passes, per_contract_profit)."""
    from src.algorithms.options_flip.pricing import expected_profit, scenarios
    from types import SimpleNamespace
    if opt_mark <= 0 or spot <= 0:
        return False, 0.0
    years = max((EXPIRY - asof).days, 0) / 365.0
    iv = _bs_price_iv(spot, strike, years, opt_mark)
    g = _bs_greeks(spot, strike, years, iv, opt_mark)
    if not g or iv <= 0:
        return False, 0.0
    contract = SimpleNamespace(delta=g["delta"], gamma=g["gamma"], vega=g["vega"], theta=g["theta"])
    outs = scenarios(
        contract, entry_underlying=entry_underlying, target_underlying=target_underlying,
        spot=spot, config=cfg,
    )
    profit = expected_profit(outs, 1, config=cfg)
    per = profit["per_contract"]
    return bool(per >= float(cfg.min_profit_per_contract)), per


# ── realized outcome on the option's own prices ───────────────────────────────
_NO_FILL = {"outcome": "NO_FILL", "entry_opt": None, "exit_opt": None, "fill_ts": None, "exit_ts": None}


def _settle_position(
    und: pd.DataFrame, opt: pd.DataFrame, day: date, fill_ts, target: float,
    max_hold: int, sessions: list[date],
) -> dict:
    """From a confirmed fill at ``fill_ts``, chase the target or the deadline on real option bars.

    Shared by every fill model this tool has -- static limit or ratcheted -- because they differ
    only in *whether and when* the bid fills, never in what happens to the position once it has.
    Timestamps are aligned on the 5m grid shared by underlying and option history.
    """
    opt["minute"] = opt["timestamp"].dt.tz_convert("America/New_York").dt.hour * 60 + opt["timestamp"].dt.tz_convert("America/New_York").dt.minute
    opt["day"] = opt["timestamp"].dt.tz_convert("America/New_York").dt.date

    def _opt_price(at_ts, *, prefer_low: bool) -> float | None:
        rel = opt[opt["timestamp"] <= at_ts]
        if rel.empty:
            return None
        # closest option bar on/before the underlying moment, same session-day
        same = rel[rel["day"] == pd.Timestamp(at_ts).tz_convert("America/New_York").date()]
        if same.empty:
            return None
        last = same.iloc[-1]
        px = float(last["low"] if prefer_low else last["close"])
        return px

    entry_opt = _opt_price(fill_ts, prefer_low=True)
    if entry_opt is None or entry_opt <= 0:
        return dict(_NO_FILL)

    # Target horizon: today's remainder + max_hold forward sessions.
    idx = sessions.index(day)
    horizon_days = sessions[idx: idx + max(1, max_hold)]
    win = None
    exit_opt = entry_opt
    deadline_ts = None
    for hday in horizon_days:
        seg = und[und["day"] == hday]
        hit = seg[seg["high"].astype(float) >= target]
        if not hit.empty:
            hit_ts = hit.iloc[0]["timestamp"]
            if win is None or hit_ts >= fill_ts:
                win = True
                exit_ts = hit_ts
                p = _opt_price(exit_ts, prefer_low=False)
                if p:
                    exit_opt = p
                return {
                    "outcome": "WIN", "entry_opt": entry_opt, "exit_opt": exit_opt,
                    "ret": exit_opt / entry_opt - 1.0,
                    "fill_ts": fill_ts, "exit_ts": exit_ts,
                }
        # end-of-session deadline exit if this is the last hold session and target not met yet
        if hday == horizon_days[-1]:
            seg_sorted = seg.sort_values("timestamp")
            if not seg_sorted.empty:
                deadline_ts = seg_sorted.iloc[-1]["timestamp"]
    # target not reached within horizon -> exit at deadline close
    exit_ts = deadline_ts or fill_ts
    p = _opt_price(exit_ts, prefer_low=False)
    if p:
        exit_opt = p
    return {
        "outcome": "LOSS", "entry_opt": entry_opt, "exit_opt": exit_opt,
        "ret": exit_opt / entry_opt - 1.0,
        "fill_ts": fill_ts, "exit_ts": exit_ts,
    }


def _realized_pnl(
    und: pd.DataFrame, opt: pd.DataFrame, day: date, decision_minute: int,
    entry: float, target: float, max_hold: int, sessions: list[date],
) -> dict:
    """Model the Sep-18 option trade off a *static* resting limit -- one price for the whole
    session, at the level ``conditional_levels`` predicted at 10:00.

    This is the tool's original fill model. It understates real fill/win rates against
    production, which never rests a static price -- see :func:`_realized_pnl_ratchet` for the
    walk-in this omits.
    """
    und["minute"] = und["timestamp"].dt.tz_convert("America/New_York").dt.hour * 60 + und["timestamp"].dt.tz_convert("America/New_York").dt.minute
    und["day"] = und["timestamp"].dt.tz_convert("America/New_York").dt.date
    after = und[(und["day"] == day) & (und["minute"] > decision_minute)]
    if after.empty:
        return dict(_NO_FILL)
    fill_bar = after[after["low"].astype(float) <= entry]
    if fill_bar.empty:
        return dict(_NO_FILL)
    fill_ts = fill_bar.iloc[0]["timestamp"]
    return _settle_position(und, opt, day, fill_ts, target, max_hold, sessions)


def _fraction_remaining_at(minute: int) -> float:
    """:func:`~.excursion.session_fraction_remaining`, worked in minutes-of-day instead of
    timestamps -- the backtest walks 5m bars, not wall-clock moments."""
    open_minute, close_minute = 9 * 60 + 30, 16 * 60
    total = close_minute - open_minute
    remaining = close_minute - min(max(int(minute), open_minute), close_minute)
    return min(max(remaining / total, 0.0), 1.0)


def _realized_pnl_ratchet(
    und: pd.DataFrame, opt: pd.DataFrame, day: date, decision_minute: int,
    entry: float, target: float, max_hold: int, sessions: list[date], cfg,
) -> dict:
    """Model the trade off the *actual* resting order: a bid that walks toward the market as the
    session runs out, exactly as ``lifecycle._flat_or_bidding`` computes it every 5-minute run.

    The static model (:func:`_realized_pnl`) checks one fixed price against the whole session and
    materially understates fills -- production's ``entry_patience`` ratchet gives ground toward
    the mark precisely to catch the near-misses that leaves on the table. This walks the same 5m
    bars the static model does, but at each one recomputes the resting price the way production
    would at that cron tick: ``entry`` is the floor (this tool's fixed 10:00 prediction --
    ``conditional_levels`` is deliberately *not* re-run bar to bar, which would also move the
    entry level itself and confound two effects), and the limit gives ground toward the market on
    the same ``(1 - fraction_remaining) ** entry_patience`` curve production applies to premium.

    **The reference price for each tick's limit is the *prior* bar's close, never the bar being
    tested.** A candle's low is always <= its own close by construction, so pricing a bar's
    ceiling off its own close and then checking that same bar's low against it is not a
    simulation -- it is a tautology that "fills" on nearly every bar near the close, however far
    price actually was from the resting order in real time. The order the venue is holding at the
    start of an interval was priced off whatever was known *before* that interval, so that is
    what this checks against: the last cron tick's reading, not the interval's own outcome.
    """
    und["minute"] = und["timestamp"].dt.tz_convert("America/New_York").dt.hour * 60 + und["timestamp"].dt.tz_convert("America/New_York").dt.minute
    und["day"] = und["timestamp"].dt.tz_convert("America/New_York").dt.date
    today = und[und["day"] == day].sort_values("timestamp")
    decision_bar = today[today["minute"] <= decision_minute]
    after = today[today["minute"] > decision_minute]
    if decision_bar.empty or after.empty:
        return dict(_NO_FILL, limit_at_fill=None)

    patience = max(float(getattr(cfg, "entry_patience", 1.0)), 0.01)
    # The price known when the order was first placed at 10:00 -- the only ceiling available for
    # the first interval, since no cron tick has fired since then to reprice it.
    known_price = float(decision_bar.iloc[-1]["close"])
    fill_ts, limit_at_fill = None, None
    for _, bar in after.iterrows():
        given_up = (1.0 - _fraction_remaining_at(int(bar["minute"]))) ** patience
        limit = min(entry + (known_price - entry) * given_up, known_price)
        if float(bar["low"]) <= limit:
            fill_ts, limit_at_fill = bar["timestamp"], limit
            break
        known_price = float(bar["close"])
    if fill_ts is None:
        return dict(_NO_FILL, limit_at_fill=None)

    result = _settle_position(und, opt, day, fill_ts, target, max_hold, sessions)
    result["limit_at_fill"] = limit_at_fill
    return result


# ── one session through the contract gates ────────────────────────────────────
def _contract_gates(
    und: pd.DataFrame, opt: pd.DataFrame, daily: pd.DataFrame, cfg,
    day: date, decision_minute: int, price: float, spot: float, entries_target: dict,
    oi_floor: int, max_hold: int, sessions: list[date],
) -> dict:
    asof_vol = _realized_ann_vol(daily, day)
    opt_bars = opt[opt["timestamp"].dt.tz_convert("America/New_York").dt.date == day]
    chosen = None
    dte_values = []
    # strikes present on this day's option history
    strikes = sorted(opt_bars["strike"].unique())
    candidates = []
    for strike in strikes:
        delta = _bs_delta(spot, strike, day, asof_vol)
        dte = (EXPIRY - day).days
        dte_values.append(dte)
        candidates.append({"strike": strike, "delta": delta, "dte": dte, "oi": 0})
    # OI proxy from the snapshot (loaded once per strike into opt frame already)
    oi_map = dict(zip(opt["strike"], opt["oi"]))
    for c in candidates:
        c["oi"] = int(oi_map.get(c["strike"], 0))

    result = {
        "n_strikes": len(candidates),
        "min_dte": entries_target and (min(dte_values) if dte_values else 0) >= int(cfg.min_dte),
        "delta_band": False, "oi_floor": False,
        "chosen_strike": None, "chosen_delta": None, "chosen_oi": 0,
    }
    if not candidates:
        return result

    # delta band selection (strike closest to target delta within tolerance)
    in_band = [c for c in candidates if abs(c["delta"] - TARGET_DELTA) <= DELTA_TOLERANCE and c["oi"] >= oi_floor]
    result["oi_floor"] = all(c["oi"] >= oi_floor for c in candidates) or bool(in_band)
    if in_band:
        chosen = min(in_band, key=lambda c: abs(c["delta"] - TARGET_DELTA))
        result["delta_band"] = True
        result["chosen_strike"] = chosen["strike"]
        result["chosen_delta"] = round(chosen["delta"], 3)
        result["chosen_oi"] = chosen["oi"]
    return result


# ── driver ────────────────────────────────────────────────────────────────────
def analyze(cfg) -> pd.DataFrame:
    rows = []
    for symbol in SYMBOLS:
        daily = _load_daily(symbol)
        intraday = _load_intraday(symbol)
        opt = _load_option_strikes(symbol)
        if opt.empty:
            logger.warning("%s: no option history, skipping", symbol)
            continue
        # sessionize underlying
        stamps = pd.to_datetime(intraday["timestamp"], utc=True).dt.tz_convert("America/New_York")
        sframe = intraday.copy()
        sframe["ts"] = stamps
        sframe["day"] = stamps.dt.date
        sframe["minute"] = stamps.dt.hour * 60 + stamps.dt.minute
        counts = sframe.groupby("day").size()
        full = counts[counts >= 60].index
        sframe = sframe[sframe["day"].isin(full)].reset_index(drop=True)
        sessions = sorted(sframe["day"].unique())
        if len(sessions) < 3:
            continue

        # Only days with option price history (the Aug 1-27 window) are analyzable; earlier days
        # come from the underlying's cached history and have no option data to measure.
        opt_ok_days = set(opt["timestamp"].dt.tz_convert("America/New_York").dt.date.unique())
        sessions = [d for d in sessions if d in opt_ok_days]

        for idx in range(1, len(sessions) - 1):
            day = sessions[idx]
            history = sframe[sframe["day"] < day]
            today = sframe[sframe["day"] == day]
            today_cut = today[today["minute"] <= DECISION_MINUTE]
            if today_cut.empty:
                continue
            daily_through = _market_daily_through(daily, day, today_cut)
            price = float(today_cut["close"].astype(float).iloc[-1])
            session_open = float(today_cut["open"].astype(float).iloc[0])
            atr = average_true_range(daily_through, int(cfg.atr_days))
            params = scoring_parameters()
            strength_score = trend_strength(daily_through, params)
            trending = bool(strength_score >= float(cfg.min_trend_strength))
            vol_split = directional_volume(today_cut)
            levels = conditional_levels(
                history, minute=DECISION_MINUTE, price=price,
                session_open=session_open, atr=atr, config=cfg,
            )
            entry, target = float(levels["entry"]), float(levels["target"])
            if entry <= 0 or target <= 0:
                continue
            cg = _contract_gates(
                sframe, opt, daily_through, cfg, day, DECISION_MINUTE, price,
                spot=price, entries_target=True, oi_floor=int(cfg.min_open_interest),
                max_hold=int(cfg.max_hold_sessions), sessions=sessions,
            )
            # realized option P&L on the in-band chosen strike
            chosen_strike = cg.get("chosen_strike")
            outcome = {"outcome": "NO_FILL", "ret": None}
            worth_ok, worth_profit = None, None
            if chosen_strike is not None:
                opt_chosen = opt[opt["strike"] == chosen_strike]
                # option mark at the decision minute (10:00) from real price history
                omd = opt_chosen[opt_chosen["timestamp"].dt.tz_convert("America/New_York").dt.date == day]
                omd = omd[omd["timestamp"].dt.tz_convert("America/New_York").dt.hour * 60
                           + omd["timestamp"].dt.tz_convert("America/New_York").dt.minute <= DECISION_MINUTE]
                if not omd.empty:
                    mark = float(omd.sort_values("timestamp").iloc[-1]["close"])
                    worth_ok, worth_profit = _worth_trading_gate(
                        price, chosen_strike, day, mark, entry, target, cfg)
                outcome = _realized_pnl(
                    sframe, opt_chosen, day, DECISION_MINUTE, entry, target,
                    int(cfg.max_hold_sessions), sessions,
                )
                outcome_ratchet = _realized_pnl_ratchet(
                    sframe, opt_chosen, day, DECISION_MINUTE, entry, target,
                    int(cfg.max_hold_sessions), sessions, cfg,
                )
            else:
                outcome_ratchet = {"outcome": "NO_FILL", "ret": None}
            # The bull-regime gate (Trend/VWAP/gap) production requires alongside ``trending`` --
            # not run anywhere else in this function, so its own eligibility was invisible in the
            # report until asked for. Read at the same 10:00 decision point as everything else.
            bull_eligible, bull_readings, bull_checks = bull_regime(
                daily_through, today_cut, price=price, config=cfg,
            )
            vwap = float(bull_readings.get("vwap", 0.0) or 0.0)
            vwap_dist_atr = ((price - vwap) / atr) if (atr > 0 and vwap > 0) else None
            gap_atr = float(bull_readings.get("gap_atr", 0.0) or 0.0)
            # Would this session actually have armed in production? Both gates are required
            # (``direction = CALL if eligible and vol_ok and levels_ok and trending ...``) --
            # ``analyze`` computes the realized outcome regardless, on purpose (it isolates the
            # *contract* gates from the regime ones), so this flag is what lets the report filter
            # back down to the trades production would genuinely have taken.
            would_arm = bool(trending and bull_eligible)
            band_placed = band_filled = band_filled_ratchet = None
            fill_ts = outcome.get("fill_ts")
            fill_ts_ratchet = outcome_ratchet.get("fill_ts")
            if chosen_strike is not None:
                band_placed = _entry_band_snapshot(
                    daily, sframe, opt, chosen_strike, day, DECISION_MINUTE, cfg,
                )
                if fill_ts is not None:
                    ny_ts = pd.Timestamp(fill_ts).tz_convert("America/New_York")
                    fill_minute = _clamped_minute(ny_ts.hour * 60 + ny_ts.minute, cfg)
                    band_filled = _entry_band_snapshot(
                        daily, sframe, opt, chosen_strike, day, fill_minute, cfg,
                    )
                if fill_ts_ratchet is not None:
                    ny_ts_r = pd.Timestamp(fill_ts_ratchet).tz_convert("America/New_York")
                    fill_minute_r = _clamped_minute(ny_ts_r.hour * 60 + ny_ts_r.minute, cfg)
                    band_filled_ratchet = _entry_band_snapshot(
                        daily, sframe, opt, chosen_strike, day, fill_minute_r, cfg,
                    )
            rows.append({
                "symbol": symbol, "day": day.isoformat(), "price": price,
                "trending": trending, "strength_score": strength_score,
                "bull_eligible": bull_eligible, "would_arm": would_arm,
                "bull_blocking": ", ".join(
                    c.label for c in bull_checks if c.blocking
                ) or None,
                "volume_imbalance": vol_split["imbalance"],
                "vwap_dist_atr": vwap_dist_atr, "gap_atr": gap_atr,
                "entry_underlying": entry, "target_underlying": target,
                "chosen_strike": chosen_strike,
                "chosen_delta": cg.get("chosen_delta"),
                "dte_ok": cg.get("min_dte"),
                "delta_band_ok": cg.get("delta_band"),
                "oi_ok": cg.get("oi_floor"),
                "worth_ok": worth_ok, "worth_profit": worth_profit,
                "n_strikes": cg.get("n_strikes"),
                "outcome": outcome.get("outcome"),
                "opt_ret": outcome.get("ret"),
                "entry_opt": outcome.get("entry_opt"),
                "exit_opt": outcome.get("exit_opt"),
                "fill_ts": fill_ts,
                "exit_ts": outcome.get("exit_ts"),
                # The ratchet model: same session, same predicted entry, but the resting price
                # walks toward the market on ``entry_patience`` instead of sitting fixed all day.
                "outcome_ratchet": outcome_ratchet.get("outcome"),
                "opt_ret_ratchet": outcome_ratchet.get("ret"),
                "entry_opt_ratchet": outcome_ratchet.get("entry_opt"),
                "exit_opt_ratchet": outcome_ratchet.get("exit_opt"),
                "fill_ts_ratchet": outcome_ratchet.get("fill_ts"),
                "exit_ts_ratchet": outcome_ratchet.get("exit_ts"),
                "limit_at_fill_ratchet": outcome_ratchet.get("limit_at_fill"),
                "band_entry_placed": band_placed.get("entry_premium") if band_placed else None,
                "band_target_placed": band_placed.get("target_premium") if band_placed else None,
                "band_source_placed": band_placed.get("source") if band_placed else None,
                "band_entry_filled": band_filled.get("entry_premium") if band_filled else None,
                "band_target_filled": band_filled.get("target_premium") if band_filled else None,
                "band_source_filled": band_filled.get("source") if band_filled else None,
                "band_entry_filled_ratchet": (
                    band_filled_ratchet.get("entry_premium") if band_filled_ratchet else None
                ),
                "band_source_filled_ratchet": (
                    band_filled_ratchet.get("source") if band_filled_ratchet else None
                ),
            })
    return pd.DataFrame(rows)


def report(rows: pd.DataFrame) -> None:
    print(f"\nContract-gate backtest (Sep-18 2026, {rows['symbol'].nunique()} symbols, "
          f"{len(rows)} sessions / Aug 1-27)")
    print(f"contract-gate gate outcomes vs realized option outcome:")
    header = (f"{'contract gate':26s} {'rej':>5s} {'kept':>5s} {'kept WIN':>9s} "
              f"{'rej WIN':>8s} {'kept LOSS':>10s} {'kept NOFILL':>12s}  verdict")
    print(header)
    print("-" * len(header))
    gate_cols = [("would_arm", "Bull regime + trending"),
                 ("dte_ok", "DTE ≥ min_dte"), ("delta_band_ok", "Strike in delta band"),
                 ("oi_ok", "OI ≥ 100 (proxy)"), ("worth_ok", "Worth trading (≥$15/ctr)")]
    for col, label in gate_cols:
        kept = rows[rows[col].fillna(True)]
        rej = rows[rows[col] == False]  # noqa: E712
        kw = (kept["outcome"] == "WIN").mean() if len(kept) else float("nan")
        rw = (rej["outcome"] == "WIN").mean() if len(rej) else float("nan")
        kl = (kept["outcome"] == "LOSS").mean() if len(kept) else float("nan")
        kf = (kept["outcome"] == "NO_FILL").mean() if len(kept) else float("nan")
        verdict = "inert (never rejects)" if len(rej) == 0 else \
            ("necessary (rejects losers)" if rw < kw - 0.03 else "borderline")
        print(f"{label:26s} {len(rej):>5d} {len(kept):>5d} {kw:>8.0%} {rw:>7.0%} "
              f"{kl:>9.0%} {kf:>11.0%}  {verdict}")
    # realized P&L of kept trades
    kept = rows[rows["opt_ret"].notna()]
    if len(kept):
        print(f"\nFilled sessions (option bought on an entry touch): {len(kept)}")
        print(kept.groupby("outcome")["opt_ret"].agg(["count", "mean"]).round(3).to_string())
        w = kept[kept["outcome"] == "WIN"]
        if len(w):
            print(f"  WIN: mean option return {w['opt_ret'].mean():+.1%} (grade A: {w['opt_ret'].median():+.1%} median)")
    misses = rows[rows["opt_ret"].isna()]
    if len(misses):
        print(f"  NO_FILL (entry never touched): {len(misses)}")

    if len(kept):
        print(f"\nPer-trade fill/exit detail ({len(kept)} filled sessions):")
        header = (f"  {'symbol':6s} {'day':11s} {'filled at':17s} {'entry $':>8s}  "
                  f"{'exited at':17s} {'exit $':>7s}  {'ret':>7s}  outcome")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, r in kept.sort_values(["symbol", "day"]).iterrows():
            fill_local = pd.Timestamp(r["fill_ts"]).tz_convert("America/New_York") if pd.notna(r["fill_ts"]) else None
            exit_local = pd.Timestamp(r["exit_ts"]).tz_convert("America/New_York") if pd.notna(r["exit_ts"]) else None
            fill_str = fill_local.strftime("%Y-%m-%d %H:%M") if fill_local is not None else "-"
            exit_str = exit_local.strftime("%Y-%m-%d %H:%M") if exit_local is not None else "-"
            print(f"  {r['symbol']:6s} {r['day']:11s} {fill_str:17s} {r['entry_opt']:>8.2f}  "
                  f"{exit_str:17s} {r['exit_opt']:>7.2f}  {r['opt_ret']:>+6.1%}  {r['outcome']}")

        print(f"\nOption-band entry prediction: at order placement (10:00) vs. at the moment it filled:")
        header2 = (f"  {'symbol':6s} {'day':11s} {'placed $':>9s} {'src':>4s}  {'filled $':>9s} "
                   f"{'src':>4s}  {'actual fill $':>13s}  {'placed->fill Δ':>14s}")
        print(header2)
        print("  " + "-" * (len(header2) - 2))
        for _, r in kept.sort_values(["symbol", "day"]).iterrows():
            bp, bf = r.get("band_entry_placed"), r.get("band_entry_filled")
            bp_s = f"{bp:>9.2f}" if pd.notna(bp) else "      n/a"
            bf_s = f"{bf:>9.2f}" if pd.notna(bf) else "      n/a"
            delta = f"{(bf - bp):>+13.2f}" if pd.notna(bp) and pd.notna(bf) else "          n/a"
            print(f"  {r['symbol']:6s} {r['day']:11s} {bp_s}  {str(r.get('band_source_placed') or '-'):>4s}  "
                  f"{bf_s}  {str(r.get('band_source_filled') or '-'):>4s}  {r['entry_opt']:>13.2f}  {delta}")
        print("  (src: 'option' = the contract's own history; 'underlying' = the delta-translation fallback)")

    # Worth-trading, judged only against the sessions that would actually have filled.
    filled = rows[rows["opt_ret"].notna() & rows["worth_ok"].notna()]
    if len(filled):
        print("\n'Worth trading' gate, on would-have-filled sessions only:")
        print(filled.groupby("worth_ok")["outcome"].value_counts().unstack(fill_value=0).to_string())
        wt = filled[filled["worth_ok"]]
        print(f"  passed: {len(wt)} sessions, WIN {float((wt['outcome']=='WIN').mean()):.0%}, "
              f"mean opt ret {wt['opt_ret'].mean():+.1%}")
        print("  -> the worth-trading gate does not select for wins within a pinned, rich expiry.")

    _screen_candidate_gates(rows)
    _report_ratchet_comparison(rows)
    _report_full_stats(rows)


def _report_full_stats(rows: pd.DataFrame) -> None:
    """The ratcheted model's full trade log: every production-eligible session, filled or not,
    with dollar P&L (1 contract, no notional cap applied) and the band at placement vs. at fill.
    """
    base = rows[rows["would_arm"] & rows["chosen_strike"].notna()].copy()
    filled = base[base["outcome_ratchet"] != "NO_FILL"].copy()
    print(f"\nFull backtest stats, ratcheted model, {len(base)} production-eligible sessions "
          f"({len(filled)} filled, 1 contract, no notional cap):")
    if filled.empty:
        print("  nothing filled.")
        return

    filled["profit_per_contract"] = (filled["exit_opt_ratchet"] - filled["entry_opt_ratchet"]) * 100.0
    wins = filled[filled["outcome_ratchet"] == "WIN"]
    losses = filled[filled["outcome_ratchet"] == "LOSS"]
    total_profit = filled["profit_per_contract"].sum()
    print(f"  WIN {len(wins)}  LOSS {len(losses)}   total P&L ${total_profit:+,.0f}   "
          f"mean ${filled['profit_per_contract'].mean():+,.0f}/trade   "
          f"mean return {filled['opt_ret_ratchet'].mean():+.1%}")
    if len(wins):
        print(f"    WIN:  total ${wins['profit_per_contract'].sum():+,.0f}   "
              f"mean ${wins['profit_per_contract'].mean():+,.0f}   mean return {wins['opt_ret_ratchet'].mean():+.1%}")
    if len(losses):
        print(f"    LOSS: total ${losses['profit_per_contract'].sum():+,.0f}   "
              f"mean ${losses['profit_per_contract'].mean():+,.0f}   mean return {losses['opt_ret_ratchet'].mean():+.1%}")

    header = (f"  {'symbol':6s} {'day':11s} {'filled at':17s} {'entry $':>8s} {'exit $':>8s} "
              f"{'profit $':>9s} {'ret':>7s}  {'band@place':>10s} {'band@fill':>9s}  outcome")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in filled.sort_values(["symbol", "day"]).iterrows():
        fts = pd.Timestamp(r["fill_ts_ratchet"]).tz_convert("America/New_York")
        bp = r.get("band_entry_placed")
        bf = r.get("band_entry_filled_ratchet")
        bp_s = f"{bp:>10.2f}" if pd.notna(bp) else "       n/a"
        bf_s = f"{bf:>9.2f}" if pd.notna(bf) else "      n/a"
        print(f"  {r['symbol']:6s} {r['day']:11s} {fts.strftime('%Y-%m-%d %H:%M'):17s} "
              f"{r['entry_opt_ratchet']:>8.2f} {r['exit_opt_ratchet']:>8.2f} "
              f"{r['profit_per_contract']:>+9.0f} {r['opt_ret_ratchet']:>+6.1%}  "
              f"{bp_s}  {bf_s}  {r['outcome_ratchet']}")


def _report_ratchet_comparison(rows: pd.DataFrame) -> None:
    """Static resting limit vs. the ratcheted one, on the same production-eligible sessions.

    Restricted to ``would_arm`` because that is the set production would actually have armed --
    the same restriction the candidate-gate screen uses, for the same reason.
    """
    base = rows[rows["would_arm"] & rows["chosen_strike"].notna()]
    print(f"\nStatic vs. ratcheted entry, on the {len(base)} production-eligible sessions "
          f"(bull regime + trending both passed):")
    for col_prefix, label in (("", "static (fixed 10:00 price all day)"),
                               ("_ratchet", "ratcheted (walks toward the mark, entry_patience)")):
        outcomes = base[f"outcome{col_prefix}"].value_counts()
        rets = base[f"opt_ret{col_prefix}"]
        w = int(outcomes.get("WIN", 0)); l = int(outcomes.get("LOSS", 0))
        nf = int(outcomes.get("NO_FILL", 0))
        mean_ret = rets.mean() if rets.notna().any() else float("nan")
        print(f"  {label:52s}: fills {w + l:>2d}  (WIN {w}  LOSS {l})   NO_FILL {nf:>2d}   "
              f"mean ret {mean_ret:+.1%}")

    flipped = base[(base["outcome"] == "NO_FILL") & (base["outcome_ratchet"] != "NO_FILL")]
    if not flipped.empty:
        print(f"\n  Sessions the ratchet fills that the static model missed ({len(flipped)}):")
        header = (f"    {'symbol':6s} {'day':11s} {'ratchet filled at':18s} {'limit $':>8s}  "
                  f"{'ret':>7s}  outcome")
        print(header)
        print("    " + "-" * (len(header) - 4))
        for _, r in flipped.sort_values(["symbol", "day"]).iterrows():
            fts = pd.Timestamp(r["fill_ts_ratchet"]).tz_convert("America/New_York")
            ret = r["opt_ret_ratchet"]
            ret_s = f"{ret:+.1%}" if pd.notna(ret) else "n/a"
            print(f"    {r['symbol']:6s} {r['day']:11s} {fts.strftime('%Y-%m-%d %H:%M'):18s} "
                  f"{r['limit_at_fill_ratchet']:>8.2f}  {ret_s:>7s}  {r['outcome_ratchet']}")
    else:
        print("\n  The ratchet filled nothing the static model missed on this sample.")


#: Candidate signals to screen as possible new gates: (column, split point, label).
#: Each splits the sample at the point named, "favourable" side listed first. Every one of
#: these is a *reading* already computed somewhere in the algorithm (``directional_volume``,
#: ``bull_regime``'s VWAP/gap readings, ``trend_strength``) -- none is a gate today.
_CANDIDATE_SIGNALS = [
    ("volume_imbalance", 0.0, "Directional volume ≥ 0 (buyer-heavy)"),
    ("vwap_dist_atr", 0.0, "Price ≥ VWAP, in ATR"),
    ("gap_atr", 0.0, "No gap down at the open"),
    # Every ``would_arm`` row already cleared ``min_trend_strength`` (0.50σ default) to be here
    # at all -- splitting at 0.0 would be vacuous. 1.0σ asks whether trading *more strongly*
    # trending sessions, not just trending ones, discriminates further.
    ("strength_score", 1.0, "Trend strength ≥ 1.0σ (double the arming floor)"),
]


def _screen_candidate_gates(rows: pd.DataFrame) -> None:
    """Would a new gate on an existing *reading* have kept the losers out?

    Restricted to ``would_arm`` sessions -- the ones production actually would have traded --
    because screening against sessions the regime gate already refused answers a question
    nobody is asking. Sample sizes here are small (single digits per side) and printed for
    exactly that reason: a split this thin is evidence to watch, not a gate to ship.
    """
    base = rows[rows["would_arm"] & rows["opt_ret"].notna()]
    print(f"\nCandidate new gates, screened on the {len(base)} production-eligible fills "
          f"(bull regime + trending both passed):")
    if base.empty:
        print("  no production-eligible fills to screen.")
        return
    header = (f"  {'candidate signal':32s} {'favourable':>10s} {'unfav.':>7s} "
              f"{'fav WIN':>8s} {'fav mean':>9s}  {'unfav WIN':>10s} {'unfav mean':>11s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for col, split, label in _CANDIDATE_SIGNALS:
        sub = base[base[col].notna()] if col in base else base.iloc[0:0]
        if sub.empty:
            continue
        fav = sub[sub[col] >= split]
        unfav = sub[sub[col] < split]
        fw = (fav["outcome"] == "WIN").mean() if len(fav) else float("nan")
        fm = fav["opt_ret"].mean() if len(fav) else float("nan")
        uw = (unfav["outcome"] == "WIN").mean() if len(unfav) else float("nan")
        um = unfav["opt_ret"].mean() if len(unfav) else float("nan")
        print(f"  {label:32s} {len(fav):>10d} {len(unfav):>7d} "
              f"{fw:>7.0%} {fm:>+8.1%}  {uw:>9.0%} {um:>+10.1%}")
    print("  -> read the counts before the percentages: each side is single digits here, so a")
    print("     wide split is a lead to re-check on more sessions, not a result to gate on yet.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--band", action="store_true",
        help="also walk the new option-band sell logic against the option's own history",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    from src.core.config import get_config
    config = get_config()
    cfg = OptionsFlipAlgorithm(config).tuning(config)
    if not os.path.exists(os.path.join(CACHE, "IBIT_intraday5.csv")):
        print("Cache missing. Run tools/_optcache/fetch.py then tools/_optcache/fetch_options.py first.")
        return 1
    rows = analyze(cfg)
    if rows.empty:
        print("No analyzable sessions.")
        return 1
    report(rows)
    if args.band:
        band = option_band_analyze(cfg)
        report_band(band)
    return 0


def _underlying_levels(daily, sframe, day, cfg, minute: int = DECISION_MINUTE):
    """Underlying history/today cut at ``minute``, plus its entry->target band as of that point.

    ``minute`` defaults to the fixed 10:00 decision point every other gate in this tool is
    measured at, but the entry-band snapshot below needs it to move -- the production algorithm
    re-derives this same band on every 5-minute cron fire, at whatever minute is "now", so the
    band that priced the resting bid at 10:00 is not the band that was live when the bid actually
    got filled three hours later.
    """
    history = sframe[sframe["day"] < day]
    today = sframe[sframe["day"] == day]
    today_cut = today[today["minute"] <= minute]
    if today_cut.empty:
        return None
    daily_through = _market_daily_through(daily, day, today_cut)
    price = float(today_cut["close"].astype(float).iloc[-1])
    session_open = float(today_cut["open"].astype(float).iloc[0])
    atr = average_true_range(daily_through, int(cfg.atr_days))
    levels = conditional_levels(
        history, minute=minute, price=price,
        session_open=session_open, atr=atr, config=cfg,
    )
    return {"price": price, "entry": float(levels["entry"]), "target": float(levels["target"]),
            "history": history, "today": today, "today_cut": today_cut,
            "daily_through": daily_through, "levels": levels,
            "asof_ts": today_cut.sort_values("timestamp").iloc[-1]["timestamp"]}


def _option_history_through(opt: pd.DataFrame, strike: float, day: date, minute: int = DECISION_MINUTE) -> pd.DataFrame:
    """The chosen contract's own 5m bars up to (not past) ``minute`` on ``day``.

    Mirrors what the production ``option_history`` capability returns at the decision moment:
    past sessions in full, today only up to ``minute`` -- bars after it have not happened yet at
    that point in the session and would leak the rest of the day into the prediction.
    """
    sub = opt[opt["strike"] == strike]
    if sub.empty:
        return pd.DataFrame()
    local = sub.copy()
    local["ts"] = pd.to_datetime(local["timestamp"], utc=True).dt.tz_convert("America/New_York")
    local["minute"] = local["ts"].dt.hour * 60 + local["ts"].dt.minute
    local["day"] = local["ts"].dt.date
    past = local[local["day"] < day]
    today = local[(local["day"] == day) & (local["minute"] <= minute)]
    frame = pd.concat([past, today], ignore_index=True)
    return prepare_option_bars(frame)


def _clamped_minute(raw_minute: int, cfg) -> int:
    """Mirror ``algorithm._decision_minute``'s clamp, without the ``session`` dict it reads from.

    Never before 10:00 (the first fire), never past the entry cutoff -- a fill after the cutoff
    is not supposed to happen (the entry is abandoned unfilled at the close), but the clamp is
    applied anyway so this can never read a minute the production band never would have.
    """
    close_minute = 16 * 60
    first_fire = 10 * 60
    cutoff = int(close_minute - float(cfg.entry_cutoff_fraction) * 390)
    return max(min(int(raw_minute), cutoff), first_fire)


def _entry_band_snapshot(daily, sframe, opt, strike, day, minute, cfg) -> dict | None:
    """The option-band entry/target premium exactly as ``_option_band_for`` would compute it,
    as of ``minute`` on ``day`` -- the production algorithm's own snapshot at that cron fire.

    Two calls to this, at two different minutes, are what let the report contrast the band that
    priced the *resting* bid (at arming) against the band that was live the moment the bid was
    actually *filled* -- the same option history, walked forward to where the fill happened.
    """
    li = _underlying_levels(daily, sframe, day, cfg, minute=minute)
    if li is None or li["entry"] <= 0 or li["target"] <= 0:
        return None
    mark = _option_mark_at(opt, strike, day, on_or_before_ts=li["asof_ts"])
    if mark is None or mark <= 0:
        return None
    asof_vol = _realized_ann_vol(daily, day)
    delta = _bs_delta(li["price"], strike, day, asof_vol)
    under_translation = {
        "entry": option_price_for(li["entry"], underlying_now=li["price"], option_mark=mark, delta=delta),
        "target": option_price_for(li["target"], underlying_now=li["price"], option_mark=mark, delta=delta),
    }
    opt_hist = _option_history_through(opt, strike, day, minute=minute)
    session_open = mark
    if not opt_hist.empty:
        tod = opt_hist[opt_hist["day"] == day]
        if not tod.empty:
            session_open = float(tod["open"].iloc[0]) or mark
    proxy = type("_C", (), {"delta": delta, "midpoint": mark})()
    band = choose_band(
        opt_hist if not opt_hist.empty else None, minute=minute, option_mark=mark,
        session_open=session_open, config=cfg, max_hold=int(cfg.max_hold_sessions) or 1,
        underlying_translation=under_translation, contract=proxy, underlying_now=li["price"],
    )
    return {
        "entry_premium": float(band.get("entry", 0.0)),
        "target_premium": float(band.get("target", 0.0)),
        "source": str(band.get("source") or "none"),
        "sample": int(band.get("sample", 0)),
        "mark": mark,
    }


def _option_mark_at(opt, strike, day, on_or_before_ts=None):
    """The chosen contract's 5m close at the decision minute (or on/before ``on_or_before_ts``)."""
    sub = opt[opt["strike"] == strike]
    local = sub.copy()
    local["ts"] = pd.to_datetime(local["timestamp"], utc=True).dt.tz_convert("America/New_York")
    local["minute"] = local["ts"].dt.hour * 60 + local["ts"].dt.minute
    local["day"] = local["ts"].dt.date
    if on_or_before_ts is not None:
        rel = local[local["timestamp"] <= on_or_before_ts]
        if rel.empty:
            return None
        return float(rel.sort_values("timestamp").iloc[-1]["close"])
    d = local[(local["day"] == day) & (local["minute"] <= DECISION_MINUTE)]
    if d.empty:
        return None
    return float(d.sort_values("timestamp").iloc[-1]["close"])


def _option_runs_to(opt, strike, day, target_premium, start_ts=None):
    """The first option bar after 10:00 (or ``start_ts``) whose high reaches the resting sell target.

    A resting sell limit fills when the market trades *up* to it, so the condition is the option's
    high reaching the target -- not its low. Returns ``(hit_ts, hit_price)`` or None.
    """
    sub = opt[opt["strike"] == strike]
    local = sub.copy()
    local["ts"] = pd.to_datetime(local["timestamp"], utc=True).dt.tz_convert("America/New_York")
    local["minute"] = local["ts"].dt.hour * 60 + local["ts"].dt.minute
    local["day"] = local["ts"].dt.date
    seg = local[(local["day"] == day) & (local["minute"] > DECISION_MINUTE)]
    if start_ts is not None:
        seg = seg[seg["timestamp"] > start_ts]
    hit = seg[seg["high"].astype(float) >= target_premium]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return row["timestamp"], float(row["high"])


def _sell_decision(memory, *, cfg, und, opt, strike, day, start_ts, band: bool, monkey):
    """The resting sell limit for one held session, via the *production* lifecycle.

    Runs the real ``plan_symbol`` (``_held``) so the decision is whatever ships: with ``band`` the
    option's own history feeds ``target_premium`` and the bull gate is ``sell_ok``; without it the
    static underlying translation is the target and the gate is ignored.
    """
    levels = _underlying_levels(und["daily"], und["sframe"], day, cfg)
    if levels is None:
        return None, None, None
    mark = _option_mark_at(opt, strike, day, on_or_before_ts=start_ts)
    if mark is None:
        mark = memory.get("mark", 0.0)
    delta = float(memory.get("delta", 0.0) or 0.0)
    target_premium = None
    sell_ok = True
    under_translation = None
    if levels["target"] > 0 and mark > 0:
        under_translation = {
            "entry": option_price_for(levels["entry"], underlying_now=levels["price"],
                                      option_mark=mark, delta=delta),
            "target": option_price_for(levels["target"], underlying_now=levels["price"],
                                       option_mark=mark, delta=delta),
        }
    if band:
        opt_hist = _option_history_through(opt, strike, day)
        session_open = mark
        tod = opt_hist[opt_hist["day"] == day] if not opt_hist.empty and "day" in opt_hist else pd.DataFrame()
        if not tod.empty:
            session_open = float(tod["open"].iloc[0]) or mark
        proxy = type("_C", (), {"delta": delta, "midpoint": mark})()
        b = choose_band(
            opt_hist if not opt_hist.empty else None, minute=DECISION_MINUTE,
            option_mark=mark, session_open=session_open, config=cfg,
            max_hold=int(cfg.max_hold_sessions) or 1,
            underlying_translation=under_translation, contract=proxy,
            underlying_now=levels["price"],
        )
        target_premium = float(b.get("target", 0.0)) or None
        band_source = str(b.get("source") or "none")
        sell_ok = bool(bull_regime(levels["daily_through"], levels["today_cut"],
                                   price=levels["price"], config=cfg)[0])
    else:
        band_source = "static"
    mem = dict(memory)
    mem["mark"] = mark
    mem["delta"] = delta
    out = plan_symbol(
        "OPTB", memory=mem, held_contract="OSI", direction=CALL, contract=None, contracts=1,
        underlying_now=levels["price"], entry_target=0.0,
        exit_target=levels["target"], checks=[], config=cfg,
        session={"market_day": day.isoformat(), "fraction_remaining": 0.5},
        target_premium=target_premium, sell_ok=sell_ok,
    )
    resting = float(out.memory.get("target", 0.0) or 0.0)
    return resting, sell_ok, out.memory, band_source


def option_band_analyze(cfg) -> pd.DataFrame:
    """Walk the option-band sell side against real option history, vs the static translation.

    For every session whose entry would fill (underlying touches the entry after 10:00 and the
    option is bought at its own low), hold forward and decide each session's sell limit with the
    production lifecycle -- once with the option's own band (+ bull gate), once with the static
    underlying translation. The realized exit is the first option bar that reaches the resting
    limit, or the deadline close. This isolates what the option band actually *changes*.
    """
    rows = []
    for symbol in SYMBOLS:
        daily = _load_daily(symbol)
        intraday = _load_intraday(symbol)
        opt = _load_option_strikes(symbol)
        if opt.empty or intraday.empty:
            continue
        stamps = pd.to_datetime(intraday["timestamp"], utc=True).dt.tz_convert("America/New_York")
        sframe = intraday.copy()
        sframe["ts"] = stamps
        sframe["day"] = stamps.dt.date
        sframe["minute"] = stamps.dt.hour * 60 + stamps.dt.minute
        counts = sframe.groupby("day").size()
        sframe = sframe[sframe["day"].isin(counts[counts >= 60].index)].reset_index(drop=True)
        sessions = sorted(sframe["day"].unique())
        opt_ok = set(opt["timestamp"].dt.tz_convert("America/New_York").dt.date.unique())
        sessions = [d for d in sessions if d in opt_ok]
        if len(sessions) < 3:
            continue
        und = {"daily": daily, "sframe": sframe}

        for idx in range(1, len(sessions) - 1):
            day = sessions[idx]
            li = _underlying_levels(daily, sframe, day, cfg)
            if li is None or li["entry"] <= 0 or li["target"] <= 0:
                continue
            spot = li["price"]
            asof_vol = _realized_ann_vol(daily, day)
            # in-band strike, like the algorithm
            strikes = sorted(opt[opt["timestamp"].dt.tz_convert("America/New_York").dt.date == day]["strike"].unique())
            oi_map = dict(zip(opt["strike"], opt["oi"]))
            in_band = [s for s in strikes
                       if abs(_bs_delta(spot, s, day, asof_vol) - TARGET_DELTA) <= DELTA_TOLERANCE
                       and oi_map.get(s, 0) >= int(cfg.min_open_interest)]
            if not in_band:
                continue
            strike = min(in_band, key=lambda s: abs(_bs_delta(spot, s, day, asof_vol) - TARGET_DELTA))
            delta = _bs_delta(spot, strike, day, asof_vol)

            # entry: does the chosen option actually get bought? The underlying pulls back to the
            # *underlying* entry level after 10:00; at that moment the option is bought at its own
            # price (preferring its low, exactly as ``_realized_pnl`` does).
            und_after = sframe[(sframe["day"] == day) & (sframe["minute"] > DECISION_MINUTE)]
            fill_bar = und_after[und_after["low"].astype(float) <= li["entry"]]
            if fill_bar.empty:
                rows.append({"symbol": symbol, "day": day.isoformat(), "strike": strike,
                             "filled": False})
                continue
            fill_ts = fill_bar.iloc[0]["timestamp"]
            sub = opt[opt["strike"] == strike]
            local = sub.copy()
            local["ts"] = pd.to_datetime(local["timestamp"], utc=True).dt.tz_convert("America/New_York")
            local["minute"] = local["ts"].dt.hour * 60 + local["ts"].dt.minute
            local["day"] = local["ts"].dt.date
            rel = local[local["timestamp"] <= fill_ts]
            if rel.empty:
                rows.append({"symbol": symbol, "day": day.isoformat(), "strike": strike,
                             "filled": False})
                continue
            entry_opt = float(rel.sort_values("timestamp").iloc[-1]["low"])
            if entry_opt <= 0:
                rows.append({"symbol": symbol, "day": day.isoformat(), "strike": strike,
                             "filled": False})
                continue

            # The sell side is evaluated from the session *after* the fill, as production does on
            # each subsequent run -- the entry day establishes the position, it does not reprice
            # its own exit at 10:00 before the fill has even happened.
            horizon = sessions[idx + 1: idx + 1 + max(1, int(cfg.max_hold_sessions) or 1)]
            for band in (True, False):
                memory = {"state": HELD, "direction": CALL, "contracts": 1,
                          "fill_price": entry_opt, "bid": entry_opt, "mark": entry_opt,
                          "delta": delta, "sessions_held": 1, "target": 0.0, "stop": 0.0,
                          "filled_day": day.isoformat()}
                outcome_kind = "WIN"; exit_px = entry_opt; found = False
                last_src = "static"
                for h, hday in enumerate(horizon):
                    resting, sell_ok, mem, band_source = _sell_decision(
                        memory, cfg=cfg, und=und, opt=opt, strike=strike, day=hday,
                        start_ts=None, band=band, monkey=None,
                    )
                    last_src = band_source
                    if resting is None or resting <= 0:
                        break
                    hit = _option_runs_to(opt, strike, hday, resting, start_ts=None)
                    if hit is not None:
                        outcome_kind = "WIN"
                        exit_px = hit[1]
                        found = True
                        break
                    memory = dict(mem)
                    memory["sessions_held"] = int(memory.get("sessions_held", 0) or 0) + 1
                    # deadline: exit at the last horizon session's close if never reached
                    if h == len(horizon) - 1:
                        last = opt[opt["strike"] == strike]
                        ll = last.copy()
                        ll["ts"] = pd.to_datetime(ll["timestamp"], utc=True).dt.tz_convert("America/New_York")
                        ll["day"] = ll["ts"].dt.date
                        seg = ll[ll["day"] == hday].sort_values("timestamp")
                        if not seg.empty:
                            exit_px = float(seg.iloc[-1]["close"])
                            outcome_kind = "LOSS"
                            found = True
                if not found:
                    outcome_kind = "LOSS"  # never resolved -> treat as deadline loss at entry
                    exit_px = entry_opt
                ret = (exit_px / entry_opt - 1.0) if entry_opt > 0 else 0.0
                rows.append({
                    "symbol": symbol, "day": day.isoformat(), "strike": strike,
                    "filled": True, "band": band, "entry_opt": entry_opt,
                    "outcome": outcome_kind, "exit_opt": exit_px, "opt_ret": ret,
                    "band_source": last_src,
                })
    return pd.DataFrame(rows)


def report_band(df: pd.DataFrame) -> None:
    """Contrast the option-band sell against the static translation, on the same fills."""
    print(f"\nOption-band sell walk ({len(df[df['filled']])} filled positions, "
          f"{df['symbol'].nunique()} symbols, Aug 1-27)")
    filled = df[df["filled"]]
    if filled.empty:
        print("  no session filled an entry -> nothing to compare.")
        return
    for band, label in ((True, "option band + bull gate"), (False, "static translation")):
        sub = filled[filled["band"] == band]
        w = (sub["outcome"] == "WIN").sum()
        l = (sub["outcome"] == "LOSS").sum()
        mean = sub["opt_ret"].mean()
        med = sub["opt_ret"].median()
        print(f"  {label:26s}: {len(sub)}   WIN {w}  LOSS {l}   mean {mean:+.1%}  median {med:+.1%}")
        if band and "band_source" in sub:
            src = sub["band_source"].value_counts()
            print(f"    exit-deciding band source: "
                  f"{', '.join(f'{k}={v}' for k, v in src.items())}")

    # paired: the same entry, one exit decided by each mode
    piv = filled.pivot_table(index=["symbol", "day"], columns="band",
                             values=["outcome", "opt_ret", "entry_opt"], aggfunc="first")
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    b, s = True, False
    piv["delta"] = piv[f"opt_ret_{b}"] - piv[f"opt_ret_{s}"]
    piv["changed"] = piv[f"outcome_{b}"] != piv[f"outcome_{s}"]
    print(f"\n  Paired on the same fills ({len(piv)} sessions), option-band vs static:")
    print(f"    sessions where the two modes disagree on WIN/LOSS: {int(piv['changed'].sum())}")
    print(f"    option-band mean minus static mean: {piv['delta'].mean():+.1%}")
    pivs = piv.sort_values("day")
    header = f"    {'symbol':6s} {'day':12s} {'band_ret':>9s} {'static_ret':>11s} {'delta':>8s}"
    print(header)
    print("    " + "-" * (len(header) - 4))
    for (sym, day), row in pivs.iterrows():
        print(f"    {sym:6s} {str(day):12s} {row[f'opt_ret_{b}']:>+8.1%} "
              f"{row[f'opt_ret_{s}']:>+10.1%} {row['delta']:>+7.1%} "
              f"{'<' if row['changed'] else ''}")
    return


if __name__ == "__main__":
    raise SystemExit(main())
