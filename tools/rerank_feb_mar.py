"""Day-by-day with rerank markers + health signal for Feb and Mar."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.registry import get_algorithm_class
from src.api.payloads.backtest import (
    _backtest_starting_equity,
    _fetch_backtest_history,
    _configured_history_providers,
)
from src.core.config import get_config
from src.data.duckdb_store import pooled_connections
from src.execution.replay import replay

import logging
logging.basicConfig(level=logging.WARNING)


def parse_json(val):
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return val if isinstance(val, dict) else {}


def compute_health(daily_history: dict, sym: str, ts: pd.Timestamp) -> dict:
    """Compute health signal for a symbol at a given timestamp."""
    if sym not in daily_history:
        return {}
    df = daily_history[sym]
    if ts not in df.index:
        return {}
    close = df["close"]
    price = float(close.loc[ts])
    result = {"price": price}

    # 100-day MA
    lookback_100 = close.loc[:ts].tail(100)
    if len(lookback_100) >= 50:
        ma100 = float(lookback_100.mean())
        result["ma100"] = ma100
        result["vs_ma100"] = (price / ma100 - 1.0) * 100  # % above/below MA

    # 60-day return
    lookback_60 = close.loc[:ts].tail(60)
    if len(lookback_60) >= 20:
        result["ret_60d"] = (price / float(lookback_60.iloc[0]) - 1.0) * 100

    # 20-day return
    lookback_20 = close.loc[:ts].tail(20)
    if len(lookback_20) >= 5:
        result["ret_20d"] = (price / float(lookback_20.iloc[0]) - 1.0) * 100

    return result


def health_status(health: dict) -> str:
    """Return health status string."""
    parts = []
    vs_ma = health.get("vs_ma100")
    ret_60 = health.get("ret_60d")
    ret_20 = health.get("ret_20d")

    if vs_ma is not None:
        parts.append(f"MA100:{vs_ma:+.1f}%")
    if ret_60 is not None:
        parts.append(f"60d:{ret_60:+.1f}%")
    if ret_20 is not None:
        parts.append(f"20d:{ret_20:+.1f}%")

    # Health score: above MA + positive 60d + positive 20d
    score = 0
    if vs_ma is not None and vs_ma > 0:
        score += 1
    if ret_60 is not None and ret_60 > 0:
        score += 1
    if ret_20 is not None and ret_20 > 0:
        score += 1

    status = "HEALTHY" if score >= 2 else "WEAK" if score == 1 else "SICK"
    return f"[{status}] {' | '.join(parts)}"


def run():
    strategy = "rally_rotation"
    period = "2026-01-01:2026-12-31"
    starting_equity = _backtest_starting_equity()
    config = get_config(strategy_id=strategy)
    algorithm = get_algorithm_class(strategy).from_config(config)
    schedule = algorithm.schedule

    with pooled_connections(read_only=True):
        daily_history = _fetch_backtest_history(strategy, period, config)
        trade_dates_all = sorted(set.intersection(*(set(df.index) for df in daily_history.values())))
        trade_dates = [d for d in trade_dates_all if d >= pd.Timestamp("2026-01-01", tz="UTC")]

        history_df, coverage = replay(
            algorithm, config, daily_history=daily_history, trade_dates=trade_dates,
            should_run=lambda date: int(date.dayofweek) in schedule.weekdays,
            starting_equity=starting_equity,
            history_providers=_configured_history_providers(config),
        )

    if history_df.empty:
        print("ERROR: No history.")
        return

    history_df = history_df.copy()
    scale = starting_equity / float(history_df["equity"].iloc[0]) if float(history_df["equity"].iloc[0]) else 1.0
    for col in ["equity", "cash", "invested", "turnover"]:
        if col in history_df.columns:
            history_df[col] = pd.to_numeric(history_df[col], errors="coerce") * scale
    if "positions" in history_df.columns:
        history_df["positions"] = history_df["positions"].apply(
            lambda p: {s: v * scale for s, v in p.items()} if isinstance(p, dict) else {}
        )

    # Daily returns
    daily_returns = {}
    for sym in daily_history:
        df = daily_history[sym]
        daily_returns[sym] = df["close"].pct_change()

    hist = history_df.reset_index()
    hist["month"] = pd.to_datetime(hist["timestamp"]).dt.to_period("M")
    defensive = {"SGOV", "BIL", "IEF", "AGG"}

    # Simulate rerank schedule
    rerank_interval = 3
    run_index = 0
    last_rerank_run = -999
    rerank_sessions = set()

    for _, row in hist.iterrows():
        run_index += 1
        if run_index - last_rerank_run >= rerank_interval:
            rerank_sessions.add(run_index)
            last_rerank_run = run_index

    # Print Feb and Mar
    for month_filter in ["2026-02", "2026-03"]:
        mdf = hist[hist["month"] == month_filter]
        if mdf.empty:
            continue

        m_start = float(mdf.iloc[0]["equity"])
        m_end = float(mdf.iloc[-1]["equity"])
        m_return = m_end / m_start - 1.0 if m_start else 0.0

        print(f"\n{'='*130}")
        print(f"  {month_filter}  |  Month Return: {m_return:+.2%}  |  {len(mdf)} sessions")
        print(f"{'='*130}")

        prev_positions = {}
        for idx, (_, row) in enumerate(mdf.iterrows()):
            run_num = int(hist.index.get_loc(row.name)) + 1
            is_rerank = run_num in rerank_sessions

            date_str = pd.to_datetime(row["timestamp"]).strftime("%Y-%m-%d")
            dow = pd.to_datetime(row["timestamp"]).strftime("%a")
            eq = float(row["equity"])
            pos = parse_json(row.get("positions", {}))
            risk = {k: v for k, v in pos.items() if k not in defensive}

            # Get daily returns for held positions
            daily_rets = {}
            for sym in risk:
                if sym in daily_returns:
                    ts = pd.to_datetime(row["timestamp"])
                    if ts in daily_returns[sym].index:
                        ret = daily_returns[sym].loc[ts]
                        if pd.notna(ret):
                            daily_rets[sym] = float(ret)

            # Detect changes from previous day
            changed = set()
            for sym in set(list(prev_positions.keys()) + list(risk.keys())):
                old_w = prev_positions.get(sym, 0)
                new_w = risk.get(sym, 0)
                if abs(new_w - old_w) > 0.001:
                    changed.add(sym)

            # Format
            marker = ">>> RERANK" if is_rerank else "           "
            pos_str = " / ".join(f"{k}:{v/eq:.0%}" for k, v in sorted(risk.items(), key=lambda x: -x[1])) if risk else "[cash]"

            # Daily return for each position
            ret_parts = []
            for sym in sorted(risk.keys()):
                if sym in daily_rets:
                    ret_parts.append(f"{sym}:{daily_rets[sym]:+.1%}")
            ret_str = " | ".join(ret_parts) if ret_parts else ""

            # Change indicator
            change_str = ""
            if changed and prev_positions:
                added = changed - set(prev_positions.keys())
                removed = set(prev_positions.keys()) - changed
                adjusted = changed & set(prev_positions.keys())
                parts = []
                if added:
                    parts.append(f"+{','.join(sorted(added))}")
                if removed:
                    parts.append(f"-{','.join(sorted(removed))}")
                if adjusted:
                    parts.append(f"~{','.join(sorted(adjusted))}")
                change_str = " ".join(parts)

            print(f"  {marker} {date_str} {dow}  ${eq:>10,.0f}  {pos_str}")
            if ret_str:
                print(f"               Daily: {ret_str}")

            # Show health for each held position on rerank days
            if is_rerank and risk:
                for sym in sorted(risk.keys()):
                    ts = pd.to_datetime(row["timestamp"])
                    health = compute_health(daily_history, sym, ts)
                    if health:
                        hs = health_status(health)
                        print(f"                 {sym}: {hs}")

            if change_str:
                print(f"               Changed: {change_str}")

            prev_positions = risk

    print(f"\n{'='*130}\n")


if __name__ == "__main__":
    run()
