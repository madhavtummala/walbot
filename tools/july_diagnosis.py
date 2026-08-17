"""Extract July 2026 dual momentum signals and eligibility for diagnosis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.registry import get_algorithm_class
from src.api.payloads.backtest import (
    _backtest_starting_equity,
    _fetch_backtest_history,
    _configured_history_providers,
)
from src.algorithms.dual_momentum.config import DualMomentumConfig
from src.algorithms.dual_momentum.proposal import analyze_universe
from src.core.config import get_config
from src.core.market_context import build_algorithm_context
from src.core.interfaces import AlgorithmRequirements
from src.execution.replay import ReplayContextSource, HistoryCache, Coverage
from src.data.duckdb_store import pooled_connections


def run():
    strategy = "dual_momentum"
    config = get_config(strategy_id=strategy)
    algorithm = get_algorithm_class(strategy).from_config(config)
    dm_config = DualMomentumConfig.from_runtime_config(config)
    requirements = algorithm.requirements(config, {})
    starting_equity = _backtest_starting_equity()

    print("Fetching bars...")
    daily_history = _fetch_backtest_history(strategy, "12m", config)

    start = pd.Timestamp("2025-07-01", tz="UTC")
    trade_dates_all = sorted(set.intersection(*(set(df.index) for df in daily_history.values())))
    in_period = [date for date in trade_dates_all if date >= start]
    trade_dates = in_period if len(in_period) >= 2 else trade_dates_all[-264:]

    # Build history cache once
    history = HistoryCache(
        sorted(daily_history),
        trade_dates,
        providers=["yfinance"],
        lookback_minutes=requirements.history_lookback_minutes,
    )

    schedule = algorithm.schedule

    print(f"\n{'='*100}")
    print(f"  JULY 2026 - DUAL MOMENTUM SIGNAL DIAGNOSIS")
    print(f"  Universe: {', '.join(dm_config.risk_on_universe)}")
    print(f"  Eligibility: MA{dm_config.etf_ma_days}, abs_ret {dm_config.etf_abs_return_days}d>{dm_config.etf_min_abs_return:+.0%}, fast_ret {dm_config.etf_fast_return_days}d>{dm_config.etf_min_fast_return:+.1%}")
    print(f"  Ranking: max_positions={dm_config.max_positions}, entry_rank_max={dm_config.entry_rank_max}, min_score={dm_config.min_base_score}")
    print(f"{'='*100}\n")

    # Filter to July 2026
    july_dates = [d for d in trade_dates if d.year == 2026 and d.month == 7]
    if not july_dates:
        print("No July 2026 dates found")
        return

    # Iterate through each date in July
    for i, trade_date in enumerate(july_dates):
        if i == 0:
            # No prior date signal on first date
            continue

        signal_date = july_dates[i - 1]
        closes = {
            symbol: float(frame.loc[trade_date, "close"])
            for symbol, frame in daily_history.items()
            if trade_date in frame.index
        }

        # Build a minimal portfolio snapshot for the context
        # We'll just use a fixed equity and empty positions for signal analysis
        coverage = Coverage()
        context = build_algorithm_context(
            config,
            requirements,
            positions={},
            equity=starting_equity,
            source=ReplayContextSource(
                signal_date, daily_history, history=history, coverage=coverage
            ),
        )

        outcome = analyze_universe(context, dm_config)
        scored = outcome["scored"]
        ranked = outcome["ranked"]
        data = outcome["data"]
        weights = outcome["weights"]

        date_str = trade_date.strftime("%Y-%m-%d %a")
        print(f"{'─'*100}")
        print(f"  {date_str}  (signal from {signal_date.strftime('%Y-%m-%d')})  Data OK: {data['data_ok']}  Coverage: {data['coverage']:.0%}")
        print(f"{'─'*100}")

        # Print all risk-on universe members with their eligibility and scores
        print(f"  {'Symbol':>6s}  {'Eligible':>8s}  {'Score':>7s}  {'Rank':>5s}  {'AbsRet':>8s}  {'FastRet':>8s}  {'AboveMA':>8s}  {'MA_Dist':>8s}  {'Reason'}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*7}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*30}")

        for symbol in dm_config.risk_on_universe:
            row = scored.get(symbol)
            if not row:
                print(f"  {symbol:>6s}  {'N/A':>8s}  {'N/A':>7s}  {'N/A':>5s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  No data")
                continue

            eligible = "YES" if row.get("eligible") else "NO"
            score = float(row.get("base_score", 0.0))
            rank = int(row.get("rank", 0)) if row.get("rank") else "-"
            abs_ret = float(row.get("abs_return", 0.0))
            fast_ret = float(row.get("fast_return", 0.0))
            above_ma = "YES" if row.get("above_moving_average") else "NO"
            ma_dist = float(row.get("ma_distance", 0.0))
            reason = str(row.get("eligibility_reason", ""))
            weight = float(weights.get(symbol, 0.0))

            marker = " <<<" if weight > 0 else ""
            print(f"  {symbol:>6s}  {eligible:>8s}  {score:>7.4f}  {str(rank):>5s}  {abs_ret:>+7.2%}  {fast_ret:>+7.2%}  {above_ma:>8s}  {ma_dist:>+7.2%}  {reason}{marker}")

        # Print defensive
        for symbol in dm_config.defensive_universe:
            row = scored.get(symbol)
            if row:
                weight = float(weights.get(symbol, 0.0))
                marker = " <<<" if weight > 0 else ""
                print(f"  {symbol:>6s}  {'DEF':>8s}  {float(row.get('base_score', 0)):>7.4f}  {'DEF':>5s}  {float(row.get('abs_return', 0)):>+7.2%}  {float(row.get('fast_return', 0)):>+7.2%}  {'YES' if row.get('above_moving_average') else 'NO':>8s}  {float(row.get('ma_distance', 0)):>+7.2%}  Defensive sleeve{marker}")

        # Print proposed weights
        held_weights = {s: w for s, w in weights.items() if w > 0}
        if held_weights:
            print(f"  Proposed weights: {', '.join(f'{s}={w:.1%}' for s, w in sorted(held_weights.items(), key=lambda x: -x[1]))}")
        else:
            print(f"  Proposed weights: [empty - defensive sleeve]")

        print()

    print(f"\n{'='*100}")
    print(f"  CONFIG OVERRIDES vs DEFAULTS")
    print(f"{'='*100}")
    print(f"  max_positions:        {dm_config.max_positions}")
    print(f"  entry_rank_max:       {dm_config.entry_rank_max}")
    print(f"  exit_rank_max:        {dm_config.exit_rank_max}")
    print(f"  min_base_score:       {dm_config.min_base_score}")
    print(f"  etf_ma_days:          {dm_config.etf_ma_days}")
    print(f"  etf_abs_return_days:  {dm_config.etf_abs_return_days}")
    print(f"  etf_min_abs_return:   {dm_config.etf_min_abs_return}")
    print(f"  etf_fast_return_days: {dm_config.etf_fast_return_days}")
    print(f"  etf_min_fast_return:  {dm_config.etf_min_fast_return}")
    print(f"  risk_on_gross_max:    {dm_config.risk_on_gross_max}")
    print(f"  volatility_tilt:      {dm_config.volatility_tilt}")
    print(f"  rerank_interval_days: {dm_config.rerank_interval_days}")
    print(f"  cooldown_after_exit:  {dm_config.cooldown_after_exit}")


if __name__ == "__main__":
    run()
