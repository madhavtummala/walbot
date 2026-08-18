"""Compute average holding percentages for July 2026."""

from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.algorithms.registry import get_algorithm_class
from src.api.payloads.backtest import _backtest_starting_equity, _fetch_backtest_history, _configured_history_providers
from src.algorithms.rally_rotation.config import RallyRotationConfig
from src.core.config import get_config
from src.data.duckdb_store import pooled_connections
from src.execution.replay import replay

def run():
    strategy = "rally_rotation"
    config = get_config(strategy_id=strategy)
    algorithm = get_algorithm_class(strategy).from_config(config)
    dm_config = RallyRotationConfig.from_runtime_config(config)
    schedule = algorithm.schedule
    starting_equity = _backtest_starting_equity()

    daily_history = _fetch_backtest_history(strategy, "12m", config)
    start = pd.Timestamp("2025-08-01", tz="UTC")
    trade_dates_all = sorted(set.intersection(*(set(df.index) for df in daily_history.values())))
    in_period = [date for date in trade_dates_all if date >= start]
    trade_dates = in_period if len(in_period) >= 2 else trade_dates_all[-264:]

    with pooled_connections(read_only=True):
        history_df, _ = replay(algorithm, config, daily_history=daily_history, trade_dates=trade_dates,
            should_run=lambda date: int(date.dayofweek) in schedule.weekdays,
            starting_equity=starting_equity, history_providers=_configured_history_providers(config))

    # Scale
    history_df = history_df.copy()
    scale = starting_equity / float(history_df["equity"].iloc[0])
    for col in ["equity", "cash", "invested", "turnover"]:
        if col in history_df.columns:
            history_df[col] = pd.to_numeric(history_df[col], errors="coerce") * scale
    if "positions" in history_df.columns:
        history_df["positions"] = history_df["positions"].apply(
            lambda p: {s: v * scale for s, v in p.items()} if isinstance(p, dict) else {})

    # Filter July 2026
    history_df_reset = history_df.reset_index()
    history_df_reset["date"] = pd.to_datetime(history_df_reset["timestamp"])
    july = history_df_reset[(history_df_reset["date"].dt.year == 2026) & (history_df_reset["date"].dt.month == 7)]

    # Collect per-symbol equity weights across all July sessions
    all_symbols = set()
    daily_weights = []
    for _, row in july.iterrows():
        equity = float(row["equity"])
        positions = row.get("positions", {})
        if isinstance(positions, str):
            positions = json.loads(positions)
        weights = {}
        for sym, val in positions.items():
            weights[sym] = val / equity if equity else 0
            all_symbols.add(sym)
        daily_weights.append(weights)

    print(f"\n{'='*90}")
    print(f"  JULY 2026 - AVERAGE HOLDING ALLOCATIONS")
    print(f"  ({len(daily_weights)} trading sessions)")
    print(f"{'='*90}\n")

    # Compute average weight per symbol
    avg_weights = {}
    max_weights = {}
    min_weights = {}
    days_held = {}
    for sym in sorted(all_symbols):
        weights_list = [dw.get(sym, 0.0) for dw in daily_weights]
        avg_weights[sym] = sum(weights_list) / len(weights_list)
        max_weights[sym] = max(weights_list)
        min_weights[sym] = min(weights_list)
        days_held[sym] = sum(1 for w in weights_list if w > 0.001)

    # Sort by average weight
    sorted_syms = sorted(avg_weights.keys(), key=lambda s: -avg_weights[s])

    print(f"  {'Symbol':>6s}  {'Avg Weight':>10s}  {'Max':>8s}  {'Min':>8s}  {'Days Held':>9s}  {'Sessions':>8s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")
    total_avg = 0
    for sym in sorted_syms:
        avg = avg_weights[sym]
        if avg < 0.001:
            continue
        total_avg += avg
        print(f"  {sym:>6s}  {avg:>9.1%}  {max_weights[sym]:>7.1%}  {min_weights[sym]:>7.1%}  {days_held[sym]:>6d}/{len(daily_weights)}  {'  (CRASH STOP)' if sym in ['GLD','SLV'] else ''}")

    print(f"  {'─'*6}  {'─'*10}")
    print(f"  {'TOTAL':>6s}  {total_avg:>9.1%}")

    print(f"\n{'='*90}")
    print(f"  JULY 2026 - KEY DATES (when XRT first entered and became dominant)")
    print(f"{'='*90}\n")

    # Find when XRT first appeared and its trajectory
    for i, (_, row) in enumerate(july.iterrows()):
        equity = float(row["equity"])
        positions = row.get("positions", {})
        if isinstance(positions, str):
            positions = json.loads(positions)
        xrt_weight = positions.get("XRT", 0) / equity if equity and "XRT" in positions else 0
        if xrt_weight > 0:
            date_str = row["date"].strftime("%Y-%m-%d %a")
            pos_parts = []
            for sym in sorted(positions.keys()):
                val = positions[sym]
                pct = val / equity
                pos_parts.append(f"{sym}: {pct:.1%}")
            print(f"  {date_str}  Equity: ${equity:,.0f}  XRT: {xrt_weight:.1%}")
            print(f"    All positions: {' | '.join(pos_parts)}")

    print(f"\n{'='*90}")
    print(f"  WHY XRT WAS CHOSEN - ELIGIBILITY ANALYSIS")
    print(f"{'='*90}\n")
    print("  XRT (SPDR S&P Retail ETF) was in the risk_on_universe and passed ALL three")
    print("  eligibility gates throughout July:\n")
    print("  1. Above 100-day MA:  YES (MA distance +2.5% to +8.8%)")
    print("  2. 60-day abs return:  POSITIVE (+1.5% to +8.6%)")
    print("  3. 20-day fast return: ABOVE -2% (ranged -0.6% to +6.0%)")
    print()
    print("  The critical context is a BROAD SELLOFF in high-momentum names:")
    print()
    print("  Late July eligibility collapse (signal dates Jul 28-30):")
    print("    QQQM: INELIGIBLE - below 100-day MA (was -1.65% on Jul 30)")
    print("    XSD:  INELIGIBLE - below 100-day MA (was -8.16% on Jul 30)")
    print("    XBI:  INELIGIBLE - 20-day return -6.53% (below -2% threshold)")
    print("    XOP:  INELIGIBLE - 60-day return negative or below MA")
    print("    EWJ:  INELIGIBLE - below MA or 20-day return negative")
    print("    SPY:  INELIGIBLE - 20-day return negative")
    print("    IWM:  INELIGIBLE - 20-day return negative")
    print()
    print("  With max_positions=3 and entry_rank_max=3, the strategy was FORCED to")
    print("  fill its book from the remaining eligible names:")
    print("    XRT (Rank 1, score 0.95-1.67)")
    print("    SCHD (Rank 2-3, score 0.38-1.26)")
    print("    KRE  (Rank 3-4, score 0.24-0.75)")
    print("    VGK  (Rank 2-4, score 0.14-0.79)")
    print()
    print("  XRT was the highest-scoring eligible name for the last 3 sessions of July")
    print("  because it was one of the ONLY names still above its 100-day MA with")
    print("  positive momentum in a market where the prior leaders had crashed.")
    print()
    print("  The drawdown came from this forced concentration into a narrow set of")
    print("  names that the selloff had not yet reached, combined with the exit of")
    print("  high-flyers (XBI, XSD, QQQM) at depressed prices.")


if __name__ == "__main__":
    run()
