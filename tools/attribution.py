"""Explain one replay: where the return came from, and what the turnover was spent on.

The sweep ranks configurations. This says *why* one of them scored what it did, which is the
question a ranking cannot answer -- two variants can post the same return by holding different
things for different reasons, and only one of them is repeatable.

Three reports, because a momentum book has three separable problems:

**Contribution** -- which symbols and which months produced the return, and how much of the
window was spent in the defensive sleeve rather than in the market.

**Turnover by cause** -- the one the deployed configuration most needs. A daily turnover total
says how much was traded; it does not distinguish opening a new theme (a change of view) from
re-splitting an existing theme's budget between two ETFs (a sizing tweak that costs the same
spread). ``theme_allocation`` recomputes the second every session with no confirmation at all,
so it is the obvious suspect for a book turning over ~48% of itself per day, and it has never
been measured separately.

**Cost sensitivity** -- the return net of a per-trade cost, swept, so "does the edge survive
execution" is answered as a curve rather than at one assumed basis-point number.

    python -m tools.attribution --period 2023-01-01:2023-12-31
    python -m tools.attribution --period 2023-01-01:2023-12-31 --set volatility_tilt=0.0
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import pandas as pd

from src.algorithms.dual_momentum.config import DualMomentumConfig
from tools.config_sweep import Sweep, deployed_tuning

logger = logging.getLogger(__name__)


# =========================================================================================
# Turnover attribution
# =========================================================================================


CAUSES = (
    "theme_entry",
    "theme_exit",
    "intra_theme_rotation",
    "resize",
)


def _theme_map(tuning: dict[str, Any]) -> dict[str, str]:
    config = DualMomentumConfig(**{k: v for k, v in tuning.items()
                                   if k in DualMomentumConfig.__dataclass_fields__})
    return {symbol.upper(): theme for symbol, theme in (config.themes or {}).items()}


def _themes_held(held: dict[str, float], themes: dict[str, str], defensive: set[str]) -> set[str]:
    return {themes.get(symbol, symbol) for symbol in held if symbol not in defensive}


def turnover_by_cause(curve: pd.DataFrame, tuning: dict[str, Any]) -> dict[str, float]:
    """Split every dollar traded into the decision that caused it.

    Attribution is by comparing consecutive days' holdings, not by instrumenting the algorithm:
    the decision layers do not record why they moved a weight, and threading a reason through
    them would put backtest bookkeeping inside live trading code. What a day's trades did to
    the book is recoverable from the book itself.

    - ``theme_entry`` / ``theme_exit``: the set of *themes* held changed. This is the strategy
      changing its mind, and it is the turnover the strategy exists to spend.
    - ``intra_theme_rotation``: a name was opened or closed while its theme was held both days.
      Swapping QQQM for XSD inside ``us_growth`` -- same view, same spread paid.
    - ``resize``: a position that was held on both days changed size.
    """
    themes = _theme_map(tuning)
    defensive = {str(s).upper() for s in (tuning.get("defensive_universe") or [])}
    totals = dict.fromkeys(CAUSES, 0.0)
    totals["defensive"] = 0.0

    rows = list(curve.itertuples())
    for previous, row in zip(rows, rows[1:]):
        before = getattr(previous, "positions", {}) or {}
        after = getattr(row, "positions", {}) or {}
        traded = getattr(row, "trades", {}) or {}
        if not traded:
            continue
        themes_before = _themes_held(before, themes, defensive)
        themes_after = _themes_held(after, themes, defensive)

        for symbol, value in traded.items():
            value = abs(float(value))
            symbol = str(symbol).upper()
            if symbol in defensive:
                # Moving in and out of bills is the residual of every other decision rather
                # than a decision of its own, so it is counted apart rather than attributed.
                totals["defensive"] += value
                continue
            theme = themes.get(symbol, symbol)
            was_held = symbol in before
            is_held = symbol in after
            if was_held and is_held:
                totals["resize"] += value
            elif theme in themes_before and theme in themes_after:
                totals["intra_theme_rotation"] += value
            elif is_held:
                totals["theme_entry"] += value
            else:
                totals["theme_exit"] += value
    return totals


# =========================================================================================
# Contribution
# =========================================================================================


def monthly_returns(curve: pd.DataFrame) -> pd.Series:
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    return equity.resample("ME").last().pct_change().dropna()


def holding_periods(curve: pd.DataFrame) -> dict[str, float]:
    """Mean unbroken run length per symbol, in sessions, and the share of days that traded."""
    runs: list[int] = []
    open_runs: dict[str, int] = {}
    traded_days = 0
    for row in curve.itertuples():
        held = set(getattr(row, "positions", {}) or {})
        if getattr(row, "trades", {}):
            traded_days += 1
        for symbol in list(open_runs):
            if symbol not in held:
                runs.append(open_runs.pop(symbol))
        for symbol in held:
            open_runs[symbol] = open_runs.get(symbol, 0) + 1
    runs.extend(open_runs.values())
    return {
        "mean_holding_sessions": round(sum(runs) / len(runs), 1) if runs else 0.0,
        "positions_opened": len(runs),
        "pct_days_traded": round(traded_days / max(len(curve), 1), 3),
    }


def cost_curve(curve: pd.DataFrame, starting_equity: float) -> dict[int, float]:
    """Total return net of a per-side cost, at a range of basis-point assumptions.

    Applied to the turnover the run actually generated rather than simulated in the fills, so
    it is an estimate -- but it is the same estimate at every point, which is what makes the
    shape of the curve meaningful even where its level is approximate.
    """
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    gross = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    turnover = float(pd.to_numeric(curve["turnover"], errors="coerce").fillna(0.0).sum())
    return {bps: gross - turnover * (bps / 10_000.0) / max(starting_equity, 1.0)
            for bps in (0, 1, 2, 5, 10, 20)}


# =========================================================================================
# Report
# =========================================================================================


def report(curve: pd.DataFrame, tuning: dict[str, Any], starting_equity: float, label: str) -> None:
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    gross = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    turnover = float(pd.to_numeric(curve["turnover"], errors="coerce").fillna(0.0).sum())

    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"gross return {gross:+.2%}   turnover {turnover / starting_equity:.1f}x stake   "
          f"{len(curve)} sessions")

    print("\n-- turnover by cause -------------------------------------------------------")
    causes = turnover_by_cause(curve, tuning)
    total = sum(causes.values()) or 1.0
    for cause, value in sorted(causes.items(), key=lambda item: -item[1]):
        print(f"  {cause:22s} {value / starting_equity:7.1f}x stake   {value / total:6.1%}")

    print("\n-- holding behaviour -------------------------------------------------------")
    for key, value in holding_periods(curve).items():
        print(f"  {key:22s} {value}")

    print("\n-- monthly return ----------------------------------------------------------")
    for stamp, value in monthly_returns(curve).items():
        bar = "#" * int(abs(value) * 200)
        print(f"  {stamp.date().strftime('%Y-%m')}  {value:+7.2%}  {bar}")

    print("\n-- net of per-side cost ----------------------------------------------------")
    for bps, value in cost_curve(curve, starting_equity).items():
        print(f"  {bps:3d} bps   {value:+7.2%}")


def _parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """``key=value`` overrides, typed by the dataclass field rather than guessed."""
    fields = DualMomentumConfig.__dataclass_fields__
    out: dict[str, Any] = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        key = key.strip()
        if key not in fields:
            raise SystemExit(f"unknown dual_momentum key: {key}")
        kind = fields[key].type
        if "bool" in str(kind):
            out[key] = raw.strip().lower() in {"1", "true", "yes"}
        elif "int" in str(kind):
            out[key] = int(raw)
        elif "float" in str(kind):
            out[key] = float(raw)
        elif "list" in str(kind):
            out[key] = [item.strip().upper() for item in raw.split(",") if item.strip()]
        else:
            out[key] = raw
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="2023-01-01:2023-12-31")
    parser.add_argument("--set", dest="overrides", action="append", default=[],
                        help="override a tuning key, e.g. --set volatility_tilt=0.0")
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("src.core.orders", "src.brokerages.providers.paper",
                  "src.algorithms.dual_momentum.algorithm", "src.data.provider_cache", "src.connectors"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    tuning = {**deployed_tuning("dual_momentum"), **_parse_overrides(args.overrides)}
    sweep = Sweep([args.period])
    run = sweep.run("dual_momentum", args.label or "deployed", tuning, args.period)
    curve, _coverage = sweep.last_curve
    report(curve, tuning, sweep.starting_equity, f"{run.label}  /  {args.period}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
