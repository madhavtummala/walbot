"""Replay one algorithm under many configurations and rank what came out.

An operator tool, deliberately outside ``src/``: it drives the production replay engine rather
than reimplementing anything, so a result here is the same arithmetic the dashboard's backtest
tab performs. The last time this repo grew a "config optimizer" it was a second, drifting
implementation of dual momentum; this one has no strategy logic in it at all.

Two things to keep in mind when reading the output.

**No transaction costs.** The replay fills at the close with no commission, spread or slippage,
even though ``config.transaction_cost_bps`` exists. A configuration that churns is therefore
flattered. Every row reports turnover and a post-hoc cost drag at 1 and 5 basis points so the
churn can be priced back in; ``net_return_5bps`` is the number to rank on if you care.

**One window is not evidence.** A ranking over a single period is a statement about that
period. Run the finalists over more than one window and prefer what survives all of them.

    python -m tools.config_sweep --algorithm dual_momentum --stage axes --period 12m
    python -m tools.config_sweep --algorithm both --stage finalists --period 12m,6m,24m
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
import time
from typing import Any, Callable, Iterable

import pandas as pd

from src.algorithms.registry import get_algorithm_class
from src.api.payloads.backtest import (
    _backtest_starting_equity,
    _configured_history_providers,
    _fetch_backtest_history,
    _period_row_count,
    _period_start,
)
from src.common.config_utils import tuning_section
from src.core.config import get_config
from src.execution.metrics import calculate_performance_metrics
from src.data.duckdb_store import pooled_connections
from src.execution.replay import HistoryCache, replay

logger = logging.getLogger(__name__)


# =========================================================================================
# Candidate universes
# =========================================================================================
# Named rather than generated, because a universe is a judgement about what the strategy is
# allowed to be -- and because the cross-sectional z-score means adding a name changes every
# other name's score, so these are genuinely different strategies, not a size knob.

#: What is deployed for dual_momentum now: the thematic set plus broad US and quality/value
#: exposure, the exploration's ``wide`` -- less thematic, more index-like.
CURRENT = ["QQQM", "XSD", "XBI", "XOP", "KRE", "XRT", "IJH", "IEMG", "EWJ", "VGK", "GLD", "IBIT", "USO",
           "SPY", "IWM", "RSP", "QUAL", "VTV", "SCHD", "SLV"]

#: Alias for the ``wide`` stage functions, which take the deployed set as their starting point.
WIDE = CURRENT

#: Everything tradable that is a plausible long, including the low-beta and income sleeves.
BROAD = WIDE + ["INDA", "ASHR", "SCZ", "IWO", "HYG", "KBE", "PBW", "USMV", "JEPQ"]

#: Thematic only -- the high-dispersion end, to test whether breadth or dispersion is what pays.
NARROW = ["QQQM", "XSD", "XBI", "XOP", "KRE", "XRT", "IBIT", "USO", "GLD"]

UNIVERSES = {"narrow": NARROW, "current": CURRENT, "broad": BROAD}

DEFENSIVE_SETS = {
    "sgov": ["SGOV"],
    "sgov_gld": ["SGOV", "GLD"],
    "bil_tlt": ["BIL", "TLT"],
    "sgov_tlt_gld": ["SGOV", "TLT", "GLD"],
}


# =========================================================================================
# Running one configuration
# =========================================================================================


@dataclasses.dataclass
class Run:
    algorithm: str
    label: str
    period: str
    tuning: dict[str, Any]
    metrics: dict[str, float]

    def row(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm, "label": self.label, "period": self.period, **self.metrics}


class Sweep:
    """Holds the market data every run shares, so the cost of a sweep is the scoring alone."""

    def __init__(self, periods: list[str], starting_equity: float | None = None) -> None:
        self.base = get_config()
        self.providers = _configured_history_providers(self.base)
        # Overridable because a DCA plan has an absolute dollar size: the deployed plan wants
        # $2,675/month, which exhausts the default $10,000 stake in under four months. Every
        # variant then measures "ran out of cash", not its trigger.
        self.starting_equity = starting_equity or _backtest_starting_equity()
        self._daily: dict[str, dict[str, pd.DataFrame]] = {}
        self._dates: dict[str, list[pd.Timestamp]] = {}
        self._history: dict[tuple[str, str, int], HistoryCache] = {}
        #: The last run's raw equity/positions frame, for callers that want to explain a
        #: result rather than just rank it -- see ``tools/explain_run.py``.
        self.last_curve: tuple[pd.DataFrame, Any] | None = None
        for period in periods:
            self._load_period(period)

    @staticmethod
    def _bounds(period: str) -> tuple[pd.Timestamp, pd.Timestamp | None]:
        """A window, either as ``12m`` (the last N months) or ``START:END`` (explicit dates).

        Explicit dates exist because ``_period_start`` anchors everything to *now*, so every
        ``Nm`` window ends today and they nest inside one another. Nested windows cannot be
        independent evidence -- see ``docs/config-exploration.md``.
        """
        if ":" in period:
            start, _, end = period.partition(":")
            return pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
        return _period_start(period), None

    def _load_period(self, period: str) -> None:
        """Daily bars for every candidate symbol, and one date axis shared by every variant.

        The dates are the intersection across the *whole* candidate set, not per variant: two
        universes scored over different date axes are not comparable, and the difference would
        show up as skill.
        """
        symbols = sorted({symbol for names in UNIVERSES.values() for symbol in names}
                         | {symbol for names in DEFENSIVE_SETS.values() for symbol in names})
        # Reuse the backtest tab's own fetch, so the bars are the ones it would have used --
        # but that fetch takes its symbol list from the algorithm's *requirements*, so the
        # candidate set has to be handed over as an algorithm universe rather than as
        # ``config.symbols``, which it would otherwise ignore.
        frames = _fetch_backtest_history_for(symbols, period, self.base)

        missing = [symbol for symbol in symbols if symbol not in frames]
        if missing:
            logger.warning("No daily bars for %s; they are dropped from every universe", ", ".join(missing))

        start, end = self._bounds(period)
        common = sorted(set.intersection(*(set(frame.index) for frame in frames.values())))
        in_period = [date for date in common if date >= start and (end is None or date <= end)]
        dates = in_period if len(in_period) >= 2 else common[-_period_row_count(period):]
        if len(dates) < 2:
            raise RuntimeError(f"No common trading dates for {period}")

        self._daily[period] = frames
        self._dates[period] = dates
        logger.info(
            "%s: %d symbols, %d trade dates %s -> %s",
            period, len(frames), len(dates), dates[0].date(), dates[-1].date(),
        )

    def _history_for(self, period: str, symbols: list[str], lookback_minutes: int) -> HistoryCache:
        key = (period, ",".join(sorted(symbols)), lookback_minutes)
        if key not in self._history:
            self._history[key] = HistoryCache(
                sorted(symbols),
                self._dates[period],
                providers=self.providers,
                lookback_minutes=lookback_minutes,
            )
        return self._history[key]

    def run(self, algorithm_id: str, label: str, tuning: dict[str, Any], period: str) -> Run:
        config = dataclasses.replace(self.base, algorithm_configs={algorithm_id: tuning})
        algorithm = get_algorithm_class(algorithm_id).from_config(config)
        requirements = algorithm.requirements(config, {})

        symbols = [s for s in requirements.price_symbols if s in self._daily[period]]
        daily = {symbol: self._daily[period][symbol] for symbol in symbols}
        if not daily:
            raise RuntimeError(f"{label}: none of its symbols have daily bars")

        started = time.time()
        curve, coverage = replay(
            algorithm,
            config,
            daily_history=daily,
            trade_dates=self._dates[period],
            should_run=lambda date: int(date.dayofweek) in algorithm.schedule.weekdays,
            starting_equity=self.starting_equity,
            history_providers=self.providers,
            history=self._history_for(period, symbols, requirements.history_lookback_minutes),
        )
        self.last_curve = (curve, coverage)
        defensive = {str(s).upper() for s in (tuning.get("defensive_universe") or [])}
        return Run(algorithm_id, label, period, tuning,
                   _measure(curve, coverage, self.starting_equity, time.time() - started, defensive))


def _fetch_backtest_history_for(symbols: list[str], period: str, config) -> dict[str, pd.DataFrame]:
    """The backtest tab's daily fetch, for an explicit symbol list.

    Fast Momentum is used only as the vehicle: its ``requirements`` ask for the deepest daily
    window of any algorithm here (180 days against dual momentum's 105), so one fetch serves
    both without either being short.
    """
    forced = dataclasses.replace(
        config,
        symbols=list(symbols),
        algorithm_configs={"fast_momentum": {"risk_on_universe": list(symbols), "defensive_universe": []}},
    )
    return _fetch_backtest_history("fast_momentum", period, forced)


def _exposure(curve: pd.DataFrame, defensive: set[str]) -> dict[str, Any]:
    """How the book was actually positioned, day by day.

    A return of nearly zero has two very different explanations -- sitting in the defensive
    sleeve earning the T-bill rate, or holding risk assets that went nowhere -- and the equity
    curve alone cannot tell them apart.
    """
    defensive_days = cash_days = 0
    name_count = 0
    days_by_symbol: dict[str, int] = {}
    dates = 0
    for positions in curve.get("positions", pd.Series(dtype=object)):
        held = {
            str(symbol): float(value)
            for symbol, value in (positions or {}).items()
            if abs(float(value or 0.0)) > 0.005
        } if isinstance(positions, dict) else {}
        dates += 1
        name_count += len(held)
        if not held:
            cash_days += 1
        elif set(held) <= defensive:
            defensive_days += 1
        for symbol in held:
            days_by_symbol[symbol] = days_by_symbol.get(symbol, 0) + 1
    if not dates:
        return {"pct_defensive": 0.0, "pct_cash": 0.0, "pct_risk_on": 0.0, "avg_names": 0.0, "held": ""}
    ranked = sorted(days_by_symbol.items(), key=lambda item: -item[1])
    return {
        "pct_defensive": round(defensive_days / dates, 4),
        "pct_cash": round(cash_days / dates, 4),
        "pct_risk_on": round((dates - defensive_days - cash_days) / dates, 4),
        "avg_names": round(name_count / dates, 2),
        # Symbol:days, most-held first -- what the configuration actually owned.
        "held": " ".join(f"{symbol}:{days}" for symbol, days in ranked[:12]),
    }


def _measure(curve: pd.DataFrame, coverage, starting_equity: float, seconds: float,
             defensive: set[str] | None = None) -> dict[str, float]:
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    metrics = calculate_performance_metrics(equity)
    turnover = float(pd.to_numeric(curve.get("turnover", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    orders = int(pd.to_numeric(curve.get("order_count", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    dividends = float(pd.to_numeric(curve.get("dividend_income", pd.Series([0.0])), errors="coerce").fillna(0.0).iloc[-1])
    # Cumulative buy value. For a momentum book this is half the turnover; for DCA it is the
    # whole question -- a plan that never deploys is a cash pile with extra steps.
    deployed = float(pd.to_numeric(curve.get("dca_contributions", pd.Series([0.0])), errors="coerce").fillna(0.0).iloc[-1])

    # The replay charges nothing to trade, so churn is free in these numbers. Price it back in
    # from the turnover the run actually generated, as a fraction of the starting stake.
    drag = {bps: turnover * (bps / 10_000.0) / max(starting_equity, 1.0) for bps in (1, 5)}
    drawdown = abs(float(metrics["max_drawdown"])) or float("nan")
    return {
        "total_return": round(total_return, 6),
        "net_return_1bps": round(total_return - drag[1], 6),
        "net_return_5bps": round(total_return - drag[5], 6),
        "cagr": round(float(metrics["cagr"]), 6),
        "max_drawdown": round(float(metrics["max_drawdown"]), 6),
        "sharpe": round(float(metrics["sharpe"]), 4),
        "calmar": round(total_return / drawdown, 4) if drawdown == drawdown else 0.0,
        "turnover_x_stake": round(turnover / max(starting_equity, 1.0), 2),
        "cost_drag_5bps": round(drag[5], 6),
        "orders": orders,
        "deployed": round(deployed, 2),
        "pnl": round(float(equity.iloc[-1] - equity.iloc[0]), 2),
        "dividend_income": round(dividends, 2),
        "coverage": round(coverage.history_ratio, 4),
        "seconds": round(seconds, 1),
        **_exposure(curve, defensive or set()),
    }


# =========================================================================================
# The grids
# =========================================================================================
# One factor at a time from the deployed baseline. A full cartesian product over ten knobs is
# both unaffordable and a good way to find noise; moving one axis at a time says which axis
# actually carries the result, and the finalists stage recombines only what moved.


def deployed_tuning(algorithm_id: str) -> dict[str, Any]:
    """The tuning currently in ``config/walbot.yaml``, which is what every axis moves *from*.

    Read rather than restated. A hand-copied baseline is a baseline for a configuration nobody
    is running, and every comparison against it inherits the mistake -- the first draft of this
    file quietly disagreed with the deployed file on six keys.
    """
    return dict(tuning_section(get_config(strategy_id=algorithm_id), algorithm_id))


def _dual_baseline() -> dict[str, Any]:
    return deployed_tuning("dual_momentum")


def _fast_baseline() -> dict[str, Any]:
    return deployed_tuning("fast_momentum")


def _axis(baseline: dict[str, Any], name: str, **overrides: Any) -> tuple[str, dict[str, Any]]:
    return name, {**baseline, **overrides}


def dual_axes() -> list[tuple[str, dict[str, Any]]]:
    base = _dual_baseline()
    variants = [("baseline", dict(base))]
    for name, symbols in UNIVERSES.items():
        # Skip whichever set the algorithm deploys now -- by value, not by name, so the skip
        # still holds when a config exploration promotes a different set to deployed.
        if list(symbols) == base.get("risk_on_universe"):
            continue
        variants.append(_axis(base, f"universe={name}", risk_on_universe=list(symbols)))
    for positions in (3, 5, 6, 8):
        # Ranks move with the book: entry as wide as the book, exit two wider, which is the
        # asymmetry the strategy is built on.
        variants.append(_axis(base, f"max_positions={positions}", max_positions=positions,
                              entry_rank_max=positions, exit_rank_max=positions + 2))
    for tilt in (-1.0, 0.0, 0.5):
        variants.append(_axis(base, f"volatility_tilt={tilt}", volatility_tilt=tilt))
    variants.append(_axis(base, "risk_adjusted_score=True", risk_adjusted_score=True))
    for floor in (0.25, 0.5):
        variants.append(_axis(base, f"min_base_score={floor}", min_base_score=floor))
    for breadth in (0.4, 0.6):
        variants.append(_axis(base, f"breadth_min={breadth}", breadth_min=breadth))
    for cap in (0.25, 0.35):
        variants.append(_axis(base, f"name_weight_max={cap}", name_weight_max=cap))
    for delta in (0.1, 0.6):
        variants.append(_axis(base, f"delta_to_replace={delta}", min_score_delta_to_replace=delta))
    # The deployed config turns over ~46% of the book every session, which no return survives
    # once trading is priced. These two are the brakes, and they are the axis most likely to
    # matter live -- the replay charges nothing, so only these rows show the trade-off.
    for threshold in (0.06, 0.10):
        variants.append(_axis(base, f"rebalance_threshold={threshold}", rebalance_weight_threshold=threshold))
    for fraction in (0.02, 0.05):
        variants.append(_axis(base, f"min_trade_nav={fraction}", minimum_trade_nav_fraction=fraction))
    variants.append(_axis(base, "low_churn_combo", rebalance_weight_threshold=0.08,
                          minimum_trade_nav_fraction=0.02, min_score_delta_to_replace=0.6))
    for name, symbols in DEFENSIVE_SETS.items():
        if name != "sgov":
            variants.append(_axis(base, f"defensive={name}", defensive_universe=list(symbols)))
    variants.append(_axis(base, "benchmark=SPY", benchmark="SPY"))
    # A variant that equals the deployed baseline is not an axis, it is a restatement -- and a
    # configuration change can promote one silently. Drop it rather than let it read as a win.
    return [variant for variant in variants if variant[0] == "baseline" or variant[1] != base]


def fast_axes() -> list[tuple[str, dict[str, Any]]]:
    base = _fast_baseline()
    variants = [("baseline", dict(base))]
    for name, symbols in UNIVERSES.items():
        if list(symbols) == base.get("risk_on_universe"):
            continue
        variants.append(_axis(base, f"universe={name}", risk_on_universe=list(symbols)))
    for positions in (3, 5, 6, 8):
        variants.append(_axis(base, f"max_positions={positions}", max_positions=positions))
    for cap in (0.25, 0.35):
        variants.append(_axis(base, f"max_single_weight={cap}", max_single_position_weight=cap))
    for delta in (0.0, 0.6):
        variants.append(_axis(base, f"delta_to_replace={delta}", min_score_delta_to_replace=delta))
    for pullback in (0.0, 0.25):
        variants.append(_axis(base, f"w_pullback={pullback}", w_pullback_uptrend=pullback))
    # No sentiment axis: ``ReplayContextSource`` returns neutral scores, so ``w_sentiment``
    # cannot move a backtest at all. Anything it appears to show would be noise.
    variants.append(_axis(base, "horizons=slow", w_price_nano=0.10, w_price_micro=0.20,
                          w_price_meso=0.40, w_price_macro=0.30))
    variants.append(_axis(base, "horizons=fast", w_price_nano=0.40, w_price_micro=0.40,
                          w_price_meso=0.15, w_price_macro=0.05))
    # The brakes, as for dual momentum: the replay trades for free, so the cost of churn is
    # only visible by comparing these rows against the turnover column.
    for threshold in (0.03, 0.06):
        variants.append(_axis(base, f"rebalance_threshold={threshold}", rebalance_threshold=threshold))
    for minimum in (150.0, 300.0):
        variants.append(_axis(base, f"per_trade_min={minimum:.0f}", per_trade_value_min=minimum))
    variants.append(_axis(base, "low_churn_combo", rebalance_threshold=0.05,
                          per_trade_value_min=200.0, min_score_delta_to_replace=0.6))
    for name, symbols in DEFENSIVE_SETS.items():
        if name != "sgov":
            variants.append(_axis(base, f"defensive={name}", defensive_universe=list(symbols)))
    return [variant for variant in variants if variant[0] == "baseline" or variant[1] != base]


def dual_wide() -> list[tuple[str, dict[str, Any]]]:
    """The wide universe as the new starting point, then the knobs that matter on top of it.

    A separate stage because ``axes`` moves one knob from the *deployed* config, and the wide
    universe changed the answer enough that the other knobs deserve re-asking against it --
    ``max_positions`` in particular, since a wider candidate pool is the reason to hold more.
    """
    base = {**deployed_tuning("dual_momentum"), "risk_on_universe": list(WIDE)}
    variants = [("wide/deployed-positions", dict(base))]

    for positions in (5, 6, 8, 10):
        variants.append(_axis(base, f"wide/max_positions={positions}", max_positions=positions,
                              entry_rank_max=positions, exit_rank_max=positions + 2))

    # The turnover pair, asked against the wide universe rather than the narrow one.
    for delta in (0.0, 0.1, 0.6, 1.0):
        variants.append(_axis(base, f"wide/delta_to_replace={delta}", min_score_delta_to_replace=delta))
    for threshold in (0.06, 0.10, 0.15):
        variants.append(_axis(base, f"wide/rebalance_threshold={threshold}",
                              rebalance_weight_threshold=threshold))
    return variants


def bursty_axes() -> list[tuple[str, dict[str, Any]]]:
    """Bursty DCA's trigger, plus the backlog relaxation.

    For DCA the headline metric is not return but ``deployed``: the plan states a monthly
    contribution, and a trigger picky enough to skip it is not timing the market, it is
    declining to invest. Read ``deployed`` and ``pnl`` together -- a variant that deploys less
    and earns less has simply been out of the market.
    """
    base = deployed_tuning("bursty_dca")
    variants = [("baseline", dict(base))]

    # Both directions, because "increase the RSI" is ambiguous: raising the *threshold* fires on
    # shallower dips (less picky), while raising the *lookback* makes RSI sluggish and rarely
    # extreme, which quietly disables that leg of the trigger.
    for threshold in (5.0, 20.0, 30.0):
        variants.append(_axis(base, f"rsi_threshold={threshold:.0f}", rsi_threshold=threshold))
    for lookback in (3, 5, 14):
        variants.append(_axis(base, f"rsi_lookback={lookback}", rsi_lookback=lookback))
    for threshold in (-0.1, 0.1, 0.2):
        variants.append(_axis(base, f"percent_b_threshold={threshold}", percent_b_threshold=threshold))
    for days in (100, 300):
        variants.append(_axis(base, f"regime_ma_days={days}", regime_ma_days=days))
    for months in (1.0, 2.0, 3.0, 6.0):
        variants.append(_axis(base, f"backlog_relax={months:.0f}m", backlog_relax_months=months))
    variants.append(_axis(base, "value_averaging=False", value_averaging=False))
    for multiple in (1.0, 6.0):
        variants.append(_axis(base, f"max_trade_multiple={multiple:.0f}", max_trade_multiple=multiple))
    return variants


def dca_axes() -> list[tuple[str, dict[str, Any]]]:
    """Plain DCA, as the benchmark Bursty has to beat. It has no trigger to tune."""
    return [("baseline", dict(deployed_tuning("dca")))]


def dual_wide_churn() -> list[tuple[str, dict[str, Any]]]:
    """Every brake on membership churn, asked against the wide universe.

    A wider universe raises turnover -- more candidates means more ways for the ranking to
    change -- so whether widening is a good trade depends on what it costs to hold. The
    ``wide`` stage covers the two weight-level brakes; these are the ones that act on
    *membership* instead, plus the two that shrink the eligible set.

    ``signal_refresh_minutes`` is the interesting one. It throttles how often membership may
    change at all, and at its deployed 60 it cannot bind in a daily replay: consecutive dates
    are 1440 minutes apart, so the throttle is always already satisfied. Only values above a
    day are testable here -- and live, they are the difference between re-ranking hourly and
    re-ranking weekly.
    """
    base = {**deployed_tuning("dual_momentum"), "risk_on_universe": list(WIDE)}
    variants = [("wide/baseline", dict(base))]

    for minutes, name in ((1440, "1d"), (4320, "3d"), (10080, "7d")):
        variants.append(_axis(base, f"wide/signal_refresh={name}", signal_refresh_minutes=minutes))
    for cap in (0.20, 0.25, 0.35):
        variants.append(_axis(base, f"wide/name_weight_max={cap}", name_weight_max=cap))
    for floor in (0.25, 0.5, 0.75):
        variants.append(_axis(base, f"wide/min_base_score={floor}", min_base_score=floor))
    # The plausible combination: re-rank weekly, trade only on real drift, hold more names so
    # each one matters less.
    variants.append(_axis(base, "wide/low_churn_combo", signal_refresh_minutes=10080,
                          rebalance_weight_threshold=0.10, max_positions=8,
                          entry_rank_max=8, exit_rank_max=10))
    return variants


GRIDS: dict[str, dict[str, Callable[[], list[tuple[str, dict[str, Any]]]]]] = {
    "dual_momentum": {"axes": dual_axes, "wide": dual_wide, "wide_churn": dual_wide_churn},
    "fast_momentum": {"axes": fast_axes},
    "bursty_dca": {"axes": bursty_axes},
    "dca": {"axes": dca_axes},
}


def finalists(algorithm_id: str, axes_results: str, metric: str = "net_return_5bps",
              margin: float = 0.005) -> list[tuple[str, dict[str, Any]]]:
    """Recombine the axes that actually moved, from a completed axes run.

    Derived from results rather than written in advance, because which axes matter is the thing
    the axes stage is for. Each winning axis contributes the *single* best value it found; the
    combination is then run alongside every one-axis-removed ablation, so a combined result that
    rests entirely on one change says so instead of looking like a synergy.

    ``margin`` is the improvement an axis has to clear to be counted at all -- half a point of
    net return over twelve months, which is above the noise of a single window but not by much.
    """
    with open(axes_results.replace(".csv", ".json"), encoding="utf-8") as handle:
        rows = [row for row in json.load(handle) if row["algorithm"] == algorithm_id]
    if not rows:
        raise RuntimeError(f"no {algorithm_id} rows in {axes_results}")

    baseline_row = next(row for row in rows if row["label"] == "baseline")
    baseline = baseline_row["tuning"]

    # Best value per changed key, among variants that beat the baseline by the margin.
    best_by_key: dict[str, tuple[float, Any, str]] = {}
    for row in rows:
        if row["label"] == "baseline" or row[metric] - baseline_row[metric] < margin:
            continue
        changed = {key for key in set(baseline) | set(row["tuning"])
                   if baseline.get(key) != row["tuning"].get(key)}
        for key in changed:
            if key not in best_by_key or row[metric] > best_by_key[key][0]:
                best_by_key[key] = (row[metric], row["tuning"].get(key), row["label"])

    if not best_by_key:
        logger.warning("%s: no axis beat the baseline by %.1f%%; nothing to combine",
                       algorithm_id, margin * 100)
        return [("baseline", dict(baseline))]

    combined = {**baseline, **{key: value for key, (_score, value, _label) in best_by_key.items()}}
    contributors = sorted({label for _score, _value, label in best_by_key.values()})
    logger.info("%s: combining %s", algorithm_id, ", ".join(contributors))

    variants = [("baseline", dict(baseline)), ("combined", combined)]
    for key in sorted(best_by_key):
        variants.append((f"combined-minus:{key}", {**combined, key: baseline.get(key)}))
    return variants


# =========================================================================================
# Driver
# =========================================================================================


def execute(sweep: Sweep, algorithm_id: str, variants: Iterable[tuple[str, dict[str, Any]]],
            periods: list[str]) -> list[Run]:
    """Run every variant. Pure reading, so it takes the read-only pool the backtest tab uses.

    The daily fetch that fills ``Sweep`` is deliberately outside it: that one writes provider
    bars into the cache, and holding a connection across minutes of network calls locks other
    processes out of the file.
    """
    runs: list[Run] = []
    variants = list(variants)
    with pooled_connections(read_only=True):
      for index, (label, tuning) in enumerate(variants, start=1):
          for period in periods:
              try:
                  run = sweep.run(algorithm_id, label, tuning, period)
              except Exception as exc:  # noqa: BLE001 - one bad variant must not end the sweep
                  logger.warning("%s / %s / %s failed: %s", algorithm_id, label, period, exc)
                  continue
              runs.append(run)
              print(
                  f"[{index:>3}/{len(variants)}] {algorithm_id:14s} {period:4s} {label:28s} "
                  f"ret={run.metrics['total_return']:+7.2%} net5={run.metrics['net_return_5bps']:+7.2%} "
                  f"dd={run.metrics['max_drawdown']:+6.2%} sharpe={run.metrics['sharpe']:5.2f} "
                  f"turn={run.metrics['turnover_x_stake']:5.1f}x ({run.metrics['seconds']:.0f}s)",
                  flush=True,
              )
    return runs


def write_results(runs: list[Run], path: str) -> None:
    if not runs:
        return
    rows = [run.row() for run in runs]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(path.replace(".csv", ".json"), "w", encoding="utf-8") as handle:
        json.dump([{**run.row(), "tuning": run.tuning} for run in runs], handle, indent=2, sort_keys=True)
    print(f"\nwrote {len(rows)} rows to {path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", default="both",
                        choices=["dual_momentum", "fast_momentum", "bursty_dca", "dca", "both"])
    parser.add_argument("--starting-equity", type=float, default=None,
                        help="override the backtest stake; a DCA plan needs one that funds it")
    parser.add_argument("--stage", default="axes", choices=["axes", "wide", "wide_churn", "finalists"])
    parser.add_argument("--from", dest="axes_results", default="data/config_sweep_12m.csv",
                        help="finalists stage: the axes run to recombine")
    parser.add_argument("--period", default="12m", help="comma-separated, e.g. 12m,6m")
    parser.add_argument("--out", default="data/config_sweep.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("src.core.orders", "src.brokerages.providers.paper", "src.algorithms.dual_momentum.algorithm",
                  "src.algorithms.fast_momentum", "src.data.provider_cache", "src.connectors"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    periods = [item.strip() for item in args.period.split(",") if item.strip()]
    algorithms = ["dual_momentum", "fast_momentum"] if args.algorithm == "both" else [args.algorithm]

    sweep = Sweep(periods, starting_equity=args.starting_equity)
    runs: list[Run] = []
    for algorithm_id in algorithms:
        if args.stage == "finalists":
            variants = finalists(algorithm_id, args.axes_results)
        else:
            variants = GRIDS[algorithm_id][args.stage]()
        runs.extend(execute(sweep, algorithm_id, variants, periods))
    write_results(runs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
