"""Tuning for the dual-momentum algorithm."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


EPSILON = 1e-9

#: Trading days per year, for annualising a daily volatility estimate.
TRADING_DAYS = 252

#: Share of the risk-on universe that must have usable history before the algorithm trades at
#: all -- a data-integrity guard, not a market view. Below it the book holds the defensive sleeve.
MIN_UNIVERSE_COVERAGE = 0.50

#: A name may not be *newly entered* while it scores below this (cross-sectional z units).
#: Hardcoded at zero -- the universe median -- rather than a knob, since any other value is a
#: guess about a distribution renormalised every run. Holdings are exempt: an owned name is
#: judged by eligibility and rank alone.
MIN_ENTRY_SCORE = 0.0


@dataclass(frozen=True)
class RallyRotationConfig:
    """Every knob, ordered by how much thought it deserves.

    Roughly: what it may hold, how many, how often it changes its mind, what it refuses to
    hold, how large each position is, what is too small to trade -- and only then the internals
    of the score itself, which are the least likely thing to want changing and the easiest to
    break.

    These are research defaults, not recommended live settings.
    """

    # =====================================================================================
    # 1. What it may hold
    # =====================================================================================
    risk_on_universe: list[str] = field(default_factory=lambda: ["QQQM", "VTI", "IWM", "IEMG", "XSD"])
    #: Where the book sits when nothing qualifies, and where undeployed gross is parked.
    defensive_universe: list[str] = field(default_factory=lambda: ["BIL", "IEF", "AGG", "GLD"])

    # =====================================================================================
    # 2. How many, and how hard it is to displace one
    # =====================================================================================
    #: 5 rather than the spec's 3: measured across 6M/4M/3M replay windows, five holdings
    #: added roughly 5pp of return with no increase in drawdown. Momentum concentration
    #: sounds decisive but a single wrong leader dominates a 3-name book.
    max_positions: int = 5
    #: Worst rank that may be newly *entered*.
    entry_rank_max: int = 5
    #: Rank at which an incumbent is finally dropped. Wider than ``entry_rank_max`` so a holding
    #: that slips a place or two isn't sold for it -- but it only protects rank slippage; a name
    #: that fails a gate is unranked and cannot be retained however wide this is set.
    exit_rank_max: int = 7
    #: Score advantage a challenger needs to displace a holding, as opposed to filling a free
    #: slot. The anti-churn knob: at 0 the book swaps on any improvement.
    min_score_delta_to_replace: float = 0.35
    defensive_max_positions: int = 2

    # =====================================================================================
    # 3. How often it changes its mind
    # =====================================================================================
    #: How often the cross-section is re-ranked, in elapsed *trading days* since the last
    #: re-rank (not runs -- the binding's cron cadence has nothing to say about how often a
    #: medium-term signal should act). Selection, entry and replacement all sit on this clock;
    #: :func:`crash_stop` does not, since a name can gap 30% while the algorithm waits its turn.
    #: 0 means every run; do not tune this from a one-at-a-time sweep, it interacts badly.
    rerank_interval_days: int = 5
    #: Market days of history kept per symbol, for the settling count in :mod:`.memory`. Counted
    #: in market days rather than runs, so a pause makes a name take longer to qualify rather
    #: than stopping its clock mid-window or disqualifying it outright.
    eligibility_window: int = 10
    #: Market days, out of ``eligibility_window``, a name must have been ranked inside
    #: ``entry_rank_max`` before it may be opened. 1 makes entry stateless. An entry condition
    #: only -- set too high it locks the book out of names that already started moving.
    entry_min_eligible_days: int = 8

    # =====================================================================================
    # 4. What it refuses to hold at any rank
    # =====================================================================================
    # Absolute momentum: relative strength decides the order, these decide whether a name may
    # be held at all. Stated in raw percent (a known weakness: the same number is loose for a
    # calm name and tight for a wild one).
    etf_ma_days: int = 100
    etf_abs_return_days: int = 60
    etf_min_abs_return: float = 0.0
    etf_fast_return_days: int = 20
    etf_min_fast_return: float = -0.02
    #: Reject any name whose 20-day annualised volatility exceeds this ceiling. 0 = off. Written
    #: as an entry filter but not one in practice: eligibility drives ranking, so a *held* name
    #: whose volatility rises through this ceiling is de-ranked and sold too.
    vol_ceiling: float = 0.0
    #: A holding that falls this much in a single session is sold outright, ahead of every other
    #: exit rule. 0 turns it off. Reads close-to-close -- a circuit breaker, not risk management.
    max_daily_drop: float = 0.10

    # =====================================================================================
    # 5. How large each position is
    # =====================================================================================
    #: Cap on total invested fraction of equity, and the only lever on gross exposure. Below
    #: 1.0 the remainder is parked in the defensive sleeve rather than held as cash.
    risk_on_gross_max: float = 1.0
    #: How much volatility should move a position's size, as an exponent: weight is
    #: proportional to score x sigma ** volatility_tilt.
    #:
    #:   -1.0  risk parity -- divide by volatility, so calm names get the big positions
    #:    0.0  score alone -- volatility does not enter sizing at all
    #:   +1.0  lean in -- scale up with volatility, which is what an ungated momentum book
    #:         does implicitly by never dividing
    #:
    #: One number rather than a boolean because the useful settings are not binary: the
    #: question is how hard to press, and the answer is a market regime opinion.
    volatility_tilt: float = 1.0
    #: Daily window for the per-name volatility estimate that ``volatility_tilt`` reads.
    vol_estimation_days: int = 20

    # =====================================================================================
    # 6. What is too small to be worth trading
    # =====================================================================================
    #: Smallest weight change worth trading. Suppresses drift, never a full exit.
    rebalance_weight_threshold: float = 0.03
    minimum_trade_notional: float = 100.0
    #: The same floor as a fraction of equity; the larger of the two applies.
    minimum_trade_nav_fraction: float = 0.005

    # =====================================================================================
    # 7. Inside the score
    # =====================================================================================
    # Last because it's the least likely thing to want changed and the easiest to break: the
    # score is a robust cross-sectional z-score, so changing one horizon changes what every name
    # scores. Selection horizons in market days; :func:`days_knob` migrates two older units
    # (an assumed 15-minute bar grid, then market minutes) still on disk.
    #
    # ``nano_days`` is exposed but should be left at 1 -- removing the horizon outright measured
    # worse than zeroing its weight, and the reason isn't established. Treat as unexplained.
    nano_days: int = field(default=1, metadata={
        "legacy_days_key": "selection_horizon_nano_days",
        "legacy_minutes_key": "selection_horizon_nano_minutes",
        "legacy_key": "selection_horizon_nano"})
    micro_days: int = field(default=2, metadata={
        "legacy_days_key": "selection_horizon_micro_days",
        "legacy_minutes_key": "selection_horizon_micro_minutes",
        "legacy_key": "selection_horizon_micro"})
    meso_days: int = field(default=3, metadata={
        "legacy_days_key": "selection_horizon_meso_days",
        "legacy_minutes_key": "selection_horizon_meso_minutes",
        "legacy_key": "selection_horizon_meso"})
    macro_days: int = field(default=12, metadata={
        "legacy_days_key": "selection_horizon_macro_days",
        "legacy_minutes_key": "selection_horizon_macro_minutes",
        "legacy_key": "selection_horizon_macro"})
    #: Blend weights, one per horizon. Slow-dominant by design; a slow-heavier blend
    #: (.10/.25/.60) measured worse on return in all three replay windows.
    w_nano: float = 0.05
    w_micro: float = 0.15
    w_meso: float = 0.30
    w_macro: float = 0.50
    #: Market days of smoothing on the composite score. Was 45 *minutes*, which on daily bars
    #: rounds to a single sample -- the smoothing was silently switched off wherever the cache
    #: held no intraday bars.
    score_ema_days: int = field(default=3, metadata={"legacy_minutes_key": "score_ema_minutes"})
    #: Median/MAD rather than mean/standard deviation, so one event-driven spike does not
    #: flatten everyone else's score.
    robust_zscore: bool = True
    #: Rank on return-per-unit-of-volatility rather than raw return. Off, the cross-section
    #: rewards amplitude and the highest-volatility name that happened to rise wins.
    #:
    #: Off because that is what the measurements said, but it is a genuine trade rather than a
    #: settled question: turning it on was +2.6% over 3m and -8.6% over 6m.
    risk_adjusted_score: bool = False

    @property
    def symbols(self) -> list[str]:
        """Everything the algorithm needs priced."""
        return sorted(set(self.risk_on_universe) | set(self.defensive_universe))

    @property
    def required_history_minutes(self) -> int:
        """None. Every feature this algorithm computes now comes from daily bars.

        Deliberately zero rather than "some intraday window we then ignore": a non-zero value
        makes ``LiveContextSource`` fetch an intraday window on every run and makes the
        backtester build a ``HistoryCache`` over it, which was where a twelve-month replay
        spent 97 of its 99 seconds.
        """
        return 0

    @property
    def selection_horizon_days(self) -> int:
        """The slowest selection horizon, for the daily-bar depth calculation."""
        return max(self.nano_days, self.micro_days, self.meso_days, self.macro_days)

    @property
    def required_daily_bars(self) -> int:
        return (
            max(
                self.etf_ma_days,
                self.etf_abs_return_days,
                self.vol_estimation_days,
                self.selection_horizon_days + max(self.score_ema_days, 1),
            )
            + 5
        )


# =========================================================================================
# Pure feature maths
