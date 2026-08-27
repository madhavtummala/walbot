"""Tuning for the dual-momentum algorithm.
"""

from __future__ import annotations



import logging
from dataclasses import dataclass, field



logger = logging.getLogger(__name__)


EPSILON = 1e-9

#: Trading days per year, for annualising a daily volatility estimate.
TRADING_DAYS = 252

#: Share of the risk-on universe that must have usable history before the algorithm will trade
#: at all. Below it the book holds the defensive sleeve and reports a data gap.
#:
#: A constant rather than a knob: it is a data-integrity guard, not a market view, and it never
#: bound once in a twelve-month replay. A thin cache reads as "nothing is above its average",
#: which is a bear market that never happened -- that is the only thing this is here to catch.
MIN_UNIVERSE_COVERAGE = 0.50

#: A name may not be *newly entered* while it scores below this, in cross-sectional z units.
#:
#: Zero, and hardcoded, because zero is the only value in these units that means anything on its
#: own: the score is a robust z-score, so 0 is the universe median and "do not buy a name that is
#: below the middle of its own field" is a statement that survives a change of tape. Any other
#: number is a guess about a distribution that is renormalised every run.
#:
#: It was a knob, ``min_base_score``, deployed at 0. Removing it entirely cost ~6pp over a
#: 12-month replay and ~1.7pp over 3m, so the *check* earns its place even though the *dial*
#: did not. Holdings are exempt: this is an entry condition, and a name already owned is judged
#: by eligibility and rank alone.
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
    #: Rank at which an incumbent is finally dropped. Wider than ``entry_rank_max`` on purpose,
    #: so a holding that slips a place or two is not sold for it.
    #:
    #: It only protects a name that slipped in *rank*. A name that fails a gate is ineligible,
    #: an ineligible name is never ranked, and an unranked name cannot be retained however wide
    #: this is set. That is how a volatility ceiling comes to force a sale.
    exit_rank_max: int = 7
    #: Score advantage a challenger needs to displace a holding, as opposed to filling a free
    #: slot. The anti-churn knob: at 0 the book swaps on any improvement.
    min_score_delta_to_replace: float = 0.35
    defensive_max_positions: int = 2

    # =====================================================================================
    # 3. How often it changes its mind
    # =====================================================================================
    #: How often the cross-section is re-ranked, in *trading* days -- elapsed sessions since the
    #: last re-rank, not runs since. How often the algorithm looks is the binding's cron and has
    #: nothing to say about how often it should act; collapsing the two is what made a
    #: medium-term signal trade like a fast one.
    #:
    #: This used to count runs, on the premise that the algorithm fires once a session. A binding
    #: is free to state any cron it likes, and at the five-a-session one in ``walbot.yaml`` that
    #: premise made this knob mean "re-rank daily" -- five times the intended rate, with nothing
    #: on the deck saying so.
    #:
    #: Selection, entry and replacement all happen on this clock. :func:`crash_stop` does not:
    #: it runs every session whatever this says, because a name can gap 30% across a week of
    #: throttled sessions while the algorithm waits its turn.
    #:
    #: 0 means every run, which is what the algorithm did before this existed. Measured on its
    #: own that looks like a large improvement; measured in combination with everything else it
    #: is the single worst change available. Do not tune it from a one-at-a-time sweep.
    rerank_interval_days: int = 5
    #: Market days of history kept per symbol -- the window the settling count is taken over.
    #:
    #: Counted in **market days**, never in runs, however often the binding's cron fires; within
    #: a day the last run wins. Counting runs made this a function of the schedule: at five fires
    #: a session a three-"day" settling period was served by 36 minutes of one morning, and
    #: pausing the binding stopped the clock where it stood. Only days the algorithm actually ran
    #: are on record, so a pause makes a name take longer to qualify rather than disqualifying
    #: it. See :mod:`.memory`.
    eligibility_window: int = 10
    #: Market days, out of ``eligibility_window``, a name must have been ranked inside
    #: ``entry_rank_max`` before it may be opened. 1 makes entry stateless.
    #:
    #: This is an entry condition only. Set high it does not steady the book, it locks the book
    #: out of names that have already started moving: at 3 it kept XBI out for three weeks of a
    #: +23% advance after one ordinary pullback.
    entry_min_eligible_days: int = 8

    # =====================================================================================
    # 4. What it refuses to hold at any rank
    # =====================================================================================
    # Absolute momentum. Relative strength decides the order; these decide whether a name may
    # be held at all, so a thin qualifying set means holding less rather than lowering the bar.
    #
    # Every floor here is stated in raw percent, which is the same number for a 12%-volatility
    # index fund and a 65%-volatility thematic ETF. That is the known weakness of this block:
    # simultaneously too loose for a calm name and too tight for a wild one.
    etf_ma_days: int = 100
    etf_abs_return_days: int = 60
    etf_min_abs_return: float = 0.0
    etf_fast_return_days: int = 20
    etf_min_fast_return: float = -0.02
    #: Reject any name whose 20-day annualised volatility exceeds this ceiling. 0 = off.
    #:
    #: The one volatility gate left, and it earns its place: without it a 12-month replay lost
    #: ~28pp of return and ~25pp of drawdown. It is what keeps the 44-76% volatility names out
    #: of the book, and the drops they take with them.
    #:
    #: It is written as an entry filter but it is not one. Eligibility drives ranking, so a
    #: *held* name whose volatility rises through this ceiling is de-ranked and sold -- which
    #: happened once in twelve months, to SLV, a month before the top and 35% below it.
    vol_ceiling: float = 0.0
    #: A holding that falls this much in a single session is sold outright, ahead of every other
    #: exit rule. 0 turns it off.
    #:
    #: The algorithm's only stop, and it did not fire once in a twelve-month replay -- it reads
    #: close-to-close, so it is a far rarer event than an intraday touch of the same size. Read
    #: it as a circuit breaker, not as risk management.
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
    # Last because it is the least likely thing to want changed and the easiest to break: the
    # score is a robust cross-sectional z-score, so every value here is relative to the rest of
    # the universe on the same day, and changing one horizon changes what every name scores.
    #
    # Selection horizons, in market days. They have been counted three ways -- bars on an assumed
    # 15-minute grid, then market minutes (every value an exact multiple of 390, which each
    # reader had to divide back out), now days. :func:`days_knob` migrates the older two.
    #
    # The ladder is inherited rather than chosen. When the score moved from intraday to daily
    # bars the old 0.15/0.62/3.08/12.31-session horizons were converted to their nearest daily
    # equivalents instead of being re-picked, which is why the default has three of its four
    # horizons inside a single week.
    #
    # ``nano_days`` is exposed but should be left at 1. Removing the horizon outright cost 6.2pp
    # over a 12-month replay while setting its weight to zero cost only 1.5pp, so the two are not
    # equivalent and the reason is not established. Treat any change here as unexplained until
    # someone measures it.
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
        # The selection horizons are served from these bars now, so they have to be counted
        # here: they used to be answered from a separate intraday window that no longer
        # exists. The smoothing tail rides on top, since the score is EMA'd across samples.
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
