"""Tuning for the dual-momentum algorithm.
"""

from __future__ import annotations



import logging
import math
from dataclasses import dataclass, field
from typing import Any


from ...common.config_utils import load_tuning, tuning_section
from ...data.bars import TRADING_MINUTES_PER_DAY

logger = logging.getLogger(__name__)

STATE_KEY = "rally_rotation_runtime"

EPSILON = 1e-9

#: Trading days per year, for annualising a daily volatility estimate.
TRADING_DAYS = 252


@dataclass(frozen=True)
class RallyRotationConfig:
    """Every knob, with the spec's starting values.

    These are research defaults, not recommended live settings: they need converting to your
    bar frequency and evaluating post-cost, walk-forward, before any of them means anything.
    """

    # -- universes ------------------------------------------------------------------------
    risk_on_universe: list[str] = field(default_factory=lambda: ["QQQM", "VTI", "IWM", "IEMG", "XSD"])
    defensive_universe: list[str] = field(default_factory=lambda: ["BIL", "IEF", "AGG", "GLD"])
    # -- decision cadence -------------------------------------------------------------------
    #: How often the cross-section is re-ranked, in *trading* days -- counted in runs, since
    #: the algorithm runs once a session. The algorithm runs every session; that is the rate at
    #: which it looks, not the rate at which it should act, and collapsing the two is what made
    #: a medium-term signal trade like a fast one. The slowest selection horizon is twelve
    #: sessions, so re-ranking daily asks the score a question it cannot answer that fast.
    #:
    #: Selection, entry, replacement and the considered exits all happen on this clock.
    #: :func:`crash_stop` does not: it runs every session whatever this says, because a name
    #: can gap 30% across a week of throttled sessions while the algorithm waits its turn.
    #:
    #: 0 means every run, which is what the algorithm did before this existed.
    rerank_interval_days: int = 5

    # -- selection score ------------------------------------------------------------------
    #: Selection horizons, in minutes the market is open, at *daily* granularity: one, two,
    #: three and twelve sessions (``TRADING_MINUTES_PER_DAY`` is 390).
    #:
    #: These used to be 60/240/1200/4800 -- a quarter-session through twelve -- because the
    #: score was computed from intraday bars. It no longer is. The two fastest horizons were
    #: unresolvable on daily bars and the blend silently answered them from whatever
    #: resolution the cache happened to hold, so the same backtest scored its first months
    #: from ~18 daily closes and the rest from ~1,300 five-minute bars.
    #:
    #: The ladder is deliberately the closest daily equivalent of what was measured, not a
    #: re-tuning: 1/2/3/12 sessions against the old 0.15/0.62/3.08/12.31. It is the obvious
    #: thing to sweep now that a four-year window is reachable.
    #: The legacy keys drop the ``_minutes`` suffix rather than ending ``_bars``, so they are
    #: named here for :func:`load_tuning` instead of following the usual convention.
    selection_horizon_nano_minutes: int = field(default=390, metadata={"legacy_key": "selection_horizon_nano"})
    selection_horizon_micro_minutes: int = field(default=780, metadata={"legacy_key": "selection_horizon_micro"})
    selection_horizon_meso_minutes: int = field(default=1170, metadata={"legacy_key": "selection_horizon_meso"})
    selection_horizon_macro_minutes: int = field(default=4680, metadata={"legacy_key": "selection_horizon_macro"})
    w_nano: float = 0.05
    w_micro: float = 0.15
    w_meso: float = 0.30
    w_macro: float = 0.50
    robust_zscore: bool = True
    #: Rank on return-per-unit-of-volatility rather than raw return. Off, the cross-section
    #: rewards amplitude and the highest-volatility name that happened to rise wins.
    #:
    #: Defaults to off because that is what the measurements said: over 6M/4M replay windows
    #: raw ranking earned ~3.4pp more, at ~1pp more drawdown. The risk-adjusted variant won
    #: the choppy most-recent quarter and always ran lower volatility, so this is a genuine
    #: trade rather than a settled question -- which is why it is a dashboard knob.
    risk_adjusted_score: bool = False
    #: Three sessions. Was 45 minutes, which on daily bars rounds to a single sample -- the
    #: smoothing was silently switched off wherever the cache had no intraday bars.
    score_ema_minutes: int = 1170

    # -- data sufficiency -----------------------------------------------------------------
    #: How much of the risk-on universe must have usable history before the algorithm will
    #: trade at all. Below this it holds the defensive sleeve and reports a data gap.
    #:
    #: All that is left of the breadth regime gate. That gate asked whether enough of the
    #: universe was in an uptrend, which is the question ``eligibility`` already asks of each
    #: name individually -- and it answered it worse, by vetoing names that had passed every
    #: test the strategy makes of them because other, unrelated names were weak. In March 2026
    #: that meant QQQM at -5.0% vetoing USO at +55.3%, the book's own top-ranked name. It ran
    #: at ``breadth_min: 0.0`` long enough to confirm it only ever subtracted.
    #:
    #: This survives because it is not a market view: a thin cache reads as "nothing is above
    #: its average", which is a bear market that never happened.
    min_universe_coverage: float = 0.50

    # -- per-ETF absolute eligibility -----------------------------------------------------
    etf_ma_days: int = 100
    etf_abs_return_days: int = 60
    etf_min_abs_return: float = 0.0
    etf_fast_return_days: int = 20
    etf_min_fast_return: float = -0.02
    #: How much slack a name already held gets on each eligibility floor before it is
    #: sold. Entry and exit used the same thresholds, so a 20-day return sitting near
    #: ``etf_min_fast_return`` flipped a holding in and out on consecutive sessions --
    #: membership changed on 143 of 250 days in 2023, 138 entries against 134 exits.
    #: A band means a holding leaves because the asset actually broke down, not because
    #: it grazed the line it entered through.
    exit_threshold_slack: float = 0.05
    #: A holding that falls this much in a single session is sold outright, ahead of
    #: every other exit rule. 0 turns it off.
    #:
    #: This is the algorithm's only stop. There used to be a portfolio-level session
    #: breaker beside it, ``intraday_drawdown_limit``, which could not fire: it rebases
    #: its session high on every run, so at ``DAILY_AT_OPEN`` the drawdown it measured
    #: was always exactly zero. A knob that reads as protection and provides none is
    #: worse than no knob, so it is gone.
    max_daily_drop: float = 0.10
    # -- trend filter (medium-term) -------------------------------------------------------
    #: Names must be above this MA to be eligible. Filters out short-term rallies
    #: in names still in a medium-term downtrend. 0 = off.
    trend_ma_days: int = 20
    #: Names must have positive return over this many days to be eligible. 0 = off.
    trend_return_days: int = 20
    #: Minimum return over trend_return_days. 0 = off.
    trend_min_return: float = 0.0
    # -- eligibility persistence ----------------------------------------------------------
    #: Eligibility is a stateless per-day test, so a name sitting near any of its floors
    #: flipped in and out on consecutive sessions: membership changed on 126 of 250 days in
    #: 2023 with 114 entries. These three turn it into a *state*: a name is judged on how much
    #: of the recent window it qualified for, not on today alone.
    #:
    #: The band between the two counts is where a holding is neither bought nor sold, which is
    #: what lets a holding stay put until something genuinely changes.
    eligibility_window: int = 10
    #: Runs, out of ``eligibility_window``, a name must have been eligible for before it may
    #: be opened.
    entry_min_eligible_days: int = 8
    #: A holding is sold once it has been eligible on this many runs or fewer.
    exit_max_eligible_days: int = 3

    # -- ranking and hysteresis -----------------------------------------------------------
    #: 5 rather than the spec's 3: measured across 6M/4M/3M replay windows, five holdings
    #: added roughly 5pp of return with no increase in drawdown. Momentum concentration
    #: sounds decisive but a single wrong leader dominates a 3-name book.
    max_positions: int = 5
    min_base_score: float = 0.25
    entry_rank_max: int = 5
    exit_rank_max: int = 7
    min_score_delta_to_replace: float = 0.35

    # -- sentiment (phased in last; both weights default to off) --------------------------
    sentiment_weight: float = 0.0
    sentiment_size_scale: float = 0.0
    sentiment_clip: float = 2.0
    sentiment_lookback_minutes: int = 120

    # -- sizing and risk ------------------------------------------------------------------
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
    #: +1.0 measured best across 6M/4M/3M at full coverage, and unusually it improved return
    #: *and* drawdown together (19.4% / -8.0% against 17.0% / -10.8% at zero). Portfolio
    #: volatility barely moved, so it is selecting better rather than simply betting bigger.
    #: Expect that to invert in a sharp reversal -- this presses on the wildest names.
    volatility_tilt: float = 1.0
    #: Daily window for the per-name volatility estimate that ``volatility_tilt`` reads.
    vol_estimation_days: int = 20
    #: Reject any name whose 20-day annualized volatility exceeds this ceiling. 0 = off.
    vol_ceiling: float = 0.0
    #: Reject any name whose 5-day vol is above its 20-day vol by more than this ratio
    #: (e.g. 0.3 means 5d vol must not exceed 20d vol by 30%). 0 = off.
    vol_rising_threshold: float = 0.0
    #: Maximum ratio of current intraday range (high-low)/close to the 20-day average range.
    #: Names with range expansion beyond this multiple are flagged. 0 = off.
    range_expansion_limit: float = 0.0
    #: Minimum distance (as fraction of price) above the 100-day MA to treat a climax
    #: signal as a sell. Below this distance, the same pattern is a buy-the-dip. 0 = off.
    climax_ma_distance_min: float = 0.0
    #: Minimum volume ratio (current / 20d avg) to confirm a climax signal. 0 = off.
    climax_volume_ratio_min: float = 0.0

    rebalance_weight_threshold: float = 0.03
    #: How far to move toward the new target on each run, as a fraction: the book becomes
    #: ``(1 - lambda) * current + lambda * target``. 1.0 jumps straight there, which is what
    #: this did before the knob existed.
    #:
    #: A no-trade band and a partial adjustment brake different things, which is why both are
    #: here. The band is a *filter* -- once a move clears it, the position goes all the way to
    #: target -- so it suppresses small drift but does nothing about a target that swings hard
    #: every session. The partial move is a *regulariser*: it lets the book track a genuine
    #: trend within a few runs while a one-session target spike moves it only a fraction of the
    #: way, and reverts for free when the spike does.
    #:
    #: Exits are exempt: a name the strategy has decided to drop is dropped, not decayed to
    #: zero over a week, which would leave a broken holding on the book for as long as the
    #: strategy took to notice.
    rebalance_step: float = 1.0
    minimum_trade_notional: float = 100.0
    minimum_trade_nav_fraction: float = 0.005
    defensive_max_positions: int = 2

    @classmethod
    def from_runtime_config(cls, config: Any) -> "RallyRotationConfig":
        return load_tuning(cls, tuning_section(config, "rally_rotation"))

    @property
    def symbols(self) -> list[str]:
        """Everything the algorithm needs priced."""
        return sorted(set(self.risk_on_universe) | set(self.defensive_universe))

    @property
    def uses_sentiment(self) -> bool:
        """Sentiment is opt-in, so a baseline run costs no provider calls at all."""
        return abs(self.sentiment_weight) > 0 or abs(self.sentiment_size_scale) > 0

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
        """The slowest selection horizon, in sessions, for the daily-bar depth calculation."""
        return math.ceil(
            max(
                self.selection_horizon_nano_minutes,
                self.selection_horizon_micro_minutes,
                self.selection_horizon_meso_minutes,
                self.selection_horizon_macro_minutes,
            )
            / TRADING_MINUTES_PER_DAY
        )

    @property
    def required_daily_bars(self) -> int:
        # The selection horizons are served from these bars now, so they have to be counted
        # here: they used to be answered from a separate intraday window that no longer
        # exists. The smoothing tail rides on top, since the score is EMA'd across samples.
        smoothing_days = math.ceil(self.score_ema_minutes / TRADING_MINUTES_PER_DAY)
        return (
            max(
                self.etf_ma_days,
                self.etf_abs_return_days,
                self.vol_estimation_days,
                self.trend_ma_days,
                self.trend_return_days,
                self.selection_horizon_days + smoothing_days,
            )
            + 5
        )


# =========================================================================================
# Pure feature maths
