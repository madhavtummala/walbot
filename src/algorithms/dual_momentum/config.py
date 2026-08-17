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

STATE_KEY = "dual_momentum_runtime"

EPSILON = 1e-9

#: Trading days per year, for annualising a daily volatility estimate.
TRADING_DAYS = 252


@dataclass(frozen=True)
class DualMomentumConfig:
    """Every knob, with the spec's starting values.

    These are research defaults, not recommended live settings: they need converting to your
    bar frequency and evaluating post-cost, walk-forward, before any of them means anything.
    """

    # -- universes ------------------------------------------------------------------------
    risk_on_universe: list[str] = field(default_factory=lambda: ["QQQM", "VTI", "IWM", "IEMG", "XSD"])
    defensive_universe: list[str] = field(default_factory=lambda: ["BIL", "IEF", "AGG", "GLD"])
    benchmark: str = field(default="QQQM", metadata={"coerce": "symbol"})

    # -- decision cadence -------------------------------------------------------------------
    #: How often each *kind* of action may be taken, in days. The algorithm runs every
    #: session; that is the rate at which it looks, not the rate at which it should act, and
    #: collapsing the two is what made a medium-term signal trade like a fast one.
    #:
    #: The asymmetry is the whole point. Risk is continuous and opportunity is not: a holding
    #: that breaks down leaves today, while a better one is bought when the relevant clock next
    #: comes round. Measured over 2026 the theme set changed on 50% of sessions and theme
    #: entries and exits were 44% of all turnover, against a score whose slowest horizon is
    #: twelve sessions.
    #:
    #: 0 means "every run", which is the behaviour all four had before they existed.

    #: Entering or leaving a *theme* -- a change of view, and the most expensive kind.
    theme_rotation_interval_days: int = 0
    #: Opening any new position, including a new name inside a theme already held.
    entry_interval_days: int = 0
    #: Re-splitting weight between the names inside a theme that is already held, and
    #: re-sizing to the volatility target. Pure sizing: no change of view, same spread paid.
    intra_theme_interval_days: int = 0
    #: Closing a holding that failed eligibility or the crash stop. Keep at 0: an exit that
    #: waits for a clock is a stop-loss that does not stop anything.
    exit_interval_days: int = 0

    risk_refresh_minutes: int = 15

    # -- selection score ------------------------------------------------------------------
    #: Kept at 0 -- this algorithm reads daily bars only, so there is no intraday grid to
    #: prefer. See ``requirements``, which asks for no history window at all.
    intraday_bar_minutes: int = 0
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
    w_nano: float = 0.10
    w_micro: float = 0.20
    w_meso: float = 0.35
    w_macro: float = 0.35
    robust_zscore: bool = True
    #: Rank on return-per-unit-of-volatility rather than raw return. Off, the cross-section
    #: rewards amplitude and the highest-volatility theme that happened to rise wins.
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
    #: worse than no knob, so it is gone. ``session_drawdown_breached`` remains in
    #: ``algorithms/risk.py`` for Fast Momentum, which genuinely runs intraday.
    max_daily_drop: float = 0.10
    # -- eligibility persistence ----------------------------------------------------------
    #: Eligibility is a stateless per-day test, so a name sitting near any of its floors
    #: flipped in and out on consecutive sessions: membership changed on 126 of 250 days in
    #: 2023 with 114 entries. These three turn it into a *state*: a name is judged on how much
    #: of the recent window it qualified for, not on today alone.
    #:
    #: The band between the two counts is where a holding is neither bought nor sold, which is
    #: what lets a theme stay put until something genuinely changes.
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
    #: How many holdings may come from one theme. 0 is off.
    #:
    #: The cross-sectional score treats the universe as N independent choices, which it is
    #: not: measured on daily returns, eight of the twenty deployed names correlate 0.79+ with
    #: SPY, and {SPY, IWM, RSP, QUAL, VTV} cluster at 0.80 all-pairs. A four-position book
    #: drawn from that block is one bet sliced four ways -- in June 2023 it was exactly that,
    #: rotating daily among QQQM/XSD/QUAL/SPY/IJH/IWM/RSP and returning +0.86% against SPY's
    #: +5.09% while fully invested.
    #:
    #: Capping per theme forces the book to span genuinely different exposures, and lets the
    #: ranking still pick the strongest name *within* a theme rather than pruning the universe.
    max_positions_per_theme: int = 0
    #: Symbol -> theme label. Symbols absent from the map are each their own theme, so an
    #: incomplete map constrains nothing rather than silently lumping names together.
    themes: dict[str, str] = field(default_factory=dict)
    #: How much a sibling in the *same* theme must beat the sitting name by before the book
    #: swaps between them. Separate from ``min_score_delta_to_replace``, which governs whether
    #: a whole theme is displaced: rotating QQQM for XSD is a different decision from dropping
    #: growth for energy, and only the second is a change of view.
    intra_theme_delta_to_replace: float = 0.5
    min_base_score: float = 0.25
    entry_rank_max: int = 5
    exit_rank_max: int = 7
    min_score_delta_to_replace: float = 0.35
    cooldown_after_exit: int = 4

    # -- sentiment (phased in last; both weights default to off) --------------------------
    sentiment_weight: float = 0.0
    sentiment_size_scale: float = 0.0
    sentiment_clip: float = 2.0
    sentiment_lookback_minutes: int = 120

    # -- sizing and risk ------------------------------------------------------------------
    name_weight_max: float = 0.35
    risk_on_gross_max: float = 1.0
    #: How much volatility should move a position's size, as an exponent: weight is
    #: proportional to score x sigma ** volatility_tilt.
    #:
    #:   -1.0  risk parity -- divide by volatility, so calm names get the big positions
    #:    0.0  score alone -- volatility does not enter sizing at all
    #:   +1.0  lean in -- scale up with volatility, which is what an ungated momentum book
    #:         like Fast Momentum does implicitly by never dividing
    #:
    #: One number rather than a boolean because the useful settings are not binary: the
    #: question is how hard to press, and the answer is a market regime opinion.
    #: +1.0 measured best across 6M/4M/3M at full coverage, and unusually it improved return
    #: *and* drawdown together (19.4% / -8.0% against 17.0% / -10.8% at zero). Portfolio
    #: volatility barely moved, so it is selecting better rather than simply betting bigger.
    #: Expect that to invert in a sharp reversal -- this presses on the wildest names.
    volatility_tilt: float = 1.0
    #: A crash brake, not a governor. The spec's 12% target was written for a diversified
    #: book; against a 23-58% volatility ETF universe it would scale the portfolio to 28-55%
    #: invested *permanently*, which is a large structural drag on a strategy whose whole
    #: job is to be in the market when the market is working. At 0.30 with a 1.3 trigger,
    #: scaling engages only above ~39% ex-ante volatility.
    target_portfolio_vol: float = 0.30
    vol_estimation_days: int = 20
    vol_scale_floor: float = 0.25
    high_vol_trigger: float = 1.3
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
    #: theme it belonged to took to fade.
    rebalance_step: float = 1.0
    minimum_trade_notional: float = 100.0
    minimum_trade_nav_fraction: float = 0.005
    defensive_max_positions: int = 2

    @classmethod
    def from_runtime_config(cls, config: Any) -> "DualMomentumConfig":
        return load_tuning(cls, tuning_section(config, "dual_momentum"))

    @property
    def symbols(self) -> list[str]:
        """Everything the algorithm needs priced, benchmark included.

        ``benchmark`` gates nothing -- the regime layer that used it is gone -- but it is still
        carried here so the dashboard can chart it, and it is normally in the risk-on
        universe anyway.
        """
        return sorted(set(self.risk_on_universe) | set(self.defensive_universe) | {self.benchmark})

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
                self.selection_horizon_days + smoothing_days,
            )
            + 5
        )


# =========================================================================================
# Pure feature maths
