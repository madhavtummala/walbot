"""Human explanations of what each algorithm does and what its knobs move.

Kept in one module rather than on the classes so the wording stays consistent across
algorithms and the dashboard has a single place to read from. The trade-off is that a knob
renamed in an algorithm will not automatically update here -- ``tests/test_explainers.py``
guards that by checking every documented parameter still exists in the saved config.

Each entry is:
  summary   -- one paragraph on what the algorithm is trying to do
  formula   -- the actual arithmetic, in the same terms the code uses
  behavior  -- what it does across a day/month, including when it deliberately does nothing
  parameters -- per knob: what it is, and which direction to move it for which effect
"""

from __future__ import annotations

from typing import Any

EXPLAINERS: dict[str, dict[str, Any]] = {
    "bursty_dca": {
        "summary": (
            "Accrues exactly the same monthly budget as DCA, but sizes each buy proportionally "
            "to how far the price has fallen from its peak. Deeper dips get bigger buys. "
            "Buys at 11 AM, sells at 3 PM (requires ``frequency: 1hr`` on the dashboard)."
        ),
        "formula": [
            "drawdown = max(0, (peak - price) / peak)",
            "size = monthly_budget × (1 + drawdown × scaling_factor)",
            "cap = max_monthly_multiple × monthly_budget × (1 + drawdown × cap_boost)",
            "buys execute at 11 AM, sells execute at 3 PM",
        ],
        "behavior": (
            "Runs twice per day: buys at 11 AM when spreads are tightest, sells at 3 PM when "
            "institutional liquidity peaks. Deeper dips produce larger buys automatically. "
            "The monthly cap scales with drawdown via cap_boost, so crash months can deploy "
            "more capital without a hard threshold."
        ),
        "parameters": {
            "plan": {
                "what": "The monthly dollar budget for each symbol, set on the bubble board.",
                "effect": (
                    "Its own plan, separate from DCA's: the two accrue the same way but deploy "
                    "on different terms, so they are tuned independently. Higher budget accrues "
                    "faster, so a qualifying dip buys more."
                ),
            },
            "regime_ma_days": {
                "what": "Reference moving average period used to compute drawdown from peak.",
                "effect": (
                    "Longer (300) uses a longer lookback for the peak reference. Shorter (50) "
                    "uses a shorter lookback, making the peak recency-sensitive."
                ),
            },
            "scaling_factor": {
                "what": "How much to scale buy size per unit of drawdown.",
                "effect": (
                    "size = budget × (1 + scaling_factor × drawdown). At 20 and 5% drawdown: "
                    "2× budget. At 20 and 10% drawdown: 3× budget. Higher values concentrate "
                    "more budget into deep dips."
                ),
            },
            "cap_boost": {
                "what": "How much to scale the monthly cap with drawdown.",
                "effect": (
                    "cap = max_monthly_multiple × budget × (1 + cap_boost × drawdown). "
                    "At 1 and 10% drawdown: cap doubles. At 1 and 20% drawdown: cap triples. "
                    "Set to 0 for a fixed cap regardless of drawdown."
                ),
            },
            "max_monthly_multiple": {
                "what": "Ceiling on total deployment per symbol per month, in multiples of the monthly budget.",
                "effect": (
                    "Bounds how far ahead of schedule it can get. At 3.0 a month of repeated dips can "
                    "spend three months of budget. Set to 1.0 to never exceed the plan rate."
                ),
            },
        },
    },
    "rally_rotation": {
        "summary": (
            "Relative momentum picks the leaders, absolute momentum decides whether anything "
            "may be held at all. Every ETF is scored against the others, must clear its own "
            "trend and return floors to be ranked, and the top few are held. Sizing is "
            "proportional to score and inversely proportional to volatility. A crash stop "
            "fires every session; all other decisions happen on a slower rerank clock."
        ),
        "formula": [
            "eligible(i) = price above MA(etf_ma_days) and R60 > etf_min_abs_return",
            "              and R20 > etf_min_fast_return",
            "score(i) = w_nano.z(nano) + w_micro.z(micro) + w_meso.z(meso) + w_macro.z(macro)",
            "z(.) is a robust cross-sectional z-score (median/MAD), smoothed over score_ema_minutes",
            "",
            "every rerank_interval_days -- rank the eligible names and re-select:",
            "  enter if ranked <= entry_rank_max, score >= min_base_score, eligible on",
            "    >= entry_min_eligible_days of the last eligibility_window runs",
            "  hold while ranked <= exit_rank_max; a held name faces eligibility alone, not the",
            "    quality floor or the settling period",
            "  exit when eligible on <= exit_max_eligible_days runs, or it breaks the exit band",
            "    (the entry floors widened by exit_threshold_slack)",
            "  a challenger needs min_score_delta_to_replace over the weakest incumbent to",
            "    displace it; a free slot is filled without any margin",
            "",
            "every session, whatever that clock says:",
            "  sell any holding down max_daily_drop in a single session",
            "",
            "w(i) = max(score - min_base_score, 0) x sigma(i)^volatility_tilt, normalised to",
            "  risk_on_gross_max. No per-name cap, so one qualifying name can take the book.",
            "whatever is left undeployed, and anything held when nothing qualifies, sits in",
            "  the defensive universe rather than in cash",
        ],
        "behavior": (
            "Runs once per session on daily bars, but only re-ranks every rerank_interval_days: "
            "the rate at which it looks and the rate at which it acts are separate, and the "
            "score's slowest horizon is twelve sessions, so asking it for a new answer daily "
            "mostly buys noise. The exception is the crash stop, which runs every session. "
            "It holds the defensive sleeve rather than the least-bad risk-on ETF, and fewer "
            "qualifying names means holding less, never a lower bar. Entry and exit are "
            "deliberately asymmetric: a name that would not be bought today is not thereby "
            "worth selling."
        ),
        "parameters": {
            "risk_on_universe": {"what": "The ETFs it may hold.", "effect": "Scores are cross-sectional, so adding or removing a name changes every other name's z-score."},
            "defensive_universe": {"what": "Where the book sits when nothing qualifies, and where undeployed gross is parked.", "effect": "Short-duration choices (BIL) make risk-off flat; TLT or GLD make it an active macro bet."},
            "rerank_interval_days": {
                "what": "Trading days between re-rankings. Counted in runs, so 5 means five sessions.",
                "effect": (
                    "Selection, entry, replacement and the considered exits all happen on this clock; "
                    "the -10% crash stop does not and runs every session regardless. The slowest "
                    "selection horizon is twelve sessions, so re-ranking daily asks the score a "
                    "question it cannot answer that fast. 0 re-ranks every run."
                ),
            },
            "selection_horizon_nano_minutes": {"what": "Fastest return horizon, in minutes. 390 is one session.", "effect": "One session is the floor on daily bars; this horizon carries the least score weight by design."},
            "selection_horizon_micro_minutes": {"what": "Short return horizon, in minutes. 780 is two sessions.", "effect": "Bridges the fast and trend horizons."},
            "selection_horizon_meso_minutes": {"what": "Medium return horizon, in minutes. 1170 is three sessions.", "effect": "Half the selection weight sits here and in macro; this is what 'leadership' means."},
            "selection_horizon_macro_minutes": {"what": "Slowest return horizon, in minutes. 4680 is twelve sessions.", "effect": "Longer favours established trends and cuts turnover, at the cost of reacting late."},
            "w_nano": {"what": "Score weight on the fastest horizon.", "effect": "Raising it makes selection chase intraday moves, which is what the fork was meant to stop."},
            "w_micro": {"what": "Score weight on the short horizon.", "effect": "Moderate influence; useful for breaking ties between similar trends."},
            "w_meso": {"what": "Score weight on the medium horizon.", "effect": "Raising it favours persistent leadership over recent strength."},
            "w_macro": {"what": "Score weight on the slowest horizon.", "effect": "Raising it makes the book slower and stickier."},
            "robust_zscore": {"what": "Use median/MAD z-scores instead of mean/standard deviation.", "effect": "On, one event-driven ETF spike stops distorting everyone else's score. Off is the classic z-score."},
            "risk_adjusted_score": {"what": "Rank on return divided by the symbol's own volatility, not raw return.", "effect": "On, a 58%-vol theme and a 14%-vol index compete on trend quality. Off, the ranking rewards amplitude and the wildest riser almost always wins."},
            "score_ema_minutes": {"what": "Minutes of smoothing applied to the composite score. 1170 is three sessions.", "effect": "More smoothing means fewer rank flips on noise, and a slower response to a real change. Below one session it rounds to no smoothing at all."},
            "min_universe_coverage": {"what": "Share of the risk-on universe that must have enough history before it will trade at all.", "effect": "Below it the algorithm reports a data gap and holds the defensive sleeve, rather than reading a thin cache as a bear market."},
            "etf_ma_days": {"what": "Each ETF's own absolute-trend window.", "effect": "The core dual-momentum filter: nothing below its own trend can be held at any rank."},
            "etf_abs_return_days": {"what": "Medium-term absolute-momentum lookback per ETF.", "effect": "Longer demands a more established advance before a name is eligible."},
            "etf_min_abs_return": {"what": "Minimum return over that lookback.", "effect": "Zero means 'must have gone up'. Raising it demands a margin over flat."},
            "etf_fast_return_days": {"what": "Short lookback used to catch deterioration.", "effect": "Stops a name staying eligible on a stale long-horizon score while it collapses."},
            "max_daily_drop": {
                "what": "A holding falling this much in one session is sold immediately.",
                "effect": "The only stop this algorithm has, and it works at a daily cadence. Lower stops out on ordinary volatility; 0 turns it off.",
            },
            "eligibility_window": {"what": "How many runs the eligibility count looks back over.", "effect": "Longer makes membership slower to change in both directions."},
            "entry_min_eligible_days": {"what": "Runs in that window a name must have been eligible for before it can be opened.", "effect": "Higher demands a settled signal and enters later; at 1 entry is stateless again."},
            "exit_max_eligible_days": {"what": "A holding is sold once it has been eligible on this many runs or fewer.", "effect": "Lower holds through longer patches of ineligibility. The gap between this and entry_min_eligible_days is the band where a holding is left alone."},
            "exit_threshold_slack": {
                "what": "Slack added to every eligibility floor before a *held* name is sold.",
                "effect": "0 means entry and exit share a threshold, so anything sitting near it round-trips on noise. Higher holds through deeper dips and exits later.",
            },
            "etf_min_fast_return": {"what": "Floor on that short return.", "effect": "Less negative ejects weakening names sooner and increases turnover."},
            "max_positions": {"what": "How many risk-on names it holds at once.", "effect": "Fewer concentrates in the leader; more diversifies but dilutes the signal."},
            "min_base_score": {"what": "Score quality floor, in cross-sectional z units.", "effect": "Raise to hold fewer names more often; the shortfall stays in the defensive sleeve, not in a weaker name."},
            "entry_rank_max": {"what": "Worst rank that may be newly entered.", "effect": "Tighter than exit_rank_max on purpose: it is harder to get in than to stay in."},
            "exit_rank_max": {"what": "Rank at which an incumbent is finally dropped.", "effect": "Wider than entry_rank_max gives a holding room to wobble without being sold."},
            "min_score_delta_to_replace": {"what": "Score advantage a challenger needs to displace a holding.", "effect": "The anti-churn knob. At 0 it swaps on any improvement and trades constantly."},
            "sentiment_weight": {"what": "Coefficient on clipped sentiment inside the score.", "effect": "Start at 0. Non-zero lets news move the ranking, which is exactly what the price gates are there to prevent."},
            "sentiment_size_scale": {"what": "Coefficient on sentiment as a position-size modifier.", "effect": "0.05 with a clip of 2 gives at most a 10% size change. It can never create a position price logic rejected."},
            "sentiment_clip": {"what": "Cap on the normalised sentiment input.", "effect": "Bounds how much a single loud news cycle can move either sentiment term."},
            "sentiment_lookback_minutes": {"what": "How recent a story must be to count.", "effect": "Longer keeps stale headlines alive in the score; shorter makes sentiment sparse."},
            "volatility_tilt": {"what": "Exponent on volatility in sizing: weight follows score x sigma ** tilt.", "effect": "-1 is risk parity, so calm names get the big positions. 0 ignores volatility. +1 leans into it, concentrating in the wildest movers -- more return while a trend runs, more damage when it turns."},
            "risk_on_gross_max": {"what": "Cap on total invested fraction of equity.", "effect": "Below 1.0 it always holds cash. The simplest single lever on overall risk."},
            "vol_estimation_days": {"what": "Daily window for the per-name volatility estimate.", "effect": "Only read by volatility_tilt, which is the one place volatility now enters."},
            "vol_ceiling": {"what": "Maximum annualised volatility for eligibility. 0 = off.", "effect": "Rejects volatile names. Raise to allow more risk-on exposure; lower to be pickier."},
            "vol_rising_threshold": {"what": "Maximum ratio of 5-day vol to 20-day vol for eligibility. 0 = off.", "effect": "Rejects names whose vol is spiking. Higher allows more vol expansion before rejecting."},
            "range_expansion_limit": {"what": "Maximum intraday range expansion (current range / 20-day avg range). 0 = off.", "effect": "Flags names with unusually large daily ranges. Used by the climax exit signal."},
            "climax_ma_distance_min": {"what": "Minimum distance above the 100-day MA to treat a climax pattern as a sell. 0 = off.", "effect": "Below this distance the same pattern is a buy-the-dip, not a climax exit."},
            "climax_volume_ratio_min": {"what": "Minimum volume spike (current / 20-day avg) to confirm a climax signal. 0 = off.", "effect": "Higher demands more volume confirmation before exiting on a climax."},
            "trend_ma_days": {"what": "Names must be above this moving average to be eligible. 0 = off.", "effect": "Filters out short-term rallies in names still in a medium-term downtrend."},
            "trend_return_days": {"what": "Names must have positive return over this many days to be eligible. 0 = off.", "effect": "Requires a minimum trend duration before a name can be held."},
            "rebalance_weight_threshold": {"what": "Smallest weight change worth trading.", "effect": "Higher tolerates more drift from target in exchange for less churn."},
            "rebalance_step": {
                "what": "How far to move toward the new target each run: (1-step) x current + step x target.",
                "effect": (
                    "A different brake from rebalance_weight_threshold, which only filters small "
                    "drift. This damps a target that swings hard every session, so a one-day spike "
                    "costs a fraction of a round trip and reverts for free. Exits are exempt. "
                    "1.0 jumps straight to target; 0.25-0.5 roughly halves turnover, and over 2023 "
                    "cost more return than it saved."
                ),
            },
            "minimum_trade_notional": {"what": "Floor on the dollar size of any single order.", "effect": "Stops trivial orders whose costs exceed their benefit."},
            "minimum_trade_nav_fraction": {"what": "The same floor as a fraction of equity.", "effect": "The larger of the two applies, so the minimum scales with the account."},
            "defensive_max_positions": {"what": "How many defensive names to hold in risk-off.", "effect": "One is a pure cash-equivalent stance; two splits between, say, bills and gold."},
        },
    },
    "intraday_pick": {
        "summary": (
            "Directional options: the macro trend on the benchmark determines whether to buy "
            "calls (bull) or puts (bear). The highest-scoring candidate from the tradable "
            "universe gets a buy limit entry at a discount to fair value, and a GTC sell "
            "limit at the target exit price. If wrong, the sell limit just sits there -- "
            "the user sized the position for the loss."
        ),
        "formula": [
            "macro = bull if price > SMA(trend_ma_period) and recent_return > 0",
            "        bear if price < SMA(trend_ma_period) and recent_return < 0",
            "        flat otherwise (no trade)",
            "",
            "score(symbol) = vol_regime + range_expansion + direction_aligned",
            "                 + range_budget_remaining + volume_ratio",
            "",
            "vol_regime = realized_vol(short_window) / realized_vol(long_window)",
            "range_expansion = today_range / ATR(atr_period)",
            "direction_aligned = momentum aligned with macro AND above/below VWAP",
            "",
            "option_type = call if macro == bull, put if macro == bear",
            "entry_limit = fair_value * (1 - entry_discount_pct)",
            "exit_limit = fair_value * (1 + exit_target_pct)",
            "",
            "Two orders placed simultaneously:",
            "  1. BUY N contracts at entry_limit (limit, day)",
            "  2. SELL N contracts at exit_limit (limit, GTC)",
        ],
        "behavior": (
            "Runs intraday on a configurable schedule (default every 15 minutes during market "
            "hours). Each cycle: detect macro trend on the benchmark, score candidates, and "
            "if a qualifying name is found, place entry and exit limit orders. The GTC exit "
            "order persists until filled or cancelled. Lost trades are accepted -- the user "
            "controls risk via contracts_per_signal and max_notional_per_trade."
        ),
        "parameters": {
            "benchmark_symbol": {"what": "Symbol used to determine macro trend direction.", "effect": "SPY for broad market, QQQ for tech-heavy, IWM for small-caps."},
            "trend_ma_period": {"what": "Moving average period for the macro trend filter.", "effect": "Longer = slower to switch direction, fewer false signals."},
            "trend_lookback_days": {"what": "Days of recent return to confirm trend direction.", "effect": "Higher = needs more sustained trend to trade."},
            "vol_short_window": {"what": "Short window for volatility regime ratio.", "effect": "Lower = more responsive to recent vol spikes."},
            "vol_long_window": {"what": "Long window for volatility regime ratio.", "effect": "Higher = smoother baseline, more meaningful ratio."},
            "range_expansion_threshold": {"what": "Minimum today_range/ATR to consider a candidate.", "effect": "Higher = only trade on high-range days."},
            "entry_discount_pct": {"what": "How far below fair value to place the buy limit.", "effect": "Higher = cheaper fill but lower fill probability."},
            "exit_target_pct": {"what": "How far above fair value to place the sell limit.", "effect": "Higher = bigger win per trade but lower fill probability."},
            "contracts_per_signal": {"what": "Number of option contracts per trade.", "effect": "Directly controls risk per trade."},
            "max_notional_per_trade": {"what": "Maximum dollar cost per trade.", "effect": "Hard cap on position size."},
        },
    },
}


def explainer_for(algorithm_id: str) -> dict[str, Any]:
    """Explanation for one algorithm, with an empty shape when none is written yet."""
    entry = EXPLAINERS.get(str(algorithm_id or ""))
    if not entry:
        return {"summary": "", "formula": [], "behavior": "", "parameters": {}}
    return entry
