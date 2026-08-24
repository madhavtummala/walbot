"""Human explanations of what each algorithm does and what its knobs move.

Kept in one module rather than on the classes so the wording stays consistent across
algorithms and the dashboard has a single place to read from. The trade-off is that a knob
renamed in an algorithm will not automatically update here -- ``tests/test_explainers.py``
guards that by checking every documented parameter still exists in the saved config.

Each entry is:
  summary   -- one paragraph on what the algorithm is trying to do
  formula   -- the actual arithmetic, in the same terms the code uses
  parameters -- per knob: what it is, and which direction to move it for which effect

There was a fourth, ``behavior``: a closing paragraph on what the algorithm does across a day
or a month. It is gone because it could only ever restate its neighbours -- the summary above
it and the formula between them already say what runs, when, and in what order, so the third
telling read as filler on a card a person opens to find one knob.
"""

from __future__ import annotations

from typing import Any

EXPLAINERS: dict[str, dict[str, Any]] = {
    "bursty_dca": {
        "summary": (
            "Accrues a monthly budget per symbol, then sizes each order by two things at once: "
            "how far the price sits from its moving average, and how far ahead of or behind its "
            "plan that symbol already is. Cheap names buy more, rich names buy less or sell, "
            "and a symbol that has already overspent resists spending again until it catches "
            "up. Runs on the schedule its binding states — by default once a weekday at 11 AM."
        ),
        "formula": [
            "z = (moving_average − price) / stdev, clamped to ±3σ",
            "conviction  = max(0, 1 + scaling_factor × (z if buying else −z))",
            "willingness = 1 + relax_depth × tanh(backlog_months / relax_months)",
            "allowance   = monthly_budget × (backlog_months + relax_months × conviction)",
            "size = min(monthly_budget × conviction × willingness, allowance, cap room)",
            "cap = max_monthly_multiple × monthly_budget",
            "if size < one share, but the accrued balance covers one, send exactly one",
        ],
        "parameters": {
            "plan": {
                "what": "The monthly dollar budget for each symbol, set on the bubble board.",
                "effect": (
                    "The rate the budget accrues at, per symbol, and the base every multiple "
                    "is applied to. A higher budget accrues faster, so a dislocation buys more."
                ),
            },
            "regime_ma_days": {
                "what": "Window for the moving average and for the standard deviation that normalizes distance from it.",
                "effect": (
                    "Longer (300) measures dislocation against a slower average, so it reads a "
                    "multi-month decline as cheap. Shorter (50) reacts to recent moves and will "
                    "call a symbol fairly priced sooner after it falls."
                ),
            },
            "scaling_factor": {
                "what": "Extra multiples of the monthly budget per standard deviation of favourable dislocation.",
                "effect": (
                    "size multiple = 1 + scaling_factor × z. At 0.5 and 1σ below the average: "
                    "1.5× budget. At 0.5 and 2σ: 2×. Measured in sigma rather than percent, so "
                    "one setting works for both a broad bond ETF and a volatile single name. "
                    "Above 1/3 the buy side reaches zero before −3σ and stops buying rich names "
                    "entirely."
                ),
            },
            "relax_months": {
                "what": (
                    "Width of the backlog resistance curve, and the overdraft a neutral price "
                    "may borrow — both in months of budget."
                ),
                "effect": (
                    "Longer (6) makes backlog matter slowly and lets a symbol run months ahead "
                    "of plan. Shorter (0.5) pulls hard toward the plan rate and makes it behave "
                    "much more like straight DCA. This is the knob that governs the long-run "
                    "spend rate; max_monthly_multiple only bounds a single month."
                ),
            },
            "relax_depth": {
                "what": "How far the backlog factor swings either side of 1.0 as the backlog saturates.",
                "effect": (
                    "At 0.7 a symbol months overspent deploys 0.3× and one months behind "
                    "deploys 1.7×. Set to 0 to disable backlog resistance and size on valuation "
                    "alone. Values near 1 let an overspent symbol stop trading almost entirely."
                ),
            },
            "max_monthly_multiple": {
                "what": "Ceiling on total deployment per symbol per month, in multiples of the monthly budget.",
                "effect": (
                    "A backstop rather than the main control — relax_months governs the pacing. "
                    "At 3.0 a month of repeated dislocations can spend three months of budget. "
                    "Set to 1.0 to never exceed the plan rate within a month."
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
    "options_flip": {
        "summary": (
            "One option contract per symbol, bought at a predicted intraday low and bracketed at "
            "the exchange. A multi-day trend proposes the direction and pre-market can only veto "
            "it -- disagreement means no trade that day. The entry rests as a limit order priced "
            "from how far comparable past sessions pulled back before going the right way, and "
            "walks in as the day's budget depletes. On a fill an OCO goes to the broker: a "
            "profit limit that only ratchets up, and a stop that never moves."
        ),
        "formula": [
            "direction = call if price > SMA(trend_ma_period) and return > 0",
            "            put  if price < SMA(trend_ma_period) and return < 0",
            "            none otherwise (no trade)",
            "",
            "premarket_score = (premarket_last / prior_close - 1) / typical_daily_move",
            "trade only if sign(premarket_score) == sign(direction)",
            "        and |premarket_score| >= premarket_confirm_min      # veto, never a vote",
            "",
            "# ENTRY -- how far it pulls back before going the right way, on days that agreed",
            "adverse_d     = (open - low) / open                  # (high - open) / open for puts",
            "expected_dip  = quantile(adverse_d over the last excursion_lookback_days,",
            "                         1 - target_fill_probability)",
            "predicted_low = session_open * (1 - expected_dip)    # a price level, not an offset",
            "",
            "decay          = fraction_of_session_remaining ** entry_decay_power",
            "bid_underlying = price - (price - predicted_low) * decay",
            "bid_option     = mark + delta * (bid_underlying - price)",
            "",
            "  target_fill_probability sets WHERE the bid sits; entry_decay_power sets HOW FAST",
            "  it walks from there to the mid. It never crosses the spread.",
            "",
            "# EXIT -- the opposite tail, over the whole holding period",
            "favourable_h  = (highest high over the next max_hold_sessions - open) / open",
            "                                                     # (open - lowest low) for puts",
            "expected_run  = quantile(favourable_h over the last excursion_lookback_days,",
            "                         1 - exit_fill_probability)",
            "target_underlying = price * (1 + expected_run)       # (1 - ...) for puts",
            "target_option     = mark + delta * (target_underlying - price)",
            "",
            "  Measured on the opposite tail to the entry: a call's target comes from how far the",
            "  underlying RISES above its open, not how far it falls. Measured over",
            "  max_hold_sessions, not one day -- the move available over two sessions is much",
            "  larger, and a two-day position asking a one-day price sells its first morning.",
            "",
            "  exit_fill_probability is the exit's counterpart to target_fill_probability: it sets",
            "  the target's ambition. There is no exit decay -- the target ratchets up while the",
            "  position is held, and only converges on the mark once the deadline is reached.",
            "",
            "On fill, one OCO at the exchange:",
            "  SELL LIMIT at target_option (raised only, never lowered)",
            "  SELL STOP  at fill * (1 - stop_loss_pct), never moved",
            "Flattened after max_hold_sessions market days.",
        ],
        "parameters": {
            "symbols": {"what": "Underlyings to run, each with its own independent position.", "effect": "A portfolio of one-contract lifecycles, not a shortlist to pick a winner from."},
            "trend_ma_period": {"what": "Moving average period setting the direction.", "effect": "Longer = slower to change sides, fewer whipsaws."},
            "trend_min_return": {"what": "Move required over the trend window to count as trending.", "effect": "Higher = needs a more decisive move before trading at all."},
            "premarket_confirm_min": {"what": "Pre-market move required, as a multiple of the symbol's typical day.", "effect": "Higher = trades fewer days. Normalised, so one value works across a mixed symbol list. Pre-market can only veto, never set a direction."},
            "min_dte": {"what": "Nearest expiry the algorithm will trade, in days.", "effect": "Around 10 keeps a one-to-two-day hold off the steepest part of the theta curve."},
            "target_delta": {"what": "The delta the strike is aimed at; puts use it negated.", "effect": "Delta is the moneyness dial: ~0.50 is at the money, higher is in the money, lower is out. Higher earns more per point of underlying move -- expected profit is exactly delta x band x 100 -- but is paid for in premium that is mostly intrinsic, and a larger loss if the stop hits. Candidates within 0.15 of the target compete on volume, then open interest."},
            "min_open_interest": {"what": "Open interest floor on the chosen contract.", "effect": "The only liquidity gate, and the real test: whether a resting order finds a counterparty at all. Varies by expiry more than by symbol -- SPY's monthly medians above 2,000 across in-band strikes while its next weekly medians 78, so a floor set for monthlies rejects every weekly."},
            "target_fill_probability": {"what": "Share of comparable past sessions that would have reached the entry bid.", "effect": "Higher = shallower bid, fills more often at a worse price. The only entry-price knob."},
            "entry_decay_power": {"what": "How fast the bid walks in from the predicted low toward the contract's midpoint.", "effect": "Lower is more patient. At 1.0 it reaches the midpoint by the close, so a day that never dipped still fills near fair value; below 1.0 it is still short of the midpoint at the last fire, so it effectively never converges -- more no-trade days, every fill at a price you chose. It never crosses the spread."},
            "stop_loss_pct": {"what": "Loss cap as a fraction of the premium paid, held at the exchange.", "effect": "Never moved once set. A market order when triggered, so it sells into the bid and the realised loss can overshoot the cap on a wide market -- that slippage is accepted rather than gated."},
            "exit_fill_probability": {"what": "The profit target's counterpart to the entry knob.", "effect": "Lower = a more ambitious target, held longer."},
            "max_hold_sessions": {"what": "Market days to hold before flattening, and the horizon the exit target is priced over.", "effect": "The more consequential half is the pricing: median upside runs 0.99% in a day and 1.74% in two, so a longer hold raises what the position asks for, not just how long it waits. At the deadline the ask converges on the market the way the entry bid does."},
            "max_annual_volatility": {"what": "Ceiling on annualised realised volatility.", "effect": "Above it the premium already prices a bigger move than the model forecasts, so a correct call still loses, and the stop is likelier to be taken out by noise. There is no floor: a quiet symbol produces a narrow band and min_expected_profit already refuses it, in dollars."},
            "excursion_lookback_days": {"what": "Sessions of history the excursion distribution is estimated from.", "effect": "Shortening it does NOT make the strategy more short-term -- the excursion is already a single-session measure, and this only sets the sample size. Stable from 20 to 90 sessions; at 10 it doubles, which is the sample talking rather than the market."},
            "min_expected_profit": {"what": "Smallest expected profit worth opening a position for, in dollars, gross of commissions.", "effect": "A different axis from trend_min_return: that one asks whether the underlying is trending, this asks whether the contract stands to move enough dollars to be worth the round trip. Set it above your own commissions with room to spare; 0 takes any positive expectation."},
            "contracts_per_position": {"what": "Contracts per open position.", "effect": "Directly controls risk per trade."},
            "max_notional_per_trade": {"what": "Hard dollar cap per position, priced at the ask.", "effect": "Reduces the contract count when the premium is rich."},
        },
    },
}


def explainer_for(algorithm_id: str) -> dict[str, Any]:
    """Explanation for one algorithm, with an empty shape when none is written yet."""
    entry = EXPLAINERS.get(str(algorithm_id or ""))
    if not entry:
        return {"summary": "", "formula": [], "parameters": {}}
    return entry
