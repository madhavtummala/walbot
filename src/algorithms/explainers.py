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
                "effect": "Raise a symbol's budget to buy more contracts of it; set it to zero and that symbol stops trading. A budget below one contract's premium opens nothing.",
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
                "effect": "Higher buys far more into a dislocation: at 0.5, one sigma below the average deploys 1.5x the budget and two sigma deploys 2x. Above 1/3 it stops buying rich names at all.",
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
            "Ranks every ETF against the others and holds the best few, but only from among "
            "those already in an uptrend of their own. The ranking is relative -- a robust "
            "z-score blended over four horizons -- so it says which name is leading its field, "
            "never whether the field is worth being in; that second question is what the "
            "eligibility floors and the volatility ceiling answer, and a name failing them is "
            "unranked and therefore unheld whatever it scores. When too few qualify the book "
            "sits in the defensive sleeve rather than in the least bad name. Positions are "
            "sized in proportion to score, tilted by volatility, with no per-name cap, so a "
            "single qualifying name can hold the whole book. Ranking, entry and replacement "
            "happen on the rerank clock; the one-session crash stop answers to no clock."
        ),
        "formula": [
            "eligible(i) = price above MA(etf_ma_days) and R60 > etf_min_abs_return",
            "              and R20 > etf_min_fast_return and vol(i) <= vol_ceiling",
            "score(i) = 0.05.z(1d) + w_micro.z(micro) + w_meso.z(meso) + w_macro.z(macro)",
            "z(.) is a robust cross-sectional z-score (median/MAD), smoothed over score_ema_days",
            "",
            "every rerank_interval_days -- rank the eligible names and re-select:",
            "  enter if ranked <= entry_rank_max, score >= 0 (the universe median), and ranked",
            "    there on >= entry_min_eligible_days of the last eligibility_window runs",
            "  hold while ranked <= exit_rank_max -- a held name is exempt from the settling",
            "    period, but NOT from eligibility: an ineligible name is never ranked, so it",
            "    cannot be retained and the position is sold",
            "  a challenger needs min_score_delta_to_replace over the weakest incumbent to",
            "    displace it; a free slot is filled without any margin",
            "",
            "every session, whatever that clock says:",
            "  sell any holding down max_daily_drop in a single session",
            "",
            "w(i) = max(score, 0) x sigma(i)^volatility_tilt, normalised to risk_on_gross_max.",
            "  No per-name cap, so one qualifying name can take the book; risk_on_gross_max is",
            "  the only lever on how much is deployed in total.",
            "whatever is left undeployed, and anything held when nothing qualifies, sits in",
            "  the defensive universe rather than in cash",
        ],
        "parameters": {
            "risk_on_universe": {"what": "The ETFs it may hold.", "effect": "A wider list gives the ranking more to choose from; a narrower one concentrates the book. Either way every existing name's z-score moves, since the score is cross-sectional."},
            "defensive_universe": {"what": "Where the book sits when nothing qualifies, and where undeployed gross is parked.", "effect": "Short-duration choices (BIL) make risk-off flat and cash-like. TLT or GLD make it an active macro bet that can lose while risk-on is closed."},
            "rerank_interval_days": {
                "what": "Trading days between re-rankings. Counted in runs, so 5 means five sessions.",
                "effect": (
                    "Selection, entry, replacement and the considered exits all happen on this clock; "
                    "the -10% crash stop does not and runs every session regardless. The slowest "
                    "selection horizon is twelve sessions, so re-ranking daily asks the score a "
                    "question it cannot answer that fast. 0 re-ranks every run."
                ),
            },
            "micro_days": {"what": "Short return horizon, in market days.", "effect": "Longer blends toward the trend horizons and reacts slower; shorter tracks the last few sessions and flips rank on noise. About a week suits daily bars."},
            "meso_days": {"what": "Medium return horizon, in market days.", "effect": "Longer measures leadership over a slower window and holds through more chop; shorter re-ranks on recent strength. Half the selection weight sits here and in macro."},
            "nano_days": {"what": "Fastest return horizon, in market days.", "effect": "1 is the shortest a daily bar can express, and is the setting. Longer overlaps micro and stops being a distinct reading."},
            "macro_days": {"what": "Slowest return horizon, in market days. Carries half the score.", "effect": "Longer favours established trends and cuts turnover sharply, at the cost of reacting late. Past ~40 days it decays rather than improving."},
            "w_nano": {"what": "Score weight on the one-day horizon.", "effect": "Higher leans the score on a single day's move; lower defers to the slower horizons. The horizon's presence matters more than its weight."},
            "w_micro": {"what": "Score weight on the short horizon.", "effect": "Higher decides more ties on recent strength; lower defers to the slower horizons."},
            "w_meso": {"what": "Score weight on the medium horizon.", "effect": "Higher favours persistent leadership over recent strength; lower lets a fast mover displace a steady one."},
            "w_macro": {"what": "Score weight on the slowest horizon.", "effect": "Raising it makes the book slower and stickier."},
            "robust_zscore": {"what": "Use median/MAD z-scores instead of mean/standard deviation.", "effect": "On, one event-driven spike stops distorting every other name's score. Off is the classic mean/standard-deviation z-score, which that spike moves."},
            "risk_adjusted_score": {"what": "Rank on return divided by the symbol's own volatility, not raw return.", "effect": "On, a 58%-vol theme and a 14%-vol index compete on trend quality. Off, the ranking rewards amplitude and the wildest riser usually wins."},
            "score_ema_days": {"what": "Market days of smoothing applied to the composite score.", "effect": "More smoothing means fewer rank flips on noise, and a slower response to a real change. Below one day it rounds to no smoothing at all."},
            "etf_ma_days": {"what": "Each ETF's own absolute-trend window.", "effect": "The core dual-momentum filter: nothing below its own trend can be held at any rank."},
            "etf_abs_return_days": {"what": "Medium-term absolute-momentum lookback per ETF.", "effect": "Longer demands a more established advance before a name is eligible."},
            "etf_min_abs_return": {"what": "Minimum return over that lookback.", "effect": "Zero means 'must have gone up'. Raising it demands a margin over flat."},
            "etf_fast_return_days": {"what": "Short lookback used to catch deterioration.", "effect": "Longer needs a more sustained decline before a holding is dropped; shorter reacts to a single bad week and raises turnover."},
            "max_daily_drop": {
                "what": "A holding falling this much in one session is sold immediately.",
                "effect": "The only stop this algorithm has, and it works at a daily cadence. Lower stops out on ordinary volatility; 0 turns it off.",
            },
            "eligibility_window": {"what": "How many runs the eligibility count looks back over.", "effect": "Longer makes membership slower to change in both directions."},
            "entry_min_eligible_days": {"what": "Runs in that window a name must have been ranked inside entry_rank_max before it can be opened.", "effect": "Higher demands a settled signal and enters later; at 1 entry is stateless again."},
            "etf_min_fast_return": {"what": "Floor on that short return.", "effect": "Less negative ejects weakening names sooner and increases turnover."},
            "max_positions": {"what": "How many risk-on names it holds at once.", "effect": "Fewer concentrates in the leader; more diversifies but dilutes the signal."},
            "entry_rank_max": {"what": "Worst rank that may be newly entered.", "effect": "Tighter than exit_rank_max on purpose: it is harder to get in than to stay in."},
            "exit_rank_max": {"what": "Rank at which an incumbent is finally dropped.", "effect": "Wider than entry_rank_max gives a holding room to wobble without being sold. It only protects a name that slipped in rank -- one that fails a gate is unranked, and is sold whatever this says."},
            "min_score_delta_to_replace": {"what": "Score advantage a challenger needs to displace a holding.", "effect": "The anti-churn knob. At 0 it swaps on any improvement and trades constantly."},
            "volatility_tilt": {"what": "Exponent on volatility in sizing: weight follows score x sigma ** tilt.", "effect": "-1 is risk parity, so calm names get the big positions. 0 ignores volatility. +1 leans into it, concentrating in the wildest movers -- more return while a trend runs, more damage when it turns."},
            "risk_on_gross_max": {"what": "Cap on total invested fraction of equity.", "effect": "Below 1.0 it always holds cash. The simplest single lever on overall risk."},
            "vol_estimation_days": {"what": "Daily window for the per-name volatility estimate.", "effect": "Longer smooths the per-name volatility that volatility_tilt sizes on; shorter lets one violent week resize the book."},
            "vol_ceiling": {"what": "Maximum annualised volatility for eligibility. 0 = off.", "effect": "Lower excludes more volatile names and sells a holding whose volatility rises through it. 0 turns the gate off entirely."},
            "rebalance_weight_threshold": {"what": "Smallest weight change worth trading.", "effect": "Higher tolerates more drift from target in exchange for less churn."},
            "minimum_trade_notional": {"what": "Floor on the dollar size of any single order.", "effect": "Higher suppresses more small adjustments and lets the book drift further from target; lower corrects sooner and trades more often."},
            "minimum_trade_nav_fraction": {"what": "The same floor as a fraction of equity.", "effect": "The same floor as a share of equity, so it grows with the account. Whichever of the two is larger applies."},
            "defensive_max_positions": {"what": "How many defensive names to hold in risk-off.", "effect": "One is a pure cash-equivalent stance; two or more splits risk-off across, say, bills and gold."},
        },
    },
    "options_flip": {
        "summary": (
            "Buys calls on trending symbols, bidding at a pullback level comparable sessions "
            "actually reached and selling into the rebound. Every price is a limit, so a missed "
            "fill is an outcome the model prices rather than one it assumes away."
        ),
        "formula": [
            "strength(i) = sum over horizons of w_h x return_h / (annual_vol x sqrt(days_h/252))",
            "  a threshold, not a rank: every symbol with strength >= min_trend_strength is",
            "  a candidate, judged alone rather than against the others",
            "",
            "three gates, all in ATR of the symbol's own range -- never in percent:",
            "  1. bull intact:  price > SMA(regime_fast_ma_days), price > VWAP or recovering",
            "     off the opening-range low, and (open - prior_close)/ATR >= -max_gap_down_atr",
            "  2. levels reachable:  E = P - k_entry x ATR,  T = E + k_target x ATR",
            "     k_entry  = dip quantile at entry_reach, over comparable past sessions",
            "     k_target = rebound quantile at exit_reach, over those that dipped",
            "     and fraction_of_session_remaining >= entry_cutoff_fraction",
            "  3. base case pays:  profit >= min_profit_per_contract",
            "",
            "the contract itself has to clear four more, and any one refuses the trade:",
            "  expiry >= min_dte · strike within the delta band of target_delta",
            "  open interest >= min_open_interest · quoted spread <= max_spread_pct",
            "  the quote is younger than max_quote_age_seconds",
            "  the symbol's board budget buys at least one contract at the ask",
            "and the underlying's own annualised vol must sit under max_annual_volatility",
            "",
            "priced through the full greeks, not delta alone:",
            "  dC = delta x dS + 0.5 x gamma x dS^2 + vega x dIV + theta x dt",
            "  base: dS = T - E, IV +iv_change_base   ->  profit = dC x 100, gross",
            "  bad:  dS = 0,     IV +iv_change_bad    ->  max_debit, the most the case supports",
            "",
            "entry -- a pullback limit, never a chase:",
            "  starts near the bid and steps toward the mark by entry_patience, but only",
            "  while price is in the entry zone and the thesis holds",
            "  never above max_debit; cancelled at the entry cutoff",
            "",
            "exit -- a target that concedes rather than ratchets:",
            "  sell limit = entry + (exit_gain_share - exit_ask_decay x days_held) x modelled_gain",
            "  converges on the mark at the hold deadline, or once the bull gate has read",
            "  closed for several runs -- one closed read barely moves it",
            "  stop_loss_pct, when set, is the only thing that ends a reversal early",
        ],
        "parameters": {
            # Ordered by how much each one moves the outcome, most consequential first.
            "plan": {
                "what": "The dollar budget for each symbol, per position, set on the bubble board.",
                "effect": (
                    "Raise a symbol's budget to buy more contracts of it; set it to zero and "
                    "that symbol stops trading. A budget below one contract's premium opens "
                    "nothing. The budget is also the loss cap, since a long option cannot lose "
                    "more than its premium."
                ),
            },
            "stop_loss_pct": {"what": "Loss cap as a fraction of the debit. Zero disables the stop entirely.", "effect": "Tighter (0.25) ends a reversal sooner but cuts winners on ordinary theta and IV drift. 0 removes the stop, leaving the hold deadline as the only exit."},
            "max_hold_sessions": {"what": "Sessions to hold before the deadline exit takes over.", "effect": "Longer gives the target more sessions to arrive, and prices a wider target to match. Shorter forces the deadline exit sooner, on a target priced for less room."},
            "target_delta": {"what": "The delta the strike is aimed at.", "effect": "Higher earns more per point of underlying move, costs premium that is mostly intrinsic, and buys a contract fewer people trade -- flow concentrates at and out of the money."},
            "entry_reach": {"what": "Where the entry sits, as the share of comparable sessions that reached it.", "effect": "Lower is a deeper, cheaper entry that fills less often. Pairs with exit_reach; both are probabilities rather than offsets."},
            "exit_reach": {"what": "Where the target sits, as the share of comparable pulled-back sessions that reached it.", "effect": "Lower asks a more ambitious target, reached on fewer of the days that dipped. Higher asks less and is met more often."},
            "entry_patience": {"what": "How stubbornly the buy holds its price as the session runs out. Higher is more patient.", "effect": "Higher holds the bid at the pullback level: fewer fills, better prices. Lower walks it toward the mark, filling more often and paying more."},
            "exit_patience": {"what": "How stubbornly the sell holds its ask as the deadline approaches. Higher is more patient.", "effect": "Higher holds out for the modelled target and risks meeting the deadline unsold. Lower concedes toward the mark earlier, taking less but taking it."},
            "exit_gain_share": {"what": "Share of the modelled gain the sell limit asks for on the day of entry.", "effect": "Asking for part of the move is what makes the exit executable rather than theoretical. It concedes each session, reaching zero at the deadline."},
            "min_profit_per_contract": {"what": "Smallest predicted move worth opening for, in dollars per contract, gross of commission.", "effect": "Higher trades less often and only on wider predicted moves. Lower admits thinner setups, where spread and commission eat more of the edge."},
            "level_lookback_days": {"what": "Sessions the dip and run quantiles are learned from.", "effect": "Longer averages more sessions, so one violent stretch cannot set the level. Shorter tracks the current regime and can read a recent tail back out as a forecast."},
            "min_trend_strength": {"what": "Smallest trend strength a candidate must carry, in the symbol's own sigma.", "effect": "Higher demands a stronger trend and arms fewer symbols. Lower admits weaker ones; at 0 anything trending at all becomes a candidate."},
            "min_dte": {"what": "Nearest expiry to trade, in days.", "effect": "Higher buys more time and less decay per day, for more premium. Lower is cheaper and decays faster -- under a week theta outweighs direction."},
            "min_open_interest": {"what": "Open interest floor on the chosen contract.", "effect": "Higher restricts to contracts with a counterparty waiting, so fewer strikes qualify. Lower admits thin ones a resting order may never trade against."},
            "max_spread_pct": {"what": "Ceiling on the quoted spread, as a fraction of the mid.", "effect": "Lower refuses wide markets, where the midpoint every estimate is built on is a guess. Higher admits them and pays that width on the way out."},
            "max_gap_down_atr": {"what": "Largest opening gap DOWN still an ordinary session, in ATR.", "effect": "Downside only. An up-gap is followed by a smaller pullback, so it is directionally favourable and merely harder to fill into -- which the reach probability already prices."},
            "max_annual_volatility": {"what": "Ceiling on annualised realised volatility.", "effect": "Lower excludes more volatile names, where the premium already prices a bigger move than the model forecasts and a correct call still loses. Higher admits them."},
        },
    },
}


def explainer_for(algorithm_id: str) -> dict[str, Any]:
    """Explanation for one algorithm, with an empty shape when none is written yet."""
    entry = EXPLAINERS.get(str(algorithm_id or ""))
    if not entry:
        return {"summary": "", "formula": [], "parameters": {}}
    return entry
