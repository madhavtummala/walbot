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
            "micro_days": {"what": "Short return horizon, in market days.", "effect": "Bridges the one-day horizon and the trend horizons. Around a week is the natural setting on daily bars."},
            "meso_days": {"what": "Medium return horizon, in market days.", "effect": "Half the selection weight sits here and in macro; this is what 'leadership' means."},
            "nano_days": {"what": "Fastest return horizon, in market days.", "effect": "Leave at 1. Removing this horizon cost 6.2pp over a 12-month replay while zeroing its weight cost 1.5pp, so the two are not equivalent and the reason is not established."},
            "macro_days": {"what": "Slowest return horizon, in market days. Carries half the score.", "effect": "Longer favours established trends and cuts turnover sharply, at the cost of reacting late. Past ~40 days it decays: 120 was the worst long ladder measured, on both return and drawdown."},
            "w_nano": {"what": "Score weight on the one-day horizon.", "effect": "Almost inert as a weight -- 0.05 to 0 moved a 12-month replay by 1.5pp. The horizon's presence matters more than its weight."},
            "w_micro": {"what": "Score weight on the short horizon.", "effect": "Moderate influence; useful for breaking ties between similar trends."},
            "w_meso": {"what": "Score weight on the medium horizon.", "effect": "Raising it favours persistent leadership over recent strength."},
            "w_macro": {"what": "Score weight on the slowest horizon.", "effect": "Raising it makes the book slower and stickier."},
            "robust_zscore": {"what": "Use median/MAD z-scores instead of mean/standard deviation.", "effect": "On, one event-driven ETF spike stops distorting everyone else's score. Off is the classic z-score."},
            "risk_adjusted_score": {"what": "Rank on return divided by the symbol's own volatility, not raw return.", "effect": "On, a 58%-vol theme and a 14%-vol index compete on trend quality. Off, the ranking rewards amplitude and the wildest riser almost always wins."},
            "score_ema_days": {"what": "Market days of smoothing applied to the composite score.", "effect": "More smoothing means fewer rank flips on noise, and a slower response to a real change. Below one day it rounds to no smoothing at all."},
            "etf_ma_days": {"what": "Each ETF's own absolute-trend window.", "effect": "The core dual-momentum filter: nothing below its own trend can be held at any rank."},
            "etf_abs_return_days": {"what": "Medium-term absolute-momentum lookback per ETF.", "effect": "Longer demands a more established advance before a name is eligible."},
            "etf_min_abs_return": {"what": "Minimum return over that lookback.", "effect": "Zero means 'must have gone up'. Raising it demands a margin over flat."},
            "etf_fast_return_days": {"what": "Short lookback used to catch deterioration.", "effect": "Stops a name staying eligible on a stale long-horizon score while it collapses."},
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
            "vol_estimation_days": {"what": "Daily window for the per-name volatility estimate.", "effect": "Only read by volatility_tilt, which is the one place volatility now enters."},
            "vol_ceiling": {"what": "Maximum annualised volatility for eligibility. 0 = off.", "effect": "The one volatility gate left, and it is worth ~28pp of return and ~25pp of drawdown over a 12-month replay. Note it also applies to holdings through the ranking, so a name whose volatility rises through the ceiling is sold."},
            "rebalance_weight_threshold": {"what": "Smallest weight change worth trading.", "effect": "Higher tolerates more drift from target in exchange for less churn."},
            "minimum_trade_notional": {"what": "Floor on the dollar size of any single order.", "effect": "Stops trivial orders whose costs exceed their benefit."},
            "minimum_trade_nav_fraction": {"what": "The same floor as a fraction of equity.", "effect": "The larger of the two applies, so the minimum scales with the account."},
            "defensive_max_positions": {"what": "How many defensive names to hold in risk-off.", "effect": "One is a pure cash-equivalent stance; two splits between, say, bills and gold."},
        },
    },
    "options_flip": {
        "summary": (
            "A bull-regime, pullback-entry, rebound-exit long-call trade on a deadline-aware "
            "limit schedule. Candidates come from Rally Rotation's ranking of its risk-on "
            "universe. Three gates decide whether one trades: the bull thesis has to be intact "
            "today, the entry and target levels have to be ones comparable sessions actually "
            "reached, and the base case -- priced through the full greeks, net of execution and "
            "fee reserves -- has to clear the floor. A limit controls price and never guarantees "
            "a fill, so missed entries and missed exits are modelled rather than assumed away."
        ),
        "formula": [
            "# CANDIDATES -- each symbol judged alone, never against the others",
            "strength = sum over horizons of  w_h x return_h / (annual_vol x sqrt(days_h/252))",
            "trade every symbol with strength >= min_trend_strength -- no ranking, no limit",
            "",
            "  Nothing compares symbols. Adding or removing a name cannot change what the",
            "  others do. A cross-sectional score could not express 'all of them are",
            "  trending' or 'none is' -- its mean is zero by construction.",
            "",
            "# GATE 1 -- is the bull thesis intact today?",
            "price > SMA(regime_fast_ma_days)      # the 50d stack and the slope are readings",
            "",
            "  The slope gate was removed: a 20-day mean turns only after a move has",
            "  happened. Over 2026-08-17..20 IBIT ran 36.40 -> 41.19 while its slope was",
            "  still negative, and the strategy sat out every session of it. The broad-",
            "  market check went too: candidates are already chosen cross-sectionally, so",
            "  a name outperforming a flat index is the trade, not a disqualifier.",
            "(open - prior_close) / ATR(atr_days) >= -max_gap_down_atr   # up-gaps allowed",
            "price > VWAP, or recovering off the opening-range low",
            "",
            "  Distances in ATR, never in percent: percent cannot say how big a move is for the",
            "  name making it. An extreme gap is a different distribution, not a better one, so",
            "  the pullback statistics below do not apply to it.",
            "",
            "# GATE 2 -- where would we transact, and how often is each level reached?",
            "E = P - k_entry  * ATR        k_entry  from the conditional dip quantile",
            "T = E + k_target * ATR        k_target from the conditional rebound quantile",
            "",
            "  Both quantiles are taken over comparable past sessions -- bucketed on how far",
            "  price sits from its open -- measuring travel from THIS minute to the close, not",
            "  from the open. The rebound is measured AFTER the dip: counting a rally that",
            "  happened before the entry would have filled credits profit the position never had.",
            "",
            "P_touch  = share of comparable sessions that reached E",
            "P_target = share of THOSE that then reached T",
            "E is placed at the entry_reach reach quantile of the dip distribution",
            "T is placed at the target_reach quantile of the run distribution ON THE DAYS",
            "                 THAT DIPPED -- so each level means the reach it asks for,",
            "                 and no separate minimum can disagree with it",
            "         and fraction_of_session_remaining >= entry_cutoff_fraction",
            "",
            "  This is what replaces a fixed offset. A bid always the same distance below the",
            "  open is a filter that admits days which fell: over 185 sessions per symbol the",
            "  days a fixed 0.40% bid MISSED were worth +1.73% (IBIT) and +0.67% (GLD) against",
            "  -0.34% and -0.42% for the days it filled, at p < 0.001.",
            "",
            "# GATE 3 -- does the base case pay, through the full greeks?",
            "dC ~ delta*dS + 0.5*gamma*dS^2 + vega*dIV + theta*dt",
            "",
            "  bad : dS = 0            IV +iv_change_bad    -- a correct-looking entry that fails",
            "  base: dS = T - E        IV +iv_change_base   -- the trade decision",
            "  good: dS = 1.5*(T - E)  IV flat              -- profit-cap context only",
            "",
            "profit = base_change * 100          # gross: the predicted band, in dollars",
            "max_debit = the largest price the base case still supports",
            "",
            "  Delta alone prices a small move and nothing else. Gamma bends the delta over a",
            "  move worth trading; theta is 4% of premium a day on a near-dated contract; vega",
            "  carries the IV crush that loses money on a correctly-called direction.",
            "",
            "# ENTRY -- a pullback limit, never a chase",
            "buy limit starts near the bid, inside the spread",
            "raise it ONLY while price is in the entry zone and the thesis holds",
            "never above max_debit; cancel at the entry cutoff",
            "",
            "  Good convergence: the underlying reaches the planned pullback and the limit steps",
            "  toward the mid. Bad convergence: the underlying rallies away and the limit is",
            "  raised just to get filled -- which converts a pullback strategy into buying an",
            "  extended move.",
            "",
            "# EXIT -- a schedule that gives ground, not a ratchet",
            "sell limit = entry + (exit_gain_share - exit_ask_decay * days_held) * modelled_gain",
            "at the deadline: converge on the market across what is left of the session",
            "",
            "  The reverse of the ratchet this replaced. A target that only ever rose asked more",
            "  of a position the longer it failed to deliver, which is how a winner becomes a",
            "  deadline exit at the bid.",
        ],
        "parameters": {
            # Ordered by how much each one moves the outcome, most consequential first.
            "symbols": {"what": "Symbols to consider. Empty means the account's tradable universe.", "effect": "This algorithm's own list -- another strategy's universe is never read, so retuning one cannot silently change what the other trades."},
            "contracts_per_trade": {"what": "Contracts per position -- the unit of risk.", "effect": "A long call cannot lose more than its premium, so the unit IS the loss cap. That is what makes running without a stop coherent, and why this matters most when the stop is off."},
            "stop_loss_pct": {"what": "Loss cap as a fraction of the debit. Zero disables the stop entirely.", "effect": "Disabled, the bracket becomes a lone profit target. Off by default because premium falls on theta and implied volatility with the directional case intact, so a premium stop cuts winners for reasons unrelated to the thesis."},
            "max_hold_sessions": {"what": "Sessions to hold before the deadline exit takes over.", "effect": "It also sets the horizon the target is priced over -- the run available grows with the hold -- so the two cannot be set apart. Pricing a long hold off a short target makes a correct direction unprofitable."},
            "target_reach": {"what": "Where the target sits, as the share of comparable pulled-back sessions that reached it.", "effect": "Measured on the days that actually dipped, so it means what it says. Lower is more ambitious and reached less often."},
            "exit_gain_share": {"what": "Share of the modelled gain the sell limit asks for on the day of entry.", "effect": "Asking for part of the move is what makes the exit executable rather than theoretical. It concedes each session, reaching zero at the deadline."},
            "exit_patience": {"what": "How stubbornly the sell holds its ask as the deadline approaches. Higher is more patient.", "effect": "The impatient side by design: a position reaching its deadline unsold is sold at whatever is offered, so conceding early is cheaper than conceding at gunpoint."},
            "entry_reach": {"what": "Where the entry sits, as the share of comparable sessions that reached it.", "effect": "Lower is a deeper, cheaper entry that fills less often. Pairs with target_reach; both are probabilities rather than offsets."},
            "entry_patience": {"what": "How stubbornly the buy holds its price as the session runs out. Higher is more patient.", "effect": "The patient side by design: chasing a rising ask turns a pullback trade into a momentum one, and an unfilled entry costs only the opportunity. It never crosses the mark."},
            "min_profit_per_contract": {"what": "Smallest predicted move worth opening for, in dollars per contract, gross of commission.", "effect": "The predicted band priced through the greeks. The strictest gate in the set, and the one that decides how often this trades at all."},
            "target_delta": {"what": "The delta the strike is aimed at.", "effect": "Higher earns more per point of underlying move, costs premium that is mostly intrinsic, and buys a contract fewer people trade -- flow concentrates at and out of the money."},
            "level_lookback_days": {"what": "Sessions the dip and run quantiles are learned from.", "effect": "Long enough that one exceptional stretch cannot set the tail. A short window is read back out as a forecast, which puts the target far beyond anything the symbol normally does."},
            "min_trend_strength": {"what": "Smallest trend strength a candidate must carry, in the symbol's own sigma.", "effect": "A threshold, not a rank: it means the same on a quiet symbol as on a violent one. Every symbol clearing it may be taken and none need be."},
            "max_notional_per_trade": {"what": "Dollar ceiling per position, priced at the ask. Zero means no cap.", "effect": "It trims the unit and never sets it. Set it as a fraction of equity: on a cheap premium the unit buys a lot of contracts."},
            "min_dte": {"what": "Nearest expiry to trade, in days.", "effect": "Under a week the theta curve is steepest, and a very near expiry is theta-negative before direction is considered."},
            "min_open_interest": {"what": "Open interest floor on the chosen contract.", "effect": "Asks whether a resting order finds a counterparty at all. It does not catch cost -- that is max_spread_pct."},
            "max_spread_pct": {"what": "Ceiling on the quoted spread, as a fraction of the mid.", "effect": "The entry rests and never crosses, but the exit has to get out and the stop is denominated in premium. Open interest alone does not catch a wide market."},
            "max_gap_down_atr": {"what": "Largest opening gap DOWN still an ordinary session, in ATR.", "effect": "Downside only. An up-gap is followed by a smaller pullback, so it is directionally favourable and merely harder to fill into -- which the reach probability already prices."},
            "max_annual_volatility": {"what": "Ceiling on annualised realised volatility.", "effect": "Past it the premium already prices a bigger move than the model forecasts, so a correct call still loses."},
        },
    },
}


def explainer_for(algorithm_id: str) -> dict[str, Any]:
    """Explanation for one algorithm, with an empty shape when none is written yet."""
    entry = EXPLAINERS.get(str(algorithm_id or ""))
    if not entry:
        return {"summary": "", "formula": [], "parameters": {}}
    return entry
