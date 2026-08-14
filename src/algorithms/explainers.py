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
    "dca": {
        "summary": (
            "Buys a fixed dollar amount per symbol per month, spread continuously rather than "
            "in one lump. Each symbol accrues budget against elapsed wall-clock time and buys "
            "as soon as the accrued amount can clear the broker's minimum trade size."
        ),
        "formula": [
            "accrued = monthly_budget x (hours_since_last_buy / 730.5)",
            "min_executable = max(min_trade_dollars, one whole share if fractional shares are unavailable)",
            "buy when accrued >= min_executable, for the accrued amount",
        ],
        "behavior": (
            "Runs Mondays inside the trading window. Between runs the budget keeps accruing, so "
            "a missed run is not a missed contribution -- the next one simply buys more. A symbol "
            "whose accrued amount is still under the minimum trade size does nothing and keeps "
            "accruing, which is why small budgets on expensive shares buy infrequently."
        ),
        "parameters": {
            "__plan__": {
                "what": "The monthly dollar budget for each symbol, set on the bubble board.",
                "effect": (
                    "Higher budget accrues faster, so it trades more often and in larger size. "
                    "The bubble area is proportional to the dollar amount."
                ),
            },
        },
    },
    "bursty_dca": {
        "summary": (
            "Accrues exactly the same monthly budget as DCA, but refuses to spend it at an "
            "arbitrary moment. It waits for a dip inside an uptrend, then deploys the backlog. "
            "The spend rate over a month is the same; the timing is not."
        ),
        "formula": [
            "regime gate: close >= SMA(regime_ma_days)",
            "oversold: %B(percent_b_lookback, percent_b_num_std) < percent_b_threshold  OR  RSI(rsi_lookback) < rsi_threshold",
            "buy fires only when regime gate AND oversold are both true",
            "sell mirrors it: %B > (1 - percent_b_threshold) OR RSI > (100 - rsi_threshold)",
            "size (value averaging): gap = monthly_budget x elapsed_months - held_value",
            "clamped per trade to max_trade_multiple x monthly_budget",
            "clamped per month to max_monthly_multiple x monthly_budget (cumulative)",
        ],
        "behavior": (
            "Runs daily at the open rather than weekly, because a dip worth buying can open and "
            "close inside a week. On days when the trigger does not fire it buys nothing and the "
            "budget keeps accruing, so the next qualifying dip buys a larger amount. In a market "
            "that never dips below the bands it can sit out for weeks by design -- that backlog "
            "is the strategy, not a failure."
        ),
        "parameters": {
            "regime_ma_days": {
                "what": "Length of the moving average that defines 'uptrend'. Price must be above it to buy.",
                "effect": (
                    "Longer (300) means a slower, stickier trend filter that keeps buying through "
                    "corrections. Shorter (50) turns the gate off sooner in a decline, so it stops "
                    "buying earlier but also resumes earlier."
                ),
            },
            "percent_b_lookback": {
                "what": "Bollinger Band window used to compute %B, the position of price within the bands.",
                "effect": "Longer smooths the bands so only larger dips register. Shorter makes it twitchier.",
            },
            "percent_b_num_std": {
                "what": "How many standard deviations wide the Bollinger Bands are.",
                "effect": "Wider (2.5) means only deeper dips reach the lower band. Narrower (1.5) fires more often.",
            },
            "percent_b_threshold": {
                "what": "%B below this counts as oversold. 0.0 is exactly the lower band.",
                "effect": (
                    "Raise toward 0.2 to buy shallower dips more frequently. Push negative to demand "
                    "price break below the band before buying at all."
                ),
            },
            "rsi_lookback": {
                "what": "RSI period. 2 is the Connors-style very short RSI, not the classic 14.",
                "effect": "Longer makes RSI sluggish and rarely extreme, which effectively disables this leg of the trigger.",
            },
            "rsi_threshold": {
                "what": "RSI below this counts as oversold. It is an OR with the %B test, so either one fires the buy.",
                "effect": "Raise toward 20 to fire more often. Lower toward 5 to demand a sharper flush.",
            },
            "max_trade_multiple": {
                "what": "Ceiling on a single trade, in multiples of the monthly budget.",
                "effect": (
                    "This is the main risk control on catch-up. At 3.0 one dip can deploy three months "
                    "of budget at once. Lower to 1.0 to spread the backlog across several dips."
                ),
            },
            "max_monthly_multiple": {
                "what": "Ceiling on total deployment per symbol per month, in multiples of the monthly budget.",
                "effect": (
                    "Bounds how far ahead of schedule it can get. At 3.0 a month of repeated dips can "
                    "spend three months of budget. Set to 1.0 to never exceed the plan rate."
                ),
            },
            "value_averaging": {
                "what": "When on, trade size targets a value path rather than a fixed contribution.",
                "effect": (
                    "On, it buys more after the position has fallen behind its path and less after it "
                    "has run ahead. Off, it falls back to plain accrued-amount buying and the two "
                    "clamps above stop applying."
                ),
            },
        },
    },
    "fast_momentum": {
        "summary": (
            "Ranks a risk-on ETF universe against a defensive one on momentum measured at four "
            "horizons plus news sentiment, holds the best few, and sizes them with hard caps. It "
            "re-evaluates intraday on 15-minute bars."
        ),
        "formula": [
            "score = w_price_nano.z(nano) + w_price_micro.z(micro) + w_price_meso.z(meso)",
            "        + w_price_macro.z(macro) + w_sentiment.sentiment + pullback_bonus",
            "z(.) is a cross-sectional z-score, so scores are relative to the rest of the universe",
            "pullback_bonus = w_pullback_uptrend x z(meso) x min(|z(nano)|, pullback_nano_z_cap)",
            "  applied only when the macro trend is intact, z(meso) >= pullback_meso_z_threshold,",
            "  z(nano) <= pullback_nano_z_threshold, and micro_return >= pullback_min_micro_return",
            "hold the top max_positions candidates clearing min_risk_on_score and min_risk_on_micro_return",
            "an incumbent is only replaced if the challenger beats it by min_score_delta_to_replace",
        ],
        "behavior": (
            "Re-scores every refresh during market hours. Weights are capped per name and in total, "
            "and orders below the per-trade minimum or inside the rebalance threshold are skipped, "
            "so it does not churn on noise. If intraday volatility exceeds the limit it scales all "
            "weights down; if the intraday drawdown limit trips it goes defensive for the rest of "
            "the session."
        ),
        "parameters": {
            "risk_on_universe": {
                "what": "The ETFs it may hold when conditions are risk-on.",
                "effect": "Scores are cross-sectional, so adding or removing names changes every other name's z-score.",
            },
            "defensive_universe": {
                "what": "Where it hides when risk-on candidates fail their thresholds.",
                "effect": "Shorter-duration choices (BIL) make the defensive leg flatter; TLT makes it a rate bet.",
            },
            "max_positions": {
                "what": "How many risk-on names it holds at once.",
                "effect": "Fewer means a more concentrated bet on the top-ranked name. More diversifies but dilutes the signal.",
            },
            "max_single_position_weight": {
                "what": "Hard cap on any one position as a fraction of equity.",
                "effect": "Binds before max_positions does when few names qualify. At 0.25 nothing can exceed a quarter of the account.",
            },
            "max_gross_exposure": {
                "what": "Cap on total invested fraction of equity.",
                "effect": "Below 1.0 it always holds cash. This is the simplest single lever on overall risk.",
            },
            "min_risk_on_score": {
                "what": "Minimum composite score for a risk-on name to be eligible at all.",
                "effect": "Raise to demand clearly positive momentum, which means more time sitting defensive.",
            },
            "min_risk_on_micro_return": {
                "what": "Minimum raw intraday return, independent of the z-scored ranking.",
                "effect": "A sanity floor: stops it buying the best of a uniformly falling universe just because it is least bad.",
            },
            "min_score_delta_to_replace": {
                "what": "How much better a challenger must score before it displaces a held position.",
                "effect": (
                    "This is the anti-churn knob. At 0 it swaps on any improvement and trades constantly; "
                    "at 0.10 it only rotates on a decisive change."
                ),
            },
            "w_price_nano": {"what": "Weight on the shortest momentum horizon (last few bars).", "effect": "Higher makes it chase very recent moves and turn over faster."},
            "w_price_micro": {"what": "Weight on intraday momentum, roughly one session of bars.", "effect": "The default primary driver; raising it makes the strategy more intraday."},
            "w_price_meso": {"what": "Weight on the multi-week trend.", "effect": "Higher tilts toward established trends and reduces turnover."},
            "w_price_macro": {"what": "Weight on the multi-month trend.", "effect": "Higher makes it slow and position-like rather than tactical."},
            "w_sentiment": {"what": "Weight on news sentiment, clamped to [-1, 1] before weighting.", "effect": "Set to 0 to ignore news entirely and score on price alone."},
            "w_pullback_uptrend": {
                "what": "Weight on the buy-the-dip bonus for names in an uptrend that just pulled back.",
                "effect": "Higher favours entering on weakness inside a trend rather than on strength. 0 disables the bonus.",
            },
            "pullback_meso_z_threshold": {"what": "How strong the medium trend must be for a pullback to count.", "effect": "Higher restricts the bonus to the strongest trends only."},
            "pullback_nano_z_threshold": {"what": "How weak the immediate move must be to count as a pullback.", "effect": "More negative demands a deeper dip before the bonus applies."},
            "pullback_nano_z_cap": {"what": "Ceiling on how much dip depth can amplify the bonus.", "effect": "Stops a crash being scored as an unusually attractive pullback."},
            "pullback_min_micro_return": {"what": "Floor on intraday return for a pullback to still qualify.", "effect": "Prevents catching a name that is actively collapsing intraday."},
            "per_trade_value_min": {"what": "Minimum dollar value for any single order.", "effect": "Raise to avoid dust trades and commissions on tiny rebalances."},
            "rebalance_threshold": {"what": "Weight drift that must be exceeded before it trades to correct.", "effect": "Higher tolerates more drift and trades far less. The main turnover control alongside min_score_delta_to_replace."},
            "max_intraday_volatility": {"what": "Realised intraday volatility above which it de-risks.", "effect": "Lower makes it flinch earlier in choppy sessions."},
            "high_volatility_weight_scale": {"what": "How much weights are scaled when that volatility limit trips.", "effect": "0.5 halves every position; 1.0 disables the response."},
            "intraday_drawdown_limit": {"what": "Session drawdown that flips it defensive for the rest of the day.", "effect": "Closer to zero stops it out sooner and more often."},
            "nano_momentum_lookback_bars": {"what": "Bars in the shortest momentum window.", "effect": "Fewer bars means a noisier, faster signal."},
            "micro_momentum_lookback_bars": {"what": "Bars in the intraday momentum window (78 bars is about one session).", "effect": "Longer smooths the primary signal."},
            "meso_trend_lookback_days": {"what": "Days in the medium trend window.", "effect": "Longer defines trend over a broader stretch and reacts later."},
            "macro_trend_lookback_days": {"what": "Days in the long trend window used for the regime check.", "effect": "Longer keeps the macro gate open through deeper corrections."},
            "sentiment_lookback_minutes": {"what": "How recent a news item must be to count.", "effect": "Longer keeps stale headlines in the score."},
            "volatility_lookback_bars": {"what": "Bars used to measure realised intraday volatility.", "effect": "Shorter reacts faster to a volatility spike."},
            "min_defensive_score": {"what": "Minimum score for a defensive name to be chosen.", "effect": "Raise to demand the hiding place is itself trending, not merely flat."},
        },
    },
    "dual_momentum": {
        "summary": (
            "Relative momentum picks the leaders, absolute momentum decides whether anything "
            "risk-on may be held at all. A benchmark trend and breadth gate sets the regime, "
            "each ETF must clear its own absolute-trend test to be ranked, and a separate fast "
            "signal decides when to enter. Weights are score-over-volatility, scaled toward a "
            "portfolio volatility target."
        ),
        "formula": [
            "regime = benchmark above its MA and its 60-day return positive and breadth >= breadth_min",
            "         confirmed over regime_confirm_bars runs before risk-on is allowed",
            "eligible(i) = price above MA(etf_ma_days) and R60 > etf_min_abs_return",
            "              and R20 > etf_min_fast_return",
            "score(i) = w_nano.z(nano) + w_micro.z(micro) + w_meso.z(meso) + w_macro.z(macro)",
            "z(.) is a robust cross-sectional z-score (median/MAD), smoothed over score_ema_bars",
            "timing(i) = EMA(nano/sigma_nano - meso/sigma_meso) > momentum_change_enter",
            "            or the pullback setup (macro and meso leading while nano dips)",
            "enter if eligible and rank <= entry_rank_max and score >= min_base_score and timing",
            "hold while eligible and rank <= exit_rank_max, even if timing is false",
            "w(i) = max(score - min_base_score, 0) / sigma(i), normalised, capped at name_weight_max",
            "k = clip(target_portfolio_vol / sqrt(w' Sigma w), vol_scale_floor, 1) once vol exceeds",
            "    high_vol_trigger x target; below the floor it goes defensive instead",
        ],
        "behavior": (
            "Runs every 15 minutes for timing and risk, but only re-ranks every "
            "signal_refresh_minutes, so membership changes slowly while stops and scaling stay "
            "fast. In risk-off it holds the defensive universe rather than the least-bad risk-on "
            "ETF. Fewer qualifying names means holding less, never a lower bar. A challenger must "
            "beat an incumbent by min_score_delta_to_replace, an exited name waits out "
            "cooldown_after_exit, and breaching the intraday drawdown limit parks the book in the "
            "defensive sleeve for the rest of the session."
        ),
        "parameters": {
            "risk_on_universe": {"what": "The ETFs it may hold when the regime is risk-on.", "effect": "Scores are cross-sectional, so adding or removing a name changes every other name's z-score and the breadth reading."},
            "defensive_universe": {"what": "Where the book sits when risk-on is not permitted.", "effect": "Short-duration choices (BIL) make risk-off flat; TLT or GLD make it an active macro bet."},
            "benchmark": {"what": "The index proxy whose trend defines the market regime.", "effect": "QQQM gates a growth basket tightly; SPY is broader and flips less often."},
            "signal_refresh_minutes": {"what": "How often membership may change.", "effect": "Longer cuts turnover and lets winners run; shorter reacts to leadership changes sooner."},
            "risk_refresh_minutes": {"what": "How often timing and risk are re-evaluated.", "effect": "Also the unit for cooldown_after_exit. Shorter reacts faster but re-checks cost more."},
            "selection_horizon_nano": {"what": "Fastest return horizon, in intraday bars.", "effect": "Small values make the ranking twitchy; this horizon carries the least score weight by design."},
            "selection_horizon_micro": {"what": "Short return horizon, in bars.", "effect": "Bridges the fast and trend horizons."},
            "selection_horizon_meso": {"what": "Medium return horizon, in bars.", "effect": "Half the selection weight sits here and in macro; this is what 'leadership' means."},
            "selection_horizon_macro": {"what": "Slowest return horizon, in bars.", "effect": "Longer favours established trends and cuts turnover, at the cost of reacting late."},
            "w_nano": {"what": "Score weight on the fastest horizon.", "effect": "Raising it makes selection chase intraday moves, which is what the fork was meant to stop."},
            "w_micro": {"what": "Score weight on the short horizon.", "effect": "Moderate influence; useful for breaking ties between similar trends."},
            "w_meso": {"what": "Score weight on the medium horizon.", "effect": "Raising it favours persistent leadership over recent strength."},
            "w_macro": {"what": "Score weight on the slowest horizon.", "effect": "Raising it makes the book slower and stickier."},
            "robust_zscore": {"what": "Use median/MAD z-scores instead of mean/standard deviation.", "effect": "On, one event-driven ETF spike stops distorting everyone else's score. Off is the classic z-score."},
            "risk_adjusted_score": {"what": "Rank on return divided by the symbol's own volatility, not raw return.", "effect": "On, a 58%-vol theme and a 14%-vol index compete on trend quality. Off, the ranking rewards amplitude and the wildest riser almost always wins."},
            "score_ema_bars": {"what": "Bars of smoothing applied to the composite score.", "effect": "More smoothing means fewer rank flips on noise, and a slower response to a real change."},
            "benchmark_ma_days": {"what": "Trend window for the benchmark regime test.", "effect": "Longer keeps it risk-on through corrections; shorter exits earlier and re-enters more often."},
            "benchmark_return_days": {"what": "Absolute-momentum window for the benchmark.", "effect": "The 'is the market itself going up' test. Shorter reacts faster and whipsaws more."},
            "breadth_ma_days": {"what": "Trend window used for each name in the breadth count.", "effect": "Usually matched to benchmark_ma_days so the two gates agree on what a trend is."},
            "breadth_min": {"what": "Share of the risk-on universe that must be above its average.", "effect": "Higher demands a broad advance, so it sits defensive during narrow rallies."},
            "regime_confirm_bars": {"what": "Consecutive passing reads before risk-on is allowed.", "effect": "Higher avoids entering on one borderline reading, at the cost of entering later."},
            "regime_exit_confirm_bars": {"what": "Consecutive failing reads before leaving risk-on.", "effect": "Higher rides out single-bar scares; lower exits faster and pays the spread more often."},
            "etf_ma_days": {"what": "Each ETF's own absolute-trend window.", "effect": "The core dual-momentum filter: nothing below its own trend can be held at any rank."},
            "etf_abs_return_days": {"what": "Medium-term absolute-momentum lookback per ETF.", "effect": "Longer demands a more established advance before a name is eligible."},
            "etf_min_abs_return": {"what": "Minimum return over that lookback.", "effect": "Zero means 'must have gone up'. Raising it demands a margin over flat."},
            "etf_fast_return_days": {"what": "Short lookback used to catch deterioration.", "effect": "Stops a name staying eligible on a stale long-horizon score while it collapses."},
            "etf_min_fast_return": {"what": "Floor on that short return.", "effect": "Less negative ejects weakening names sooner and increases turnover."},
            "max_positions": {"what": "How many risk-on names it holds at once.", "effect": "Fewer concentrates in the leader; more diversifies but dilutes the signal."},
            "min_base_score": {"what": "Score quality floor, in cross-sectional z units.", "effect": "Raise to hold fewer names more often; the shortfall stays in the defensive sleeve, not in a weaker name."},
            "entry_rank_max": {"what": "Worst rank that may be newly entered.", "effect": "Tighter than exit_rank_max on purpose: it is harder to get in than to stay in."},
            "exit_rank_max": {"what": "Rank at which an incumbent is finally dropped.", "effect": "Wider than entry_rank_max gives a holding room to wobble without being sold."},
            "min_score_delta_to_replace": {"what": "Score advantage a challenger needs to displace a holding.", "effect": "The anti-churn knob. At 0 it swaps on any improvement and trades constantly."},
            "cooldown_after_exit": {"what": "Refresh intervals a name must wait before re-entry.", "effect": "Stops a name being sold and re-bought around a threshold. Multiplied by risk_refresh_minutes."},
            "momentum_change_ema_bars": {"what": "Smoothing on the fast momentum-change signal.", "effect": "More smoothing means fewer false entry triggers and slightly later entries."},
            "momentum_change_enter": {"what": "Minimum smoothed momentum change to enter.", "effect": "Above zero demands visible acceleration; below zero lets it enter into softness."},
            "pullback_macro_z_min": {"what": "Macro leadership required for a pullback entry.", "effect": "Higher restricts dip-buying to names that are clear long-horizon leaders."},
            "pullback_meso_z_min": {"what": "Medium-horizon leadership required for a pullback entry.", "effect": "This is what keeps 'buy the dip' from meaning 'buy the loser'."},
            "pullback_nano_z_max": {"what": "How far the fast horizon must have dipped.", "effect": "More negative demands a deeper pullback, so the setup fires less often."},
            "pullback_micro_return_min": {"what": "Short-horizon return that must still hold up.", "effect": "Confirms the dip is a pause, not the start of a decline."},
            "sentiment_weight": {"what": "Coefficient on clipped sentiment inside the score.", "effect": "Start at 0. Non-zero lets news move the ranking, which is exactly what the price gates are there to prevent."},
            "sentiment_size_scale": {"what": "Coefficient on sentiment as a position-size modifier.", "effect": "0.05 with a clip of 2 gives at most a 10% size change. It can never create a position price logic rejected."},
            "sentiment_clip": {"what": "Cap on the normalised sentiment input.", "effect": "Bounds how much a single loud news cycle can move either sentiment term."},
            "sentiment_lookback_minutes": {"what": "How recent a story must be to count.", "effect": "Longer keeps stale headlines alive in the score; shorter makes sentiment sparse."},
            "volatility_tilt": {"what": "Exponent on volatility in sizing: weight follows score x sigma ** tilt.", "effect": "-1 is risk parity, so calm names get the big positions. 0 ignores volatility. +1 leans into it, concentrating in the wildest movers -- more return while a trend runs, more damage when it turns."},
            "name_weight_max": {"what": "Cap on any one ETF before portfolio scaling.", "effect": "Binds before max_positions when few names qualify."},
            "risk_on_gross_max": {"what": "Cap on total invested fraction of equity.", "effect": "Below 1.0 it always holds cash. The simplest single lever on overall risk."},
            "target_portfolio_vol": {"what": "Annualised ex-ante volatility target.", "effect": "Lower means smaller positions in volatile markets. A risk budget, not a return booster."},
            "vol_estimation_days": {"what": "Daily window for volatility and covariance.", "effect": "Shorter reacts to volatility spikes faster and re-sizes more often."},
            "vol_scale_floor": {"what": "Smallest scale factor before switching to defensive.", "effect": "Below this, holding a shrunken risk position is worse than holding none."},
            "high_vol_trigger": {"what": "Multiple of target volatility at which scaling engages.", "effect": "Above 1.0 it stops re-sizing a portfolio already inside its budget, which saves turnover."},
            "intraday_drawdown_limit": {"what": "Session loss that parks the book in the defensive sleeve.", "effect": "Tighter stops the day sooner and locks in the loss; it stays tripped until the next session."},
            "rebalance_weight_threshold": {"what": "Smallest weight change worth trading.", "effect": "Higher tolerates more drift from target in exchange for less churn."},
            "minimum_trade_notional": {"what": "Floor on the dollar size of any single order.", "effect": "Stops trivial orders whose costs exceed their benefit."},
            "minimum_trade_nav_fraction": {"what": "The same floor as a fraction of equity.", "effect": "The larger of the two applies, so the minimum scales with the account."},
            "defensive_max_positions": {"what": "How many defensive names to hold in risk-off.", "effect": "One is a pure cash-equivalent stance; two splits between, say, bills and gold."},
        },
    },
    "spy_rotation": {
        "summary": (
            "Classifies SPY into one of four states -- growing, flat, falling, or crisis -- from "
            "momentum at three horizons plus sentiment, then rotates between growth, covered-call "
            "income, cash, and a capped hedge depending on which state it is in."
        ),
        "formula": [
            "state is decided from SPY's micro, meso and macro returns against the growth/falling thresholds",
            "growing  -> hold the growth sleeve",
            "flat     -> shift flat_equity_income_exposure into the covered-call income sleeve",
            "falling  -> rotate into the defensive universe, up to max_defensive_positions",
            "crisis   -> add hedges, capped by max_crisis_hedge_exposure and max_crisis_hedge_weight",
        ],
        "behavior": (
            "It is a regime switcher, not a stock picker: most of the time it holds one sleeve and "
            "does nothing. Turnover happens at state changes, so the thresholds that define the "
            "states matter far more than the per-name scores."
        ),
        "parameters": {
            "spy_symbol": {"what": "The index proxy whose state drives every decision.", "effect": "Changing it changes what 'the market' means to this strategy."},
            "equity_income_universe": {"what": "Covered-call ETFs used in the flat state.", "effect": "These trade income for capped upside; they are what it holds when it expects chop."},
            "defensive_universe": {"what": "Where it rotates when SPY is falling.", "effect": "BIL is cash-like; longer-duration bonds add a rate bet to the defensive stance."},
            "crisis_hedge_universe": {"what": "Instruments used only in the crisis state.", "effect": "Inverse and volatility products decay when held; the exposure caps exist to bound that."},
            "flat_equity_income_exposure": {"what": "Fraction rotated into the income sleeve in the flat state.", "effect": "Higher commits harder to chop and gives up more upside if the market resumes rising."},
            "max_crisis_hedge_exposure": {"what": "Total cap on hedges as a fraction of equity.", "effect": "The main control on how expensive a false crisis signal is."},
            "max_crisis_hedge_weight": {"what": "Cap on any single hedge instrument.", "effect": "Stops the whole hedge budget landing in one decaying product."},
            "max_defensive_positions": {"what": "How many defensive names it may hold at once.", "effect": "More spreads the defensive sleeve; fewer concentrates it."},
            "max_crisis_hedge_positions": {"what": "How many hedge instruments it may hold.", "effect": "Usually 1; raising it splits the hedge across products."},
            "growth_macro_return": {"what": "Macro return above which SPY counts as growing.", "effect": "Higher demands a stronger market before taking the growth stance, so it sits in income or cash more."},
            "max_gross_exposure": {"what": "Cap on total invested fraction of equity.", "effect": "Below 1.0 it always holds cash regardless of state."},
            "max_single_position_weight": {"what": "Cap on any one position.", "effect": "At 1.0 a single-name sleeve is allowed to be the whole portfolio."},
            "micro_momentum_lookback_bars": {"what": "Bars in the short window used for state detection.", "effect": "Fewer makes state changes twitchier."},
            "meso_momentum_lookback_bars": {"what": "Bars in the medium window.", "effect": "Longer makes state changes rarer and more deliberate."},
            "macro_trend_lookback_days": {"what": "Days in the long window that gates the growth state.", "effect": "Longer keeps it constructive through short corrections."},
            "sentiment_lookback_minutes": {"what": "How recent news must be to influence the state.", "effect": "Longer keeps stale headlines in the classification."},
            "min_income_score": {"what": "Minimum score for an income name to be selected.", "effect": "Raise to skip the income sleeve unless it is genuinely attractive."},
            "min_defensive_score": {"what": "Minimum score for a defensive name.", "effect": "Raise to demand the defensive choice is itself working."},
            "min_crisis_hedge_score": {"what": "Minimum score for a hedge to be used.", "effect": "Raise to avoid hedging with an instrument that is not responding."},
            "min_crisis_hedge_meso_return": {"what": "Minimum medium-horizon return for a hedge to qualify.", "effect": "Stops buying a hedge that is decaying while the crisis persists."},
        },
    },
}


def explainer_for(algorithm_id: str) -> dict[str, Any]:
    """Explanation for one algorithm, with an empty shape when none is written yet."""
    entry = EXPLAINERS.get(str(algorithm_id or ""))
    if not entry:
        return {"summary": "", "formula": [], "behavior": "", "parameters": {}}
    return entry
