# Options Flip

A bull-regime, pullback-entry, rebound-exit long-call strategy on a deadline-aware limit
schedule. Candidates are scored on their own trend in their own sigma; three gates decide
whether one of them trades.

**Status: implemented, not validated.** Every measurement below is real and most of them are
small. The honest summary is at the bottom under "What is and is not established" — read it
before trusting any number here.


## The shape

```
CANDIDATES   score each symbol on its OWN trend, in its own sigma -- no universe consulted
             strength = sum_h  w_h x return_h / (annual_vol x sqrt(days_h / 252))
             keep the top max_candidates with strength >= min_trend_strength

GATE 1       is the bull thesis intact today?
  regime       price > SMA(20), gap down within max_gap_down_atr, holding VWAP,
               volatility under the ceiling

GATE 2       where would we transact, and how often is each level reached?
  levels       E = P - k_entry  x ATR      k_entry  = quantile(dip, 1 - entry_reach)
               T = E + k_target x ATR      k_target = quantile(run | dipped, 1 - exit_reach)
               dip over ONE session (the entry is abandoned at the close);
               run over max_hold_sessions (the target has the whole hold)

GATE 3       does the base case pay, through the full greeks?
  economics    dC ~ delta*dS + 0.5*gamma*dS^2 + vega*dIV + theta*dt
               profit = base_change * 100 >= min_profit_per_contract   (gross)

ENTRY        a limit at E, translated to premium through delta. It ratchets toward the mark on
             an entry_patience curve, never crosses it, and is abandoned unfilled at the close.
EXIT         a limit at exit_gain_share of the modelled gain, conceding on an exit_patience
             curve and converging on the market at the deadline.
```


## Configuration

Twenty fields. Everything else is a constant in `config.py` -- window lengths, tolerances,
the ATR period -- because none of those is a decision anyone would make differently.

| field | default |
| --- | --- |
| `symbols` | `[]` (empty means the account's tradable universe) |
| `min_trend_strength` | `0.50` |
| `max_candidates` | `3` |
| `contracts_per_trade` | `1` |
| `max_notional_per_trade` | `3500.0` |
| `max_hold_sessions` | `6` |
| `stop_loss_pct` | `0.5` |
| `entry_reach` | `0.40` |
| `entry_patience` | `2.0` |
| `exit_reach` | `0.35` |
| `exit_gain_share` | `0.70` |
| `exit_patience` | `1.5` |
| `target_delta` | `0.8` |
| `min_dte` | `7` |
| `min_open_interest` | `100` |
| `max_spread_pct` | `0.06` |
| `min_profit_per_contract` | `15.0` |
| `max_gap_down_atr` | `1.0` |
| `max_annual_volatility` | `0.8` |
| `level_lookback_days` | `80` |

Two pairs read together. ``entry_reach``/``exit_reach`` are both "the share of comparable
sessions that reached this level", so they are probabilities rather than offsets.
``entry_patience``/``exit_patience`` are both "how stubbornly this side holds its price as its
clock runs out", higher being more patient. They started deliberately asymmetric (entry patient,
exit impatient); a combo sweep over a real month (``tools/options_flip_config_combo_sweep.py``)
found holding the exit firmer instead was the more reliable choice on that data, so both are
tuned patient now -- see the field-level rationale in ``config.py`` if a later month argues the
original asymmetry back.

An ``atr_multiple`` knob briefly widened both levels on top of the quantiles. It was removed:
setting it to 1.15 was exactly ``entry_reach: 0.55`` and ``exit_reach: 0.42``, the same levels
to a tenth of a cent, but expressed as a multiplier whose reach probability you only learned
afterwards. Two knobs for one decision, and the worse of the two units.

Per-field reasoning lives in `config.py` and in the tuning descriptions in
`src/algorithms/explainers.py`; both are checked against the dataclass, so neither can drift.


## What each measurement actually showed

**The stop was the result, not a safeguard.** At 0.35 the payoff inverted: 62% of trades won at
a +2.4% median while the nine that stopped lost 35% each, and a majority-winning strategy lost
$3,396. Tightening to 0.10 cut the size of the losses and not the asymmetry -- on real option
bars, two ~2% winners still could not pay for one 10% stop. Across a 128-configuration sweep,
**all 64 configurations with the stop off were profitable and only 19% of those with it on
were.** It is off by default. A long call cannot lose more than its premium, so with a fixed
`contracts_per_trade` the position is bounded and the deadline is the risk control that remains.

**A fixed-offset entry is adversely selected; a conditional one need not be.** A bid always
0.40% below the open filled on 86% of IBIT sessions, and the days it *missed* were worth +1.73%
against -0.34% for the days it filled (p < 0.001 on both symbols). The mechanism is not
ordering, it is depth: a day that opens and runs has a shallow low, so the fill condition
literally selects for days that fell. Conditioning the depth on the day in front of you is the
answer; a different constant is not.

**A long sample teaches the model about days the regime already rejected.** `k_entry` rises
monotonically with lookback -- 0.08 ATR at 10 sessions against 0.22 at 174 -- because a long
pool mixes in the deep pullbacks of declines. At full history the target landed *below* the
price the signal fired at, which made the strategy a mean-reversion scalp rather than a rebound
trade. At 20 sessions the target sits 0.13-0.30% above it and the fill rate roughly doubled.
Below about ten sessions a quantile becomes an order statistic and the targets invert again.

**Gate strictness, measured independently across 1,872 opportunities.** The sequential funnel
is order-dependent and understates whatever runs last, so each gate was evaluated against every
opportunity:

| gate | rejects alone | exclusive | survivors without it |
| --- | --- | --- | --- |
| min profit | 75.6% | **11.1%** | 334 |
| strike near delta | 35.8% | 0.53% | 125 |
| above the trend | 34.1% | 0.80% | 130 |
| time to work | 25.0% | 0.00% | 115 |
| holding VWAP | 23.9% | **2.88%** | 169 |
| open not a gap down | 2.6% | 0.05% | 116 |
| volatility ok | 0.0% | 0.00% | 115 |
| levels available | 0.0% | 0.00% | 115 |

`min_profit_per_contract` is the throttle. **Exclusive rejection is only meaningful relative to the
current gate set:** VWAP measured 0.21% exclusive while a redundant 20/50 stack gate sat in
front of it catching the same sessions, and jumped to 2.88% once that was removed. A gate that
looks inert may be masked rather than useless.

The MA-slope and broad-market gates were removed on that evidence. The slope was a lagging
confirmation used as leading permission: over 2026-08-17..20 IBIT ran 36.40 -> 41.19 while its
20-day slope was still negative, and the strategy sat out every session of it. The market check
asked a question the cross-sectional candidate scoring had already answered.

**115 survivors is not 115 trades.** The gates are almost all session-level, so a qualifying
session has ~9 of its 18 in-window fire times qualify -- the same setup counted nine times. The
algorithm arms one position per session, so the ceiling is the *session* count, and the hold
then blocks the sessions after it.


## What is and is not established

**Established.** The plumbing runs end to end on real traded option bars. Fill modelling is
calibrated -- 65% modelled touch probability against a 60-67% realised fill rate, on two
independent runs. Implied volatility is backed out of actual prices rather than assumed. The
gate table and funnel are legible enough to find design faults, and did: three of the last five
bugs were found by them and not by tests.

**Not established: whether any of it makes money.** The largest real-price sample is 33 sessions
across three symbols, of which one (XSD) traded 846 contracts in six weeks across nine strikes
and is not a market. GLD supplied 114 of 115 surviving opportunities. Four to eight trades is
the whole evidence base, in a single bull regime, and no configuration should be tuned on it --
a 128-config sweep on that sample will find something excellent by chance.

The six-month synthetic run has 125 sessions and real power, but prices options with Black-Scholes
at *realised* volatility. Implied normally sits above realised, so those options are
systematically too cheap and every number from it is optimistic. Correcting that bias is the
next thing worth doing.

**Known modelling gaps.** `pricehistory` returns trades and not quotes, so there is no historical
bid/ask and fills are approximated by "a trade printed through the limit", ignoring queue
position. The backtest's exit loop only runs at the start of a following session, so a position
cannot exit the day it opened -- which is why holds average ~20 hours and every exit lands in the
first ten minutes of the next session. The live path allows a same-day exit.


## Live-verified Schwab behaviour

Established by placing and cancelling real orders on account 0000-0000:

- `TRIGGER -> OCO(limit, stop_limit)` on an option rests correctly, and a child leg can be
  replaced by its own order id without disturbing its sibling.
- A replacement payload may not carry children -- `400 "Replacing order cannot have child
  orders."` Re-pricing a trigger parent reissues new ids for every node beneath it.
- **A 201 does not mean the order is live.** A bracket was accepted with a `Location` header and
  rejected moments later; one bad leg rejects the entire tree, and the reason is written on the
  offending leg, not the parent.
- Relative stop pricing is unavailable: `stopPriceLinkBasis: TRIGGER` with `PERCENT` returns a
  bare 500. The link fields drive trailing stops only.
- Expired contracts return no history, and option bars reach back about 33 sessions.
- Options approval on that account is unconfirmed -- a rejection returned *"The account is not
  approved for this level of options trading."* Untriggered legs are not approval-checked until
  they activate.
