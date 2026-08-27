# Rally Rotation: what the knobs actually do, and which ones to keep

Every number here comes from replaying the deployed configuration over three windows against
the DuckDB bar store, scripts in the scratchpad, on 2026-08-24. Read the caveats at the bottom
before treating any single figure as a fact about the strategy rather than about this tape.

Baseline (deployed config, `config/walbot.yaml`):

| window | return | max drawdown | turnover |
| --- | --- | --- | --- |
| 3m  | -9.65%  | -10.05% | 15x |
| 6m  | +32.66% | -13.97% | 43x |
| 12m | +47.43% | -10.49% | 63x |


## The finding everything else follows from

The algorithm states its momentum thresholds in **raw percent**, but ranks and sizes on a
**cross-sectional z-score**. Neither unit says how big a move is *for the name making it*, and
the two failures below are the same defect seen from opposite ends.

**Too loose for a calm name.** VEA held ~100% of the account on a `base_score` of +0.21 -- a
fifth of a MAD above the universe median -- on a 20-day move of +3.0% against 17% annualised
volatility. That is 0.6 sigma: noise. A cross-sectional score cannot report this, because the
top name in a flat tape scores positive by construction.

**Too tight for a volatile name.** XBI rose +23.1% over the 3m window and *lost* the book 3.24%.
It was bought 12% into the move, sold in full at the local low on 2026-07-29, and re-entered
three weeks later 15% higher. The lockout was `etf_min_fast_return: -0.02` catching a 20-day
return of -8.4% during an ordinary pullback -- about -1.7 sigma for a 31%-volatility ETF.

The 3m window contained exactly one real rally (XBI, 97% of days above its 100-day average, at
>=1 sigma on a third of all days). The equal-weight universe returned +0.24%. Every configuration
tested lost money there, because the loss is a **timing** failure and every knob swept only
changes **size**.


## Measured: what each knob is worth

Ablation, one knob at a time, set to its neutral value. Positive `dd` = shallower drawdown.

| knob -> neutral | 3m | 6m | 12m | verdict |
| --- | --- | --- | --- | --- |
| `vol_rising_threshold` -> 0 | -1.18% | +2.39% | **+33.20%** | delete |
| `vol_ceiling` -> 0 | -2.72% | -15.75% | **-28.33%** (dd -25.17%) | keep |
| `range_expansion_limit` -> 0 | 0.00% | +13.65% | +0.60% | delete |
| `climax_ma_distance_min` -> 0 | 0.00% | +13.65% | +0.60% | delete |
| `climax_volume_ratio_min` -> 0 | 0.00% | 0.00% | 0.00% | delete |
| `entry_min_eligible_days` -> 1 | +4.13% | +3.45% | +2.43% | set to 1 |
| `score_ema_minutes` -> 0 | +0.37% | +7.44% | -13.56% | keep, two-sided |
| `risk_adjusted_score` -> on | +2.59% | -8.61% | -4.21% | keep, two-sided |
| `robust_zscore` -> off | +2.02% | +0.20% | +2.53% | investigate |
| `volatility_tilt` -> 0 | -0.79% | +0.84% | +2.14% | keep |
| `max_daily_drop` -> 0 | 0.00% | 0.00% | 0.00% | never fired in 12m |
| `exit_threshold_slack` -> 0 | 0.00% | 0.00% | 0.00% | display-only, delete |
| `exit_max_eligible_days` -> 0 | 0.00% | 0.00% | 0.00% | never fired |
| `exit_rank_max` -> 12 | 0.00% | 0.00% | 0.00% | see below -- **keep** |
| `min_universe_coverage` -> 0 | 0.00% | 0.00% | 0.00% | hardcode |

**One-at-a-time ablation is not reliable here.** `rerank_interval_days -> 0` showed +29.45% on
12m alone and is actively harmful in combination (variant E below). `entry_min_eligible_days`
showed +2.43% alone and is worth ~+30pp in combination. Interactions dominate; nothing in the
table above should be acted on without a combined run.


## Why two gates ablate to exactly zero but are not equivalent

`exit_rank_max` **is** read (`memory.py:232`): entry needs rank <= `entry_rank_max` (3), but a
holding is kept while rank <= `exit_rank_max` (6). It ablates to 0.00% because `_rank()` ranks
only *eligible* names, and the eligible set is small -- mean 5.0 per run over 12m, <= 6 on 66% of
runs -- so 6 already means "any eligible name". Tightening it is what shows the mechanism:

| variant | 6m | 12m |
| --- | --- | --- |
| cleaned, `exit_rank_max: 6` | +65.42% | **+115.02%** (64x) |
| cleaned, `exit_rank_max: 3` | +65.36% | +106.26% (66x) |
| cleaned, `exit_rank_max: 12` | +65.42% | +115.02% (64x) |

The wide band is worth ~9pp on 12m at slightly lower turnover. **Keep it.** Not ejecting a
holding on rank is the behaviour that helps; the current value already delivers it.

`exit_threshold_slack` is different: `exit_checks()` is called only from `signals.py:122`, for
**display**. It never affects a sell decision. The dashboard shows holdings being judged against
a band that does nothing. Delete the knob and fix the display.


## The volatility gates never caused an exit

Both are entry-only. Over 12m, across every run:

```
vol_rising:  283 blocks, 12 names, 0 while held
  forward return of the blocked name: 5d +1.62%  10d +2.68%  20d +4.31%
  gate was right (name fell over 20d): 39%
    USO  +12.94% fwd (9% fell) | XSD +16.06% (28%) | SLV +6.93% (47%) | XBI +5.40% (18%)

vol_ceiling: 331 blocks, 5 names, 0 while held
  forward return: 5d +0.30%  10d -0.02%  20d -2.25%
  gate was right: 54%
    SLV -5.27% (66% fell) | XSD -5.55% (74%) | IAUM -3.78% (56%)
    USO +1.62% (38%)      | IBIT +3.16% (26%)
```

`vol_rising_threshold` is wrong 61% of the time and the names it rejects *rise* 4.3% over the
next 20 days. It blocked XBI 23 times. The premise is backwards -- a 5-day volatility spike is
the start of a move as often as the end of one. Delete rather than retune.

`vol_ceiling` is only 54% right on direction but earns its keep on drawdown: it excludes the
44-76% volatility names carrying 22-31% drawdowns. USO and IBIT show it also costs real upside.


## Recommended configuration

Variant C -- five edits to `config/walbot.yaml`, no code change required:

```yaml
vol_rising_threshold: 0
range_expansion_limit: 0
climax_ma_distance_min: 0
climax_volume_ratio_min: 0
entry_min_eligible_days: 1
```

| variant | 3m | 6m | 12m | turnover (12m) |
| --- | --- | --- | --- | --- |
| deployed | -9.65% / -10.05% | +32.66% / -13.97% | +47.43% / -10.49% | 63x |
| A: `vol_rising` off | -10.83% / -11.21% | +35.05% / -13.72% | +80.63% / -11.96% | 67x |
| B: A + climax off | -10.83% / -11.21% | +50.94% / -13.59% | +85.21% / -11.64% | 62x |
| **C: B + entry_days 1** | **-1.62% / -7.45%** | **+65.42% / -10.99%** | **+115.02% / -7.20%** | 64x |
| D: C + `full_exposure_strength: 1.5` | -3.50% / -6.17% | +52.43% / -7.98% | +92.12% / -6.31% | **46x** |
| E: D + `rerank_interval_days: 0` | -9.47% / -14.78% | +29.53% / -14.48% | +64.62% / -14.48% | 92x |

Net of 5bps per trade, C returns +109.82% on 12m against the deployed +43.60%.

**C or D is a risk preference, not a settled question.** D gives up ~23pp of 12m return for ~1pp
less drawdown and 30% less turnover, and is what directly addresses a single weak name taking
the whole book. That is the argument for keeping `full_exposure_strength` as a knob.


## Deletions

**Delete outright** (18 knobs, ~45 -> ~27):

| knob(s) | warrant |
| --- | --- |
| `vol_rising_threshold` | -33pp/12m; wrong 61% of the time |
| `range_expansion_limit`, `climax_ma_distance_min`, `climax_volume_ratio_min` | one signal, three knobs, -13.65%/6m. Removes `climax_check` and 3 features |
| `exit_threshold_slack`, `exit_max_eligible_days` | display-only / never fired |
| `sentiment_weight`, `sentiment_size_scale`, `sentiment_clip`, `sentiment_lookback_minutes` | no provider configured; `uses_sentiment` always False. Removes `sentiment_adjusted` |
| `trend_ma_days`, `trend_return_days`, `trend_min_return` | deployed at 0 and duplicate `etf_ma_days` / `etf_abs_return` |
| `min_base_score` | deployed at 0, stated in relative z-units |
| `w_nano`, `selection_horizon_nano_minutes` | 0.05 weight on a one-session horizon |
| `rebalance_step` | no-op at its deployed value |

**Hardcode, do not expose:** `max_daily_drop` (never fired in 12m -- but do not delete a crash
stop on the strength of one bull tape), `min_universe_coverage` (data-safety guard, not a view).

**Keep as knobs:** `vol_ceiling`, `exit_rank_max`, `rerank_interval_days`, `max_positions`,
`entry_rank_max`, `volatility_tilt`, `score_ema_minutes`, `risk_adjusted_score`,
`full_exposure_strength`.


## Follow-ups, not costed

**Collapse the momentum thresholds into sigma.** Five knobs answer "is this move big enough" in
three incompatible units: `etf_min_abs_return` and `etf_min_fast_return` (raw %),
`exit_threshold_slack` (raw %, inert), `min_base_score` (cross-sectional z), and
`min_trend_strength` / `full_exposure_strength` (own-volatility sigma). Only the last unit
travels across names. Keep the two lookback windows, replace the five thresholds with an entry
floor and a full-exposure level, both in sigma. XBI's -8.4% dip then reads as -1.7 sigma, which a
-2.0 sigma floor tolerates while still rejecting VEA's +0.6 sigma. **Mechanism argument only --
unmeasured.**

**Exchange-side stop instead of `max_daily_drop`.** The machinery exists (`options_flip` places
OCO brackets). Two hazards: a resting stop and a weight-target rebalancer fight unless the
stop-fill writes into algorithm state; and `max_daily_drop` is a *close-to-close* test while a
resting stop is a *path* trigger, so a knob that fired zero times in 12m becomes one that fires
often. Needs its own backtest.

**A re-entry cooldown is a prerequisite for the stop.** With `entry_min_eligible_days: 1`,
nothing stops a stopped-out name being re-bought on the next run -- a mechanised version of the
XBI round trip. `entry_min_eligible_days: 1` is safe *only* because the stop never fires today.
Ship the cooldown with the stop, not after it.


## Caveats

1. **Three nested windows on one tape.** 3m is inside 6m is inside 12m. The store reaches back
   only to 2025-11-17, so a 24m request silently returns the 12m window. Nested windows are not
   independent evidence (`docs/config-exploration.md` says so).
2. **Single-name concentration.** +115% over 12m on a window containing XBI +23% in its last
   quarter is very likely flattered by one name.
3. **Costs.** Post-cost figures assume 5bps per trade. At 46-64x turnover the assumption is
   load-bearing.
4. **The deletions rest on mechanism plus measurement and are defensible. The exact values are
   not established.** Re-run against an out-of-sample window before trusting the magnitudes.
