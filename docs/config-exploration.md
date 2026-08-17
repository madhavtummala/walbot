# Configuration exploration: Dual Momentum vs Fast Momentum

Re-run after the market-data corrections (dividends booked as cash, Schwab back-adjusted daily
bars, total-return signal prices), because the previous conclusion -- that widening the universe
and raising `max_positions` helped Dual Momentum -- was drawn on price series that under-counted
every dividend payer.

## How it is measured

`tools/config_sweep.py` drives the production replay engine. It contains no strategy logic: a
result here is the same arithmetic the dashboard's backtest tab performs, on the same bars. That
is deliberate -- the last "config optimizer" in this repo was a second, drifting implementation
of dual momentum, and it is the thing this exercise must not become.

- **Baseline is read from `config/walbot.yaml`,** not restated. A hand-copied baseline is a
  baseline for a configuration nobody runs. (`tests/test_architecture.py` asserts the two match.)
- **One factor at a time** from that baseline. A full product over ten knobs is unaffordable and
  a good way to find noise; moving one axis says which axis carries the result.
- **One date axis for every variant**, intersected across the whole candidate symbol set. Two
  universes scored over different dates are not comparable, and the difference would read as
  skill.
- Window: **12M, 250 trade dates, 2025-08-18 to 2026-08-14**, history coverage ~1.0.

## The two things that make raw returns misleading here

**1. The replay charges nothing to trade.** There is no commission, spread or slippage, even
though `config.transaction_cost_bps` exists and is set. A configuration that churns is therefore
flattered, and this universe is full of thematic ETFs (XSD, XBI, XOP, KRE, XRT) whose real
spreads are not small. Every row reports turnover as a multiple of the stake and a post-hoc
drag at 1 and 5 bps; `net_return_5bps` is the honest column.

This matters more than expected. The deployed Dual Momentum config turns over **161x the stake
in twelve months** -- about 64% of the book every single session. At 5bps that is 8.1 points of
return, more than half of the 14.9% gross.

**2. Sentiment is neutral under replay.** `ReplayContextSource.sentiment()` returns no scores,
so `w_sentiment` and `sentiment_weight` cannot move a backtest at all. No sentiment axis is
swept, because anything it appeared to show would be noise.

A third, smaller one: the drawdown breaker never fires in a daily replay (each date opens a new
session, so intraday drawdown is always zero). `intraday_drawdown_limit` is therefore untestable
here and is not swept either.

## Axes

Universes are named rather than sized, because the score is a cross-sectional z-score: adding a
name changes every other name's score, so these are different strategies, not a size knob.

| set | count | what it is |
|---|---|---|
| `narrow` | 9 | thematic only -- the high-dispersion end |
| `current` | 13 | what is deployed |
| `wide` | 20 | plus broad US and quality/value (SPY, IWM, RSP, QUAL, VTV, SCHD, SLV) |
| `broad` | 29 | plus international, credit, low-vol and income sleeves |

**Dual Momentum** (29 variants): universe, `max_positions` (with entry/exit ranks moving with
it), `volatility_tilt`, `risk_adjusted_score`, `min_base_score`, `breadth_min`,
`name_weight_max`, `min_score_delta_to_replace`, `rebalance_weight_threshold`,
`minimum_trade_nav_fraction`, a combined low-churn setting, the defensive sleeve, and the
benchmark.

**Fast Momentum** (24 variants): universe, `max_positions`, `max_single_position_weight`,
`min_score_delta_to_replace`, `w_pullback_uptrend`, two horizon-weight mixes, `rebalance_threshold`,
`per_trade_value_min`, a combined low-churn setting, and the defensive sleeve.

## Reproducing

```bash
python -m tools.config_sweep --algorithm both --period 12m --out data/config_sweep_12m.csv
python -m tools.sweep_report data/config_sweep_12m.csv            # ranked by net_return_5bps
python -m tools.sweep_report data/config_sweep_12m.csv --sort sharpe
```

## Results

53 configurations, 12M window. Full tables: `data/config_sweep_12m.csv` (the `.json` carries
each row's complete tuning dict). Regenerate any ranking with
`python -m tools.sweep_report data/config_sweep_12m.csv --sort <metric>`.

`breakeven` is the one-way trading cost that would erase the configuration's entire return --
the assumption-free version of `net@5bps`.

### Dual Momentum: the deployed config ranks 21st of 29

| # | variant | gross | net@5bps | vs base | maxDD | Sharpe | turnover | breakeven |
|---|---|---|---|---|---|---|---|---|
| 1 | universe=wide | +41.9% | +31.8% | +25.0 | −7.2% | 2.51 | 202× | 20.8bp |
| 2 | universe=broad | +33.8% | +23.4% | +16.5 | −12.2% | 1.80 | 208× | 16.2bp |
| 3 | defensive=sgov_gld | +26.9% | +18.9% | +12.0 | −9.2% | 1.51 | 160× | 16.8bp |
| 4 | low_churn_combo | +21.8% | +15.6% | +8.8 | **−4.5%** | 1.66 | 123× | 17.7bp |
| 5 | rebalance_threshold=0.1 | +21.5% | +15.4% | +8.5 | −4.7% | 1.64 | 122× | 17.6bp |
| 6 | volatility_tilt=−1.0 | +23.2% | +14.4% | +7.6 | −4.7% | 1.76 | 176× | 13.2bp |
| 7 | benchmark=SPY | +22.7% | +13.9% | +7.0 | −7.5% | 1.57 | 177× | 12.8bp |
| 8 | max_positions=8 | +20.9% | +13.3% | +6.5 | −5.1% | 1.65 | 151× | 13.8bp |
| **21** | **baseline (deployed)** | **+14.9%** | **+6.9%** | — | −6.8% | 1.13 | 161× | 9.2bp |
| 29 | defensive=bil_tlt | +12.4% | +4.4% | −2.5 | −7.7% | 0.93 | 160× | 7.7bp |

1. **Widen to ~20 names, not 29.** The largest single effect, at unchanged drawdown and double
   the Sharpe. `broad` gives back a third of it and doubles drawdown to −12.2%.
2. **`volatility_tilt` has reversed sign.** Monotone across four levels: −1.0 > 0.0 > 0.5 >
   +1.0 (deployed), and −1.0 also carries the lowest drawdown. This contradicts the note in
   `dual_momentum/config.py` -- "+1.0 measured best across 6M/4M/3M ... improved return *and*
   drawdown together". That measurement predates the dividend and total-return corrections, and
   `+1.0` concentrates in exactly the high-volatility names where the raw-price bias bit
   hardest. **The re-run reversed a documented decision, which is what it was for.**
3. **The churn was destroying value, not merely costing money.** `rebalance_threshold=0.1` cut
   turnover 161x -> 122x and raised *gross* return from +14.9% to +21.5%, with drawdown
   improving to −4.7%. The deployed 0.03 is too tight.
4. **Three knobs are dead controls** at current settings: `min_score_delta_to_replace`
   (±0.06pp), `minimum_trade_nav_fraction` (0 to −0.29pp), `breadth_min` (±0.9pp).
5. **More positions helps, weakly.** 8 is best but the axis is non-monotone (6 < 5).

### Fast Momentum: the deployed config ranks 12th of 24

| # | variant | gross | net@5bps | vs base | maxDD | Sharpe | turnover | breakeven |
|---|---|---|---|---|---|---|---|---|
| 1 | universe=narrow | +35.8% | +24.0% | +12.2 | −21.6% | 1.33 | 237× | 15.1bp |
| 2 | universe=broad | +30.8% | +19.0% | +7.2 | −22.0% | 1.04 | 237× | 13.0bp |
| 3 | defensive=sgov_gld | +30.0% | +18.6% | +6.8 | −20.6% | 1.22 | 228× | 13.2bp |
| 4 | universe=wide | +30.0% | +18.6% | +6.8 | −20.7% | 1.04 | 228× | 13.1bp |
| 9 | max_single_weight=0.35 | +22.8% | +12.8% | +1.0 | −16.6% | 1.10 | 201× | 11.3bp |
| **12** | **baseline (deployed)** | **+23.3%** | **+11.8%** | — | −20.8% | 0.97 | 231× | 10.1bp |
| 20 | max_single_weight=0.25 | +17.4% | +9.6% | −2.2 | **−12.5%** | 1.01 | 156× | 11.1bp |
| 24 | max_positions=8 | +18.4% | +6.2% | −5.6 | −24.1% | 0.81 | 244× | 7.5bp |

1. **Every variant takes 12-25% drawdown**, against Dual Momentum's 4.5-12%.
2. **It wants the *opposite* universe**: `narrow` (9 names) wins by 12.2pp where Dual Momentum's
   `narrow` was near-worst. Coherent -- Dual Momentum gates on absolute trend and breadth and can
   sit defensive, so more candidates means more chances to find something that qualifies; Fast
   Momentum always holds its top-N regardless of quality, so diluting the cross-section with
   low-dispersion index names averages the signal away.
3. **`max_positions` runs the other way too**: monotone *down* past 3 (3 > 4 > 5 > 6 > 8).
4. **The turnover brakes do nothing here** (+0.26, +0.08, and `low_churn_combo` is −1.1pp).
5. **`max_single_weight=0.25` is the only real risk control**: drawdown −20.8% -> −12.5% and
   turnover 231x -> 156x, for −2.2pp of net return.

### Head to head

| | Dual Momentum | Fast Momentum |
|---|---|---|
| baseline net@5bps | +6.9% | **+11.8%** |
| best net@5bps | **+31.8%** | +24.0% |
| baseline maxDD | **−6.8%** | −20.8% |
| best-config maxDD | **−7.2%** | −21.6% |
| baseline Sharpe | 1.13 | 0.97 |
| best Sharpe | **2.51** | 1.33 |
| turnover range | 115-208x | 156-254x |
| breakeven range | 7.7-20.8bp | 7.5-15.1bp |

**As deployed**, Fast Momentum earns more but takes three times the drawdown, and is worse
risk-adjusted on both Sharpe and peak-to-trough.

**Tuned**, Dual Momentum wins decisively: +31.8% net at −7.2% and Sharpe 2.51, against Fast
Momentum's best of +24.0% at −21.6% and Sharpe 1.33. Similar return, three times the pain. Fast
Momentum is also the more fragile to costs -- lower breakeven on higher turnover.

## Reading the result honestly

A single 12-month window ranks 53 configurations, so the top of that ranking is partly luck:
with this many candidates the best one is expected to look better than it is. Treat the axes
that move the result *consistently and for a reason* as the finding, and the exact winning
combination as a hypothesis to check on other windows before deploying.

By that standard, what survives scrutiny is:

- **Trustworthy** -- a mechanism *and* a monotone shape. Dual Momentum's `volatility_tilt`
  (four levels, ordered, and it reverses a measurement taken on pre-correction data); Fast
  Momentum's `max_positions` (monotone down past 3); the opposite universe preferences, which
  follow from one algorithm being able to sit defensive and the other not.
- **Suspect** -- single winners with no shape. `defensive=sgov_gld` ranks 3rd in *both* grids,
  which is most likely "gold did well in this window" rather than an edge; GLD is already in
  the risk-on universe, so it is partly a double allocation. `min_base_score` is non-monotone
  (0.25 worse than both 0.0 and 0.5), which is what noise looks like.

Two limits on the window itself:

- The intraday cache begins **2025-11-28**, but this window opens 2025-08-18. Its first ~3.3
  months resolve the "intraday" horizons from daily bars, which compresses Dual Momentum's four
  selection horizons (60/240/1200/4800 market-minutes) toward roughly {1d, 1d, 3d, 12d} and
  makes nano and micro the same number. Coverage reads 1.0 because the *window* was filled --
  by daily bars standing in for intraday ones.
- Genuinely out-of-sample validation is therefore capped at about **8.5 months**, not 24.

> **Superseded for Dual Momentum.** That 8.5-month cap was a statement about the *intraday*
> cache, and Dual Momentum no longer reads intraday bars at all -- `required_history_minutes`
> is 0 and every feature comes from daily bars. The daily store reaches back to **2022-03-28**,
> so disjoint calendar-year windows are now available to it. Fast Momentum still reads intraday
> history, so the cap continues to apply there.

### Multi-year windows

`--period` takes explicit `START:END` dates as well as relative `Nm` windows, and those are the
only way to get independent evidence: `_period_start` anchors every relative window to "now
minus N months", so 6M/12M/24M end on the same day and are strictly nested -- 6M is inside 12M
is inside 24M. A configuration "surviving all three" has been confirmed once on overlapping
data, with the most recent months counted three times.

```bash
python -m tools.config_sweep --algorithm dual_momentum --stage confirm \
  --period 2023-01-01:2023-12-31,2024-01-01:2024-12-31,2025-01-01:2025-12-31
```

Two things make this work that did not before:

- The fetch sizes its buffer from the window's own start rather than from `_period_row_count`,
  which only parses relative windows and used to fall through to a 4-month default. A 2023
  request previously replayed the most recent four months and reported it *as* 2023.
- The shared date axis drops symbols that did not exist over the window instead of truncating
  the window for everyone. IBIT listed 2024-01-11 and by itself put every window inside the
  post-IBIT era; a 2023 replay holding it would have been holding a fund that did not trade.

A `START:END` window that finds no bars now raises rather than silently substituting another
period.

### Costs

`--cost-bps` charges a half-spread inside the fills -- a buy pays above the close, a sell
receives below it -- so it compounds and can make an order unaffordable rather than merely
expensive. It is an alternative to the post-hoc `net_return_*` columns, not a layer on top:
with both, the same churn is charged twice. Default is 0, which is what every result above was
measured under.

For a book turning over 100x its stake a year this number matters more than the gross return.
`tools/attribution.py` prints the whole cost curve for one configuration, which is the honest
form of the question -- Dual Momentum's deployed 2023 run breaks even at roughly **17bps**.
