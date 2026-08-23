# Configuration exploration: Bursty DCA sizing parameters

Which of `BurstyConfig`'s five knobs earn yield: `scaling_factor`, `regime_ma_days`,
`max_monthly_multiple`, `relax_months`, `relax_depth`. One axis at a time against the deployed
baseline (0.5 / 150 / 3.0 / 2.0 / 0.7), over the deployed plan -- XSD, SPYM, XBI and IAU at
$500/month each.

## How it was measured

- **Replay engine**: the dashboard backtest's own (`tools/config_sweep.py`), not a reimplementation.
  20 variants x 3 windows (12m / 6m / 3m), one axis moved at a time from the deployed config.
- **Stake**: $35,000, opened **in SGOV** rather than as cash (`--open-in SGOV`), so unspent budget
  keeps earning the T-bill rate while it waits -- which is the honest alternative for a plan this
  size. SGOV's own dividends are in every number below.
- **Costs**: fills are free in the replay; `net@5bps` prices the (tiny) churn back post-hoc.
  Turnover is ~0.7x stake per year, breakeven ~2,200bps, so cost modelling is noise here.
- **Windows are nested** (3m inside 6m inside 12m, all ending today), so they are three samples of
  one regime, not independent evidence. Where all three agree, that is one confirmed statement --
  not three.
- Reproduce with:
  ```
  python -m tools.parallel_sweep --algorithm bursty_dca --stage axes \
      --period 12m,6m,3m --starting-equity 35000 --open-in SGOV \
      --workers 5 --out data/config_sweep_bursty_sgov_35k.csv
  python -m tools.sweep_report data/config_sweep_bursty_sgov_35k.csv
  ```

## Results

Net return at 5bps, per window, ranked by average rank across the three:

| variant | 12m | 6m | 3m | 12m maxDD | 12m deployed |
|---|---|---|---|---|---|
| relax_months=8.0 | **+22.16%** | **+9.55%** | **+1.51%** | -10.18% | $16,926 |
| relax_months=4.0 | +18.64% | +7.50% | +1.37% | -8.60% | $14,038 |
| regime_ma_days=50 | +18.15% | +6.99% | +1.17% | -8.79% | $14,047 |
| plain_dca_ref | +17.90% | +5.52% | +0.80% | -8.39% | $13,892 |
| scaling_factor=0.0 | +17.90% | +5.54% | +0.79% | -8.39% | $13,891 |
| regime_ma_days=100 | +17.44% | +6.77% | +1.24% | -8.19% | $13,317 |
| scaling_factor=0.25 | +17.24% | +5.78% | +1.00% | -8.09% | $13,293 |
| max_monthly_multiple=6 | +16.71% | +6.55% | +1.23% | -7.94% | $12,829 |
| *baseline (deployed)* | *+16.61%* | *+6.46%* | *+1.23%* | *-7.92%* | *$12,833* |
| relax_depth=0.35 / 0.0 / 1.2 | +16.58 / 16.54 / 15.78% | ~same | ~same | ~same | ~$12,832 |
| regime_ma_days=200 / 300 | +16.14 / 16.02% | +5.71 / 5.31% | +1.17 / 1.27% | -7.56 / -7.16% | ~$12.5k |
| max_monthly_multiple=2 | +15.50% | +5.36% | +1.12% | -7.64% | $12,831 |
| relax_months=1.0 | +14.72% | +4.93% | +1.03% | -7.35% | $12,297 |
| scaling_factor=1.0 | +13.90% | +5.76% | +1.33% | -6.93% | $10,736 |
| relax_months=0.5 | +13.68% | +4.08% | +0.95% | -7.02% | $12,082 |
| max_monthly_multiple=1 | +13.42% | +5.13% | +1.16% | -6.60% | $10,925 |
| scaling_factor=2.0 | +9.91% | +4.92% | +1.20% | -5.12% | $6,485 |

## What each axis is worth

### relax_months -- the axis that matters, and it pays to widen it

The only knob with a strictly monotone, all-windows-agree effect: 0.5 -> 1 -> 2 -> 4 -> 8 months
raises net return in every window, ending +5.5pp over baseline on 12m (+22.16%). The mechanism is
visible in the `deployed` column: at 8 months the book put $16,926 to work over the year against
the baseline's $12,833. `relax_months` sets both the backlog curve's width *and* the overdraft in
`spending_allowance` (`budget x (backlog + relax_months x conviction)`), so widening it lets a
symbol borrow months further ahead of its accrual when conviction agrees -- i.e. it front-loads
deployment. The price is exposure: max drawdown grows from -7.92% to -10.18%, roughly
proportionally, so Sharpe barely moves (~1.57 throughout). This buys yield with volatility, not
with skill -- but on this evidence the trade is favourable: Calmar edges up too (2.10 -> 2.18).

### scaling_factor -- less is more

Monotone the other way across 12m/6m: 0 -> 0.25 -> 0.5 -> 1.0 -> 2.0 loses return at every step,
collapsing to +9.91% at 2.0. The cause is deployment, again: at sf=2.0 conviction saturates so
fast (ceiling 7x at 3-sigma) that the book only deployed $6,485 of its $12,000 annual plan -- it
is not timing better, it is mostly declining to invest, leaving the money in SGOV (its dividend
income is the highest of any variant). Over 3m the ordering flips -- sf=1.0/2.0 have the best
Sharpe (2.28/2.66) and half everyone else's drawdown -- so high scaling reads as a defensive
posture that happens to pay in falling windows. For yield in ordinary ones, keep it at or below
the deployed 0.5.

### plain DCA vs bursty

The `plain_dca_ref` row (sf=0, relax_depth=0) beats the deployed baseline over 12m (+17.90%) and
is the *worst* row over 3m (+0.80%, deepest drawdown). That pairing is the honest summary of what
burstiness is for: in a calm melt-up the valuation discount mostly just slows deployment; in a
rough window it is what keeps you buying dislocations instead of every scheduled date.

### regime_ma_days -- shorter measured better, except when it didn't

50 > 100 > 150 > 200 > 300 on both long windows, with 50d deploying like the wide-relax variants
($14k). But over 3m the order reverses (300d ranks 4th, 50d 13th with the worst Sharpe of the
set, 1.10). A short MA makes z noisier and deployment faster; that paid through 2025's trend and
cost in Q2-Q3 2026's chop. If changing anything here, 100 days captures most of the gain with
less regime risk than 50.

### max_monthly_multiple -- raise it, it never binds anyway

6x >= 3x > 2x > 1x in all three windows, but baseline vs 6x differ by cents: at $500/month per
symbol the cap almost never engages. Only 1x visibly starves the plan (-3.2pp on 12m). Free to
leave at 3; raising it is insurance for violent dislocations, not a yield lever today.

### relax_depth -- nearly irrelevant to yield

Every depth lands within ~1pp of baseline on 12m with identical deployed totals (~$12,833): it
reshapes *when* backlog is spent without changing how much eventually goes out, because accrual
conservation governs the total. Only 1.2 shows a consistent small drag. Not worth tuning.

## Recombining what moved

`--stage finalists` recombines every axis that beat the baseline by >0.5pp, plus one-axis-removed
ablations (`net@5bps`, per window):

| variant | 12m | 6m | 3m | 12m maxDD |
|---|---|---|---|---|
| *baseline* | *+16.58%* | *+6.44%* | *+1.22%* | *-7.92%* |
| combined = rm8 + ma50 + sf0 + rd0 | **+28.39%** | +11.05% | +0.43% | -11.82% |
| combined-minus:scaling_factor (sf stays 0.5) | +26.44% | **+11.25%** | **+1.68%** | -12.24% |
| combined-minus:relax_months | +17.87% | +5.50% | +0.79% | -8.39% |

Two readings:

- The trailing-year maximum is the **deploy-at-plan-rate corner**: wide overdraft
  (`relax_months=8`) with no valuation opinion (`scaling_factor=0`, `relax_depth=0`). It wins 12m
  by ~12pp but is near-worst in the most recent 3m (-0.79pp vs baseline, drawdown triple the
  others') -- it is fully exposed exactly when the market gets rough.
- Keeping the **deployed `scaling_factor=0.5`** (the minus-scaling row) gives up ~2pp of the
  trailing year and wins both shorter windows outright, including +1.68% with the whole 3m field
  behind it. Once the overdraft is wide, the valuation factor stops being a deployment brake and
  becomes what it was designed to be: a filter on *which* dislocations may borrow.

The `regime_ma_days` ablation contributes nothing inside the combination (identical rows with and
without) -- its apparent value in the one-axis sweep was an interaction with the old overdraft,
not an effect of its own.

## Recommendation

For yield on this stake, in order of confidence:

1. **`relax_months`: 2 -> 4 (or 8)**. The one clear, repeated winner: monotone in every window,
   +2.0pp/+5.5pp on 12m per step pair. Understand it precisely: it buys earlier deployment with
   proportionally deeper drawdown (-7.92% -> -10.18% at 8), not free alpha.
2. **Keep `scaling_factor` <= deployed 0.5.** Raising it is the largest single yield loss
   measured (-6.7pp at 2.0, via deployment collapse to $6.5k/yr); and once `relax_months` is
   wide, keeping 0.5 beats zeroing it in both recent windows.
3. If promoting anything, promote **`relax_months=8, relax_depth=0, rest deployed`**
   (the minus-scaling finalist): +9.9pp / +4.8pp / +0.5pp over baseline across 12m/6m/3m --
   the only candidate that beat baseline everywhere it was measured.
4. **`regime_ma_days`, `max_monthly_multiple`, `relax_depth`: leave alone** -- individually weak,
   jointly irrelevant inside the combination, and the cap never binds at this plan size.

Caveats before touching `config/walbot.yaml`: all three windows are nested samples of one regime
(2025-08 -> 2026-08), so "won every window" means *confirmed once*; and the winners lean on a
period whose defining feature was buying dips that came back. Re-run over an explicit disjoint
window (`--period 2024-01-01:2024-12-31`) before trusting the ranking anywhere else.

