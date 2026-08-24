"""Tuning for Options Flip.

**Kept deliberately small.** Every field here is a decision someone might genuinely want to make
differently. Everything else is either derived from one of these or fixed at a value there is no
good reason to vary -- window lengths, anti-churn thresholds, bar resolutions. Those were config
once, and with thirty-one knobs nobody could tell which six mattered.

Three worth understanding before changing anything:

``target_fill_probability`` is not a discount. It says what fraction of comparable past sessions
would have reached the entry bid, so 0.6 means "bid where six days in ten would have filled me".
It is the only entry-price knob, and unlike a percentage it means the same thing on every symbol.

``target_delta`` chooses the *strike*, and delta is the moneyness dial: ~0.50 is at the money,
higher is in the money, lower is out. The default 0.45 is slightly out of the money, which for a
one-to-two-day hold balances paying for intrinsic value against needing a move the model is not
predicting.

``stop_loss_pct`` is the loss cap, as a fraction of the premium paid. The stop is a *market*
order once triggered, so it sells into the bid and the realised loss can overshoot the cap on a
wide market. That slippage is accepted rather than gated: there is no spread ceiling, because
both legs of this strategy rest as limits and never cross the spread. ``min_open_interest`` is
the liquidity test -- whether a resting order finds a counterparty at all, which is the question
a spread figure does not answer.

``entry_decay_power`` shapes the walk-in, and it is the knob that decides how often this trades
at all. See the field for the curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Fixed rather than configurable. Each of these was a knob; none of them is a decision.
#:
#: ``EXCURSION_LOOKBACK_DAYS`` is a statistical window -- long enough for a stable quantile, short
#: ``VOLATILITY_WINDOW`` is the conventional 20 sessions. ``PREMARKET_MIN_BARS`` is a
#: data-quality floor, not a strategy view.
#: How far from ``target_delta`` a strike may sit and still be considered. Wide enough that a
#: chain with coarse strikes still offers candidates, narrow enough that the delta genuinely
#: characterises the contract.
DELTA_TOLERANCE = 0.15
#: Delta distances are compared in buckets of this size, so strikes that are equally close to the
#: target compete on liquidity rather than on a third decimal place. Without it the ranking below
#: would be decided entirely by delta and the liquidity keys would never be reached.
DELTA_BUCKET = 0.05

VOLATILITY_WINDOW = 20
PREMARKET_MIN_BARS = 15

#: Anti-churn on re-pricing a resting order, as a fraction of the wanted price and as a fraction
#: of the bid/ask spread. The larger of the two applies.
#:
#: The spread term is what makes this work on an illiquid contract. 2% of a $1.27 option is 2.5
#: cents, which on a market quoted 1.20/1.35 is a sixth of the spread -- a move the market cannot
#: distinguish, and re-placing the order for it is a round trip that buys nothing. On a tight
#: name (SPY at 0.5% wide) the price term governs and nothing changes.
REPRICE_MIN_PRICE_FRACTION = 0.02
REPRICE_MIN_SPREAD_FRACTION = 0.25
#: The same idea for the exit bracket, which only moves when the target improves materially.
RATCHET_MIN_IMPROVEMENT = 0.03
#: The window the trend's return is measured over.
TREND_LOOKBACK_DAYS = 5
INTRADAY_BAR_MINUTES = 5
#: How far past ``min_dte`` to ask the chain for. Affects the size of the request, nothing else.
CHAIN_DTE_SPAN = 35


@dataclass(frozen=True)
class OptionsFlipConfig:
    # ── universe ─────────────────────────────────────────────────────
    #: Each symbol runs its own independent lifecycle, so this is a portfolio of one-contract
    #: positions rather than a shortlist to pick a winner from.
    symbols: list[str] = field(default_factory=list)

    # ── direction ────────────────────────────────────────────────────
    #: Longer = slower to change sides, fewer whipsaws.
    trend_ma_period: int = 20
    #: How far the symbol must have moved, over ``TREND_LOOKBACK_DAYS``, to count as trending.
    trend_min_return: float = 0.005
    #: Pre-market move required in the trend's direction, as a multiple of the symbol's own
    #: typical daily move -- normalised, so one value works across a mixed symbol list.
    #: Pre-market can only ever veto; it never sets a direction. Raise it to trade fewer days.
    premarket_confirm_min: float = 0.25

    # ── contract ─────────────────────────────────────────────────────
    #: Roughly two weeks. Far enough out that a one-to-two-day hold is not fighting the steepest
    #: part of the theta curve, near enough that the contract still moves with the underlying.
    min_dte: int = 10
    #: The delta to aim the strike at. One number rather than a band, because a band's midpoint
    #: was doing the choosing anyway and stating the two edges only created a way to set them
    #: inconsistently -- ``delta_max: 1.2`` is unreachable for a call, and shifted the target
    #: without ever saying so.
    #:
    #: Delta is the moneyness dial: ~0.50 is at the money, higher is in the money, lower is out.
    #: Higher also means more dollars per point of underlying move -- expected profit works out to
    #: exactly ``delta × band × 100``, since the contract's own price cancels -- but it is paid for
    #: with premium that is mostly intrinsic value, and a larger loss if the stop hits.
    #:
    #: Puts use the same number negated. Candidates within :data:`DELTA_TOLERANCE` of it are
    #: eligible; see ``contracts.select_contract`` for how they are ranked.
    target_delta: float = 0.45
    #: Open interest floor. The real liquidity test -- it asks whether a resting order will find
    #: a counterparty at all, which no spread figure answers.
    #:
    #: Deliberately modest, because open interest varies by more than an order of magnitude with
    #: the expiry rather than the symbol: measured across in-band strikes, SPY's monthly carries a
    #: median above 2,000 while its next weekly carries 78, and some weeklies are at zero. A floor
    #: set for monthlies rejects every weekly; this one admits a liquid near-dated strike while
    #: still refusing a dead one.
    min_open_interest: int = 100

    # ── entry ────────────────────────────────────────────────────────
    target_fill_probability: float = 0.6
    #: How the bid walks in from the predicted low toward the contract's mid through the session:
    #: ``decay = fraction_of_session_remaining ** entry_decay_power``, and the bid sits that
    #: fraction of the way back toward the predicted low.
    #:
    #: **Lower is more patient.** With a predicted low of 99.00 against a mid of 100.00:
    #:
    #: ===========  ==========  ==========  ==========
    #: time         power 1.0   power 0.5   power 0.25
    #: ===========  ==========  ==========  ==========
    #: 12:45          99.50       99.29       99.16
    #: 14:22          99.75       99.50       99.29
    #: 15:55          99.99       99.89       99.66
    #: ===========  ==========  ==========  ==========
    #:
    #: At 1.0 it converges on the mid by the close, so a day that never dipped still fills near
    #: fair value. Below 1.0 the bid is still short of the mid at the last fire of the day, so it
    #: effectively never converges -- more no-trade days, and every fill at a price you chose.
    #:
    #: It never crosses the spread. The bid converges toward the *midpoint*, not the ask: this
    #: order exists to be paid for patience, and paying the offer to guarantee a fill discards
    #: the whole edge on exactly the days the prediction was wrong.
    entry_decay_power: float = 1.0

    # ── exit ─────────────────────────────────────────────────────────
    #: The loss cap, as a fraction of the premium paid. Held at the exchange, never moved.
    stop_loss_pct: float = 0.10
    #: The profit target's counterpart to ``target_fill_probability``. Lower is more ambitious.
    exit_fill_probability: float = 0.5
    #: Sessions to hold before flattening, counted in market days from the fill.
    #:
    #: It also sets the horizon the *exit target* is priced over, which is the more consequential
    #: half. The move available over two sessions is much larger than over one -- median upside on
    #: these symbols runs 0.99% in a day and 1.74% in two -- so a longer hold does not merely give
    #: the position more time, it raises what the position is asking for.
    max_hold_sessions: int = 2
    #: Sessions of history the excursion distribution is estimated from.
    #:
    #: **Shortening this does not make the strategy more short-term.** The excursion is already a
    #: single-session measure; this only controls how many samples the quantile is taken over.
    #: Measured across these symbols the estimate is stable from 20 to 90 sessions and moves by
    #: less than a tenth of a percent; at 10 it doubles, which is a sample of ten talking, not the
    #: market. Shorten it only to track a genuine regime change, and not below about 30.
    excursion_lookback_days: int = 60

    # ── gates ────────────────────────────────────────────────────────
    #: Ceiling on annualised realised volatility. Past it, the premium already prices a bigger
    #: move than the model is forecasting, so a correctly-called direction still loses -- and the
    #: stop is far likelier to be taken out by noise on the way.
    #:
    #: There is deliberately no *floor*. A quiet symbol produces a narrow band, a narrow band
    #: produces a small expected profit, and ``min_expected_profit`` already refuses it -- in
    #: dollars, which is what actually matters, rather than in a volatility percentage that has to
    #: be translated in your head.
    max_annual_volatility: float = 0.80

    #: Smallest expected profit worth trading, in dollars **per contract**, gross of commissions.
    #:
    #: Per contract rather than per position on purpose: a floor on the position total is
    #: satisfiable by buying more of a marginal trade, so raising ``contracts_per_position``
    #: would quietly loosen the quality bar. Whether a setup is worth taking cannot depend on how
    #: much of it you buy -- size is a separate decision, bounded by ``max_notional_per_trade``.
    #:
    #: A different axis from ``trend_min_return``, which is why both exist. That one asks whether
    #: the *underlying* is trending hard enough to have a direction at all; this asks whether the
    #: *contract* stands to move enough dollars to be worth the round trip. A strong trend on a
    #: cheap low-delta contract clears the first and fails this one -- which is precisely the
    #: trade that looks right on every other gate and still loses money once costs are paid.
    #:
    #: Gross because the deck reports gross; set it above your own round-trip commission with
    #: room to spare. Set to 0 to take any positive expectation.
    min_expected_profit: float = 25.0

    # ── sizing ───────────────────────────────────────────────────────
    contracts_per_position: int = 1
    max_notional_per_trade: float = 1000.0

    # ── derived ──────────────────────────────────────────────────────
    # Views of the fields above, exposed as properties so call sites read exactly as they did
    # while the configurable surface stays small. None of these is read from saved config:
    # ``load_tuning`` walks dataclass fields, and a property is not one.

    @property
    def max_dte(self) -> int:
        return self.min_dte + CHAIN_DTE_SPAN

    @property
    def trend_lookback_days(self) -> int:
        return TREND_LOOKBACK_DAYS

    @property
    def volatility_window(self) -> int:
        return VOLATILITY_WINDOW

    @property
    def premarket_min_bars(self) -> int:
        return PREMARKET_MIN_BARS

    @property
    def ratchet_min_improvement(self) -> float:
        return RATCHET_MIN_IMPROVEMENT

    @property
    def intraday_bar_minutes(self) -> int:
        return INTRADAY_BAR_MINUTES

    @property
    def required_daily_bars(self) -> int:
        """Enough history for the slowest of the daily measures, plus room for short months."""
        return max(self.trend_ma_period, self.excursion_lookback_days, VOLATILITY_WINDOW) + 10

    @property
    def required_intraday_minutes(self) -> int:
        """One full session. Today's high, low and open are all this needs to see."""
        return 390
