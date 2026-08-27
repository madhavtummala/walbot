"""Tuning for Options Flip.

A bull-regime, pullback-entry, rebound-exit long-call strategy. Three gates decide whether a
candidate trades: is the bull thesis intact, are the entry and target levels ones comparable
sessions actually reached, and does the base case pay through the full greeks.

Everything not here is a constant below -- window lengths, tolerances, the ATR period -- because
none of those is a decision anyone would make differently. The measurements behind these defaults
are in ``docs/options-flip.md``; the reasoning is on each field.

Two pairs read together and should be set together. ``entry_reach`` and ``target_reach`` are both
"the share of comparable sessions that reached this level", so they are probabilities you can
reason about rather than offsets. ``entry_patience`` and ``exit_patience`` are both "how
stubbornly this side holds its price as its clock runs out", higher being more patient -- and they
are deliberately asymmetric, because an unfilled entry costs only the opportunity while an unsold
position at the deadline is sold at whatever is offered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

VOLATILITY_WINDOW = 20
INTRADAY_BAR_MINUTES = 5

#: Anti-churn on re-pricing a resting order, as a fraction of the wanted price and of the spread.
#: The larger applies. The spread term is what makes this work on an illiquid contract: 2% of a
#: $1.27 option is 2.5 cents, which on a market quoted 1.20/1.35 is a sixth of the spread -- a
#: move the market cannot distinguish, and re-placing for it is a round trip that buys nothing.
REPRICE_MIN_PRICE_FRACTION = 0.02
REPRICE_MIN_SPREAD_FRACTION = 0.25

#: How far from ``target_delta`` a strike may sit and still be considered.
DELTA_TOLERANCE = 0.08
#: Delta distances compete in buckets of this size, so strikes equally close to the target are
#: decided on liquidity rather than on a third decimal place.
DELTA_BUCKET = 0.05

#: How far past ``min_dte`` to ask the chain for.
#:
#: A monthly expiry inside the search lands in the same delta bucket as the weekly and then wins
#: the liquidity tiebreak on parked open interest -- and it is the worse contract. Measured live
#: at delta 0.75 on a $1,500 cap: IBIT's 10-day expiry bought 5 contracts for $174 of edge per 1%
#: move against $51 of two-day theta, while the 24-day monthly bought 3 for $100 against $20. At
#: a fixed dollar cap the same delta costs more premium the further out you go, so every dollar
#: controls less underlying.
CHAIN_DTE_SPAN = 11

#: The trend stack, and the ATR window. Conventional values, and none of them is a decision
#: anyone made differently on purpose -- 20/50 is the standard pair and 14 is Wilder's.
REGIME_FAST_MA_DAYS = 20
REGIME_SLOW_MA_DAYS = 50
ATR_DAYS = 14
#: Minutes that define the opening range.
OPENING_RANGE_MINUTES = 30
#: How close a past session's position-versus-open must be to today's to count as comparable,
#: and how many sessions of intraday history the samples are drawn from.
BUCKET_TOLERANCE = 0.01
#: Latest an entry may still be armed, as a share of the session remaining. An entry that fills
#: at the close has no session left to rebound in.
ENTRY_CUTOFF_FRACTION = 0.25
#: Oldest a chain quote may be. This codebase has already been burned by a stale one.
MAX_QUOTE_AGE_SECONDS = 300.0
#: Implied-volatility assumptions, in points. The base case eases as a move plays out; the bad
#: case firms, which is how a correct-looking entry still loses.
IV_EASE_BASE = -1.0
IV_FIRM_BAD = 1.0


@dataclass(frozen=True)
class OptionsFlipConfig:
    # Ordered by how much each one moves the outcome, most consequential first. Related knobs
    # stay adjacent where their importance is comparable.

    #: Symbols to consider. Empty means the account's tradable universe.
    symbols: list[str] = field(default_factory=list)

    #: Contracts per position -- the unit of risk. A long call cannot lose more than its
    #: premium, so the unit *is* the loss cap.
    contracts_per_trade: int = 1

    #: Loss cap as a fraction of the debit. **Zero disables the stop**, and does so completely:
    #: the bracket becomes a lone profit target. Premium falls on theta and implied volatility
    #: with the directional case intact, so a premium stop cuts winners for reasons unrelated to
    #: the thesis. With a bounded unit the deadline is the risk control that remains.
    stop_loss_pct: float = 0.0

    #: Sessions to hold before the deadline exit takes over. It also sets the horizon the target
    #: is priced over -- the run available grows with the hold -- so the two cannot be set apart.
    max_hold_sessions: int = 4

    #: Where the target sits, as the share of comparable *pulled-back* sessions that reached it.
    #: Lower is more ambitious and reached less often.
    target_reach: float = 0.42

    #: Share of the modelled gain the sell limit asks for on the day of entry. Asking for part
    #: of the move is what makes the exit executable rather than theoretical.
    exit_gain_share: float = 0.70

    #: How stubbornly the sell holds its ask as the deadline approaches; higher is more patient.
    #: The impatient side by design: a position reaching its deadline unsold is sold at whatever
    #: is offered, so conceding early is cheaper than conceding at gunpoint.
    exit_patience: float = 0.7

    #: Where the entry sits, as the share of comparable sessions that reached it. Lower is a
    #: deeper, cheaper entry that fills less often.
    entry_reach: float = 0.55

    #: How stubbornly the buy holds its price as the session runs out; higher is more patient.
    #: The patient side by design: chasing a rising ask turns a pullback trade into a momentum
    #: one, and an unfilled entry costs only the opportunity. It never crosses the mark.
    entry_patience: float = 1.5

    #: Smallest predicted move worth opening for, in dollars per contract, gross of commission.
    #: The strictest gate in the set, and the one that decides how often this trades at all.
    min_profit_per_contract: float = 15.0

    #: The delta to aim the strike at. Higher earns more per point of underlying move, costs
    #: premium that is mostly intrinsic, and buys a contract fewer people trade.
    target_delta: float = 0.62

    #: Sessions the dip and run quantiles are learned from. Long enough that one exceptional
    #: stretch cannot set the tail, since a short window is read back out as a forecast.
    level_lookback_days: int = 80

    #: Smallest trend strength a candidate must carry, in the symbol's own sigma. A threshold,
    #: not a rank: it means the same on a quiet symbol as on a violent one.
    min_trend_strength: float = 0.50


    #: Dollar ceiling per position, priced at the ask. Zero means no cap. It trims the unit and
    #: never sets it.
    max_notional_per_trade: float = 1500.0

    #: Nearest expiry to trade. Under a week the theta curve is steepest.
    min_dte: int = 7

    #: Open interest floor -- whether a resting order finds a counterparty at all. It does not
    #: catch cost; that is ``max_spread_pct``.
    min_open_interest: int = 100

    #: Ceiling on the quoted spread, as a fraction of the mid. The entry rests and never
    #: crosses, but the exit has to get out and the stop is denominated in premium.
    max_spread_pct: float = 0.06

    #: Largest opening gap *down* still an ordinary session, in ATR. Downside only: an up-gap is
    #: followed by a smaller pullback, so it is favourable and merely harder to fill into.
    max_gap_down_atr: float = 1.0

    #: Ceiling on annualised realised volatility. Past it the premium already prices a bigger
    #: move than the model forecasts, so a correct call still loses.
    max_annual_volatility: float = 0.80

    # ── derived ──────────────────────────────────────────────────────
    # Views of the fields above, exposed as properties so call sites read exactly as they did
    # while the configurable surface stays small. None of these is read from saved config:
    # ``load_tuning`` walks dataclass fields, and a property is not one.

    @property
    def regime_fast_ma_days(self) -> int:
        return REGIME_FAST_MA_DAYS

    @property
    def regime_slow_ma_days(self) -> int:
        return REGIME_SLOW_MA_DAYS

    @property
    def atr_days(self) -> int:
        return ATR_DAYS

    @property
    def opening_range_minutes(self) -> int:
        return OPENING_RANGE_MINUTES

    @property
    def bucket_tolerance(self) -> float:
        return BUCKET_TOLERANCE

    @property
    def entry_cutoff_fraction(self) -> float:
        return ENTRY_CUTOFF_FRACTION

    @property
    def max_quote_age_seconds(self) -> float:
        return MAX_QUOTE_AGE_SECONDS

    @property
    def iv_change_base(self) -> float:
        return IV_EASE_BASE

    @property
    def iv_change_bad(self) -> float:
        return IV_FIRM_BAD

    @property
    def max_dte(self) -> int:
        return self.min_dte + CHAIN_DTE_SPAN

    @property
    def volatility_window(self) -> int:
        return VOLATILITY_WINDOW

    @property
    def intraday_bar_minutes(self) -> int:
        return INTRADAY_BAR_MINUTES

    @property
    def required_daily_bars(self) -> int:
        """Enough history for the slowest of the daily measures, plus room for short months."""
        return max(self.regime_slow_ma_days, self.atr_days, VOLATILITY_WINDOW) + 10

    @property
    def required_intraday_minutes(self) -> int:
        """Enough sessions to build the band, not one.

        One session was right when the only intraday reading was today's open. The band averages
        the same minute-of-day across ``level_lookback_days`` past sessions, so a 390-minute
        window leaves it with a sample of one. Found in a live run that priced a signal from a
        one-session sample.
        """
        return 390 * (max(int(self.level_lookback_days), 1) + 1)
