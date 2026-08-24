"""Option contract identity: the OSI symbol, and the quoted contract behind it.

Deliberately imports nothing from the rest of the codebase, for the same reason
``algorithms/ids.py`` does not: identity is not behaviour. The brokerage layer needs to name a
contract on an order leg, the market data layer needs to name one on a quote, and the algorithm
needs to compare the two. If any of those owned the representation the other two would import
it, and a contract's name would depend on which half of the system was asking.

**One representation, end to end.** Schwab returns OSI symbols on chain rows, accepts them on
order legs, and reports them back on positions, so a contract parsed out of a chain can be
matched against a held position by string equality. That is the property worth protecting: the
alternative -- carrying ``(underlying, expiry, type, strike)`` around and re-deriving the symbol
at each boundary -- puts four chances to disagree between the chain and the fill.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

#: The OSI symbol is a root, the expiry as ``YYMMDD``, ``C`` or ``P``, then the strike in
#: thousandths as eight digits: ``AAPL  250117C00150000`` is the 17 Jan 2025 150-strike call.
#:
#: **Two spellings exist and both are real.** The standard pads the root to six characters, which
#: is what Schwab emits and expects (21 characters, always). Alpaca omits the padding and emits
#: ``AAPL250117C00150000``. They name the same contract, so both must parse -- but each broker
#: only accepts its own, which is why :func:`osi_symbol` takes a ``padded`` flag rather than
#: picking one. Accepting only the padded form made Alpaca's option *positions* invisible, since
#: :func:`is_osi_symbol` is what tells a held contract from a held equity.
OSI_LENGTH = 21
_OSI_PATTERN = re.compile(
    r"^(?P<root>[A-Za-z ]{1,6}?) *(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{8})$"
)

CALL = "call"
PUT = "put"
OPTION_TYPES = (CALL, PUT)

#: Strikes are carried in thousandths of a dollar, which is what makes the field an integer.
#: A strike finer than a tenth of a cent does not exist on any listed contract.
_STRIKE_SCALE = 1000


def osi_symbol(underlying: str, expiry: date, option_type: str, strike: float, *, padded: bool = True) -> str:
    """Build the OSI symbol for one contract, in the spelling the target broker accepts.

    ``padded`` gives the 21-character standard form Schwab uses; ``False`` gives Alpaca's
    unpadded one. The root is space-padded on the *right* -- ``"AAPL  "`` not ``"  AAPL"`` --
    which is worth stating, because a left-padded root parses without error and names a contract
    that does not exist.
    """
    kind = _normalize_type(option_type)
    root = str(underlying or "").strip().upper()
    if not root or len(root) > 6:
        raise ValueError(f"Underlying {underlying!r} is not a valid OSI root (1-6 characters)")
    if strike <= 0:
        raise ValueError(f"Strike must be positive, got {strike!r}")
    thousandths = round(float(strike) * _STRIKE_SCALE)
    if thousandths >= 10 ** 8:
        raise ValueError(f"Strike {strike!r} does not fit the 8-digit OSI field")
    stem = f"{root:<6}" if padded else root
    return f"{stem}{expiry:%y%m%d}{'C' if kind == CALL else 'P'}{thousandths:08d}"


def to_osi_form(symbol: str, *, padded: bool) -> str:
    """Re-spell an OSI symbol in the other broker's form, leaving a non-option string alone.

    The translation point. A contract discovered in a Schwab chain has to be *ordered* through
    whichever broker the account uses, and the two disagree about padding -- so the symbol is
    converted at the brokerage boundary rather than everywhere a contract is handled.
    """
    if not is_osi_symbol(symbol):
        return symbol
    parsed = parse_osi(symbol)
    return osi_symbol(
        parsed["underlying"], parsed["expiry"], parsed["option_type"], parsed["strike"],
        padded=padded,
    )


def parse_osi(symbol: str) -> dict[str, Any]:
    """Inverse of :func:`osi_symbol`, as ``{underlying, expiry, option_type, strike}``.

    Raises rather than returning ``None`` on a malformed symbol. A symbol that cannot be parsed
    reached us from a broker or a chain, and treating that as a missing value would let it flow
    on as "no contract" -- which reads as a flat book rather than as the data error it is.
    """
    text = str(symbol or "")
    match = _OSI_PATTERN.match(text)
    if not match:
        raise ValueError(f"{symbol!r} is not an OSI option symbol")
    return {
        "underlying": match.group("root").strip().upper(),
        "expiry": date(
            2000 + int(match.group("expiry")[0:2]),
            int(match.group("expiry")[2:4]),
            int(match.group("expiry")[4:6]),
        ),
        "option_type": CALL if match.group("kind") == "C" else PUT,
        "strike": int(match.group("strike")) / _STRIKE_SCALE,
    }


def is_osi_symbol(symbol: str) -> bool:
    """Whether ``symbol`` names a contract rather than an equity.

    Needed because Schwab reports option and equity positions in one list keyed by symbol, with
    nothing else distinguishing them -- the shape of the string is the only signal.
    """
    return bool(_OSI_PATTERN.match(str(symbol or "")))


def _normalize_type(option_type: str) -> str:
    kind = str(option_type or "").strip().lower()
    if kind in ("c", "call"):
        return CALL
    if kind in ("p", "put"):
        return PUT
    raise ValueError(f"Unknown option type {option_type!r} (expected one of {OPTION_TYPES})")


@dataclass(frozen=True)
class OptionContract:
    """One quoted contract, as a chain returned it.

    Carries the quote alongside the identity because every decision the strategy makes needs
    both at once: the delta band picks the contract, the spread decides whether it is tradable,
    and the mark prices the order. Splitting them would mean re-joining them at every use.

    ``mark`` is stored rather than derived so a provider that publishes its own mark is believed
    over the bid/ask midpoint -- see :meth:`midpoint` for the fallback.
    """

    osi_symbol: str
    underlying: str
    option_type: str
    strike: float
    expiry: date
    bid: float = 0.0
    ask: float = 0.0
    mark: float = 0.0
    delta: float = 0.0
    open_interest: int = 0
    volume: int = 0
    implied_volatility: float = 0.0

    @property
    def midpoint(self) -> float:
        """The bid/ask mid, falling back to whichever side is quoted, then to ``mark``.

        A one-sided quote is common on a thin contract near the open. Returning the quoted side
        is better than returning zero, which would read as "free" to anything doing arithmetic.
        """
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.bid or self.ask or self.mark

    @property
    def spread_pct(self) -> float:
        """Bid/ask spread as a fraction of the midpoint. ``inf`` when there is no two-sided market.

        Infinity rather than zero for an unquotable contract: this number is compared against a
        ceiling, and zero would make the widest possible market look like the tightest.
        """
        mid = self.midpoint
        if mid <= 0 or self.bid <= 0 or self.ask <= 0:
            return float("inf")
        return (self.ask - self.bid) / mid

    def moneyness(self, spot: float) -> float:
        """How far out of the money this strike sits, as a fraction of spot.

        **Positive is always out of the money**, for calls and puts alike -- a call is OTM above
        spot and a put is OTM below it, and normalising the sign means one threshold reads the
        same way on both sides. Negative is in the money.

        Distinct from delta, which is what actually selects the strike. Delta says how much the
        contract moves with the underlying; moneyness says where the strike sits. They correlate
        but are not interchangeable: on a high-volatility name a delta-0.45 call can sit several
        percent out of the money, and on a quiet one it will hug spot.
        """
        if spot <= 0:
            return 0.0
        gap = (self.strike - spot) if self.option_type == CALL else (spot - self.strike)
        return gap / spot

    def dte(self, as_of: date) -> int:
        """Calendar days to expiry. Negative once expired, which callers should treat as a gate."""
        return (self.expiry - as_of).days


def black_scholes_delta(
    spot: float, strike: float, years: float, volatility: float, option_type: str,
    *, rate: float = 0.04,
) -> float:
    """Delta from first principles, for when the provider will not supply one.

    Schwab returns ``-999`` for every greek outside the hours it computes them, and this strategy
    selects entirely on delta -- so without a fallback the algorithm simply stops choosing
    contracts, and says so in a way that points at the delta band rather than at the data.

    An estimate, and inferior to the provider's: it assumes European exercise, no dividends, and
    takes ``volatility`` from the underlying's realised moves rather than from the option's own
    implied vol, which is also missing when the greeks are. For choosing which strike is nearest a
    target it is more than accurate enough -- delta is a smooth function of moneyness and the
    ranking barely moves. It should not be used to price anything.
    """
    kind = _normalize_type(option_type)
    if spot <= 0 or strike <= 0 or years <= 0 or volatility <= 0:
        return 0.0
    d1 = (
        (math.log(spot / strike) + (rate + 0.5 * volatility ** 2) * years)
        / (volatility * math.sqrt(years))
    )
    call_delta = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    return call_delta if kind == CALL else call_delta - 1.0
