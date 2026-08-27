"""What a contract is worth if the underlying does what the model says -- and what it costs if not.

Replaces a delta-only translation that ignored everything an option position is actually exposed
to. Delta prices a *small* move and nothing else. Over the move this strategy is trying to
capture, and the one-to-three days it holds for:

* **gamma** bends the delta -- a 0.66-delta IBIT call carries gamma 0.155, so a $1 move in the
  underlying changes the delta by 0.155, and pricing the whole move at the starting delta
  understates a winner and overstates a loser;
* **theta** is not small -- that same contract decays $0.061 a day on a $1.51 mark, **4% of
  premium per day**, so a two-day hold needs an 8% move in the contract just to break even;
* **vega** carries the IV crush that can lose money on a correctly-called direction, which is
  the failure a delta-only model cannot even represent.

The estimate is a second-order Taylor expansion:

``ΔC ≈ Δ·ΔS + ½·Γ·ΔS² + Vega·ΔIV + Θ·Δt``

with ``ΔIV`` and ``Δt`` supplied per scenario. It is an approximation and is wrong for a large
move, which is why the target is a conservative quantile rather than the day's predicted high.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Scenario:
    """One outcome the trade is judged against."""

    name: str
    #: Underlying move from the entry level, in price units.
    underlying_move: float
    #: Change in implied volatility, in the same units the chain publishes (percentage points).
    iv_change: float
    #: Days held.
    days: float


def option_change(contract: Any, scenario: Scenario) -> float:
    """``ΔC`` for one contract under one scenario, per share.

    Vega is quoted per one *point* of implied volatility and the chain publishes IV in percentage
    points, so ``iv_change`` is in points too and no rescaling happens here -- a mismatch between
    those two units is a silent factor of a hundred, which is why it is stated rather than
    inferred.
    """
    move = float(scenario.underlying_move)
    delta = float(getattr(contract, "delta", 0.0) or 0.0)
    gamma = float(getattr(contract, "gamma", 0.0) or 0.0)
    vega = float(getattr(contract, "vega", 0.0) or 0.0)
    theta = float(getattr(contract, "theta", 0.0) or 0.0)
    return (
        delta * move
        + 0.5 * gamma * move * move
        + vega * float(scenario.iv_change)
        + theta * float(scenario.days)
    )


def scenarios(
    contract: Any, *, entry_underlying: float, target_underlying: float,
    spot: float, config: Any,
) -> dict[str, dict[str, float]]:
    """Bad, base and good outcomes for a position opened at ``entry_underlying``.

    All three are measured from the *entry* level rather than from the current price, because the
    position does not exist until the pullback fills and pricing it from today's spot credits the
    trade with the pullback it is still waiting for.
    """
    hold = max(float(config.max_hold_sessions), 1.0)
    to_target = target_underlying - entry_underlying
    return {
        name: {
            "underlying": entry_underlying + s.underlying_move,
            "change": option_change(contract, s),
        }
        for name, s in {
            # Thesis fails: the underlying sits where it was bought and IV firms slightly, which
            # is the ordinary way a correct-looking entry still loses.
            "bad": Scenario("bad", 0.0, float(config.iv_change_bad), hold),
            # The trade decision. IV eases as the move plays out, which is the usual direction.
            "base": Scenario("base", to_target, float(config.iv_change_base), hold),
            # Context for the profit cap, not a number anything is sized on.
            "good": Scenario("good", to_target * 1.5, 0.0, hold),
        }.items()
    }


def max_debit(contract: Any, outcomes: dict[str, dict[str, float]], *, config: Any) -> float:
    """The largest price still supported by the base case, per share.

    The ceiling the entry limit may never be raised past. Without it, a repricing loop that keeps
    stepping toward a rising ask converts a pullback strategy into buying an extended move -- the
    exact failure the reference design names, and the reason this is derived from the profit
    model rather than picked.

    Gross of costs. Both legs rest as limits and never cross, so a fill gets its price and the
    only real charge is commission -- about $1.30 on the round trip, which is noise against any
    debit worth capping.
    """
    base_gain = float(outcomes.get("base", {}).get("change", 0.0))
    justified = base_gain - (float(config.min_profit_per_contract) / 100.0)
    if justified <= 0:
        return 0.0
    # A debit above the model's own value for the contract is paying for a move already made.
    return max(min(float(contract.midpoint), float(contract.midpoint) + justified), 0.01)


def expected_profit(outcomes: dict[str, dict[str, float]], contracts: int, *, config: Any) -> dict[str, float]:
    """Base-case dollars, per contract and in total. **Gross** -- the predicted band, priced.

    No reserve is subtracted. The gate this feeds asks how big the predicted move is, and a
    commission estimate folded into that answer made it a different and worse question.
    """
    base = float(outcomes.get("base", {}).get("change", 0.0))
    reserve = 0.0
    per_contract = (base - reserve) * 100.0
    return {
        "per_contract": per_contract,
        "total": per_contract * max(int(contracts), 1),
        "bad": (float(outcomes.get("bad", {}).get("change", 0.0)) - reserve) * 100.0,
        "good": (float(outcomes.get("good", {}).get("change", 0.0)) - reserve) * 100.0,
    }
