"""What a contract is worth if the underlying does what the model says -- and what it costs if not.

A second-order Taylor expansion, so gamma/theta/vega are priced too, not just delta::

    ΔC ≈ Δ·ΔS + ½·Γ·ΔS² + Vega·ΔIV + Θ·Δt

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
    #: Change in implied volatility, in points (matches the chain's own units).
    iv_change: float
    #: Days held.
    days: float


def option_change(contract: Any, scenario: Scenario) -> float:
    """``ΔC`` for one contract under one scenario, per share."""
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

    Measured from the entry level, not today's spot -- the position does not exist until the
    pullback fills.
    """
    hold = max(float(config.max_hold_sessions), 1.0)
    to_target = target_underlying - entry_underlying
    return {
        name: {
            "underlying": entry_underlying + s.underlying_move,
            "change": option_change(contract, s),
        }
        for name, s in {
            "bad": Scenario("bad", 0.0, float(config.iv_change_bad), hold),
            "base": Scenario("base", to_target, float(config.iv_change_base), hold),
            "good": Scenario("good", to_target * 1.5, 0.0, hold),
        }.items()
    }


def max_debit(contract: Any, outcomes: dict[str, dict[str, float]], *, config: Any) -> float:
    """The largest price still supported by the base case, per share -- the entry's hard ceiling.

    Gross of costs: both legs rest as limits and never cross, so commission (~$1.30 round trip)
    is noise against any debit worth capping.
    """
    base_gain = float(outcomes.get("base", {}).get("change", 0.0))
    justified = base_gain - (float(config.min_profit_per_contract) / 100.0)
    if justified <= 0:
        return 0.0
    return max(min(float(contract.midpoint), float(contract.midpoint) + justified), 0.01)


def expected_profit(outcomes: dict[str, dict[str, float]], contracts: int, *, config: Any) -> dict[str, float]:
    """Base-case dollars, per contract and in total. Gross -- no commission reserve subtracted."""
    base = float(outcomes.get("base", {}).get("change", 0.0))
    per_contract = base * 100.0
    return {
        "per_contract": per_contract,
        "total": per_contract * max(int(contracts), 1),
        "bad": float(outcomes.get("bad", {}).get("change", 0.0)) * 100.0,
        "good": float(outcomes.get("good", {}).get("change", 0.0)) * 100.0,
    }
