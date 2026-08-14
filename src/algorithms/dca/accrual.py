"""Wall-clock budget accrual shared by DCA and Bursty DCA.

Every amount in a DCA plan means **dollars per month, per symbol**, in both algorithms, so
switching between them never silently changes the spend rate.

The budget is deliberately *not* divided by run count. At an hourly cadence that is ~141 runs
a month, so a $100 budget would yield $0.71 per run -- below every broker minimum -- and the
divisor would change whenever the schedule was edited. Accruing against elapsed wall-clock
time instead leaves cadence controlling only the *opportunity* to act, never the spend rate,
and lets missed runs self-correct because the next run observes a longer interval.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from ...data.state_store import load_state, save_state

#: Average hours in a calendar month (365.25 * 24 / 12).
HOURS_IN_MONTH = 730.5

#: Ceiling on how much budget one gap between runs may accrue. Without it, a bot that was off
#: for half a year would come back and try to deploy half a year of budget in one session.
DEFAULT_MAX_CATCHUP_MONTHS = 1.0


@dataclass(frozen=True)
class SymbolState:
    """Per-symbol accrual state, persisted between runs."""

    accrued: float = 0.0
    last_run_at: str = ""
    deployed_this_month: float = 0.0
    month: str = ""
    #: When the value-averaging path started, so Bursty DCA knows where the path should be.
    path_started_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "accrued": round(self.accrued, 6),
            "last_run_at": self.last_run_at,
            "deployed_this_month": round(self.deployed_this_month, 6),
            "month": self.month,
            "path_started_at": self.path_started_at,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> SymbolState:
        if not isinstance(raw, dict):
            return cls()
        return cls(
            accrued=_as_float(raw.get("accrued")),
            last_run_at=str(raw.get("last_run_at") or ""),
            deployed_this_month=_as_float(raw.get("deployed_this_month")),
            month=str(raw.get("month") or ""),
            path_started_at=str(raw.get("path_started_at") or ""),
        )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_timestamp(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def accrue(
    state: SymbolState,
    monthly_budget: float,
    now: datetime,
    max_catchup_months: float = DEFAULT_MAX_CATCHUP_MONTHS,
) -> SymbolState:
    """Advance ``state`` to ``now``, adding the budget earned over the elapsed wall-clock time.

    The first run only seeds the clock: budget is never accrued retroactively for time before
    the plan was known. A month boundary resets the cumulative deployment cap.
    """
    month = now.strftime("%Y-%m")
    if state.month and state.month != month:
        state = replace(state, deployed_this_month=0.0)
    state = replace(state, month=month)
    if not state.path_started_at:
        state = replace(state, path_started_at=now.isoformat())

    previous = _parse_timestamp(state.last_run_at)
    if previous is None:
        return replace(state, last_run_at=now.isoformat())

    elapsed_hours = max((now - previous).total_seconds() / 3600.0, 0.0)
    elapsed_months = min(elapsed_hours / HOURS_IN_MONTH, max(max_catchup_months, 0.0))
    return replace(
        state,
        accrued=state.accrued + (max(monthly_budget, 0.0) * elapsed_months),
        last_run_at=now.isoformat(),
    )


#: Absolute floor on a trade, for a fractional brokerage with no configured minimum. Without
#: it the floor could be zero, and a "ready" symbol would order a quantity that rounds to no
#: shares at all -- accruing forever while looking like it was trading.
MIN_TRADE_DOLLARS_FLOOR = 1.0


def min_executable(price: float, min_trade_dollars: float, supports_fractional_shares: bool) -> float:
    """Smallest trade that can actually reach the market for this symbol on this brokerage.

    On a whole-share brokerage that is at least one share, which is why a $100/month budget
    against a $500 ETF has to accrue for months before it can trade at all.
    """
    floor_dollars = max(float(min_trade_dollars), MIN_TRADE_DOLLARS_FLOOR)
    if not supports_fractional_shares:
        floor_dollars = max(floor_dollars, max(float(price), 0.0))
    return floor_dollars


def months_since(started_at: str, now: datetime) -> float:
    """Elapsed months as a float, used to locate the value-averaging path."""
    started = _parse_timestamp(started_at)
    if started is None:
        return 0.0
    return max((now - started).total_seconds() / 3600.0, 0.0) / HOURS_IN_MONTH


def path_months(state: SymbolState, fallback_now: datetime) -> float:
    """How far along the value-averaging path this symbol should be.

    Measured from the run that produced the state rather than from the wall clock: step 2 has
    no timestamp of its own, and using "now" would put the path wherever the machine's clock
    happens to be -- wrong by months in a backtest or a replayed run.
    """
    as_of = _parse_timestamp(state.last_run_at) or fallback_now
    return months_since(state.path_started_at, as_of)


# --------------------------------------------------------------------------------------
# Persistence. Same state store the paper brokerage uses.
# --------------------------------------------------------------------------------------


def state_key(algorithm_id: str, account_id: str) -> str:
    return f"dca_accrual:{algorithm_id}:{account_id or 'default'}"


def load_accrual_state(algorithm_id: str, account_id: str) -> dict[str, SymbolState]:
    raw = load_state(state_key(algorithm_id, account_id), {})
    if not isinstance(raw, dict):
        return {}
    return {str(symbol).upper(): SymbolState.from_dict(value) for symbol, value in raw.items()}


def save_accrual_state(algorithm_id: str, account_id: str, state: dict[str, SymbolState]) -> None:
    save_state(
        state_key(algorithm_id, account_id),
        {symbol: symbol_state.as_dict() for symbol, symbol_state in state.items()},
    )
