"""Every test a name has to pass, stated once, as data.

The absolute-momentum half of dual momentum: relative strength decides the *order*, these decide
whether a name may be held at all. A name failing here is never ranked, so a thin qualifying set
means holding less rather than lowering the bar.

Each gate returns a :class:`Check` -- the measurement beside the threshold -- rather than a bare
boolean, and a verdict is ``all(check.ok for check in ...)``. That is not for the dashboard's
benefit: it is what stops the reason a name was rejected being written twice, once as the test
and once as a message about the test, which is how the two drift apart.

There is one list. There used to be a second, ``exit_checks``, stating every floor widened by
``exit_threshold_slack`` -- but nothing ever called it for a decision, only for display, while
``eligible`` was computed from the entry gates all along. A holding was therefore *shown* the
band it was supposedly judged against and *sold* on a different test entirely.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...core.interfaces import Check
from .config import MIN_UNIVERSE_COVERAGE, RallyRotationConfig


def entry_checks(row: dict[str, Any], config: RallyRotationConfig) -> list[Check]:
    """Whether this name may be *opened*: history, trend, momentum, volatility."""
    checks = [
        Check(
            label=f"{config.etf_ma_days}-day history",
            ok=bool(row.get("enough_history")),
            value=f"{int(row.get('daily_bars', 0))} bars",
            limit=f"≥ {config.etf_ma_days} bars",
        ),
        Check(
            label=f"Above its {config.etf_ma_days}-day average",
            ok=bool(row.get("above_moving_average")),
            value=f"{float(row.get('ma_distance', 0.0)):+.1%}",
            limit="> 0%",
        ),
        Check(
            label=f"{config.etf_abs_return_days}-day return",
            ok=float(row.get("abs_return", 0.0)) > config.etf_min_abs_return,
            value=f"{float(row.get('abs_return', 0.0)):+.1%}",
            limit=f"> {config.etf_min_abs_return:+.1%}",
        ),
        Check(
            label=f"{config.etf_fast_return_days}-day return",
            ok=float(row.get("fast_return", 0.0)) > config.etf_min_fast_return,
            value=f"{float(row.get('fast_return', 0.0)):+.1%}",
            limit=f"> {config.etf_min_fast_return:+.1%}",
        ),
    ]
    checks.extend(_volatility_checks(row, config))
    return checks


def _volatility_checks(row: dict[str, Any], config: RallyRotationConfig) -> list[Check]:
    """A ceiling on how volatile a name may be.

    Entry only, like every gate here. The companion rule that rejected a name whose 5-day
    volatility was rising against its 20-day is gone: over a twelve-month replay it was wrong
    61% of the time, and the names it turned away rose 4.3% over the following 20 sessions.
    Volatility expansion is coincident with a move, not ahead of it.
    """
    checks: list[Check] = []
    annual_volatility = float(row.get("annual_volatility", 0.0))
    if ceiling := max(config.vol_ceiling, 0.0):
        checks.append(Check(
            label="Volatility ceiling",
            ok=annual_volatility <= ceiling,
            value=f"{annual_volatility:.0%} annualised",
            limit=f"≤ {ceiling:.0%}",
        ))
    return checks


def _crash_check(row: dict[str, Any], config: RallyRotationConfig) -> Check:
    """The one exit that answers to no clock: a single session down ``max_daily_drop``.

    Separate from the band-based tests because the two are on different cadences. Those are a
    considered judgement and can wait for the re-rank interval; this cannot, or a name can gap
    30% over a week of throttled sessions while the algorithm politely waits its turn to look.
    """
    drop = max(config.max_daily_drop, 0.0)
    session_return = float(row.get("session_return", 0.0))
    return Check(
        label="No crash this session",
        ok=not (drop and session_return <= -drop),
        value=f"{session_return:+.1%} today",
        limit=f"> {-drop:.0%}" if drop else "not checked",
    )


def crash_stop(row: dict[str, Any], config: RallyRotationConfig) -> Check:
    """The crash check on its own, for the every-session sweep that runs outside the cadence."""
    return _crash_check(row, config)


def universe_data_ok(scored: dict[str, dict[str, Any]], config: RallyRotationConfig) -> dict[str, Any]:
    """Whether enough of the universe can be judged at all. Not a view on the market.

    All that survives of the breadth regime gate. That gate asked "is enough of the universe in
    an uptrend", which duplicated the question :func:`entry_checks` already asks per name -- and
    answered it worse, by vetoing names that had passed every test the strategy makes of them
    because other, unrelated names were weak.

    This check is a different kind of thing and is kept for that reason: too little usable
    history is not a bearish reading, it is an unusable one. A cold or truncated cache would
    otherwise read as "nothing is above its average", which is a bear market that never happened.
    """
    risk_on = [scored[symbol] for symbol in config.risk_on_universe if symbol in scored]
    usable = [row for row in risk_on if row.get("enough_history")]
    coverage = len(usable) / len(risk_on) if risk_on else 0.0
    ok = bool(usable) and coverage >= MIN_UNIVERSE_COVERAGE
    return {
        "data_ok": ok,
        "coverage": coverage,
        "detail": "" if ok else (
            f"only {len(usable)} of {len(risk_on)} risk-on names have {config.etf_ma_days} daily bars"
        ),
    }


def passes(checks: list[Check]) -> bool:
    """A verdict is the conjunction of its gates, never a second opinion about them."""
    return all(check.ok for check in checks)


def blocking(checks: list[Check]) -> Check | None:
    """The gate that actually decided the outcome: the first one that failed.

    Deliberately the *first*, not all of them. A name below its moving average will usually also
    be below its return floors -- those are consequences of the same fact, and presenting four
    red rows implies four independent problems.
    """
    return next((check for check in checks if not check.ok), None)


def mark_blocking(checks: list[Check]) -> list[Check]:
    """Flag that gate, once the full list is assembled.

    Applied at the end rather than inside :func:`entry_checks`, because the market gates are only
    half the story: a name can clear every one of them and still be turned away by the selection
    that follows. Marking each list as it was built gave two "the reason" flags on one row.
    """
    culprit = blocking(checks)
    return [replace(check, blocking=True) if check is culprit else check for check in checks]
