"""Every test a name has to pass, stated once, as data.

The absolute-momentum half of dual momentum: relative strength decides the *order*, these decide
whether a name may be held at all. A name failing here is never ranked, so a thin qualifying set
means holding less rather than lowering the bar.

Each gate returns a :class:`Check` -- the measurement beside the threshold -- rather than a bare
boolean, and a verdict is ``all(check.ok for check in ...)``. That is not for the dashboard's
benefit: it is what stops the reason a name was rejected being written twice, once as the test
and once as a message about the test, which is how the two drift apart.

Entry and exit are deliberately asymmetric. Sharing one threshold for both makes every floor a
coin-flip boundary for anything sitting near it, and the round trip costs the spread twice.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ...core.interfaces import Check
from .config import RallyRotationConfig


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
    checks.extend(_trend_checks(row, config))
    return checks


def _volatility_checks(row: dict[str, Any], config: RallyRotationConfig) -> list[Check]:
    """A ceiling on how volatile a name may be, and on how fast it is getting there."""
    checks: list[Check] = []
    annual_volatility = float(row.get("annual_volatility", 0.0))
    if ceiling := max(config.vol_ceiling, 0.0):
        checks.append(Check(
            label="Volatility ceiling",
            ok=annual_volatility <= ceiling,
            value=f"{annual_volatility:.0%} annualised",
            limit=f"≤ {ceiling:.0%}",
        ))
    if rising := max(config.vol_rising_threshold, 0.0):
        vol_5d = float(row.get("vol_5d", 0.0))
        # Only meaningful against a measured baseline: with no 20-day estimate there is nothing
        # for the short window to be rising *against*.
        ok = not (annual_volatility > 0 and vol_5d > annual_volatility * (1.0 + rising))
        checks.append(Check(
            label="Short-term volatility not spiking",
            ok=ok,
            value=f"5d {vol_5d:.0%} vs 20d {annual_volatility:.0%}",
            limit=f"5d ≤ 20d x {1.0 + rising:.2f}",
        ))
    return checks


def _trend_checks(row: dict[str, Any], config: RallyRotationConfig) -> list[Check]:
    """The medium-term filter: a short rally inside a longer downtrend is not momentum."""
    checks: list[Check] = []
    if trend_ma_days := max(config.trend_ma_days, 0):
        distance = float(row.get("trend_ma_distance", 0.0))
        checks.append(Check(
            label=f"Above its {trend_ma_days}-day trend line",
            ok=distance >= 0,
            value=f"{distance:+.1%}",
            limit="≥ 0%",
        ))
    trend_return_days = max(config.trend_return_days, 0)
    trend_min_return = float(config.trend_min_return)
    if trend_return_days and trend_min_return:
        trend_return = float(row.get("trend_return", 0.0))
        checks.append(Check(
            label=f"{trend_return_days}-day trend return",
            ok=trend_return > trend_min_return,
            value=f"{trend_return:+.2%}",
            limit=f"> {trend_min_return:+.2%}",
        ))
    return checks


def exit_checks(row: dict[str, Any], config: RallyRotationConfig) -> list[Check]:
    """Whether a name already held may *stay*, on floors widened by ``exit_threshold_slack``.

    The counterpart to :func:`entry_checks`. A holding that would not be bought today is not
    thereby worth selling: the quality floor and the settling period are entry conditions only,
    and applying them symmetrically sells a name the moment it stops being a purchase.
    """
    slack = max(config.exit_threshold_slack, 0.0)
    checks = [
        Check(
            label="History still sufficient",
            ok=bool(row.get("enough_history", 1)),
            value=f"{int(row.get('daily_bars', 0))} bars",
            limit=f"≥ {config.etf_ma_days} bars",
        ),
        _crash_check(row, config),
        Check(
            label=f"Within {slack:.0%} of its {config.etf_ma_days}-day average",
            ok=float(row.get("ma_distance", 0.0)) >= -slack,
            value=f"{float(row.get('ma_distance', 0.0)):+.1%}",
            limit=f"≥ {-slack:.1%}",
        ),
        Check(
            label=f"{config.etf_abs_return_days}-day return above the exit band",
            ok=float(row.get("abs_return", 0.0)) > config.etf_min_abs_return - slack,
            value=f"{float(row.get('abs_return', 0.0)):+.1%}",
            limit=f"> {config.etf_min_abs_return - slack:+.1%}",
        ),
        Check(
            label=f"{config.etf_fast_return_days}-day return above the exit band",
            ok=float(row.get("fast_return", 0.0)) > config.etf_min_fast_return - slack,
            value=f"{float(row.get('fast_return', 0.0)):+.1%}",
            limit=f"> {config.etf_min_fast_return - slack:+.1%}",
        ),
    ]
    return checks


def _crash_check(row: dict[str, Any], config: RallyRotationConfig) -> Check:
    """The one exit that answers to no clock: a single session down ``max_daily_drop``.

    Separate from the band-based tests because the two are on different cadences. Those are a
    considered judgement and can wait for the re-rank interval; this cannot, or a name can gap
    30% over a week of throttled sessions while the algorithm politely waits its turn to look.
    """
    drop = max(config.max_daily_drop, 0.0)
    session_return = float(row.get("nano_return", 0.0))
    return Check(
        label="No crash this session",
        ok=not (drop and session_return <= -drop),
        value=f"{session_return:+.1%} today",
        limit=f"> {-drop:.0%}" if drop else "not checked",
    )


def crash_stop(row: dict[str, Any], config: RallyRotationConfig) -> Check:
    """The crash check on its own, for the every-session sweep that runs outside the cadence."""
    return _crash_check(row, config)


def climax_check(row: dict[str, Any], config: RallyRotationConfig) -> Check | None:
    """A blowoff top: far above the average, on an expanded range, on a volume spike.

    Near the moving average that same pattern -- wide range, heavy volume -- is a buy-the-dip
    signal. Far above it, the combination means exhausted buying rather than fresh demand, which
    is why the distance test comes first and switches the whole check off below it.

    ``None`` when the test is switched off, rather than a check that trivially passes. A gate
    nothing is being measured against is not evidence, and rendering it as a passed one implies
    the run considered something it did not.
    """
    distance_min = max(config.climax_ma_distance_min, 0.0)
    range_limit = max(config.range_expansion_limit, 0.0)
    if not distance_min or not range_limit:
        return None

    ma_distance = float(row.get("ma_distance", 0.0))
    range_expansion = float(row.get("range_expansion", 0.0))
    volume_ratio = float(row.get("volume_ratio", 0.0))
    volume_min = max(config.climax_volume_ratio_min, 0.0)
    exhausted = (
        ma_distance >= distance_min
        and range_expansion >= range_limit
        and (not volume_min or volume_ratio >= volume_min)
    )
    return Check(
        label="Not a climax top",
        ok=not exhausted,
        value=f"{ma_distance:+.0%} above average, range {range_expansion:.1f}x, volume {volume_ratio:.1f}x",
        limit=f"exhausted at ≥ {distance_min:.0%} and {range_limit:.1f}x and {volume_min:.1f}x",
    )


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
    ok = bool(usable) and coverage >= config.min_universe_coverage
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
