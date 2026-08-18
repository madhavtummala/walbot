"""Timestamp parsing shared across layers.

One tolerant ISO reader rather than the three private copies that had accumulated
(``rally_rotation/stateful._parse_time``, ``dca/accrual._parse_timestamp``, and the inline
handling in the connectors).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now() -> datetime:
    """The current instant, timezone-aware, in UTC.

    Only the two callers that genuinely meant this. ``provider_cache._now`` returns a
    ``pd.Timestamp`` and ``schwab_auth._now`` returns an epoch float for token expiry
    arithmetic -- same name, different types, deliberately left alone.
    """
    return datetime.now(timezone.utc)


def parse_iso_utc(value: Any) -> datetime | None:
    """Parse an ISO-8601 string, assuming UTC when it carries no offset.

    Returns ``None`` rather than raising: these values come from persisted state and from
    provider payloads, where an unreadable timestamp means "unknown" and must degrade to the
    caller's fallback instead of taking a run down.
    """
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def minutes_between(later: datetime | None, earlier: datetime | None) -> float:
    """Absolute minutes between two moments; infinite when either is unknown.

    Infinity rather than zero, because every caller is asking "has enough time passed?" and an
    unknown earlier bound must not read as "no time at all".
    """
    if later is None or earlier is None:
        return float("inf")
    return abs((later - earlier).total_seconds()) / 60.0
