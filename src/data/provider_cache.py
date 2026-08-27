from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from .duckdb_store import _connect

logger = logging.getLogger(__name__)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _timestamp(value: Any | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def load_cached_payload(
    category: str,
    provider: str,
    cache_key: str,
    *,
    db_path: str | None = None,
) -> Any | None:
    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT payload, expires_at
            FROM api_cache
            WHERE category = ? AND provider = ? AND cache_key = ?
            """,
            [category, provider, cache_key],
        ).fetchone()
    if not row:
        logger.debug("Cache miss: %s/%s [%s]", category, provider, cache_key)
        return None
    # ``expires_at`` was written on every row and read on none, so every TTL in the config was
    # decorative and a cached quote was served for ever. Found live: a run on 2026-08-25 priced
    # IBIT at 36.15 and GLD at 405.77 from rows fetched on 2026-08-17 and expired thirty minutes
    # later, while the venue was quoting 45.04 and 426.65 -- an eight-day-old price, delivered
    # with ``current: True``, to an algorithm that sizes real orders from it.
    #
    # Enforced here rather than at each caller because the TTL is the cache's own promise; a
    # caller that has to remember to check it is a caller that will eventually forget.
    expires_at = _timestamp(row[1])
    if expires_at is not None and expires_at <= _now():
        logger.debug("Cache expired: %s/%s [%s] at %s", category, provider, cache_key, expires_at)
        return None
    try:
        logger.debug("Cache hit: %s/%s [%s]", category, provider, cache_key)
        return json.loads(row[0])
    except json.JSONDecodeError:
        logger.warning("Cache corruption: Failed to decode %s/%s [%s]", category, provider, cache_key)
        return None


def save_cached_payload(
    category: str,
    provider: str,
    cache_key: str,
    payload: Any,
    ttl_seconds: int,
    *,
    db_path: str | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=max(int(ttl_seconds), 0))
    logger.debug("Saving to cache: %s/%s [%s] (ttl=%ss)", category, provider, cache_key, ttl_seconds)
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO api_cache (category, provider, cache_key, payload, fetched_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                category,
                provider,
                cache_key,
                json.dumps(payload, sort_keys=True, default=str),
                now,
                expires_at,
            ],
        )


def clear_cached_payloads(
    *,
    category: str | None = None,
    provider: str | None = None,
    cache_key_prefixes: list[str] | None = None,
    db_path: str | None = None,
) -> int:
    clauses: list[str] = []
    params: list[Any] = []
    if category:
        clauses.append("category = ?")
        params.append(category)
    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    prefixes = [prefix for prefix in (cache_key_prefixes or []) if prefix]
    if prefixes:
        clauses.append("(" + " OR ".join(["cache_key LIKE ?"] * len(prefixes)) + ")")
        params.extend([f"{prefix}%" for prefix in prefixes])
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with _connect(db_path) as connection:
        deleted = int(connection.execute(f"SELECT COUNT(*) FROM api_cache {where}", params).fetchone()[0] or 0)
        connection.execute(f"DELETE FROM api_cache {where}", params)
        return deleted


def provider_is_limited(provider: str, *, db_path: str | None = None) -> bool:
    with _connect(db_path) as connection:
        row = connection.execute(
            "SELECT limited_until FROM api_provider_state WHERE provider = ?",
            [provider],
        ).fetchone()
    if not row:
        return False
    limited_until = _timestamp(row[0])
    return bool(limited_until is not None and limited_until > _now())


def record_provider_success(category: str, provider: str, *, db_path: str | None = None) -> None:
    now = datetime.now(timezone.utc)
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO api_provider_state
                (provider, category, request_count, window_started_at, limited_until, last_error, updated_at)
            SELECT ?, ?, COALESCE(request_count, 0) + 1, ?, NULL, NULL, ?
            FROM (SELECT NULL) LEFT JOIN api_provider_state ON provider = ?
            """,
            [provider, category, now, now, provider],
        )


def record_provider_limited(
    category: str,
    provider: str,
    error: str,
    *,
    retry_after_seconds: int = 3600,
    db_path: str | None = None,
) -> None:
    now_dt = datetime.now(timezone.utc)
    limited_until = now_dt + timedelta(seconds=max(int(retry_after_seconds), 60))
    now = now_dt
    with _connect(db_path) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO api_provider_state
                (provider, category, request_count, window_started_at, limited_until, last_error, updated_at)
            SELECT ?, ?, COALESCE(request_count, 0) + 1, ?, ?, ?, ?
            FROM (SELECT NULL) LEFT JOIN api_provider_state ON provider = ?
            """,
            [provider, category, now, limited_until, error[:500], now, provider],
        )
