"""Talking to a provider: credentials, requests, and telling a rate limit from an outage.

The distinction :class:`ProviderRateLimited` draws is the point. An outage means try the next
provider; a rate limit means this one is fine and will answer later, so it is recorded against
the provider and the fallback happens anyway -- but the accounting differs.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from typing import Any

from ..data.provider_cache import record_provider_limited, record_provider_success
from .sources import ProviderRateLimited, ProviderUnavailable

logger = logging.getLogger(__name__)

def _request_json(
    provider: str,
    category: str,
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    query = urllib.parse.urlencode({key: value for key, value in (params or {}).items() if value not in {None, ""}})
    full_url = f"{url}?{query}" if query else url
    request_headers = {"User-Agent": "walbot/1.0", **(headers or {})}
    request = urllib.request.Request(full_url, headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code in {402, 403, 429}:
            record_provider_limited(category, provider, error or str(exc), retry_after_seconds=3600)
            raise ProviderRateLimited(error or str(exc)) from exc
        raise ProviderUnavailable(error or str(exc)) from exc
    except urllib.error.URLError as exc:
        raise ProviderUnavailable(str(exc)) from exc

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderUnavailable("provider returned non-JSON response") from exc

    if _looks_limited(payload):
        record_provider_limited(category, provider, json.dumps(payload)[:500], retry_after_seconds=3600)
        raise ProviderRateLimited("provider quota appears exhausted")
    record_provider_success(category, provider)
    return payload


def _looks_limited(payload: Any) -> bool:
    text = json.dumps(payload).lower() if isinstance(payload, (dict, list)) else str(payload).lower()
    markers = ("rate limit", "rate-limit", "too many requests", "quota", "api call frequency", "limit reached")
    return any(marker in text for marker in markers)


def _basic_auth_header(username: str, password: str) -> dict[str, str]:
    token = b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}", "Accept": "application/json"}


def _bearer_auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}", "Accept": "application/json"}
