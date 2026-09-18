from __future__ import annotations

import logging
import time
from typing import Any

import requests

from ...data.state_store import load_state, save_state

logger = logging.getLogger(__name__)

BASE_URL = "https://api.schwabapi.com"
TOKEN_URL = f"{BASE_URL}/v1/oauth/token"
TRADER_BASE = f"{BASE_URL}/trader/v1"
MARKETDATA_BASE = f"{BASE_URL}/marketdata/v1"

#: Where the rotating refresh token is persisted between runs.
TOKEN_STATE_KEY = "schwab_oauth_token"

#: Refresh a little before expiry so a long request cannot straddle the boundary.
TOKEN_EXPIRY_MARGIN_SECONDS = 60


class SchwabAuthError(RuntimeError):
    """Raised when Schwab credentials are missing or a token cannot be refreshed."""


class SchwabAPIError(RuntimeError):
    """Raised when a Schwab endpoint returns an error response."""

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"Schwab API returned {status_code}: {body}")


class SchwabSession:
    """Authenticated HTTP session for the Schwab Trader and Market Data APIs.

    Schwab uses three-legged OAuth: a long-lived refresh token (obtained once through the
    browser consent flow) is exchanged for a short-lived access token. Only the exchange is
    automated here -- the initial consent must be done out of band, and its refresh token
    supplied via config or the state store.
    """

    def __init__(self, config: Any, session: Any = None):
        self.app_key = str(getattr(config, "schwab_app_key", "") or "")
        self.app_secret = str(getattr(config, "schwab_app_secret", "") or "")
        self._session = session or requests.Session()
        self._access_token = ""
        self._expires_at = 0.0
        #: Account number (digits only) to Schwab's hash for it. See :func:`account_hash`.
        self.account_hashes: dict[str, str] = {}
        #: ``(monotonic stamp, is_open)`` from the last market-hours read, or ``None``. Held here
        #: rather than on the brokerage because a brokerage lasts one request and this does not.
        self.market_hours: tuple[float, bool] | None = None

        stored = load_state(TOKEN_STATE_KEY, {}) or {}
        self._refresh_token = str(getattr(config, "schwab_refresh_token", "") or stored.get("refresh_token") or "")

    # -- auth ---------------------------------------------------------------------------

    def _require_credentials(self) -> None:
        missing = [
            name
            for name, value in (
                ("schwab_app_key", self.app_key),
                ("schwab_app_secret", self.app_secret),
                ("schwab_refresh_token", self._refresh_token),
            )
            if not value
        ]
        if missing:
            raise SchwabAuthError(
                f"Missing Schwab credentials: {', '.join(missing)}. Complete the OAuth consent "
                "flow once and store the resulting refresh token."
            )

    def _refresh_access_token(self) -> None:
        self._require_credentials()
        response = self._session.post(
            TOKEN_URL,
            data={"grant_type": "refresh_token", "refresh_token": self._refresh_token},
            auth=(self.app_key, self.app_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        if response.status_code >= 400:
            raise SchwabAuthError(f"Token refresh failed ({response.status_code}): {response.text}")

        payload = response.json()
        self._access_token = str(payload.get("access_token") or "")
        if not self._access_token:
            raise SchwabAuthError(f"Token refresh returned no access_token: {payload}")
        self._expires_at = time.time() + float(payload.get("expires_in", 1800))

        # Schwab rotates the refresh token on some grants; persist it or the next run fails.
        rotated = str(payload.get("refresh_token") or "")
        if rotated and rotated != self._refresh_token:
            self._refresh_token = rotated
            # The 7-day clock runs from the original consent and a rotation does not restart
            # it, so carry the recorded issue time across rather than stamping a new one.
            stored = load_state(TOKEN_STATE_KEY, {}) or {}
            issued_at = stored.get("issued_at") if isinstance(stored, dict) else None
            record: dict[str, Any] = {"refresh_token": rotated}
            if isinstance(issued_at, (int, float)):
                record["issued_at"] = issued_at
            save_state(TOKEN_STATE_KEY, record)
            logger.info("Schwab refresh token rotated and persisted")

    def access_token(self) -> str:
        if not self._access_token or time.time() >= self._expires_at - TOKEN_EXPIRY_MARGIN_SECONDS:
            self._refresh_access_token()
        return self._access_token

    # -- requests -----------------------------------------------------------------------

    def request(self, method: str, url: str, **kwargs) -> Any:
        headers = {"Authorization": f"Bearer {self.access_token()}", "Accept": "application/json"}
        headers.update(kwargs.pop("headers", {}))
        response = self._session.request(method, url, headers=headers, timeout=30, **kwargs)
        if response.status_code >= 400:
            raise SchwabAPIError(response.status_code, response.text)
        # Order creation answers 201 with an empty body; the new order id is only in Location,
        # so it must survive the empty-body case.
        location = response.headers.get("Location", "")
        if not response.content:
            return {"location": location} if location else {}
        try:
            return response.json()
        except ValueError:
            return {"raw": response.text, "location": location}

    def get(self, url: str, **kwargs) -> Any:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> Any:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> Any:
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs) -> Any:
        return self.request("DELETE", url, **kwargs)


#: Live sessions by app key, so the access token outlives the object that fetched it. Keyed on
#: the credentials rather than the account: one token reaches all of a user's accounts.
_SESSIONS: dict[str, "SchwabSession"] = {}


def shared_session(config: Any) -> SchwabSession:
    """The session for these credentials, reused across brokerages and requests.

    ``resolve_brokerage`` builds a fresh brokerage per request and per bot run, and a fresh
    session with it -- which left the expiry check in :meth:`SchwabSession.access_token`
    unreachable, so every page load paid an OAuth round trip per account before reading
    anything. Callers passing their own session bypass this.
    """
    app_key = str(getattr(config, "schwab_app_key", "") or "")
    if not app_key:
        # Nothing to key on, and the missing-credential error belongs to the session itself.
        return SchwabSession(config)
    if app_key not in _SESSIONS:
        _SESSIONS[app_key] = SchwabSession(config)
    return _SESSIONS[app_key]


def reset_shared_sessions() -> None:
    """Drop the cached sessions. For tests and for a credentials change."""
    _SESSIONS.clear()


def _digits(value: Any) -> str:
    """An account number reduced to its digits, so formatting never decides identity."""
    return "".join(character for character in str(value or "") if character.isdigit())


def account_hash(session: SchwabSession, account_number: str = "") -> str:
    """Resolve the hash Schwab requires in account-scoped paths.

    Account-scoped endpoints reject the plain account number, so every call must go through
    ``/accounts/accountNumbers`` first. Returns the hash for ``account_number``, or the first
    account when none is specified.

    Cached on the *session*, which owns the credentials the mapping belongs to. Schwab mints a
    hash once, so re-resolving it cost a round trip per request for an answer that cannot
    change. A session carrying no cache is resolved uncached rather than refused: anything with
    a ``get`` is usable here, and an optimization is no reason to narrow that.
    """
    wanted = _digits(account_number)
    cache = getattr(session, "account_hashes", None)
    if cache is not None and cache.get(wanted):
        return cache[wanted]

    accounts = session.get(f"{TRADER_BASE}/accounts/accountNumbers") or []
    if not accounts:
        raise SchwabAPIError(404, "Schwab returned no accounts for these credentials")

    # One response lists every account, so every account is cached from it. Keeping only the one
    # asked for meant a page reading five accounts fetched this same listing five times.
    if cache is not None:
        for entry in accounts:
            number, value = _digits(entry.get("accountNumber", "")), str(entry.get("hashValue", ""))
            if number and value:
                cache[number] = value
        first = str(accounts[0].get("hashValue", ""))
        if first:
            cache[""] = first

    for entry in accounts:
        # Compared on digits alone. Schwab's API reports the number bare -- "12345678" -- while
        # every human-facing surface, statements and the website included, writes it "1234-5678".
        # Matching the raw strings made a correctly configured account 404, and because the
        # dashboard falls back to the default account on error, the Schwab tab then showed
        # Alpaca's money under Schwab's name.
        if not wanted or _digits(entry.get("accountNumber", "")) == wanted:
            resolved = str(entry.get("hashValue", ""))
            return resolved
    raise SchwabAPIError(
        404,
        f"Schwab account {account_number} not found in {len(accounts)} account(s)",
    )
