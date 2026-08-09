from __future__ import annotations

import logging
import time
from typing import Any

import requests

from ..data.state_store import load_state, save_state

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
            save_state(TOKEN_STATE_KEY, {"refresh_token": rotated})
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

    def delete(self, url: str, **kwargs) -> Any:
        return self.request("DELETE", url, **kwargs)


def account_hash(session: SchwabSession, account_number: str = "") -> str:
    """Resolve the hash Schwab requires in account-scoped paths.

    Account-scoped endpoints reject the plain account number, so every call must go through
    ``/accounts/accountNumbers`` first. Returns the hash for ``account_number``, or the first
    account when none is specified.
    """
    accounts = session.get(f"{TRADER_BASE}/accounts/accountNumbers") or []
    if not accounts:
        raise SchwabAPIError(404, "Schwab returned no accounts for these credentials")

    wanted = str(account_number or "").strip()
    for entry in accounts:
        if not wanted or str(entry.get("accountNumber", "")) == wanted:
            return str(entry.get("hashValue", ""))
    raise SchwabAPIError(404, f"Schwab account {wanted} not found in {len(accounts)} account(s)")
