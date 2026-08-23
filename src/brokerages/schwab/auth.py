from __future__ import annotations

import logging
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import requests

from ...data.state_store import load_state, save_state
from .client import BASE_URL, TOKEN_STATE_KEY, TOKEN_URL, SchwabAuthError

logger = logging.getLogger(__name__)

AUTHORIZE_URL = f"{BASE_URL}/v1/oauth/authorize"

#: Schwab refresh tokens die 7 days after the consent that minted them. Refreshing an access
#: token does not extend this, so the only cure is another trip through the browser.
REFRESH_TOKEN_TTL_SECONDS = 7 * 24 * 60 * 60

#: Turn the dashboard pill amber with this much life left, so a re-auth can happen at leisure
#: rather than in the middle of a trading session.
WARNING_THRESHOLD_SECONDS = 36 * 60 * 60

#: Where the pending CSRF nonce lives between building the authorize URL and the callback.
PENDING_STATE_KEY = "schwab_oauth_pending"

#: A consent that has not come back within this window is treated as abandoned.
PENDING_STATE_TTL_SECONDS = 15 * 60


def _now() -> float:
    return time.time()


def stored_token() -> dict[str, Any]:
    stored = load_state(TOKEN_STATE_KEY, {}) or {}
    return stored if isinstance(stored, dict) else {}


def _callback_url(config: Any) -> str:
    return str(getattr(config, "schwab_callback_url", "") or "").strip()


def auth_status(config: Any) -> dict[str, Any]:
    """Describe the Schwab consent well enough for the dashboard to colour a pill.

    ``state`` is one of ``unconfigured`` (no app key/secret/callback, so nothing to do),
    ``missing`` (configured but never consented), ``ok``, ``warning``, or ``expired``.
    """
    app_key = str(getattr(config, "schwab_app_key", "") or "")
    app_secret = str(getattr(config, "schwab_app_secret", "") or "")
    callback_url = _callback_url(config)

    configured = bool(app_key and app_secret and callback_url)
    status: dict[str, Any] = {
        "configured": configured,
        "callback_url": callback_url,
        "state": "unconfigured",
        "expires_at": None,
        "seconds_remaining": None,
        "detail": "",
    }
    if not configured:
        missing = [
            name
            for name, value in (
                ("SCHWAB_APP_KEY", app_key),
                ("SCHWAB_APP_SECRET", app_secret),
                ("SCHWAB_CALLBACK_URL", callback_url),
            )
            if not value
        ]
        status["detail"] = f"Schwab is not configured: missing {', '.join(missing)}."
        return status

    stored = stored_token()
    # A refresh token supplied through config predates the dashboard flow and carries no
    # issue time, so its age is unknowable -- report it as healthy rather than crying wolf.
    refresh_token = str(stored.get("refresh_token") or getattr(config, "schwab_refresh_token", "") or "")
    if not refresh_token:
        status["state"] = "missing"
        status["detail"] = "Schwab has never been authorized. Connect to place trades."
        return status

    issued_at = stored.get("issued_at")
    if not isinstance(issued_at, (int, float)):
        status["state"] = "ok"
        status["detail"] = "Authorized. Expiry unknown -- reconnect once to start tracking it."
        return status

    expires_at = float(issued_at) + REFRESH_TOKEN_TTL_SECONDS
    remaining = expires_at - _now()
    status["expires_at"] = expires_at
    status["seconds_remaining"] = remaining
    if remaining <= 0:
        status["state"] = "expired"
        status["detail"] = "Schwab authorization expired. Reconnect to resume trading."
    elif remaining <= WARNING_THRESHOLD_SECONDS:
        status["state"] = "warning"
        status["detail"] = f"Schwab authorization expires in {_humanize(remaining)}."
    else:
        status["state"] = "ok"
        status["detail"] = f"Authorized. Expires in {_humanize(remaining)}."
    return status


def _humanize(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = seconds / 3600.0
    if hours < 1:
        return f"{int(seconds // 60)}m"
    if hours < 48:
        return f"{int(hours)}h"
    return f"{int(hours // 24)}d"


def begin_authorization(config: Any) -> dict[str, Any]:
    """Mint a CSRF nonce and build the Schwab consent URL to send the browser to."""
    app_key = str(getattr(config, "schwab_app_key", "") or "")
    callback_url = _callback_url(config)
    if not app_key or not callback_url:
        raise SchwabAuthError("Schwab app key and callback URL must be set before authorizing.")

    nonce = secrets.token_urlsafe(24)
    save_state(PENDING_STATE_KEY, {"state": nonce, "created_at": _now()})
    query = urlencode(
        {
            "client_id": app_key,
            "redirect_uri": callback_url,
            "response_type": "code",
            "state": nonce,
        }
    )
    return {"authorize_url": f"{AUTHORIZE_URL}?{query}", "state": nonce}


def _consume_pending_state(returned_state: str) -> None:
    pending = load_state(PENDING_STATE_KEY, {}) or {}
    expected = str(pending.get("state") or "") if isinstance(pending, dict) else ""
    created_at = pending.get("created_at") if isinstance(pending, dict) else None

    # Clear first: a nonce is single-use whether or not it validates, so a replay of the same
    # callback cannot mint a second token.
    save_state(PENDING_STATE_KEY, {})

    if not expected:
        raise SchwabAuthError("No Schwab authorization is in progress. Start again from the dashboard.")
    if not secrets.compare_digest(expected, str(returned_state or "")):
        raise SchwabAuthError("Schwab authorization state did not match. Start again from the dashboard.")
    if isinstance(created_at, (int, float)) and _now() - float(created_at) > PENDING_STATE_TTL_SECONDS:
        raise SchwabAuthError("Schwab authorization took too long. Start again from the dashboard.")


def complete_authorization(config: Any, code: str, returned_state: str, session: Any = None) -> dict[str, Any]:
    """Exchange a consent code for tokens and persist the refresh token with its issue time."""
    app_key = str(getattr(config, "schwab_app_key", "") or "")
    app_secret = str(getattr(config, "schwab_app_secret", "") or "")
    callback_url = _callback_url(config)
    if not app_key or not app_secret or not callback_url:
        raise SchwabAuthError("Schwab app key, secret, and callback URL must all be set.")
    if not code:
        raise SchwabAuthError("Schwab did not return an authorization code.")

    _consume_pending_state(returned_state)

    http = session or requests
    response = http.post(
        TOKEN_URL,
        data={"grant_type": "authorization_code", "code": code, "redirect_uri": callback_url},
        auth=(app_key, app_secret),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )
    if response.status_code >= 400:
        raise SchwabAuthError(f"Schwab code exchange failed ({response.status_code}): {response.text}")

    payload = response.json()
    refresh_token = str(payload.get("refresh_token") or "")
    if not refresh_token:
        raise SchwabAuthError("Schwab code exchange returned no refresh_token.")

    issued_at = _now()
    save_state(TOKEN_STATE_KEY, {"refresh_token": refresh_token, "issued_at": issued_at})
    logger.info("Schwab authorization completed; refresh token valid for 7 days")
    return {"refresh_token": refresh_token, "issued_at": issued_at}
