from __future__ import annotations

import json
import time
from typing import Any

import pytest

from src.brokerages import schwab_auth
from src.brokerages.schwab_auth import (
    PENDING_STATE_KEY,
    REFRESH_TOKEN_TTL_SECONDS,
    auth_status,
    begin_authorization,
    complete_authorization,
)
from src.brokerages.schwab_client import TOKEN_STATE_KEY, SchwabAuthError, SchwabSession
from src.core.config import Config
from src.data.state_store import ephemeral_state


class FakeResponse:
    def __init__(self, payload: Any = None, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload) if payload is not None else ""
        self.content = self.text.encode()

    def json(self) -> Any:
        return self._payload


class FakeHTTP:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def make_config(**overrides: Any) -> Config:
    base = {
        "schwab_app_key": "key",
        "schwab_app_secret": "secret",
        "schwab_callback_url": "https://trading.example.org/schwab/callback",
    }
    base.update(overrides)
    return Config(**base)


def test_status_is_unconfigured_without_callback_url():
    with ephemeral_state():
        status = auth_status(make_config(schwab_callback_url=""))
    assert status["state"] == "unconfigured"
    assert status["configured"] is False
    assert "SCHWAB_CALLBACK_URL" in status["detail"]


def test_status_is_missing_when_never_authorized():
    with ephemeral_state():
        status = auth_status(make_config())
    assert status["state"] == "missing"


def test_status_tracks_seven_day_window():
    fresh = time.time() - 3600
    with ephemeral_state({TOKEN_STATE_KEY: {"refresh_token": "r", "issued_at": fresh}}):
        assert auth_status(make_config())["state"] == "ok"

    stale = time.time() - (REFRESH_TOKEN_TTL_SECONDS - 3600)
    with ephemeral_state({TOKEN_STATE_KEY: {"refresh_token": "r", "issued_at": stale}}):
        assert auth_status(make_config())["state"] == "warning"

    dead = time.time() - (REFRESH_TOKEN_TTL_SECONDS + 60)
    with ephemeral_state({TOKEN_STATE_KEY: {"refresh_token": "r", "issued_at": dead}}):
        assert auth_status(make_config())["state"] == "expired"


def test_status_tolerates_token_without_issue_time():
    """A refresh token from the old env-var flow has no age; it must not read as expired."""
    with ephemeral_state({TOKEN_STATE_KEY: {"refresh_token": "r"}}):
        status = auth_status(make_config())
    assert status["state"] == "ok"
    assert status["expires_at"] is None


def test_begin_authorization_builds_url_and_stores_nonce():
    with ephemeral_state() as store:
        result = begin_authorization(make_config())
        nonce = store[PENDING_STATE_KEY]["state"]
    assert result["state"] == nonce
    assert result["authorize_url"].startswith("https://api.schwabapi.com/v1/oauth/authorize?")
    assert "client_id=key" in result["authorize_url"]
    assert "response_type=code" in result["authorize_url"]
    assert f"state={nonce}" in result["authorize_url"]
    assert "redirect_uri=https%3A%2F%2Ftrading.example.org%2Fschwab%2Fcallback" in result["authorize_url"]


def test_complete_authorization_persists_token_with_issue_time():
    http = FakeHTTP(FakeResponse({"refresh_token": "new-refresh", "access_token": "a"}))
    with ephemeral_state() as store:
        nonce = begin_authorization(make_config())["state"]
        complete_authorization(make_config(), code="the-code", returned_state=nonce, session=http)
        stored = store[TOKEN_STATE_KEY]

    assert stored["refresh_token"] == "new-refresh"
    assert isinstance(stored["issued_at"], float)
    url, kwargs = http.calls[0]
    assert kwargs["data"]["grant_type"] == "authorization_code"
    assert kwargs["data"]["code"] == "the-code"
    assert kwargs["data"]["redirect_uri"] == "https://trading.example.org/schwab/callback"
    assert kwargs["auth"] == ("key", "secret")


def test_complete_authorization_rejects_mismatched_state():
    http = FakeHTTP(FakeResponse({"refresh_token": "r"}))
    with ephemeral_state():
        begin_authorization(make_config())
        with pytest.raises(SchwabAuthError, match="did not match"):
            complete_authorization(make_config(), code="c", returned_state="forged", session=http)
    assert http.calls == []


def test_nonce_is_single_use():
    http = FakeHTTP(FakeResponse({"refresh_token": "r"}))
    with ephemeral_state():
        nonce = begin_authorization(make_config())["state"]
        complete_authorization(make_config(), code="c", returned_state=nonce, session=http)
        with pytest.raises(SchwabAuthError, match="No Schwab authorization is in progress"):
            complete_authorization(make_config(), code="c", returned_state=nonce, session=http)


def test_expired_nonce_is_rejected(monkeypatch):
    http = FakeHTTP(FakeResponse({"refresh_token": "r"}))
    with ephemeral_state():
        nonce = begin_authorization(make_config())["state"]
        monkeypatch.setattr(schwab_auth, "_now", lambda: time.time() + schwab_auth.PENDING_STATE_TTL_SECONDS + 60)
        with pytest.raises(SchwabAuthError, match="took too long"):
            complete_authorization(make_config(), code="c", returned_state=nonce, session=http)


def test_rotation_preserves_original_issue_time():
    """Rotating the refresh token must not restart the 7-day clock, or the pill lies."""
    issued_at = time.time() - 4 * 24 * 60 * 60
    http = FakeHTTP(FakeResponse({"access_token": "a", "expires_in": 1800, "refresh_token": "rotated"}))
    config = make_config(schwab_refresh_token="original")
    with ephemeral_state({TOKEN_STATE_KEY: {"refresh_token": "original", "issued_at": issued_at}}) as store:
        SchwabSession(config, session=http).access_token()
        stored = store[TOKEN_STATE_KEY]

    assert stored["refresh_token"] == "rotated"
    assert stored["issued_at"] == issued_at
