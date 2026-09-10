"""Runtime switches and the Schwab OAuth handshake.

Split out of the single ``api_payloads`` module. Public names are unchanged and still
importable from ``api_payloads``.
"""

from __future__ import annotations

import logging
from typing import Any

from ...brokerages.schwab.auth import auth_status, begin_authorization, complete_authorization
from ...core.bot_runtime import bot_runtime
from ...core.config import get_config
from ...core.config.accounts import UnknownAccountError
from ..controls import load_controls, save_controls

logger = logging.getLogger(__name__)


def controls_payload() -> dict[str, Any]:
    config = get_config()
    controls = load_controls()
    if not controls.get("trading_account_id"):
        controls["trading_account_id"] = config.account_id
    return {
        "controls": controls,
        "accounts": config.account_options,
        "bot": bot_runtime.snapshot(),
    }


def schwab_auth_payload() -> dict[str, Any]:
    """Consent status, plus whether Schwab is wired up as a connector at all.

    The dashboard shows the Schwab row when the *connector* is configured, not when consent
    has been completed -- otherwise the one control that starts consent is hidden until after
    consent, which is the wrong way round.
    """
    config = get_config()
    connectors = set(getattr(config, "intraday_market_data_provider_order", []) or []) | set(
        getattr(config, "eod_market_data_provider_order", []) or []
    )
    return {**auth_status(config), "connector_enabled": "schwab" in connectors}


def start_schwab_auth_payload() -> dict[str, Any]:
    return begin_authorization(get_config())


def complete_schwab_auth_payload(code: str, state: str) -> dict[str, Any]:
    complete_authorization(get_config(), code=code, returned_state=state)
    return schwab_auth_payload()


def save_controls_payload(body: dict[str, Any]) -> dict[str, Any]:
    raw_controls = body.get("controls", body)
    controls = save_controls(raw_controls)
    return {
        "controls": controls,
        "accounts": _account_options(str(controls.get("trading_account_id") or "")),
        "bot": bot_runtime.snapshot(),
    }


def _account_options(account_id: str) -> list[dict[str, str]]:
    """The account list, which must not depend on the account being asked about -- it *is* the
    list of them, so a stale id posted from an old browser tab must not fail the whole save."""
    try:
        return get_config(account_id=account_id or None).account_options
    except UnknownAccountError:
        logger.warning(
            "Controls named account %r, which no longer exists; listing accounts from the "
            "default instead.", account_id,
        )
        return get_config().account_options
