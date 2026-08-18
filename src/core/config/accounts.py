"""Resolving an account id to its broker.

Separate because this decides where an order is sent: an account that cannot be resolved
must raise rather than fall back, which is the difference between a blank dashboard panel
and orders landing in the wrong book.
"""

from __future__ import annotations

from typing import Any

from .coercion import _normalize_keyed_items
from .defaults import UNNAMED_ACCOUNT_ID

from .yaml_io import load_accounts_config


class UnknownAccountError(KeyError):
    """A named account is not in the accounts config.

    Its own type so callers can tell "you asked for an account that does not exist" apart from
    "the broker is unreachable". The first is a configuration mistake that must never be
    papered over with a different account; the second is weather.
    """

    def __init__(self, account_id: str, known: list[str] | None = None) -> None:
        self.account_id = account_id
        self.known = known or []
        super().__init__(
            f"Unknown account {account_id!r}"
            + (f"; configured accounts are {', '.join(self.known)}" if self.known else "")
        )

    def __str__(self) -> str:
        return self.args[0] if self.args else f"Unknown account {self.account_id!r}"


def _normalize_accounts_config(raw: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    if not raw:
        return "", {}
    if isinstance(raw.get("accounts"), list):
        return str(raw.get("default") or ""), _normalize_keyed_items(raw.get("accounts"))
    accounts = raw.get("accounts") if isinstance(raw.get("accounts"), dict) else raw
    if not isinstance(accounts, dict):
        return "", {}
    items = accounts.get("items", accounts.get("accounts", []))
    return str(accounts.get("default") or raw.get("default") or ""), _normalize_keyed_items(items)


def get_account_broker_type(account_id: str) -> str:
    """Resolve the broker type for a given account ID from accounts config.

    Raises rather than guessing. This decides which brokerage an order is sent to, so
    defaulting an unrecognised account to Alpaca is the one failure mode that could move real
    money in the wrong account.
    """
    raw = load_accounts_config()
    _, items = _normalize_accounts_config(raw)
    account = items.get(account_id, {})
    if not account:
        named = bool(account_id) and account_id != UNNAMED_ACCOUNT_ID
        if items and named:
            raise UnknownAccountError(str(account_id), sorted(items))
        if items:
            default_id = raw.get("default", "")
            account = items.get(default_id, {}) if isinstance(default_id, str) else {}
            account = account or next(iter(items.values()), {})
    return str(account.get("broker", "alpaca")).strip().lower()
