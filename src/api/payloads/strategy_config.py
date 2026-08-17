"""Which ``Config`` a read-only view of an algorithm is computed against.

Signal views and backtests are not order-placing origins -- they compute a proposal and submit
nothing, so unlike ``controls.resolve_binding_for_origin`` they cannot refuse when the answer
is ambiguous. They still have to answer *for some account*, because some algorithms are
configured per account: a DCA plan is per account, so computing against the default account
rendered and backtested one plan while the dashboard's own editor wrote another, and no edit
ever showed up in either view.

This lives in the payloads package rather than beside ``account_for_strategy`` in
``controls`` so that it builds its config through the same ``get_config`` name every other
payload module resolves -- which is what lets one patch of that name cover the whole package.
"""

from __future__ import annotations

import logging
from typing import Any

from src.api.controls import account_for_strategy

from ...core.config import UnknownAccountError, get_config

logger = logging.getLogger(__name__)


def config_for_strategy_view(strategy: str, account_id: str = "") -> Any:
    """The config a signal view or backtest of ``strategy`` should read.

    ``account_id`` names the account outright; empty means "ask the binding". The one
    concession a read-only view makes that an order-placing path must not: ``sanitize_binding``
    does not check ``account_id`` against the configured accounts, so a binding can outlive the
    account it names. Refusing would take the whole dashboard down for a stale binding, so this
    falls back to the default account -- and callers report the account actually used, so the
    substitution is visible rather than silent.
    """
    resolved = str(account_id or "")[:80] or account_for_strategy(strategy)
    try:
        return get_config(account_id=resolved or None, strategy_id=strategy)
    except UnknownAccountError:
        logger.warning(
            "Binding for %s names account %r, which no longer exists; showing the default account.",
            strategy,
            resolved,
        )
        return get_config(strategy_id=strategy)
