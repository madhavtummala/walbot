"""Which ``Config`` a read-only view of an algorithm is computed against.

Signal views and backtests compute a proposal and submit nothing, so unlike
``controls.resolve_deployment_for_origin`` they cannot refuse when nothing is deployed -- but
some algorithms (e.g. a DCA plan) are configured per account, so they still have to answer for
*some* account rather than always the default.
"""

from __future__ import annotations

import logging
from typing import Any

from src.api.controls import account_for_strategy

from ...core.config import UnknownAccountError, get_config

logger = logging.getLogger(__name__)


def config_for_strategy_view(strategy: str, account_id: str = "") -> Any:
    """The config a signal view or backtest of ``strategy`` should read.

    ``account_id`` names the account outright; empty means "ask the deployment". A deployment can
    outlive the account it names (``sanitize_deployment`` doesn't check), so this falls back to
    the default account rather than taking the whole dashboard down for a stale deployment.
    """
    resolved = str(account_id or "")[:80] or account_for_strategy(strategy)
    try:
        return get_config(account_id=resolved or None, strategy_id=strategy)
    except UnknownAccountError:
        logger.warning(
            "%s is deployed to account %r, which no longer exists; showing the default account.",
            strategy,
            resolved,
        )
        return get_config(strategy_id=strategy)
