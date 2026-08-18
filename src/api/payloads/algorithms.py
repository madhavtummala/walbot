"""Per-algorithm tuning, signals and order history.

Split out of the single ``api_payloads`` module, which had grown to 1253 lines covering nine
unrelated domains. The public names are unchanged and still importable from ``api_payloads``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .strategy_config import config_for_strategy_view

from ...core.config import (
    DEFAULT_STRATEGY_ID,
    get_config,
    load_algorithms_config,
    save_algorithms_config,
)
from ...algorithms.explainers import explainer_for
from ...algorithms.registry import LEGACY_ALGORITHM_IDS, canonical_algorithm_id, get_algorithm_class
from ...data.order_journal import load_order_journal

logger = logging.getLogger(__name__)





def algorithm_config_payload(strategy: str) -> dict[str, Any]:
    """The saved tuning for one algorithm, read from the key it actually lives under."""
    strategy = canonical_algorithm_id(strategy)[:80]
    sections = load_algorithms_config().get("algorithms") or {}
    if not isinstance(sections, dict):
        sections = {}
    key = strategy if strategy in sections else next(
        (legacy for legacy in LEGACY_ALGORITHM_IDS.get(strategy, []) if legacy in sections), strategy
    )
    values = sections.get(key) if isinstance(sections.get(key), dict) else {}
    return {
        "strategy": strategy,
        # Surfaced so the dashboard can say which key on disk a value came from: several
        # algorithms are still filed under a retired id.
        "config_key": key,
        "config": values,
        "explainer": explainer_for(strategy),
    }


def save_algorithm_config_payload(strategy: str, values: Any) -> dict[str, Any]:
    strategy = canonical_algorithm_id(strategy)[:80]
    # A list or scalar here means the caller sent the wrong shape. Coercing it to {} would
    # quietly erase every tuned value for this algorithm, so refuse instead.
    if not isinstance(values, dict):
        raise ValueError("Algorithm config must be a JSON object.")
    raw = load_algorithms_config()
    sections = raw.setdefault("algorithms", {})
    if not isinstance(sections, dict):
        sections = {}
        raw["algorithms"] = sections
    # Write back to the key it was read from, so retired ids keep their tuning rather than
    # gaining a second, silently-ignored copy under the canonical name.
    key = strategy if strategy in sections else next(
        (legacy for legacy in LEGACY_ALGORITHM_IDS.get(strategy, []) if legacy in sections), strategy
    )
    sections[key] = values
    save_algorithms_config(raw)
    return algorithm_config_payload(strategy)


def algorithm_activity_payload(strategy: str = "", limit: int = 40) -> dict[str, Any]:
    """Orders this bot placed for one algorithm, from its own journal.

    The counterpart to account_activity_payload: the broker knows the fill, only the bot knows
    which algorithm asked for it. Neither view replaces the other.
    """
    strategy_id = canonical_algorithm_id(strategy) if strategy else ""
    return {
        "strategy": strategy_id,
        "rows": load_order_journal(strategy=strategy_id, limit=limit),
    }


def strategy_signals_payload(strategy: str = DEFAULT_STRATEGY_ID, account_id: str = "") -> dict[str, Any]:
    """Dashboard signal payload: dispatch to the algorithm's own ``signal_view``.

    ``none`` used to fall back to the DCA plan's view, because DCA was not selectable in the
    deck and would otherwise have had nowhere to render. DCA is an ordinary algorithm now, so
    a saved ``none`` simply resolves to it.

    The view is computed for the account this strategy is deployed on, not for the default
    account. DCA's plan is per account, so reading the default one showed the wrong plan --
    and, because ``analyze`` persists accrual, wrote that account's ledger too.
    """
    strategy = canonical_algorithm_id(strategy or DEFAULT_STRATEGY_ID)[:80]
    config = config_for_strategy_view(strategy, account_id)
    algorithm = get_algorithm_class(strategy).from_config(config)
    view = algorithm.signal_view(config)
    return {
        "strategy": strategy,
        # Which account this view describes. Per-account config (a DCA plan) means the answer
        # is not the same for every deployment, so the view has to say which one it is.
        "account_id": getattr(config, "account_id", ""),
        "wired": view.wired,
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "summary": view.summary,
        "leaders": view.leaders,
    }
