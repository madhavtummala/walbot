"""Per-algorithm tuning, signals and order history.

Split out of the single ``api_payloads`` module, which had grown to 1253 lines covering nine
unrelated domains. The public names are unchanged and still importable from ``api_payloads``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from hashlib import sha256
from typing import Any

import pandas as pd

from .strategy_config import config_for_strategy_view

from ...core.config import (
    DEFAULT_STRATEGY_ID,
    load_algorithms_config,
    save_algorithms_config,
)
from ...algorithms.explainers import explainer_for
from ...algorithms.registry import LEGACY_ALGORITHM_IDS, canonical_algorithm_id, get_algorithm_class
from ...core.runner import run_algorithm
from ...data.order_journal import load_order_journal
from ...data.state_store import load_state, save_state

logger = logging.getLogger(__name__)


SIGNALS_CACHE_STATE_KEY = "strategy_signals_cache"

#: Bumped whenever the shape of a cached snapshot changes. A version mismatch throws the whole
#: store away -- the same rule the backtest cache applies -- so a row written under an older
#: schema can never be served to a frontend that no longer understands it.
SIGNALS_CACHE_VERSION = 1


def _load_signals_cache() -> dict[str, Any]:
    cache = load_state(SIGNALS_CACHE_STATE_KEY, {"version": SIGNALS_CACHE_VERSION, "items": {}})
    if cache.get("version") == SIGNALS_CACHE_VERSION and isinstance(cache.get("items"), dict):
        return cache
    return {"version": SIGNALS_CACHE_VERSION, "items": {}}


def _save_signals_cache(cache: dict[str, Any]) -> None:
    save_state(SIGNALS_CACHE_STATE_KEY, cache)


def _signals_cache_key(strategy: str, account_id: str = "") -> str:
    """Hash of everything a cached snapshot's validity depends on.

    The same basis the backtest cache hashes -- minus the period, which a signal view does not
    have -- so a snapshot computed under one tuning is never served for another. The account is
    part of it because per-account plans make the answer differ per deployment.
    """
    config = config_for_strategy_view(strategy, account_id)
    try:
        algorithm_basis = get_algorithm_class(strategy).from_config(config).config_fingerprint(config)
    except (KeyError, TypeError, ValueError):
        # An unresolvable strategy still needs a stable key. Reporting it is the compute
        # step's job -- failing here would turn a bad request id into an error from a hash.
        algorithm_basis = {"unresolved": strategy}
    cache_basis = {
        "strategy": strategy,
        "account": getattr(config, "account_id", ""),
        "data_feed": config.alpaca_data_feed,
        "portfolio": {
            "max_weight_per_symbol": config.max_weight_per_symbol,
            "max_portfolio_exposure": config.max_portfolio_exposure,
            "cash_buffer": config.cash_buffer,
            "transaction_cost_bps": config.transaction_cost_bps,
        },
        "algorithm": algorithm_basis,
    }
    encoded = json.dumps(cache_basis, sort_keys=True, default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


def _tune_editor(strategy: str) -> str | None:
    """Which editor the Tune screen should render for this algorithm's configuration.

    Asked of the algorithm class rather than listed here, so an algorithm whose configuration
    is not a list of scalars -- Bursty DCA's per-symbol budgets -- says so itself. The
    dashboard used to carry its own list of which algorithms those were, which meant adding one
    took an edit in the frontend that nothing in the backend could enforce.
    """
    try:
        return getattr(get_algorithm_class(strategy), "tune_editor", None)
    except (KeyError, TypeError, ValueError):
        # An unknown id still has to render a config page; it just gets the generic form.
        return None



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
        # Declared by the algorithm class: None means the generic parameter form.
        "tune_editor": _tune_editor(strategy),
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


def strategy_signals_payload(
    strategy: str = DEFAULT_STRATEGY_ID,
    account_id: str = "",
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dashboard signal payload: run the algorithm, then let it render its own plan.

    Deliberately the same ``run_algorithm`` call the scheduler makes, so the deck shows the
    plan that would actually trade. Nothing is placed and nothing is written -- state is
    committed on the execution path alone -- so opening this page cannot move an account's
    ledger or shift what the next scheduled run does.

    ``none`` used to fall back to the DCA plan's view, because DCA was not selectable in the
    deck and would otherwise have had nowhere to render. DCA is an ordinary algorithm now, so
    a saved ``none`` simply resolves to it.

    The plan is computed for the account this strategy is deployed on, not for the default
    account: a DCA plan is per account, so reading the default one showed the wrong plan.

    Caching mirrors the backtest: a finished snapshot is stored per (strategy, config,
    account) and served until ``refresh`` asks for a recompute, and ``cache_only`` probes
    without ever computing -- which is what lets the dashboard open on a cached view instead
    of recomputing every time it renders.
    """
    body = body or {}
    refresh = bool(body.get("refresh"))
    cache_only = bool(body.get("cache_only") or body.get("cacheOnly"))

    strategy = canonical_algorithm_id(strategy or DEFAULT_STRATEGY_ID)[:80]
    key = _signals_cache_key(strategy, account_id)
    cache = _load_signals_cache()

    if key in cache["items"] and not refresh:
        cached_payload = dict(cache["items"][key])
        cached_payload["cached"] = True
        return cached_payload

    # A probe reports the miss rather than paying for a compute; the dashboard treats this the
    # same way it treats "no cached backtest yet" -- a state the next explicit Refresh changes.
    if not refresh or cache_only:
        return {
            "strategy": strategy,
            "cached": False,
            "error": "No cached live signals are available.",
        }

    config = config_for_strategy_view(strategy, account_id)
    algorithm = get_algorithm_class(strategy).from_config(config)
    view = algorithm.signal_view(run_algorithm(strategy, config, algorithm=algorithm))
    payload = {
        "strategy": strategy,
        # Which account this view describes. Per-account config (a DCA plan) means the answer
        # is not the same for every deployment, so the view has to say which one it is.
        "account_id": getattr(config, "account_id", ""),
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "summary": view.summary,
        # Serialised from the dataclasses rather than assembled here, so the payload cannot
        # quietly grow a field the contract in ``core.interfaces`` does not describe.
        "rows": [asdict(row) for row in view.rows],
    }
    payload["cached"] = False
    cache["items"][key] = payload
    _save_signals_cache(cache)
    return payload
