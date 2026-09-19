"""Per-algorithm tuning, signals and order history.

Split out of the single ``api_payloads`` module, which had grown to 1253 lines covering nine
unrelated domains. The public names are unchanged and still importable from ``api_payloads``.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from hashlib import sha256
from typing import Any

import pandas as pd

from .strategy_config import config_for_strategy_view
from ..controls import DEPLOYMENT_KEYS

from ...core.config import (
    DEFAULT_STRATEGY_ID,
    load_algorithms_config,
    config_transaction,
    save_algorithms_config,
)
from ...algorithms.explainers import explainer_for
from ...algorithms.registry import LEGACY_ALGORITHM_IDS, canonical_algorithm_id, get_algorithm_class
from ...core.runner import run_algorithm
from ...data.order_journal import clear_order_journal, load_order_journal
from ...data.state_store import load_state, save_state
from ...common.lazy_field import LazyField

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


#: A live signal view is one algorithm run against the market as it is now, so it ages with the
#: session rather than with the tuning -- the fingerprint in the cache key already invalidates on
#: a config change. Five minutes: long enough that clicking between tabs does not queue a run per
#: click, short enough that a view opened after lunch is not still this morning's.
SIGNALS_TTL_SECONDS = 300.0

#: The store is a read-modify-write of one row, and recomputes now land on background threads.
#: Without this, two algorithms finishing together would each write back the copy they read and
#: whichever landed second would silently drop the other's snapshot.
_SIGNALS_LOCK = threading.Lock()


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



def _tune_attr(strategy: str, name: str, default: Any) -> Any:
    """One Tune-screen declaration off the algorithm class, with a fallback for unknown ids."""
    try:
        return getattr(get_algorithm_class(strategy), name, default)
    except (KeyError, TypeError, ValueError):
        return default


def _with_seeded_board(strategy: str, values: dict[str, Any]) -> dict[str, Any]:
    """Fill in the board an algorithm would open with, when nothing is saved yet.

    The editor draws bubbles from ``config.plan``, so an algorithm that has never been tuned
    renders an empty canvas -- which reads as "this trades nothing" when in fact it trades on
    its global defaults. Asking the algorithm for its own opening board is the honest answer,
    and it is read-only: nothing is written until the reader saves.
    """
    if _tune_editor(strategy) != "budgets" or isinstance(values.get("plan"), dict) and values["plan"]:
        return values
    try:
        algorithm = get_algorithm_class(strategy).from_config(config_for_strategy_view(strategy, ""))
        board = algorithm.budget_plan(config_for_strategy_view(strategy, ""))
    except Exception as exc:  # noqa: BLE001 - a board that cannot be seeded is still editable
        logger.warning("Could not seed the %s board: %s", strategy, exc)
        return values
    return {**values, "plan": board}


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
    # The deployment keys share this section with the tuning but are not tuning: they are
    # where the algorithm trades and what drives it, edited from the deploy control and saved
    # through ``save_controls``. Left in, the generic parameter form would offer ``account_id``
    # as a knob and a save here would write one editor's view over the other's.
    values = {name: value for name, value in values.items() if name not in DEPLOYMENT_KEYS}
    values = _with_seeded_board(strategy, values)
    return {
        "strategy": strategy,
        # Surfaced so the dashboard can say which key on disk a value came from: several
        # algorithms are still filed under a retired id.
        "config_key": key,
        "config": values,
        # Declared by the algorithm class: None means the generic parameter form.
        "tune_editor": _tune_editor(strategy),
        # Which buckets that editor splits symbols across, and what an amount means. Asked of
        # the algorithm rather than hardcoded in the dashboard, which is what limited the board
        # to algorithms whose buckets happened to be named buy and sell.
        "tune_buckets": [str(b) for b in _tune_attr(strategy, "tune_buckets", ("buy", "sell"))],
        "tune_budget_hint": str(_tune_attr(strategy, "tune_budget_hint", "") or ""),
        # How the board reads and edits a bubble's number. One component, two units.
        "tune_unit": str(_tune_attr(strategy, "tune_unit", "currency")),
        "tune_max_amount": float(_tune_attr(strategy, "tune_max_amount", 2000.0)),
        "tune_step": float(_tune_attr(strategy, "tune_step", 25.0)),
        "explainer": explainer_for(strategy),
    }


def save_algorithm_config_payload(strategy: str, values: Any) -> dict[str, Any]:
    strategy = canonical_algorithm_id(strategy)[:80]
    # A list or scalar here means the caller sent the wrong shape. Coercing it to {} would
    # quietly erase every tuned value for this algorithm, so refuse instead.
    if not isinstance(values, dict):
        raise ValueError("Algorithm config must be a JSON object.")
    with config_transaction():
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
        # Carried across rather than taken from ``values``: the deployment keys live in this
        # same section but belong to the deploy control, and this payload never carries them.
        # Assigning the section wholesale would undeploy the algorithm as a side effect of
        # saving its tuning.
        existing = sections.get(key) if isinstance(sections.get(key), dict) else {}
        preserved = {name: existing[name] for name in DEPLOYMENT_KEYS if name in existing}
        sections[key] = {**values, **preserved}
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


def clear_algorithm_activity_payload(strategy: str = "") -> dict[str, Any]:
    """Drop this algorithm's journal entries and return the now-empty view.

    Display-only: the broker's own order history (the account page's "Recent orders") is
    untouched, since that is read from the broker directly rather than from this journal.
    """
    strategy_id = canonical_algorithm_id(strategy) if strategy else ""
    cleared = clear_order_journal(strategy=strategy_id)
    return {"strategy": strategy_id, "rows": [], "cleared": cleared}


def _compute_signals(key: str) -> dict[str, Any]:
    """Run the algorithm and store the view. The slow half of the signal tab, off the request.

    Deliberately the same ``run_algorithm`` call the scheduler makes, so the deck shows the plan
    that would actually trade. Nothing is placed and nothing is written -- state is committed on
    the execution path alone -- so running this on a background thread beside a scheduled run
    cannot move an account's ledger or shift what the next scheduled run does.
    """
    strategy, _, account_id = key.partition("|")
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
    with _SIGNALS_LOCK:
        cache = _load_signals_cache()
        cache["items"][_signals_cache_key(strategy, account_id)] = payload
        _save_signals_cache(cache)
    return payload


#: Keyed on the request's identity -- strategy and account -- rather than on the fingerprinted
#: cache key, because this key has to be readable back into the arguments the compute needs.
#: Config changes are still caught: the *stored* key carries the fingerprint, so a snapshot
#: computed under one tuning is never served for another.
SIGNALS = LazyField("live signals", _compute_signals, ttl_seconds=SIGNALS_TTL_SECONDS)


def _stored_signals(strategy: str, account_id: str) -> dict[str, Any] | None:
    with _SIGNALS_LOCK:
        return _load_signals_cache()["items"].get(_signals_cache_key(strategy, account_id))


def strategy_signals_payload(
    strategy: str = DEFAULT_STRATEGY_ID,
    account_id: str = "",
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The signal view for one algorithm: the last one computed, and a newer one on its way.

    Visiting is what refreshes it. A read answers instantly with the stored snapshot and starts
    a recompute behind it when that snapshot is older than :data:`SIGNALS_TTL_SECONDS`, so the
    tab is never blank while a run is in flight and nobody has to press anything. This replaces
    a ``cache_only`` probe that could only ever report a miss: the tab opened on a cached view
    and then said "hit Refresh to compute them", which made a button the only way to get a
    number the page could perfectly well have fetched itself.

    ``refresh`` is an explicit reload. The stored snapshot is dropped so the page has nothing to
    paint and draws its skeleton, and the run happens in the background -- forcing it
    synchronously would hold the request open for the length of an algorithm run.
    """
    body = body or {}
    refresh = bool(body.get("refresh"))
    strategy = canonical_algorithm_id(strategy or DEFAULT_STRATEGY_ID)[:80]
    key = f"{strategy}|{account_id}"

    if refresh:
        with _SIGNALS_LOCK:
            cache = _load_signals_cache()
            if cache["items"].pop(_signals_cache_key(strategy, account_id), None) is not None:
                _save_signals_cache(cache)
        SIGNALS.recompute(key)
        return {"strategy": strategy, "cached": False, "state": "computing",
                "refreshing": True, "error": ""}

    snapshot = SIGNALS.get(key)
    # The in-memory copy when this process has computed one, the stored copy otherwise -- which
    # is what carries a snapshot across a restart, and is why replacing the container does not
    # leave every signal tab blank.
    stored = snapshot["value"] or _stored_signals(strategy, account_id)
    if stored:
        return {**stored, "cached": True, "state": "ready",
                "refreshing": snapshot["refreshing"], "error": snapshot["error"]}
    return {"strategy": strategy, "cached": False, "state": snapshot["state"],
            "refreshing": snapshot["refreshing"], "error": snapshot["error"]}
