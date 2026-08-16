"""Dashboard payload builders.

A facade. The implementations live in ``src/api/payloads/`` -- one module per domain -- after
this file reached 1253 lines spanning accounts, universe, DCA, algorithms, watchlist, controls,
social, backtests and process status. Every name it used to export is still importable from
here, resolved on attribute access so importing one domain does not drag in the other eight.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

#: Public name -> the ``payloads`` module that defines it.
_EXPORTS = {
    "ACCOUNT_FIELDS": "accounts",
    "BACKTEST_CACHE_PATH": "backtest",
    "BACKTEST_CACHE_STATE_KEY": "backtest",
    "BACKTEST_CACHE_VERSION": "backtest",
    "BACKTEST_STARTING_EQUITY": "backtest",
    "DEFAULT_WATCHLIST": "watchlist",
    "DIVIDEND_ACTIVITY_DAYS": "accounts",
    "PROJECT_ROOT": "system",
    "WATCHLIST_STATE_KEY": "watchlist",
    "_account_items": "accounts",
    "_backtest_order_summary": "backtest",
    "_backtest_response": "backtest",
    "_backtest_starting_equity": "backtest",
    "_brokerage_positions": "accounts",
    "_cache_key": "backtest",
    "_compute_backtest": "backtest",
    "_configured_history_providers": "backtest",
    "_default_backtest_period": "backtest",
    "_display_path": "system",
    "_dividend_pl": "accounts",
    "_enum_value": "accounts",
    "_fetch_backtest_history": "backtest",
    "_file_info": "system",
    "_json_backtest_rows": "backtest",
    "_load_backtest_cache": "backtest",
    "_paper_activity": "accounts",
    "_paper_positions": "accounts",
    "_period_label": "backtest",
    "_period_months": "backtest",
    "_period_row_count": "backtest",
    "_period_start": "backtest",
    "_redact": "system",
    "_save_backtest_cache": "backtest",
    "_watchlist_symbols": "watchlist",
    "account_activity_payload": "accounts",
    "accounts_payload": "accounts",
    "algorithm_activity_payload": "algorithms",
    "algorithm_config_payload": "algorithms",
    "apply_universe_payload": "universe",
    "backtest_payload": "backtest",
    "complete_schwab_auth_payload": "controls",
    "controls_payload": "controls",
    "dca_payload": "dca",
    "delete_account_payload": "accounts",
    "positions_payload": "accounts",
    "recommend_universe_payload": "universe",
    "refresh_social_payload": "social",
    "save_account_payload": "accounts",
    "save_algorithm_config_payload": "algorithms",
    "save_controls_payload": "controls",
    "save_dca_payload": "dca",
    "save_watchlist_payload": "watchlist",
    "schwab_auth_payload": "controls",
    "social_payload": "social",
    "start_schwab_auth_payload": "controls",
    "status_payload": "system",
    "strategy_signals_payload": "algorithms",
    "universe_payload": "universe",
    "watchlist_payload": "watchlist",
}


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"src.api.payloads.{module}"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
