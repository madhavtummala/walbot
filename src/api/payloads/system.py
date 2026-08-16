"""Process, config-file and data-freshness status.

Split out of the single ``api_payloads`` module, which had grown to 1253 lines covering nine
unrelated domains. The public names are unchanged and still importable from ``api_payloads``.
"""


from __future__ import annotations

from src.api.controls import load_controls
from ...data.universe import resolve_project_path

import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd


from ...core.bot_runtime import bot_runtime
from ...core.config import (
    DEFAULT_STRATEGY_ID,
    get_config,
)

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _redact(value: str) -> str:
    config = get_config()
    secrets = [
        config.alpaca_api_key,
        config.alpaca_api_secret,
        config.alpha_vantage_api_key,
    ]
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[redacted]")
    redacted = re.sub(r"(apikey=)[^&\s]+", r"\1[redacted]", redacted, flags=re.IGNORECASE)
    return redacted


def _file_info(path: str) -> dict[str, Any]:
    if not path:
        return {"path": path, "exists": False}
    resolved = resolve_project_path(path)
    if not resolved.exists():
        return {"path": path, "exists": False}
    return {
        "path": path,
        "exists": True,
        "updated_at": pd.Timestamp.fromtimestamp(resolved.stat().st_mtime, tz="UTC").isoformat(),
        "size_bytes": resolved.stat().st_size,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def status_payload() -> dict[str, Any]:
    controls = load_controls()
    strategy = str(controls.get("active_strategy") or DEFAULT_STRATEGY_ID)
    config = get_config(strategy_id=strategy)
    social_info = _file_info(config.social_trends_csv)
    social_rows = 0
    social_symbols: set[str] = set()
    latest_social_timestamp = None

    if social_info["exists"]:
        try:
            df = pd.read_csv(resolve_project_path(config.social_trends_csv))
            if not df.empty:
                social_rows = len(df)
                if "symbol" in df:
                    social_symbols = set(df["symbol"].dropna().astype(str).str.upper())
                if "timestamp" in df:
                    timestamps = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
                    if not timestamps.empty:
                        latest_social_timestamp = timestamps.max().isoformat()
        except Exception as exc:  # pragma: no cover - status should stay available.
            social_info["error"] = _redact(str(exc))

    config_summary = asdict(config)
    for secret_field in list(config_summary):
        if secret_field.endswith("_api_key") or secret_field.endswith("_api_secret"):
            config_summary[secret_field] = bool(config_summary.get(secret_field))

    return {
        "mode": "READ_ONLY" if config.kill_switch else "ORDER_ENABLED",
        "kill_switch": config.kill_switch,
        "trading_endpoint": config.alpaca_base_url,
        "trading_account": {"id": config.account_id, "label": config.account_label},
        "bot": bot_runtime.snapshot(),
        "config": config_summary,
        "universe": {
            "count": len(config.symbols),
            "symbols": config.symbols,
            "master_list": config.tradables_csv,
        },
        "social": {
            **social_info,
            "rows": social_rows,
            "symbol_count": len(social_symbols),
            "latest_timestamp": latest_social_timestamp,
        },
        "risk": {
            "max_weight_per_symbol": config.max_weight_per_symbol,
            "max_portfolio_exposure": config.max_portfolio_exposure,
            "max_longs": config.max_longs,
            "target_annual_vol": config.target_annual_vol,
            "cash_buffer": config.cash_buffer,
        },
    }
