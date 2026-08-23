"""Process-wide access to the Schwab stream.

The Streamer API allows one websocket per user, so the client is a singleton here rather
than a per-caller object: the first caller to need real-time data starts it, everyone after
that shares its :class:`~src.connectors.market.schwab_stream.StreamStore`. Nothing in this
module is on any critical path -- when streaming is not configured, or Schwab refuses the
session, every consumer falls back to the REST path it already had.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .market.schwab_stream import SchwabStreamClient, StreamStore

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_client: SchwabStreamClient | None = None


def _streaming_enabled(config) -> bool:
    providers = [str(item).lower() for item in getattr(config, "streaming_market_data_provider_order", [])]
    return "schwab" in providers


def ensure_stream(config, symbols: list[str] | None = None) -> SchwabStreamClient | None:
    """Start (or extend) the shared stream if config enables it; ``None`` means "not streaming".

    Safe to call on every price fetch: an existing client just absorbs new symbols as ADD
    subscriptions. A client whose constructor fails (no Schwab credentials, for instance)
    is remembered as absent so the failure logs once, not per call.
    """
    global _client
    if not _streaming_enabled(config):
        return None
    with _lock:
        if _client is None:
            try:
                _client = SchwabStreamClient(config, symbols)
            except Exception as exc:  # noqa: BLE001 - streaming must never break pricing
                logger.warning("Schwab stream unavailable; continuing without it: %s", exc)
                return None
            _client.start()
            logger.info("Schwab stream started for %s symbol(s)", len(_client.status()["symbols"]))
    if symbols:
        _client.add_symbols(symbols)
    return _client


def stream_store() -> StreamStore | None:
    """The shared snapshot store, or ``None`` when streaming never started."""
    return _client.store if _client else None


def stream_status() -> dict[str, Any]:
    """Connection state for dashboards and diagnostics; ``{"enabled": False}`` when off."""
    if _client is None:
        return {"enabled": False}
    status = {"enabled": True, **_client.status()}
    status["connected"] = bool(status.get("connected"))
    return status


def reset_stream() -> None:
    """Forget the singleton. Tests only."""
    global _client
    with _lock:
        if _client is not None:
            _client.stop()
        _client = None
