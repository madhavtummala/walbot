from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Any

from src.core.config import get_config
from src.core.interfaces import MODE_TARGET, AlgorithmResult, Intent
from src.core.pipeline import (
    StaleResultError,
    UnknownBrokerageError,
    place_orders as pipeline_place_orders,
    read_snapshot,
    resolve_brokerage,
    run_algorithm,
)

logger = logging.getLogger(__name__)

DEFAULT_ALGORITHM = "fast_momentum"


def _result_payload(result: AlgorithmResult) -> dict[str, Any]:
    """Serialise a step-1 result for the agent, keeping only what step 2 and review need."""
    return {
        "strategy": result.strategy,
        "as_of": result.as_of.isoformat(),
        "mode": result.mode,
        "intents": [
            {"symbol": intent.symbol, "kind": intent.kind, "value": round(intent.value, 6)}
            for intent in result.intents
        ],
        "target_weights": {symbol: round(weight, 6) for symbol, weight in result.target_weights.items()},
        "latest_prices": {symbol: round(price, 4) for symbol, price in result.latest_prices.items()},
        "signals": {
            symbol: {
                "score": row.get("score"),
                "reason": row.get("reason"),
                "signal": row.get("signal"),
                "score_components": row.get("score_components"),
                "realized_volatility": row.get("realized_volatility"),
            }
            for symbol, row in result.signals.items()
        },
        "allocation_mode": result.metadata.get("allocation_mode"),
    }


def _result_from_payload(payload: dict[str, Any]) -> AlgorithmResult:
    """Rebuild a step-1 result from the payload the agent was given."""
    return AlgorithmResult(
        strategy=str(payload.get("strategy") or DEFAULT_ALGORITHM),
        intents=[
            Intent(symbol=str(row["symbol"]).upper(), kind=str(row.get("kind") or "weight"), value=float(row["value"]))
            for row in (payload.get("intents") or [])
        ],
        target_weights={str(k).upper(): float(v) for k, v in (payload.get("target_weights") or {}).items()},
        signals=payload.get("signals") or {},
        latest_prices={str(k).upper(): float(v) for k, v in (payload.get("latest_prices") or {}).items()},
        metadata={"allocation_mode": payload.get("allocation_mode")},
        mode=str(payload.get("mode") or MODE_TARGET),
        as_of=datetime.fromisoformat(payload["as_of"]) if payload.get("as_of") else datetime.now(timezone.utc),
    )


def _validate_target_weights(target_weights: Any) -> tuple[dict[str, float], str | None]:
    """Coerce caller-supplied weights to ``{SYMBOL: float}``, or return a reason they are unusable."""
    if not isinstance(target_weights, dict) or not target_weights:
        return {}, "target_weights must be a non-empty mapping of symbol to weight"

    cleaned: dict[str, float] = {}
    for symbol, weight in target_weights.items():
        key = str(symbol).strip().upper()
        if not key:
            return {}, "target_weights contains an empty symbol"
        try:
            value = float(weight)
        except (TypeError, ValueError):
            return {}, f"Weight for {key} is not a number: {weight!r}"
        if value != value or value in (float("inf"), float("-inf")):
            return {}, f"Weight for {key} is not finite"
        if value < 0:
            return {}, f"Weight for {key} is negative; short targets are not accepted here"
        cleaned[key] = value

    total = sum(cleaned.values())
    if total > 1.0 + 1e-6:
        return {}, f"Target weights sum to {total:.4f}, which exceeds 1.0 (100% of equity)"
    return cleaned, None


def _server(name: str, host: str, port: int):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised in container builds with mcp installed.
        raise RuntimeError("The MCP runtime is not installed. Install the 'mcp' package to use --mcp mode.") from exc

    try:
        return FastMCP(name, host=host, port=port)
    except TypeError:
        return FastMCP(name)


def create_mcp_server(host: str = "0.0.0.0", port: int = 8001):
    mcp = _server("walbot", host, port)

    @mcp.tool()
    def get_algorithm_result(algorithm: str = DEFAULT_ALGORITHM) -> dict[str, Any]:
        """Run the algorithm against market data and return the portfolio it proposes.

        Needs no brokerage. Returns target weights, the score and reason behind each symbol,
        and the prices the proposal was built from. Nothing is submitted. Pass the whole
        payload back to place_orders -- it carries the prices that step needs.
        """
        config = get_config(strategy_id=algorithm)
        if config.kill_switch:
            return {"strategy": algorithm, "status": "error", "reason": "Kill switch is enabled"}
        return {"status": "ok", **_result_payload(run_algorithm(algorithm, config))}

    @mcp.tool()
    def get_current_positions() -> dict[str, Any]:
        """Return the live holdings reported by the brokerage, with equity and cash."""
        config = get_config()
        try:
            brokerage = resolve_brokerage(config)
        except UnknownBrokerageError as exc:
            return {"status": "error", "reason": str(exc)}

        account_state = brokerage.get_account_state()
        snapshot = read_snapshot(config, brokerage)
        return {
            "status": "ok",
            "account_id": config.account_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "equity": snapshot.equity,
            "cash": float(account_state.get("cash", 0.0)),
            "buying_power": float(account_state.get("buying_power", 0.0)),
            "positions": [
                {"symbol": symbol, "shares": shares}
                for symbol, shares in sorted(snapshot.positions.items())
            ],
        }

    @mcp.tool()
    def place_orders(
        algorithm_result: dict[str, Any], target_weights: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """Submit orders for a reviewed portfolio, and return the resulting weight changes.

        ``algorithm_result`` is the payload get_algorithm_result returned -- pass it back
        unchanged, since it carries the prices used to size shares. Supply ``target_weights``
        only to override the proposal; it is then the complete intended portfolio, so any held
        symbol left out is sold to zero. The algorithm still applies its own stickiness and
        risk guards on top. Submits immediately.
        """
        result = _result_from_payload(algorithm_result)
        config = get_config(strategy_id=result.strategy)

        cleaned = None
        if target_weights is not None:
            cleaned, error = _validate_target_weights(target_weights)
            if error is not None:
                return {"strategy": result.strategy, "status": "error", "reason": error}

        if config.kill_switch:
            logger.warning("Kill switch is enabled. Skipping order placement for %s.", result.strategy)
            return {"strategy": result.strategy, "status": "skipped", "reason": "Kill switch is enabled"}

        try:
            brokerage = resolve_brokerage(config)
        except UnknownBrokerageError as exc:
            return {"strategy": result.strategy, "status": "error", "reason": str(exc)}

        try:
            return pipeline_place_orders(result, config, brokerage, target_weights=cleaned)
        except (StaleResultError, ValueError) as exc:
            return {"strategy": result.strategy, "status": "error", "reason": str(exc)}

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Walbot MCP tools.")
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", default=int(os.getenv("MCP_PORT", "8001")), type=int)
    parser.add_argument("--transport", default=os.getenv("MCP_TRANSPORT", "sse"))
    args = parser.parse_args()

    mcp = create_mcp_server(host=args.host, port=args.port)
    try:
        mcp.run(transport=args.transport, host=args.host, port=args.port)
    except TypeError:
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
