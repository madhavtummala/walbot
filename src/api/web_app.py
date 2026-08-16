from __future__ import annotations

import argparse
import os

import uvicorn

from src.api.api_app import app
from src.api.api_payloads import (
    apply_universe_payload,
    backtest_payload,
    controls_payload,
    dca_payload,
    recommend_universe_payload,
    refresh_social_payload,
    save_controls_payload,
    save_dca_payload,
    social_payload,
    status_payload,
    universe_payload,
)

__all__ = [
    "app",
    "apply_universe_payload",
    "backtest_payload",
    "controls_payload",
    "dca_payload",
    "recommend_universe_payload",
    "refresh_social_payload",
    "save_controls_payload",
    "save_dca_payload",
    "social_payload",
    "status_payload",
    "universe_payload",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Walbot dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--mcp-port", default=int(os.getenv("MCP_PORT", "8002")), type=int)
    parser.add_argument("--mcp-transport", default=os.getenv("MCP_TRANSPORT", "sse"))
    parser.add_argument("--no-mcp-server", action="store_true",
                        help="Do not start the MCP tool server alongside the dashboard.")
    args = parser.parse_args()

    # The same single-process arrangement the container uses, because the alternative -- running
    # ``python -m src.mcp_server`` in a second terminal -- puts two processes on
    # data/walbot.duckdb and DuckDB allows exactly one. Development that differs from
    # deployment here reproduces a locking bug that deployment does not have.
    #
    # Not under --reload: uvicorn's reloader runs the app in a *child* process, so a thread
    # started here would serve MCP from the parent while the dashboard holds the database in
    # the child -- the two-process split again, with the port bound by the wrong one.
    if args.reload and not args.no_mcp_server:
        print("Reload mode: MCP tool server not started (it cannot share the reloader's child process).", flush=True)
    if not args.no_mcp_server and not args.reload:
        from src.mcp_server import serve_in_thread

        try:
            serve_in_thread(host=args.host, port=args.mcp_port, transport=args.mcp_transport)
        except Exception as error:  # noqa: BLE001 - the dashboard must still come up
            print(f"MCP tool server did not start: {error}", flush=True)

    uvicorn.run(
        "src.api.web_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "warning"),
    )


if __name__ == "__main__":
    main()
