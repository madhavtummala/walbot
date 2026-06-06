from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import uvicorn


def _start_mcp_server(host: str, port: int, transport: str) -> subprocess.Popen:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "src.mcp_server",
            "--host",
            host,
            "--port",
            str(port),
            "--transport",
            transport,
        ]
    )
    time.sleep(0.5)
    if process.poll() is not None:
        raise RuntimeError(f"MCP server exited during startup with code {process.returncode}")
    return process


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Walbot container runtime.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bot", action="store_true", help="Start the dashboard and built-in bot scheduler.")
    mode.add_argument("--mcp", action="store_true", help="Start the dashboard in tool mode plus the MCP tool server.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", default=int(os.getenv("PORT", "8000")), type=int)
    parser.add_argument("--mcp-host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--mcp-port", default=int(os.getenv("MCP_PORT", "8001")), type=int)
    parser.add_argument("--mcp-transport", default=os.getenv("MCP_TRANSPORT", "sse"))
    args = parser.parse_args()

    runtime_mode = "mcp" if args.mcp else "bot"
    os.environ["TRADING_RUNTIME_MODE"] = runtime_mode

    mcp_process: subprocess.Popen | None = None
    try:
        if runtime_mode == "mcp":
            mcp_process = _start_mcp_server(args.mcp_host, args.mcp_port, args.mcp_transport)
        uvicorn.run(
            "src.api.api_app:app",
            host=args.host,
            port=args.port,
            log_level=os.getenv("UVICORN_LOG_LEVEL", "warning"),
        )
    finally:
        if mcp_process and mcp_process.poll() is None:
            mcp_process.terminate()
            try:
                mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mcp_process.kill()


if __name__ == "__main__":
    main()
