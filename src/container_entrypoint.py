from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import uvicorn

#: Where the image keeps its shipped config, deliberately outside /config. A bind-mounted
#: /config shadows anything baked in at that path, so files copied there by the Dockerfile
#: are invisible at runtime and a fresh volume starts completely empty.
CONFIG_DEFAULTS_DIR = Path(os.getenv("CONFIG_DEFAULTS_DIR", "/app/config-defaults"))

CONFIG_FILENAME = "walbot.yaml"


def prepare_config() -> str:
    """Make sure the mounted volume has a config file, and say what was done.

    Three cases, in order: an existing unified file is left alone; a volume still holding the
    seven pre-unification files is migrated into one, preserving tuning and DCA accrual; an
    empty volume is seeded from the image defaults.
    """
    from src.core.config import config_file_path, migrate_legacy_config

    target = config_file_path()
    if target.exists():
        return ""

    migrated = migrate_legacy_config()
    if migrated is not None:
        return f"Merged the previous per-section config files into {migrated}"

    source = CONFIG_DEFAULTS_DIR / CONFIG_FILENAME
    if not source.is_file():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return f"Seeded {target} from the image defaults"


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

    note = prepare_config()
    if note:
        print(note, flush=True)

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
