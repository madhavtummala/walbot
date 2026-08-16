from __future__ import annotations

import argparse
import os
import shutil
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Walbot container runtime.")
    # No --bot/--mcp modes. Whether an algorithm is driven by the scheduler or by an agent is
    # a property of its deployment -- the binding's frequency, "15m" through "1d" or "mcp" --
    # so a process-wide mode could only contradict it. Both the scheduler and the MCP server
    # always run; the bindings decide what either of them actually does.
    parser.add_argument("--bot", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mcp", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-mcp-server", action="store_true",
                        help="Do not start the MCP tool server alongside the dashboard.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", default=int(os.getenv("PORT", "8000")), type=int)
    parser.add_argument("--mcp-host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--mcp-port", default=int(os.getenv("MCP_PORT", "8001")), type=int)
    parser.add_argument("--mcp-transport", default=os.getenv("MCP_TRANSPORT", "sse"))
    args = parser.parse_args()

    note = prepare_config()
    if note:
        print(note, flush=True)

    if not args.no_mcp_server:
        # In this process, not beside it: two processes on one DuckDB file is what produced
        # "Conflicting lock is held" whenever a tool call and the dashboard overlapped. See
        # ``serve_in_thread``. The daemon thread needs no shutdown handling of its own.
        from src.mcp_server import serve_in_thread

        try:
            serve_in_thread(host=args.mcp_host, port=args.mcp_port, transport=args.mcp_transport)
        except Exception as error:  # noqa: BLE001 - the dashboard must still come up
            print(f"MCP tool server did not start: {error}", flush=True)

    uvicorn.run(
        "src.api.api_app:app",
        host=args.host,
        port=args.port,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "warning"),
    )


if __name__ == "__main__":
    main()
