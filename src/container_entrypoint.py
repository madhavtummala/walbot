from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import uvicorn
from src.mcp_server import serve_in_thread

#: Where the image keeps its shipped config, deliberately outside /config. A bind-mounted
#: /config shadows anything baked in at that path, so files copied there by the Dockerfile
#: are invisible at runtime and a fresh volume starts completely empty.
CONFIG_DEFAULTS_DIR = Path(os.getenv("CONFIG_DEFAULTS_DIR", "/app/config-defaults"))

CONFIG_FILENAME = "walbot.yaml"

#: What the image actually ships. The live config is gitignored -- it names real accounts -- so
#: the only file baked into the image is the template, and a fresh volume is seeded from that.
CONFIG_TEMPLATE_FILENAME = "walbot.yaml.sample"


def prepare_config() -> str:
    """Make sure the mounted volume has a config file, and say what was done.

    Three cases, in order: an existing unified file is left alone; a volume still holding the
    seven pre-unification files is migrated into one, preserving tuning and DCA accrual; an
    empty volume is seeded from the image defaults -- which since the live config stopped being
    committed means the template, leaving a first-run container to be configured rather than
    started on somebody else's accounts.
    """
    from src.core.config import config_file_path, migrate_legacy_config

    target = config_file_path()
    if target.exists():
        return ""

    migrated = migrate_legacy_config()
    if migrated is not None:
        return f"Merged the previous per-section config files into {migrated}"

    # A real config first, for an image built with one present, then the shipped template.
    for name in (CONFIG_FILENAME, CONFIG_TEMPLATE_FILENAME):
        source = CONFIG_DEFAULTS_DIR / name
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return f"Seeded {target} from {source.name}"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Walbot container runtime.")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", default=int(os.getenv("PORT", "8000")), type=int)
    parser.add_argument("--mcp-host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--mcp-port", default=int(os.getenv("MCP_PORT", "8001")), type=int)
    parser.add_argument("--mcp-transport", default=os.getenv("MCP_TRANSPORT", "sse"))
    args = parser.parse_args()

    note = prepare_config()
    if note:
        print(note, flush=True)

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
