from __future__ import annotations

import argparse
import os

import uvicorn

from .api_app import app
from .api_payloads import (
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
    parser = argparse.ArgumentParser(description="Serve the Trading Bot dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8001, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "src.web_app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "warning"),
    )


if __name__ == "__main__":
    main()
