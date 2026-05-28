from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Body, FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

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
    strategy_signals_payload,
    universe_payload,
)
from .bot_runtime import bot_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*")
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    bot_runtime.start()
    try:
        yield
    finally:
        bot_runtime.stop()


app = FastAPI(title="Trading Bot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", NoCacheStaticFiles(directory=WEB_ROOT / "static"), name="static")


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    response = FileResponse(WEB_ROOT / "index.html")
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/status")
def status() -> dict[str, Any]:
    return status_payload()


@app.get("/api/universe")
def universe() -> dict[str, Any]:
    return universe_payload()


@app.post("/api/universe/recommend")
def recommend_universe(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return recommend_universe_payload(body)


@app.post("/api/universe/apply")
def apply_universe(body: dict[str, Any]) -> dict[str, Any]:
    return apply_universe_payload(body)


@app.get("/api/dca")
def dca() -> dict[str, Any]:
    return dca_payload()


@app.post("/api/dca")
def save_dca(body: dict[str, Any]) -> dict[str, Any]:
    return save_dca_payload(body)


@app.get("/api/controls")
def controls() -> dict[str, Any]:
    return controls_payload()


@app.post("/api/controls")
def save_controls(body: dict[str, Any]) -> dict[str, Any]:
    return save_controls_payload(body)


@app.get("/api/social")
def social(limit: int = Query(default=250, ge=1, le=5000)) -> dict[str, Any]:
    return social_payload(limit=limit)


@app.get("/api/strategy-signals")
def strategy_signals(strategy: str = Query(default="momentum_social", max_length=80)) -> dict[str, Any]:
    return strategy_signals_payload(strategy=strategy)


@app.post("/api/refresh-social")
def refresh_social(body: dict[str, Any]) -> dict[str, Any]:
    return refresh_social_payload(body)


@app.post("/api/backtest")
def backtest(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return backtest_payload(body)
