from __future__ import annotations

import os
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.api.api_payloads import (
    account_activity_payload,
    accounts_payload,
    algorithm_activity_payload,
    algorithm_config_payload,
    apply_universe_payload,
    backtest_payload,
    complete_schwab_auth_payload,
    controls_payload,
    recommend_universe_payload,
    refresh_social_payload,
    save_controls_payload,
    delete_account_payload,
    save_account_payload,
    save_algorithm_config_payload,
    schwab_auth_payload,
    positions_payload,
    social_payload,
    start_schwab_auth_payload,
    status_payload,
    strategy_signals_payload,
    universe_payload,
)
from ..brokerages.schwab_client import SchwabAuthError
from ..core.bot_runtime import bot_runtime
from ..core.config import DEFAULT_STRATEGY_ID
from ..common.logging_utils import configure_logging, demote_uvicorn_access_logs_to_debug


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = PROJECT_ROOT / "web"
demote_uvicorn_access_logs_to_debug()


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
    configure_logging()
    demote_uvicorn_access_logs_to_debug()
    # Always started. A binding parked on "mcp" is simply never scheduled, so the loop costs
    # nothing when nothing is switched on.
    bot_runtime.start()
    try:
        yield
    finally:
        bot_runtime.stop()


app = FastAPI(title="Walbot API", lifespan=lifespan)

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
def strategy_signals(
    strategy: str = Query(default=DEFAULT_STRATEGY_ID, max_length=80),
    account_id: str = Query(default="", max_length=80),
) -> dict[str, Any]:
    # ``account_id`` is optional: omitted, the strategy's binding decides. The dashboard sends
    # it so the view it renders is the one whose plan its own editor is writing.
    return strategy_signals_payload(strategy=strategy, account_id=account_id)


@app.post("/api/refresh-social")
def refresh_social(body: dict[str, Any]) -> dict[str, Any]:
    return refresh_social_payload(body)


@app.post("/api/backtest")
def backtest(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    return backtest_payload(body)


@app.get("/api/accounts")
def accounts() -> dict[str, Any]:
    return accounts_payload()


@app.post("/api/accounts")
def save_account(body: dict[str, Any]) -> dict[str, Any]:
    try:
        return save_account_payload(body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/accounts/{account_id}")
def delete_account(account_id: str) -> dict[str, Any]:
    try:
        return delete_account_payload(account_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/algorithm-config")
def algorithm_config(strategy: str = Query(default=DEFAULT_STRATEGY_ID, max_length=80)) -> dict[str, Any]:
    return algorithm_config_payload(strategy)


@app.post("/api/algorithm-config")
def save_algorithm_config(body: dict[str, Any]) -> dict[str, Any]:
    try:
        # Passing a non-object through as {} would silently wipe the saved tuning, so the
        # bad shape has to reach save_algorithm_config_payload and be rejected there.
        return save_algorithm_config_payload(
            str(body.get("strategy") or DEFAULT_STRATEGY_ID),
            body.get("config"),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/activity")
def activity(
    account_id: str = Query(default="", max_length=80),
    limit: int = Query(default=40, ge=1, le=200),
) -> dict[str, Any]:
    return account_activity_payload(account_id=account_id, limit=limit)


@app.get("/api/algorithm-activity")
def algorithm_activity(
    strategy: str = Query(default=DEFAULT_STRATEGY_ID, max_length=80),
    limit: int = Query(default=40, ge=1, le=200),
) -> dict[str, Any]:
    return algorithm_activity_payload(strategy=strategy, limit=limit)


@app.get("/api/positions")
def positions(account_id: str = Query(default="", max_length=80)) -> dict[str, Any]:
    return positions_payload(account_id)


@app.get("/api/schwab/auth")
def schwab_auth() -> dict[str, Any]:
    return schwab_auth_payload()


@app.post("/api/schwab/auth/start")
def start_schwab_auth() -> dict[str, Any]:
    try:
        return start_schwab_auth_payload()
    except SchwabAuthError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/schwab/callback", include_in_schema=False)
def schwab_callback(
    code: str = Query(default=""),
    state: str = Query(default=""),
    error: str = Query(default=""),
) -> HTMLResponse:
    """Land the Schwab consent redirect and trade the code for a refresh token.

    Schwab drives the browser here, so the reply has to be a page rather than JSON. It hands
    the outcome back to the dashboard tab that opened it and then gets out of the way.
    """
    if error:
        return _callback_page(False, f"Schwab denied the request: {error}")
    try:
        status = complete_schwab_auth_payload(code=code, state=state)
    except SchwabAuthError as auth_error:
        return _callback_page(False, str(auth_error))
    except Exception as unexpected:  # noqa: BLE001 - surface the cause on the page, not a 500
        return _callback_page(False, f"Schwab token exchange failed: {unexpected}")
    return _callback_page(True, status.get("detail") or "Schwab connected.")


def _callback_page(ok: bool, message: str) -> HTMLResponse:
    body = f"""<!doctype html>
<meta charset="utf-8" />
<title>Schwab authorization</title>
<style>
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center;
         font-family: Inter, system-ui, -apple-system, sans-serif; background: #eef1fb; color: #111827; }}
  .card {{ padding: 28px 34px; border-radius: 18px; background: #fff; text-align: center;
           box-shadow: 0 24px 70px rgba(17, 24, 39, 0.12); max-width: 420px; }}
  .mark {{ font-size: 34px; }}
  p {{ margin: 10px 0 0; color: #657084; font-size: 14px; line-height: 1.5; }}
</style>
<div class="card">
  <div class="mark">{"&#10003;" if ok else "&#10007;"}</div>
  <h2>{"Schwab connected" if ok else "Authorization failed"}</h2>
  <p>{escape(message)}</p>
  <p>You can close this tab.</p>
</div>
<script>
  try {{ window.opener && window.opener.postMessage({{ type: "schwab-auth", ok: {str(ok).lower()} }}, "*"); }} catch (e) {{}}
  {"setTimeout(function () { window.close(); }, 1200);" if ok else ""}
</script>
"""
    return HTMLResponse(body, status_code=200 if ok else 400)
