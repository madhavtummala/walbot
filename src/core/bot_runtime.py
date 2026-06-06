from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from math import floor
from typing import Any
from zoneinfo import ZoneInfo

from ..brokerages.alpaca_client import create_data_client, create_trading_client, get_latest_price, is_market_open, submit_market_order
from src.core.config import get_config
from ..api.controls import load_controls
from ..algorithms.dca import allocation_preview, load_dca_plan
from ..execution.live_runner import run_once
from ..common.logging_utils import log_position_changes
from ..notifications.service import request_trade_approval

logger = logging.getLogger(__name__)
MARKET_TZ = ZoneInfo("America/Chicago")
MARKET_OPEN_MINUTE = (8 * 60) + 30
MARKET_CLOSE_MINUTE = 15 * 60

@dataclass
class RuntimeState:
    enabled: bool = False
    running: bool = False
    account_id: str = ""
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_error: str = ""
    last_run_date: str = ""
    last_run_key: str = ""


class _RuntimeLoop:
    def __init__(self, name: str, enabled_fn, run_fn, interval_fn, run_key_fn=None, account_id_fn=None) -> None:
        self.name = name
        self._enabled_fn = enabled_fn
        self._run_fn = run_fn
        self._interval_fn = interval_fn
        self._run_key_fn = run_key_fn
        self._account_id_fn = account_id_fn or (lambda controls: str(controls.get("trading_account_id") or ""))
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = RuntimeState()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name=f"dashboard-{self.name}-runtime", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        self.start()
        with self._lock:
            self._state.last_run_date = ""
        self._wake.set()

    def snapshot(self) -> dict[str, Any]:
        controls = load_controls()
        with self._lock:
            state = self._state.__dict__.copy()
        state["enabled"] = self._enabled_fn(controls)
        state["account_id"] = str(self._account_id_fn(controls) or state.get("account_id") or "")
        return state

    def _loop(self) -> None:
        while not self._stop.is_set():
            controls = load_controls()
            account_id = str(self._account_id_fn(controls) or "")
            enabled = self._enabled_fn(controls)
            with self._lock:
                self._state.enabled = enabled
                self._state.account_id = account_id
            if enabled:
                run_key = self._next_run_key()
                if run_key is not None:
                    self._run_guarded(account_id, run_key)
            self._wake.wait(self._check_seconds())
            self._wake.clear()

    def _check_seconds(self) -> int:
        try:
            return max(int(self._interval_fn()), 15)
        except Exception:
            return 300

    def _next_run_key(self) -> str | None:
        if self._run_key_fn is None:
            return ""
        run_key = self._run_key_fn()
        if not run_key:
            return None
        with self._lock:
            if self._state.last_run_key == run_key:
                return None
        return str(run_key)

    def _run_guarded(self, account_id: str, run_key: str) -> None:
        with self._lock:
            if self._state.running:
                return
            self._state.running = True
            self._state.last_started_at = datetime.now(timezone.utc).isoformat()
            self._state.last_error = ""
            if run_key:
                self._state.last_run_key = run_key
        try:
            self._run_fn(account_id or None)
            with self._lock:
                self._state.last_run_date = date.today().isoformat()
        except Exception as exc:  # pragma: no cover - surfaced via status payload.
            logger.exception("%s runtime failed", self.name)
            with self._lock:
                self._state.last_error = str(exc)
        finally:
            with self._lock:
                self._state.running = False
                self._state.last_finished_at = datetime.now(timezone.utc).isoformat()


def _algorithm_enabled(controls: dict[str, Any]) -> bool:
    return bool(controls.get("algorithm_enabled")) and str(controls.get("active_strategy") or "none") != "none"


def _options_enabled(controls: dict[str, Any]) -> bool:
    config = get_config()
    return (
        bool(controls.get("options_trading_enabled"))
        and str(controls.get("options_strategy") or "none") != "none"
        and not config.kill_switch
    )


def _dca_enabled(_controls: dict[str, Any]) -> bool:
    config = get_config()
    from src.api.api_payloads import universe_payload

    plan = load_dca_plan(universe_payload()["rows"])
    return bool(plan.get("enabled")) and not config.kill_switch


def _run_algorithm(account_id: str | None) -> None:
    run_once(account_id=account_id)


def _run_options(account_id: str | None) -> None:
    from .algorithms.options.swing import run_options_once

    run_options_once(account_id=account_id)


def _run_dca(account_id: str | None) -> None:
    from src.api.api_payloads import universe_payload

    config = get_config(account_id=account_id)
    if config.kill_switch:
        logger.warning("Kill switch is enabled. Exiting DCA runtime without sending orders.")
        return

    plan = load_dca_plan(universe_payload()["rows"])
    preview = allocation_preview(plan)
    if not preview:
        log_position_changes([])
        return

    trading_client = create_trading_client(config)
    if not is_market_open(trading_client):
        logger.warning("Market is closed according to Alpaca clock. Exiting DCA runtime without sending orders.")
        return

    data_client = create_data_client(config)
    planned_orders: list[dict[str, Any]] = []
    for row in preview:
        symbol = str(row["symbol"])
        side = str(row["action"])
        price = get_latest_price(symbol, data_client, data_feed=config.alpaca_data_feed)
        quantity = floor(float(row["notional"]) / price) if price > 0 else 0
        if quantity <= 0:
            logger.info("Skipping DCA %s for %s because notional is below one share", side, symbol)
            continue
        planned_orders.append(
            {
                "symbol": symbol,
                "action": side,
                "quantity": quantity,
                "notional": float(row["notional"]),
                "trade_dollars": float(row["notional"]),
            }
        )

    if not planned_orders:
        log_position_changes([])
        return

    if config.require_trade_approval:
        approval_id = f"dca-{datetime.now(timezone.utc).strftime('%H%M%S')}"
        approved = request_trade_approval(
            planned_orders,
            approval_id=approval_id,
            timeout_seconds=config.trade_approval_timeout_seconds,
            poll_seconds=config.trade_approval_poll_seconds,
        )
        if not approved:
            logger.warning("DCA approval %s was denied or timed out; skipping %s planned order(s)", approval_id, len(planned_orders))
            log_position_changes([
                {**order, "action": "skip", "quantity": 0, "approval_id": approval_id, "approval_status": "not_approved"}
                for order in planned_orders
            ])
            return

    order_results: list[dict[str, Any]] = []
    for planned_order in planned_orders:
        symbol = str(planned_order["symbol"])
        side = str(planned_order["action"])
        quantity = int(planned_order["quantity"])
        logger.info("Submitting DCA %s order for %s qty=%s", side, symbol, quantity)
        order = submit_market_order(trading_client, symbol, side, quantity)
        order_results.append({**planned_order, "order_id": getattr(order, "id", "unknown")})
    log_position_changes(order_results)


def _cron_field_matches(field: str, value: int, *, min_value: int, max_value: int) -> bool:
    field = str(field or "*").strip()
    if field == "*":
        return True
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, raw_step = part.split("/", 1)
            try:
                step = max(int(raw_step), 1)
            except ValueError:
                return False
        if part == "*":
            start, end = min_value, max_value
        elif "-" in part:
            raw_start, raw_end = part.split("-", 1)
            try:
                start, end = int(raw_start), int(raw_end)
            except ValueError:
                return False
        else:
            try:
                start = end = int(part)
            except ValueError:
                return False
        if start <= value <= end and (value - start) % step == 0:
            return True
    return False


def _cron_matches(pattern: str, now: datetime) -> bool:
    fields = str(pattern or "").split()
    if len(fields) != 5:
        return False
    minute, hour, day_of_month, month, day_of_week = fields
    cron_dow = 0 if now.weekday() == 6 else now.weekday() + 1
    return (
        _cron_field_matches(minute, now.minute, min_value=0, max_value=59)
        and _cron_field_matches(hour, now.hour, min_value=0, max_value=23)
        and _cron_field_matches(day_of_month, now.day, min_value=1, max_value=31)
        and _cron_field_matches(month, now.month, min_value=1, max_value=12)
        and (
            _cron_field_matches(day_of_week, cron_dow, min_value=0, max_value=7)
            or (cron_dow == 0 and _cron_field_matches(day_of_week, 7, min_value=0, max_value=7))
        )
    )


def _dca_run_key() -> str | None:
    from src.api.api_payloads import universe_payload

    plan = load_dca_plan(universe_payload()["rows"])
    now = datetime.now(MARKET_TZ).replace(second=0, microsecond=0)
    if not _cron_matches(str(plan.get("schedule_pattern") or ""), now):
        return None
    return f"dca:{now.isoformat(timespec='minutes')}"


def _parse_hhmm(value: str, default_minute: int) -> int:
    try:
        hour, minute = str(value or "").strip().split(":", 1)
        parsed = (int(hour) * 60) + int(minute)
    except (TypeError, ValueError):
        return default_minute
    return parsed if 0 <= parsed <= (23 * 60) + 59 else default_minute


def _is_regular_market_hours(now: datetime, start_time: str = "08:30", end_time: str = "15:00") -> bool:
    local_now = now.astimezone(MARKET_TZ) if now.tzinfo else now.replace(tzinfo=MARKET_TZ)
    if local_now.weekday() >= 5:
        return False
    minute_of_day = (local_now.hour * 60) + local_now.minute
    start_minute = max(_parse_hhmm(start_time, MARKET_OPEN_MINUTE), MARKET_OPEN_MINUTE)
    end_minute = min(_parse_hhmm(end_time, MARKET_CLOSE_MINUTE), MARKET_CLOSE_MINUTE)
    return start_minute <= minute_of_day < end_minute


def _algorithm_jitter_offset_minutes(bucket: datetime, refresh_minutes: int, jitter_minutes: int) -> int:
    jitter_minutes = max(int(jitter_minutes or 0), 0)
    if jitter_minutes <= 0:
        return 0
    seed = f"{bucket.date().isoformat()}:{bucket.hour:02d}:{bucket.minute:02d}:{refresh_minutes}".encode("utf-8")
    return int.from_bytes(sha256(seed).digest()[:4], "big") % (jitter_minutes + 1)


def _algorithm_bucket_key(
    now: datetime,
    refresh_minutes: int,
    jitter_minutes: int = 0,
    start_time: str = "08:30",
    end_time: str = "15:00",
) -> str | None:
    if not _is_regular_market_hours(now, start_time, end_time):
        return None
    refresh_minutes = max(int(refresh_minutes or 30), 1)
    local_now = now.astimezone(MARKET_TZ) if now.tzinfo else now.replace(tzinfo=MARKET_TZ)
    minute_of_day = (local_now.hour * 60) + local_now.minute
    start_minute = max(_parse_hhmm(start_time, MARKET_OPEN_MINUTE), MARKET_OPEN_MINUTE)
    bucket_minute = start_minute + (((minute_of_day - start_minute) // refresh_minutes) * refresh_minutes)
    bucket_hour, bucket_minute = divmod(bucket_minute, 60)
    bucket = local_now.replace(hour=bucket_hour, minute=bucket_minute, second=0, microsecond=0)
    jitter_offset = _algorithm_jitter_offset_minutes(bucket, refresh_minutes, jitter_minutes)
    scheduled = bucket + timedelta(minutes=jitter_offset)
    if local_now < scheduled:
        return None
    return f"algorithm:{bucket.isoformat(timespec='minutes')}"


def _algorithm_run_key() -> str | None:
    config = get_config()
    return _algorithm_bucket_key(
        datetime.now(MARKET_TZ),
        config.algorithm_market_data_refresh_minutes,
        config.algorithm_run_jitter_minutes,
        config.trading_start_time,
        config.trading_end_time,
    )


def _options_run_key() -> str | None:
    return _algorithm_run_key()


def _algorithm_check_seconds() -> int:
    return get_config().algorithm_check_seconds


def _options_check_seconds() -> int:
    return get_config().algorithm_check_seconds


def _options_account_id(controls: dict[str, Any]) -> str:
    return str(controls.get("options_trading_account_id") or controls.get("trading_account_id") or "")


def _dca_check_seconds() -> int:
    return get_config().dca_check_seconds


class BotRuntime:
    def __init__(self) -> None:
        self.algorithm = _RuntimeLoop(
            "algorithm",
            _algorithm_enabled,
            _run_algorithm,
            _algorithm_check_seconds,
            _algorithm_run_key,
        )
        self.options = _RuntimeLoop(
            "options",
            _options_enabled,
            _run_options,
            _options_check_seconds,
            _options_run_key,
            _options_account_id,
        )
        self.dca = _RuntimeLoop("dca", _dca_enabled, _run_dca, _dca_check_seconds, _dca_run_key)

    def start(self) -> None:
        self.algorithm.start()
        self.options.start()
        self.dca.start()

    def stop(self) -> None:
        self.algorithm.stop()
        self.options.stop()
        self.dca.stop()

    def wake_algorithm(self) -> None:
        self.algorithm.wake()

    def wake_options(self) -> None:
        self.options.wake()

    def wake_dca(self) -> None:
        self.dca.wake()

    def snapshot(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm.snapshot(),
            "options": self.options.snapshot(),
            "dca": self.dca.snapshot(),
        }


bot_runtime = BotRuntime()
