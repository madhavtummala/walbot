from __future__ import annotations

import json
import logging
import os
import time
from typing import Any
from urllib import parse, request

from ...common.config_utils import as_bool, direct_or_env
from ..base import NotificationConnector, NotificationMessage

logger = logging.getLogger(__name__)


def _notification_timeout() -> float:
    try:
        return max(float(os.getenv("TELEGRAM_NOTIFICATION_TIMEOUT_SECONDS", "5")), 0.1)
    except ValueError:
        return 5.0


class TelegramNotificationConnector(NotificationConnector):
    provider_name = "telegram"

    def settings(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.config.get("enabled", True)) and as_bool(os.getenv("TELEGRAM_NOTIFICATIONS_ENABLED"), True),
            "bot_token": direct_or_env(self.config, "bot_token", "bot_token_env", "TELEGRAM_BOT_TOKEN"),
            "chat_id": direct_or_env(self.config, "chat_id", "chat_id_env", "TELEGRAM_CHAT_ID"),
            "api_root": str(self.config.get("api_root") or os.getenv("TELEGRAM_API_ROOT", "https://api.telegram.org")),
            "timeout_seconds": self.config.get("timeout_seconds"),
        }

    def timeout(self, settings: dict[str, Any]) -> float:
        raw_timeout = settings.get("timeout_seconds")
        if raw_timeout is None or raw_timeout == "":
            return _notification_timeout()
        try:
            return max(float(raw_timeout), 0.1)
        except ValueError:
            return 5.0

    def _post(self, method: str, payload: dict[str, Any], timeout: float) -> dict[str, Any] | None:
        settings = self.settings()
        token = str(settings["bot_token"]).strip()
        if not token or not settings["enabled"]:
            return None
        api_root = str(settings["api_root"]).strip().rstrip("/")
        url = f"{api_root}/bot{token}/{method}"
        encoded = parse.urlencode(payload).encode("utf-8")
        telegram_request = request.Request(
            url,
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with request.urlopen(telegram_request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            return {"ok": True, "raw": body}
        return result if isinstance(result, dict) else {"ok": True, "result": result}

    def send(self, message: NotificationMessage) -> bool:
        settings = self.settings()
        token = str(settings["bot_token"]).strip()
        chat_id = str(settings["chat_id"]).strip()
        if not token or not chat_id or not settings["enabled"]:
            return False

        timeout = self.timeout(settings)

        try:
            result = self._post("sendMessage", {"chat_id": chat_id, "text": message.text}, timeout)
        except Exception as exc:
            logger.warning("Unable to send Telegram portfolio notification: %s", exc)
            return False

        if not result:
            return False
        if result.get("raw") is not None:
            return True
        if not result.get("ok", True):
            logger.warning("Telegram portfolio notification failed: %s", result.get("description") or result)
            return False
        return True

    def request_approval(
        self,
        message: NotificationMessage,
        *,
        approval_id: str,
        timeout_seconds: int = 300,
        poll_seconds: int = 5,
    ) -> bool:
        settings = self.settings()
        chat_id = str(settings["chat_id"]).strip()
        timeout = self.timeout(settings)
        if not chat_id:
            return False
        reply_markup = json.dumps(
            {
                "inline_keyboard": [
                    [
                        {"text": "Approve", "callback_data": f"approve:{approval_id}"},
                        {"text": "Deny", "callback_data": f"deny:{approval_id}"},
                    ]
                ]
            }
        )
        try:
            send_result = self._post(
                "sendMessage",
                {"chat_id": chat_id, "text": message.text, "reply_markup": reply_markup},
                timeout,
            )
        except Exception as exc:
            logger.warning("Unable to send Telegram trade approval request: %s", exc)
            return False
        if not send_result or not send_result.get("ok", True):
            return False

        deadline = time.monotonic() + max(int(timeout_seconds or 0), 1)
        offset: int | None = None
        poll_interval = max(float(poll_seconds or 1), 1.0)
        while time.monotonic() < deadline:
            remaining = max(min(poll_interval, deadline - time.monotonic()), 0.1)
            try:
                payload: dict[str, Any] = {"timeout": int(max(remaining, 1))}
                if offset is not None:
                    payload["offset"] = offset
                result = self._post("getUpdates", payload, timeout + remaining)
            except Exception as exc:
                logger.warning("Unable to poll Telegram approval updates: %s", exc)
                time.sleep(remaining)
                continue
            if not result or not result.get("ok", True):
                time.sleep(remaining)
                continue
            for update in result.get("result", []) or []:
                try:
                    offset = max(int(update.get("update_id")) + 1, offset or 0)
                except (TypeError, ValueError):
                    pass
                decision = self._approval_decision(update, approval_id, chat_id)
                if decision is not None:
                    return decision
        return False

    def _approval_decision(self, update: dict[str, Any], approval_id: str, chat_id: str) -> bool | None:
        callback = update.get("callback_query") if isinstance(update.get("callback_query"), dict) else {}
        if callback:
            data = str(callback.get("data") or "").strip().lower()
            message = callback.get("message") if isinstance(callback.get("message"), dict) else {}
            chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
            if str(chat.get("id") or "") != str(chat_id):
                return None
            if data == f"approve:{approval_id}".lower():
                return True
            if data == f"deny:{approval_id}".lower():
                return False

        message = update.get("message") if isinstance(update.get("message"), dict) else {}
        chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
        if str(chat.get("id") or "") != str(chat_id):
            return None
        text = str(message.get("text") or "").strip().lower()
        if text == f"/approve {approval_id}".lower() or text == f"approve {approval_id}".lower():
            return True
        if text == f"/deny {approval_id}".lower() or text == f"deny {approval_id}".lower():
            return False
        return None
