from __future__ import annotations

from urllib import parse

from src.notifications import service as notifications
from src.notifications.providers import telegram
from src.notifications.registry import get_notification_connector_class


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return b'{"ok": true}'


def test_format_portfolio_change_message_ignores_non_submitted_orders() -> None:
    message = notifications.format_portfolio_change_message(
        [
            {"symbol": "AAA", "action": "hold", "quantity": 0},
            {
                "symbol": "BBB",
                "action": "buy",
                "quantity": 3,
                "target_weight": 0.25,
                "current_shares": 0,
                "target_shares": 3,
                "trade_dollars": 300.0,
                "order_id": "order-1",
            },
        ]
    )

    assert message == (
        "Walbot 🤖 💰 💸\n"
        "Portfolio changes submitted: 1 order\n"
        "BUY BBB qty=3 target_weight=25.00% current=0 target=3 trade_dollars=300.00 order_id=order-1"
    )


def test_format_trade_approval_message_includes_reply_commands() -> None:
    message = notifications.format_trade_approval_message(
        [{"symbol": "AAA", "action": "buy", "quantity": 2, "trade_dollars": 200.0}],
        "abc123",
    )

    assert "Approval ID: abc123" in message
    assert "/approve abc123" in message
    assert "/deny abc123" in message
    assert "BUY AAA qty=2" in message


def test_send_telegram_message_posts_to_configured_bot(monkeypatch) -> None:
    calls = []

    def fake_urlopen(telegram_request, timeout):
        calls.append((telegram_request, timeout))
        return _Response()

    monkeypatch.setattr(
        notifications,
        "load_connectors_config",
        lambda: {
            "notifications": {
                "providers": {
                    "telegram": {
                        "bot_token": "token-123",
                        "chat_id": "chat-456",
                        "api_root": "https://telegram.example.test",
                        "timeout_seconds": 2,
                    }
                }
            }
        },
    )
    monkeypatch.setattr(telegram.request, "urlopen", fake_urlopen)

    assert notifications.send_telegram_message("Portfolio changed") is True

    telegram_request, timeout = calls[0]
    assert telegram_request.full_url == "https://telegram.example.test/bottoken-123/sendMessage"
    assert timeout == 2
    payload = parse.parse_qs(telegram_request.data.decode("utf-8"))
    assert payload == {"chat_id": ["chat-456"], "text": ["Portfolio changed"]}


def test_send_telegram_message_is_disabled_without_credentials(monkeypatch) -> None:
    monkeypatch.setattr(notifications, "load_connectors_config", lambda: {})
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    assert notifications.send_telegram_message("Portfolio changed") is False


def test_send_telegram_message_falls_back_to_environment(monkeypatch) -> None:
    calls = []

    def fake_urlopen(telegram_request, timeout):
        calls.append((telegram_request, timeout))
        return _Response()

    monkeypatch.setattr(notifications, "load_connectors_config", lambda: {})
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "env-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "env-chat")
    monkeypatch.setattr(telegram.request, "urlopen", fake_urlopen)

    assert notifications.send_telegram_message("Portfolio changed") is True

    telegram_request, _timeout = calls[0]
    assert telegram_request.full_url == "https://api.telegram.org/botenv-token/sendMessage"


def test_notification_registry_returns_telegram_connector() -> None:
    cls = get_notification_connector_class("telegram")

    assert cls.provider_name == "telegram"


def test_telegram_approval_decision_accepts_matching_chat_reply() -> None:
    connector = telegram.TelegramNotificationConnector({})

    assert connector._approval_decision(
        {"message": {"chat": {"id": "123"}, "text": "/approve abc123"}},
        "abc123",
        "123",
    ) is True
    assert connector._approval_decision(
        {"message": {"chat": {"id": "123"}, "text": "/deny abc123"}},
        "abc123",
        "123",
    ) is False
    assert connector._approval_decision(
        {"message": {"chat": {"id": "999"}, "text": "/approve abc123"}},
        "abc123",
        "123",
    ) is None
