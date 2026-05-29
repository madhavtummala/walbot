from __future__ import annotations

from urllib import parse

from src import notifications


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
        "Trading Bot 🤖 💰 💸\n"
        "Portfolio changes submitted: 1 order\n"
        "BUY BBB qty=3 target_weight=25.00% current=0 target=3 trade_dollars=300.00 order_id=order-1"
    )


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
    monkeypatch.setattr(notifications.request, "urlopen", fake_urlopen)

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
    monkeypatch.setattr(notifications.request, "urlopen", fake_urlopen)

    assert notifications.send_telegram_message("Portfolio changed") is True

    telegram_request, _timeout = calls[0]
    assert telegram_request.full_url == "https://api.telegram.org/botenv-token/sendMessage"
