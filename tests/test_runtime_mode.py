from __future__ import annotations

from src.api import api_app


def test_runtime_mode_defaults_to_bot(monkeypatch) -> None:
    monkeypatch.delenv(api_app.RUNTIME_MODE_ENV, raising=False)

    assert api_app.runtime_mode() == "bot"
    assert api_app.should_start_bot_runtime()


def test_runtime_mode_can_disable_internal_bot_for_mcp(monkeypatch) -> None:
    monkeypatch.setenv(api_app.RUNTIME_MODE_ENV, "mcp")

    assert api_app.runtime_mode() == "mcp"
    assert not api_app.should_start_bot_runtime()


def test_runtime_mode_falls_back_to_bot_for_unknown_values(monkeypatch) -> None:
    monkeypatch.setenv(api_app.RUNTIME_MODE_ENV, "surprise")

    assert api_app.runtime_mode() == "bot"
    assert api_app.should_start_bot_runtime()
