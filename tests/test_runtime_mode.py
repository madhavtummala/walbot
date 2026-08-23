from __future__ import annotations


from src.core.bot_runtime import _binding_enabled


def test_there_is_no_process_wide_runtime_mode() -> None:
    """Scheduled or agent-driven is a property of a deployment, not of the process.

    A --mcp process mode could only contradict the binding's own schedule, and it did: the
    dashboard reported "MCP mode" with every algorithm switched off.
    """
    from src.api import api_app
    from src import container_entrypoint

    assert not hasattr(api_app, "runtime_mode")
    assert not hasattr(api_app, "should_start_bot_runtime")
    source = (container_entrypoint.__file__,)
    text = open(source[0], encoding="utf-8").read()
    assert "TRADING_RUNTIME_MODE" not in text


def test_a_binding_parked_on_mcp_is_never_scheduled() -> None:
    """It is switched on, but it waits for an external request rather than a clock."""
    controls = {"bindings": [{"id": "b1", "strategy": "dca", "account_id": "paper",
                              "enabled": True, "cron": ""}]}

    assert _binding_enabled("b1")(controls) is False


def test_a_binding_with_a_cron_is_scheduled() -> None:
    controls = {"bindings": [{"id": "b1", "strategy": "dca", "account_id": "paper",
                              "enabled": True, "cron": "*/15 9-15 * * 1-5"}]}

    assert _binding_enabled("b1")(controls) is True


def test_the_mcp_server_is_always_started() -> None:
    """No switch to turn it off: an agent-driven binding needs something to call into.

    Whether an algorithm is driven by an agent is whether the binding has a ``cron``, so a
    process-wide off switch could only contradict it -- and a binding parked on ``mcp`` with
    the server suppressed is an algorithm nothing can ever run.
    """
    from src import container_entrypoint

    text = open(container_entrypoint.__file__, encoding="utf-8").read()
    assert "serve_in_thread(" in text
    assert "no_mcp_server" not in text
