from __future__ import annotations

from src.dca import allocation_preview, load_dca_plan, sanitize_dca_plan, save_dca_plan


UNIVERSE = [
    {"symbol": "SPY", "name": "SPY", "bucket": "Broad", "enabled": True},
    {"symbol": "QQQ", "name": "QQQ", "bucket": "Growth", "enabled": True},
    {"symbol": "GLD", "name": "GLD", "bucket": "Gold", "enabled": True},
]


def test_sanitize_dca_plan_keeps_only_universe_symbols() -> None:
    plan = {
        "enabled": True,
        "frequency": "daily",
        "buy": {"amount": 100, "items": [{"symbol": "SPY"}, {"symbol": "BAD"}]},
        "sell": {"amount": 50, "items": [{"symbol": "QQQ"}, {"symbol": "SPY"}]},
    }

    sanitized = sanitize_dca_plan(plan, UNIVERSE)

    assert [item["symbol"] for item in sanitized["buy"]["items"]] == ["SPY"]
    assert [item["symbol"] for item in sanitized["sell"]["items"]] == ["QQQ", "SPY"]


def test_allocation_preview_uses_exact_item_amounts() -> None:
    plan = sanitize_dca_plan(
        {
            "enabled": True,
            "frequency": "weekly",
            "max_item_amount": 50,
            "buy": {
                "items": [
                    {"symbol": "SPY", "amount": 50},
                    {"symbol": "QQQ", "amount": 30},
                    {"symbol": "GLD", "amount": 10},
                ],
            },
            "sell": {"amount": 0, "items": []},
        },
        UNIVERSE,
    )

    preview = allocation_preview(plan)

    assert [row["symbol"] for row in preview] == ["SPY", "QQQ", "GLD"]
    assert [round(row["notional"], 2) for row in preview] == [50.0, 30.0, 10.0]


def test_sanitize_dca_plan_clamps_item_amount_to_max() -> None:
    plan = sanitize_dca_plan(
        {
            "max_item_amount": 50,
            "buy": {"items": [{"symbol": "SPY", "amount": 140}]},
            "sell": {"items": []},
        },
        UNIVERSE,
    )

    assert plan["buy"]["items"][0]["amount"] == 50


def test_sanitize_dca_plan_drops_dashboard_layout_fields() -> None:
    plan = sanitize_dca_plan(
        {
            "buy": {
                "items": [
                    {"symbol": "SPY", "name": "Decorative name", "bucket": "Decorative bucket", "amount": 40, "position": {"x": 2, "y": -0.25}}
                ]
            },
            "sell": {"items": []},
        },
        UNIVERSE,
    )

    assert plan["buy"]["items"][0] == {"symbol": "SPY", "amount": 40.0}


def test_sanitize_dca_plan_preserves_schedule_pattern() -> None:
    plan = sanitize_dca_plan(
        {
            "schedule_pattern": "0 9 * * 1-5",
            "buy": {"items": [{"symbol": "SPY", "amount": 40}]},
            "sell": {"items": []},
        },
        UNIVERSE,
    )

    assert plan["schedule_pattern"] == "0 9 * * 1-5"


def test_save_and_load_dca_plan_uses_yaml_section(tmp_path) -> None:
    config_path = tmp_path / "dca_bot.yaml"

    saved = save_dca_plan(
        {
            "enabled": True,
            "schedule_pattern": "0 10 * * 1-5",
            "buy": {"items": [{"symbol": "SPY", "amount": 40}]},
            "sell": {"items": []},
        },
        UNIVERSE,
        path=str(config_path),
    )

    loaded = load_dca_plan(UNIVERSE, path=str(config_path))

    assert loaded == saved
    assert "dca_plan:" in config_path.read_text(encoding="utf-8")


def test_save_and_load_dca_plan_uses_dca_bot_file(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "dca_bot.yaml"
    monkeypatch.setenv("TRADING_DCA_BOT_FILE", str(config_path))

    saved = save_dca_plan(
        {
            "enabled": True,
            "schedule_pattern": "0 10 * * 1-5",
            "buy": {"items": [{"symbol": "SPY", "amount": 40}]},
            "sell": {"items": []},
        },
        UNIVERSE,
    )

    loaded = load_dca_plan(UNIVERSE)
    saved_yaml = config_path.read_text(encoding="utf-8")

    assert loaded == saved
    assert "dca_bot:" in saved_yaml
    assert "dca_plan:" in saved_yaml
