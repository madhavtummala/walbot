from __future__ import annotations

from src.controls import load_controls, sanitize_controls, save_controls


def test_sanitize_controls_defaults_and_bools() -> None:
    controls = sanitize_controls({"algorithm_enabled": "false", "options_trading_enabled": "yes"})

    assert controls == {
        "trading_account_id": "",
        "equities": {"enabled": False, "strategy": "momentum_social"},
        "options": {"enabled": False, "strategy": "none", "account_id": ""},
        "algorithm": {"enabled": False, "strategy": "momentum_social"},
        "options_trading": {"enabled": False, "strategy": "none", "account_id": ""},
        "algorithm_enabled": False,
        "options_trading_enabled": False,
        "active_strategy": "momentum_social",
        "options_strategy": "none",
        "options_trading_account_id": "",
    }


def test_save_and_load_controls(tmp_path) -> None:
    controls_path = tmp_path / "algorithm_bot.yaml"

    saved = save_controls(
        {"algorithm_enabled": False, "options_trading_enabled": True, "active_strategy": "breakout"},
        path=str(controls_path),
    )

    assert saved == load_controls(path=str(controls_path))
    assert saved["active_strategy"] == "breakout"
    saved_yaml = controls_path.read_text(encoding="utf-8")
    assert "algorithm_bot:" in saved_yaml
    assert "options_bot:" in saved_yaml
    assert "algorithmic_trading:" not in saved_yaml


def test_save_and_load_controls_uses_split_bot_files(tmp_path, monkeypatch) -> None:
    algorithm_bot_path = tmp_path / "algorithm_bot.yaml"
    options_bot_path = tmp_path / "options_bot.yaml"
    monkeypatch.setenv("TRADING_ALGORITHM_BOT_FILE", str(algorithm_bot_path))
    monkeypatch.setenv("TRADING_OPTIONS_BOT_FILE", str(options_bot_path))

    saved = save_controls(
        {
            "trading_account_id": "paper-equities",
            "equities": {"enabled": True, "strategy": "breakout"},
            "options": {
                "enabled": True,
                "strategy": "options_swing_dual_momentum",
                "account_id": "paper-options",
            },
        }
    )

    loaded = load_controls()

    assert loaded == saved
    assert "algorithm_bot:" in algorithm_bot_path.read_text(encoding="utf-8")
    assert "options_bot:" in options_bot_path.read_text(encoding="utf-8")
    assert "algorithmic_trading:" not in algorithm_bot_path.read_text(encoding="utf-8")


def test_flat_api_controls_are_normalized() -> None:
    controls = sanitize_controls({"algorithm_enabled": True, "active_strategy": "breakout"})

    assert controls["algorithm_enabled"] is True
    assert controls["equities"]["strategy"] == "breakout"
    assert controls["algorithm"]["strategy"] == "breakout"


def test_new_equities_and_options_sections_load() -> None:
    controls = sanitize_controls(
        {
            "equities": {"enabled": True, "strategy": "risk_parity"},
            "options": {"enabled": True, "strategy": "protective_put", "account_id": "paper-options"},
        }
    )

    assert controls["active_strategy"] == "risk_parity"
    assert controls["algorithm_enabled"] is True
    assert controls["options_strategy"] == "protective_put"
    assert controls["options_trading_enabled"] is True
    assert controls["options_trading_account_id"] == "paper-options"


def test_load_controls_reads_structured_bot_yaml(tmp_path) -> None:
    controls_path = tmp_path / "algorithm_bot.yaml"
    controls_path.write_text(
        """
algorithm_bot:
  enabled: false
  strategy: none
options_bot:
  enabled: false
  strategy: none
""",
        encoding="utf-8",
    )

    loaded = load_controls(path=str(controls_path))

    assert loaded["active_strategy"] == "none"
    assert loaded["algorithm_enabled"] is False
