from __future__ import annotations

from src.api.controls import load_controls, sanitize_controls, save_controls


def test_sanitize_controls_defaults_and_bools() -> None:
    controls = sanitize_controls({"algorithm_enabled": "false"})

    assert controls == {
        "trading_account_id": "",
        "equities": {"enabled": False, "strategy": "rally_rotation"},
        "algorithm": {"enabled": False, "strategy": "rally_rotation"},
        "algorithm_enabled": False,
        "bindings": [
            {"id": "b1", "strategy": "rally_rotation", "account_id": "", "enabled": False, "cron": "30 9 * * 1-5"},
        ],
        "active_strategy": "rally_rotation",
    }


def test_save_and_load_controls(tmp_path) -> None:
    controls_path = tmp_path / "algorithm_bot.yaml"

    saved = save_controls(
        {"algorithm_enabled": False, "active_strategy": "breakout"},
        path=str(controls_path),
    )

    assert saved == load_controls(path=str(controls_path))
    assert saved["active_strategy"] == "breakout"
    saved_yaml = controls_path.read_text(encoding="utf-8")
    assert "algorithm_bot:" in saved_yaml
    assert "algorithmic_trading:" not in saved_yaml


def test_save_and_load_controls_uses_split_bot_files(tmp_path, monkeypatch) -> None:
    algorithm_bot_path = tmp_path / "algorithm_bot.yaml"
    monkeypatch.setenv("TRADING_ALGORITHM_BOT_FILE", str(algorithm_bot_path))

    saved = save_controls(
        {
            "trading_account_id": "paper-equities",
            "equities": {"enabled": True, "strategy": "breakout"},
        }
    )

    loaded = load_controls()

    assert loaded == saved
    assert "algorithm_bot:" in algorithm_bot_path.read_text(encoding="utf-8")
    assert "algorithmic_trading:" not in algorithm_bot_path.read_text(encoding="utf-8")


def test_flat_api_controls_are_normalized() -> None:
    controls = sanitize_controls({"algorithm_enabled": True, "active_strategy": "breakout"})

    assert controls["algorithm_enabled"] is True
    assert controls["equities"]["strategy"] == "breakout"
    assert controls["algorithm"]["strategy"] == "breakout"


def test_new_equities_section_loads() -> None:
    controls = sanitize_controls(
        {
            "equities": {"enabled": True, "strategy": "risk_parity"},
        }
    )

    assert controls["active_strategy"] == "risk_parity"
    assert controls["algorithm_enabled"] is True


def test_migrating_none_lands_in_the_off_state(tmp_path) -> None:
    """A config saved as "none" could hold enabled: true while the dashboard showed off."""
    controls_path = tmp_path / "algorithm_bot.yaml"
    controls_path.write_text(
        """
algorithm_bot:
  enabled: true
  strategy: none
""",
        encoding="utf-8",
    )

    loaded = load_controls(path=str(controls_path))

    assert loaded["active_strategy"] == "bursty_dca"
    assert loaded["algorithm_enabled"] is False


def test_load_controls_reads_structured_bot_yaml(tmp_path) -> None:
    controls_path = tmp_path / "algorithm_bot.yaml"
    controls_path.write_text(
        """
algorithm_bot:
  enabled: false
  strategy: none
""",
        encoding="utf-8",
    )

    loaded = load_controls(path=str(controls_path))

    # "none" is a retired equity id that now resolves to DCA; the power toggle is what
    # keeps the bot idle.
    assert loaded["active_strategy"] == "bursty_dca"
    assert loaded["algorithm_enabled"] is False
