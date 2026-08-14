from __future__ import annotations

from pathlib import Path

import yaml

from src.container_entrypoint import CONFIG_FILENAME, prepare_config


def _bind(monkeypatch, defaults: Path, target: Path) -> None:
    monkeypatch.setattr("src.container_entrypoint.CONFIG_DEFAULTS_DIR", defaults)
    monkeypatch.setenv("TRADING_CONFIG_FILE", str(target))


def test_an_empty_volume_is_seeded_from_the_image(tmp_path, monkeypatch) -> None:
    """A bind mount shadows whatever the image baked into /config, so it starts empty."""
    defaults = tmp_path / "defaults"
    defaults.mkdir()
    (defaults / CONFIG_FILENAME).write_text("algorithms: {dca: {}}\n", encoding="utf-8")
    target = tmp_path / "config" / CONFIG_FILENAME
    _bind(monkeypatch, defaults, target)

    note = prepare_config()

    assert "Seeded" in note
    assert yaml.safe_load(target.read_text()) == {"algorithms": {"dca": {}}}


def test_a_volume_from_before_the_merge_is_migrated_not_overwritten(tmp_path, monkeypatch) -> None:
    """Those files hold the DCA plan and accrual; seeding over them would reset months."""
    defaults = tmp_path / "defaults"
    defaults.mkdir()
    (defaults / CONFIG_FILENAME).write_text("algorithms: {shipped: true}\n", encoding="utf-8")
    volume = tmp_path / "config"
    volume.mkdir()
    (volume / "accounts.yaml").write_text("default: live\naccounts: {items: {live: {}}}\n", encoding="utf-8")
    (volume / "dca_bot.yaml").write_text("dca_bot: {dca_plan: {buy: {amount: 900}}}\n", encoding="utf-8")
    _bind(monkeypatch, defaults, volume / CONFIG_FILENAME)

    note = prepare_config()

    merged = yaml.safe_load((volume / CONFIG_FILENAME).read_text())
    assert "Merged" in note
    assert merged["default"] == "live"
    assert merged["dca_bot"]["dca_plan"]["buy"]["amount"] == 900
    assert "shipped" not in merged.get("algorithms", {}), "the image must not overwrite tuning"


def test_an_existing_unified_file_is_left_alone(tmp_path, monkeypatch) -> None:
    defaults = tmp_path / "defaults"
    defaults.mkdir()
    (defaults / CONFIG_FILENAME).write_text("algorithms: {shipped: true}\n", encoding="utf-8")
    target = tmp_path / "config" / CONFIG_FILENAME
    target.parent.mkdir()
    target.write_text("algorithms: {mine: true}\n", encoding="utf-8")
    _bind(monkeypatch, defaults, target)

    assert prepare_config() == ""
    assert yaml.safe_load(target.read_text()) == {"algorithms": {"mine": True}}
