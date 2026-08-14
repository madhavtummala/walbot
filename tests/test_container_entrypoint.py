from __future__ import annotations

from pathlib import Path

from src.container_entrypoint import CONFIG_FILES, seed_config_defaults


def _defaults(tmp_path: Path) -> Path:
    source = tmp_path / "defaults"
    source.mkdir()
    for filename in CONFIG_FILES.values():
        (source / filename).write_text(f"# shipped {filename}\n", encoding="utf-8")
    return source


def _point_env(monkeypatch, defaults: Path, target_dir: Path) -> None:
    monkeypatch.setattr("src.container_entrypoint.CONFIG_DEFAULTS_DIR", defaults)
    for env_name, filename in CONFIG_FILES.items():
        monkeypatch.setenv(env_name, str(target_dir / filename))


def test_an_empty_volume_is_seeded_from_the_image(tmp_path, monkeypatch) -> None:
    """A bind mount shadows whatever the image baked into /config, so it starts empty."""
    target = tmp_path / "config"
    _point_env(monkeypatch, _defaults(tmp_path), target)

    seeded = seed_config_defaults()

    assert {path.name for path in seeded} == set(CONFIG_FILES.values())
    assert (target / "algorithms.yaml").read_text(encoding="utf-8") == "# shipped algorithms.yaml\n"


def test_seeding_never_overwrites_what_the_volume_already_has(tmp_path, monkeypatch) -> None:
    """The dashboard writes tuning to these paths; a redeploy must not discard it."""
    target = tmp_path / "config"
    target.mkdir()
    (target / "algorithms.yaml").write_text("# tuned by hand\n", encoding="utf-8")
    _point_env(monkeypatch, _defaults(tmp_path), target)

    seeded = seed_config_defaults()

    assert (target / "algorithms.yaml").read_text(encoding="utf-8") == "# tuned by hand\n"
    assert Path(target / "algorithms.yaml") not in seeded
    # Everything else it was missing still arrives.
    assert (target / "accounts.yaml").exists()


def test_seeding_is_a_no_op_without_shipped_defaults(tmp_path, monkeypatch) -> None:
    _point_env(monkeypatch, tmp_path / "absent", tmp_path / "config")

    assert seed_config_defaults() == []
