from __future__ import annotations

from src.data.universe import load_tradable_names, resolve_project_path


def test_load_tradable_names_returns_symbol_lookup(tmp_path) -> None:
    tradables_csv = tmp_path / "tradables.csv"
    tradables_csv.write_text("Ticker,Name\nAAA,Alpha Fund\nBBB,Beta Fund\nAAA,Duplicate\n", encoding="utf-8")

    names = load_tradable_names(str(tradables_csv))

    assert names == {"AAA": "Alpha Fund", "BBB": "Beta Fund"}


def test_resolve_project_path_keeps_absolute_paths(tmp_path) -> None:
    assert resolve_project_path(str(tmp_path)) == tmp_path
