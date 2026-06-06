from __future__ import annotations

from src.data.state_store import load_state, save_state


def test_state_store_round_trips_json_payload(tmp_path) -> None:
    db_path = tmp_path / "state.duckdb"
    payload = {"enabled": True, "items": [{"symbol": "SPY", "amount": 25}]}

    save_state("dca_plan", payload, db_path=str(db_path))

    assert load_state("dca_plan", {}, db_path=str(db_path)) == payload
