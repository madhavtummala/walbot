from __future__ import annotations

import yaml

from src.api.controls import (
    ORIGIN_MCP,
    ORIGIN_SCHEDULE,
    deployment_refusal,
    find_deployment,
    load_controls,
    primary_algorithm,
    sanitize_controls,
    save_controls,
)


def test_an_algorithm_without_an_account_is_not_deployed() -> None:
    """Undeployed is the absence of an account, not a separate flag."""
    controls = sanitize_controls(
        {"deployments": [{"algorithm": "bursty_dca", "enabled": True, "cron": "0 11 * * 1-5"}]}
    )

    assert controls["deployments"] == []


def test_a_deployment_carries_only_its_account_switch_and_schedule() -> None:
    controls = sanitize_controls(
        {
            "deployments": [
                {"algorithm": "bursty_dca", "account_id": "schwab1", "enabled": True, "cron": "0 11 * * 1-5"}
            ]
        }
    )

    assert controls["deployments"] == [
        {"algorithm": "bursty_dca", "account_id": "schwab1", "enabled": True, "cron": "0 11 * * 1-5"}
    ]


def test_one_algorithm_cannot_be_deployed_to_two_accounts() -> None:
    """The second entry is a malformed document, not a second deployment.

    Keeping both would start two scheduler loops sharing one algorithm-state key, which is the
    race this whole shape exists to make unrepresentable.
    """
    controls = sanitize_controls(
        {
            "deployments": [
                {"algorithm": "bursty_dca", "account_id": "schwab1", "enabled": True, "cron": ""},
                {"algorithm": "bursty_dca", "account_id": "alpaca1", "enabled": True, "cron": ""},
            ]
        }
    )

    assert [d["account_id"] for d in controls["deployments"]] == ["schwab1"]


def test_several_algorithms_may_share_one_account() -> None:
    """The forbidden direction is one algorithm to many accounts, not the reverse."""
    controls = sanitize_controls(
        {
            "deployments": [
                {"algorithm": "bursty_dca", "account_id": "alpaca1", "enabled": False, "cron": ""},
                {"algorithm": "rally_rotation", "account_id": "alpaca1", "enabled": False, "cron": ""},
            ]
        }
    )

    assert [d["algorithm"] for d in controls["deployments"]] == ["bursty_dca", "rally_rotation"]
    assert {d["account_id"] for d in controls["deployments"]} == {"alpaca1"}


def test_an_absent_cron_takes_the_algorithms_default_but_an_empty_one_is_agent_driven() -> None:
    """The difference between "never chosen" and "chosen to be agent-driven".

    A new deployment defaulting to empty would sit switched on and never run, because empty
    means "a clock does not drive this".
    """
    never_chosen = sanitize_controls(
        {"deployments": [{"algorithm": "bursty_dca", "account_id": "schwab1"}]}
    )
    chosen_empty = sanitize_controls(
        {"deployments": [{"algorithm": "bursty_dca", "account_id": "schwab1", "cron": ""}]}
    )

    assert never_chosen["deployments"][0]["cron"] != ""
    assert chosen_empty["deployments"][0]["cron"] == ""


def test_exactly_one_origin_drives_a_deployment() -> None:
    scheduled = {"algorithm": "bursty_dca", "account_id": "a", "enabled": True, "cron": "0 11 * * 1-5"}
    agent_driven = {"algorithm": "bursty_dca", "account_id": "a", "enabled": True, "cron": ""}

    assert deployment_refusal(scheduled, ORIGIN_SCHEDULE) == ""
    assert deployment_refusal(scheduled, ORIGIN_MCP) != ""
    assert deployment_refusal(agent_driven, ORIGIN_MCP) == ""
    assert deployment_refusal(agent_driven, ORIGIN_SCHEDULE) != ""


def test_a_switched_off_deployment_refuses_both_origins() -> None:
    off = {"algorithm": "bursty_dca", "account_id": "a", "enabled": False, "cron": ""}

    assert "switched off" in deployment_refusal(off, ORIGIN_MCP)
    assert "switched off" in deployment_refusal(off, ORIGIN_SCHEDULE)


def test_an_undeployed_algorithm_refuses_with_a_reason_that_says_so() -> None:
    assert "not deployed" in deployment_refusal(None, ORIGIN_MCP)


def test_deployment_keys_round_trip_without_touching_tuning(tmp_path) -> None:
    config_path = tmp_path / "walbot.yaml"
    config_path.write_text(
        """
algorithms:
  bursty_dca:
    account_id: schwab1
    enabled: false
    cron: 0 11 * * 1-5
    regime_ma_days: 150
    plan:
      buy:
        amount: 1500
""",
        encoding="utf-8",
    )

    controls = load_controls(path=str(config_path))
    assert find_deployment(controls, "bursty_dca")["account_id"] == "schwab1"

    controls["deployments"][0]["enabled"] = True
    saved = save_controls(controls, path=str(config_path))

    assert saved == load_controls(path=str(config_path))
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    section = document["algorithms"]["bursty_dca"]
    assert section["enabled"] is True
    # The tuning either side of the deployment keys is untouched.
    assert section["regime_ma_days"] == 150
    assert section["plan"] == {"buy": {"amount": 1500}}


def test_undeploying_removes_the_keys_rather_than_blanking_them(tmp_path) -> None:
    """``account_id: ""`` left behind would read back as deployed nowhere, said less clearly."""
    config_path = tmp_path / "walbot.yaml"
    config_path.write_text(
        """
algorithms:
  bursty_dca:
    account_id: schwab1
    enabled: true
    cron: ''
    regime_ma_days: 150
""",
        encoding="utf-8",
    )

    save_controls({"deployments": []}, path=str(config_path))

    section = yaml.safe_load(config_path.read_text(encoding="utf-8"))["algorithms"]["bursty_dca"]
    assert "account_id" not in section
    assert "enabled" not in section
    assert "cron" not in section
    assert section["regime_ma_days"] == 150


def test_deploying_an_untuned_algorithm_creates_its_section(tmp_path) -> None:
    config_path = tmp_path / "walbot.yaml"
    config_path.write_text("algorithms: {}\n", encoding="utf-8")

    save_controls(
        {"deployments": [{"algorithm": "rally_rotation", "account_id": "alpaca1", "cron": ""}]},
        path=str(config_path),
    )

    loaded = load_controls(path=str(config_path))
    assert find_deployment(loaded, "rally_rotation")["account_id"] == "alpaca1"


def test_primary_algorithm_falls_back_when_nothing_is_deployed() -> None:
    from src.core.config import DEFAULT_STRATEGY_ID

    assert primary_algorithm({"deployments": []}) == DEFAULT_STRATEGY_ID
    assert primary_algorithm(
        {"deployments": [{"algorithm": "options_flip", "account_id": "a"}]}
    ) == "options_flip"


def test_saving_tuning_does_not_undeploy_the_algorithm(tmp_path, monkeypatch) -> None:
    """The tuning editor and the deploy control share one section on disk.

    ``save_algorithm_config_payload`` assigns that section wholesale, and the tuning payload
    deliberately omits the deployment keys -- so without carrying them across, saving a knob
    from the Tune screen would quietly stop the algorithm trading.
    """
    from src.api.payloads.algorithms import algorithm_config_payload, save_algorithm_config_payload

    config_path = tmp_path / "walbot.yaml"
    config_path.write_text(
        """
algorithms:
  bursty_dca:
    account_id: schwab1
    enabled: true
    cron: 0 11 * * 1-5
    regime_ma_days: 150
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADING_CONFIG_FILE", str(config_path))

    # The tuning view never shows the deployment keys as knobs.
    assert set(algorithm_config_payload("bursty_dca")["config"]).isdisjoint(
        {"account_id", "enabled", "cron"}
    )

    save_algorithm_config_payload("bursty_dca", {"regime_ma_days": 200})

    deployment = find_deployment(load_controls(path=str(config_path)), "bursty_dca")
    assert deployment is not None, "saving tuning undeployed the algorithm"
    assert deployment["account_id"] == "schwab1"
    assert deployment["enabled"] is True
    assert deployment["cron"] == "0 11 * * 1-5"
