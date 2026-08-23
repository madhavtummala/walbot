from __future__ import annotations

from src.algorithms.bursty_dca.config import raw_plan, sanitize_plan, unknown_plan_symbols


#: What the account may trade. A plain set now rather than a list of dashboard payload rows:
#: filtering a plan is a data question, so the algorithm asks ``data.universe`` rather than
#: reaching up into the API layer for ``universe_payload``.
UNIVERSE = {"SPY", "QQQ", "GLD"}


def test_sanitize_dca_plan_keeps_only_universe_symbols() -> None:
    plan = {
        "enabled": True,
        "buy": {"amount": 100, "items": [{"symbol": "SPY"}, {"symbol": "BAD"}]},
        "sell": {"amount": 50, "items": [{"symbol": "QQQ"}, {"symbol": "SPY"}]},
    }

    sanitized = sanitize_plan(plan, UNIVERSE)

    assert [item["symbol"] for item in sanitized["buy"]["items"]] == ["SPY"]
    assert [item["symbol"] for item in sanitized["sell"]["items"]] == ["QQQ", "SPY"]


def test_sanitize_dca_plan_clamps_item_amount_to_max() -> None:
    from src.algorithms.bursty_dca.config import MAX_ITEM_AMOUNT

    plan = sanitize_plan(
        {
            "buy": {"items": [{"symbol": "SPY", "amount": MAX_ITEM_AMOUNT * 3}]},
            "sell": {"items": []},
        },
        UNIVERSE,
    )

    assert plan["buy"]["items"][0]["amount"] == MAX_ITEM_AMOUNT


def test_sanitize_dca_plan_drops_dashboard_layout_fields() -> None:
    plan = sanitize_plan(
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


def test_sanitize_dca_plan_drops_scheduling_and_enablement_keys() -> None:
    """Cadence lives on the algorithm class and the switch is the algorithm bot's, so a plan
    carrying either would be a second source of truth that nothing reads."""
    plan = sanitize_plan(
        {
            "enabled": True,
            "schedule_pattern": "0 9 * * 1-5",
            "next_run_date": "2026-01-01",
            "algorithm": "bursty_dca",
            "buy": {"items": [{"symbol": "SPY", "amount": 40}]},
            "sell": {"items": []},
        },
        UNIVERSE,
    )

    assert set(plan) == {"buy", "sell"}


def test_a_plan_is_ordinary_algorithm_config_keyed_by_algorithm() -> None:
    """Bursty DCA is a normal algorithm with a custom editor, so its plan is normal tuning.

    It used to live in a ``dca_bot`` section keyed by account, which had no room for the
    algorithm at all. Reading it from ``algorithms.<id>.plan`` means each algorithm that wants
    a plan gets its own, with no coordination between them.
    """

    class _Config:
        algorithm_configs = {
            "other_planner": {"plan": {"buy": {"items": [{"symbol": "SPY", "amount": 60.0}]}, "sell": {"items": []}}},
            "bursty_dca": {"plan": {"buy": {"items": [{"symbol": "QQQ", "amount": 25.0}]}, "sell": {"items": []}}},
        }

    config = _Config()
    other = sanitize_plan(raw_plan(config, "other_planner"), UNIVERSE)
    bursty = sanitize_plan(raw_plan(config, "bursty_dca"), UNIVERSE)

    assert [(i["symbol"], i["amount"]) for i in other["buy"]["items"]] == [("SPY", 60.0)]
    assert [(i["symbol"], i["amount"]) for i in bursty["buy"]["items"]] == [("QQQ", 25.0)]


def test_an_unconfigured_plan_buys_nothing_rather_than_a_built_in_basket() -> None:
    """The board renders whatever is in the config section, so a fallback here would have the
    algorithm quietly trading a basket the dashboard showed as empty."""

    class _Config:
        algorithm_configs = {"bursty_dca": {"scaling_factor": 2}}

    assert raw_plan(_Config(), "bursty_dca") == {}
    assert raw_plan(_Config(), "never_configured") == {}
    assert sanitize_plan(raw_plan(_Config(), "bursty_dca"), UNIVERSE)["buy"]["items"] == []


def test_unknown_plan_symbols_surfaces_what_sanitising_drops() -> None:
    """A typo is otherwise invisible: the row never appears and the money is never spent."""
    raw = {"buy": {"items": [{"symbol": "SPY", "amount": 10}, {"symbol": "NOPE", "amount": 10}]},
           "sell": {"items": []}}

    assert unknown_plan_symbols(raw, UNIVERSE) == ["NOPE"]
    assert [i["symbol"] for i in sanitize_plan(raw, UNIVERSE)["buy"]["items"]] == ["SPY"]


def test_dca_algorithm_reads_its_own_section(monkeypatch) -> None:
    """The plan comes from ``algorithms.bursty_dca``, never from a neighbouring section."""
    from src.algorithms.bursty_dca.algorithm import BurstyDCAAlgorithm

    monkeypatch.setattr("src.data.universe.tradable_symbols", lambda _config: {"SPY", "QQQ"})

    class _Config:
        account_id = "live"
        symbols = ["SPY", "QQQ"]
        tradables_csv = ""
        algorithm_configs = {
            "other_planner": {"plan": {"buy": {"items": [{"symbol": "SPY", "amount": 60.0}]}, "sell": {"items": []}}},
            "bursty_dca": {"plan": {"buy": {"items": [{"symbol": "QQQ", "amount": 25.0}]}, "sell": {"items": []}}},
        }

    config = _Config()
    assert [i["symbol"] for i in BurstyDCAAlgorithm(config).budget_plan(config)["buy"]["items"]] == ["QQQ"]
