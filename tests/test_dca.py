from __future__ import annotations

from src.algorithms.dca import allocation_preview, raw_plan_from_config, sanitize_dca_plan, unknown_plan_symbols


UNIVERSE = [
    {"symbol": "SPY", "name": "SPY", "bucket": "Broad", "enabled": True},
    {"symbol": "QQQ", "name": "QQQ", "bucket": "Growth", "enabled": True},
    {"symbol": "GLD", "name": "GLD", "bucket": "Gold", "enabled": True},
]


def test_sanitize_dca_plan_keeps_only_universe_symbols() -> None:
    plan = {
        "enabled": True,
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
    from src.algorithms.dca import DCA_MAX_ITEM_AMOUNT

    plan = sanitize_dca_plan(
        {
            "buy": {"items": [{"symbol": "SPY", "amount": DCA_MAX_ITEM_AMOUNT * 3}]},
            "sell": {"items": []},
        },
        UNIVERSE,
    )

    assert plan["buy"]["items"][0]["amount"] == DCA_MAX_ITEM_AMOUNT


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


def test_sanitize_dca_plan_drops_scheduling_and_enablement_keys() -> None:
    """Cadence lives on the algorithm class and the switch is the algorithm bot's, so a plan
    carrying either would be a second source of truth that nothing reads."""
    plan = sanitize_dca_plan(
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
    """DCA is a normal algorithm with a custom editor, so its plan is normal tuning.

    It used to live in a ``dca_bot`` section keyed by account, which had no room for the
    algorithm -- so ``dca`` and ``bursty_dca`` were forced to share one budget however
    differently they behave. Reading it from ``algorithms.<id>.plan`` separates them.
    """

    class _Config:
        algorithm_configs = {
            "dca": {"plan": {"buy": {"items": [{"symbol": "SPY", "amount": 60.0}]}, "sell": {"items": []}}},
            "bursty_dca": {"plan": {"buy": {"items": [{"symbol": "QQQ", "amount": 25.0}]}, "sell": {"items": []}}},
        }

    config = _Config()
    steady = sanitize_dca_plan(raw_plan_from_config(config, "dca"), UNIVERSE)
    bursty = sanitize_dca_plan(raw_plan_from_config(config, "bursty_dca"), UNIVERSE)

    assert [(i["symbol"], i["amount"]) for i in steady["buy"]["items"]] == [("SPY", 60.0)]
    assert [(i["symbol"], i["amount"]) for i in bursty["buy"]["items"]] == [("QQQ", 25.0)]


def test_an_unconfigured_plan_buys_nothing_rather_than_a_built_in_basket() -> None:
    """The board renders whatever is in the config section, so a fallback here would have the
    algorithm quietly trading a basket the dashboard showed as empty. DEFAULT_DCA_PLAN seeds a
    new config; it is not a standing order."""

    class _Config:
        algorithm_configs = {"dca": {"rsi_lookback": 2}}

    assert raw_plan_from_config(_Config(), "dca") == {}
    assert raw_plan_from_config(_Config(), "never_configured") == {}
    assert sanitize_dca_plan(raw_plan_from_config(_Config(), "dca"), UNIVERSE)["buy"]["items"] == []


def test_unknown_plan_symbols_surfaces_what_sanitising_drops() -> None:
    """A typo is otherwise invisible: the row never appears and the money is never spent."""
    raw = {"buy": {"items": [{"symbol": "SPY", "amount": 10}, {"symbol": "NOPE", "amount": 10}]},
           "sell": {"items": []}}

    assert unknown_plan_symbols(raw, UNIVERSE) == ["NOPE"]
    assert [i["symbol"] for i in sanitize_dca_plan(raw, UNIVERSE)["buy"]["items"]] == ["SPY"]


def test_dca_algorithm_reads_its_own_section(monkeypatch) -> None:
    from src.algorithms.dca import bot as dca_bot
    from src.algorithms.dca.bursty import BurstyDCAAlgorithm

    monkeypatch.setattr(
        "src.api.api_payloads.universe_payload",
        lambda: {"rows": [{"symbol": "SPY", "enabled": True}, {"symbol": "QQQ", "enabled": True}]},
    )

    class _Config:
        account_id = "live"
        algorithm_configs = {
            "dca": {"plan": {"buy": {"items": [{"symbol": "SPY", "amount": 60.0}]}, "sell": {"items": []}}},
            "bursty_dca": {"plan": {"buy": {"items": [{"symbol": "QQQ", "amount": 25.0}]}, "sell": {"items": []}}},
        }

    config = _Config()
    assert [i["symbol"] for i in dca_bot.DCAAlgorithm(config).plan(config)["buy"]["items"]] == ["SPY"]
    assert [i["symbol"] for i in BurstyDCAAlgorithm(config).plan(config)["buy"]["items"]] == ["QQQ"]
