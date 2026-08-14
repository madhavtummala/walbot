from __future__ import annotations

from src.algorithms.dca import allocation_preview, load_dca_plan, sanitize_dca_plan, save_dca_plan


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

    assert set(plan) == {"max_item_amount", "buy", "sell"}


def test_save_and_load_dca_plan_uses_yaml_section(tmp_path) -> None:
    config_path = tmp_path / "dca_bot.yaml"

    saved = save_dca_plan(
        {
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


def test_plans_are_per_account_with_the_legacy_plan_as_template(tmp_path) -> None:
    """The monthly budgets are DCA's config, and config cannot be shared across accounts."""
    from src.algorithms.dca import load_dca_plan, save_dca_plan

    path = tmp_path / "dca_bot.yaml"
    path.write_text(
        """
dca_bot:
  dca_plan:
    max_item_amount: 500.0
    buy:
      amount: 100.0
      items:
        - symbol: SPY
          amount: 60.0
    sell:
      amount: 0.0
      items: []
""",
        encoding="utf-8",
    )
    universe = [{"symbol": "SPY", "enabled": True}, {"symbol": "QQQ", "enabled": True}]

    # Every account starts from the pre-account plan, so an existing config keeps working.
    for account in ("", "paper", "live"):
        loaded = load_dca_plan(universe, path=str(path), account_id=account)
        assert [(i["symbol"], i["amount"]) for i in loaded["buy"]["items"]] == [("SPY", 60.0)]

    save_dca_plan(
        {"max_item_amount": 500.0, "buy": {"amount": 25.0, "items": [{"symbol": "QQQ", "amount": 25.0}]},
         "sell": {"amount": 0.0, "items": []}},
        universe,
        path=str(path),
        account_id="paper",
    )

    paper = load_dca_plan(universe, path=str(path), account_id="paper")
    live = load_dca_plan(universe, path=str(path), account_id="live")
    assert [(i["symbol"], i["amount"]) for i in paper["buy"]["items"]] == [("QQQ", 25.0)]
    # Untouched accounts still read the template rather than inheriting paper's edit.
    assert [(i["symbol"], i["amount"]) for i in live["buy"]["items"]] == [("SPY", 60.0)]


def test_dca_algorithm_reads_the_plan_for_the_account_it_trades(monkeypatch) -> None:
    from src.algorithms.dca import bot as dca_bot

    seen = {}

    def fake_load(rows, path=None, account_id=""):
        seen["account_id"] = account_id
        return {"buy": {"items": []}, "sell": {"items": []}}

    monkeypatch.setattr(dca_bot, "load_dca_plan", fake_load)
    monkeypatch.setattr("src.api.api_payloads.universe_payload", lambda: {"rows": []})

    class _Config:
        account_id = "live"

    config = _Config()
    dca_bot.DCAAlgorithm(config).plan(config)
    assert seen["account_id"] == "live"
