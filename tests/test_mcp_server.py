from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src import mcp_server
from src.api import controls as controls_module
from src.core import plan_cache
from src.core.config import Config
from src.core.interfaces import AlgorithmPlan, DesiredOrder, Intent, OrderRequest


@pytest.fixture(autouse=True)
def _empty_plan_cache():
    """Tokens are process-global, so one test's leftovers are another's phantom plan."""
    plan_cache.clear()
    yield
    plan_cache.clear()


class DummyMCP:
    def __init__(self) -> None:
        self.tools = []

    def tool(self):
        def decorator(func):
            self.tools.append(func)
            setattr(self, func.__name__, func)
            return func

        return decorator


def _deployment(**overrides) -> dict:
    """A deployment an agent is allowed to drive: switched on, no cron."""
    deployment = {"algorithm": "rally_rotation", "account_id": "paper", "enabled": True, "cron": ""}
    deployment.update(overrides)
    return deployment


def _build(monkeypatch, deployments: list[dict] | None = None) -> DummyMCP:
    fake_server = DummyMCP()
    monkeypatch.setattr(mcp_server, "_server", lambda *args, **kwargs: fake_server)
    controls = {"deployments": [_deployment()] if deployments is None else deployments}
    # Patched in both namespaces: the tools read controls directly, and
    # ``resolve_deployment_for_origin`` reads them through its own module.
    monkeypatch.setattr(mcp_server, "load_controls", lambda *a, **k: controls)
    monkeypatch.setattr(controls_module, "load_controls", lambda *a, **k: controls)
    mcp_server.create_mcp_server()
    return fake_server


def test_create_mcp_server_exposes_expected_tools(monkeypatch) -> None:
    fake_server = _build(monkeypatch)

    assert [tool.__name__ for tool in fake_server.tools] == [
        "list_algorithms",
        "get_algorithm_plan",
        "get_price",
        "list_accounts",
        "get_account_positions",
        "get_account_orders",
        "place_orders",
    ]


def _plan(**overrides) -> AlgorithmPlan:
    fields = {
        "strategy": "rally_rotation",
        "mode": "target",
        "intents": [Intent(symbol="AAA", kind="weight", value=0.5)],
        "latest_prices": {"AAA": 100.0},
        "signals": {"AAA": {"score": 1.0}},
        "state": {"accrued": 12.0},
    }
    fields.update(overrides)
    return AlgorithmPlan(**fields)


def _token(plan: AlgorithmPlan | None = None, *, algorithm: str = "rally_rotation", account_id: str = "default") -> str:
    """Stash a plan the way get_algorithm_plan would, and return its token.

    ``account_id`` matches ``Config()``'s own default, so a test that does not care about the
    account is not tripped by the guard that a plan must execute against the book it was sized
    against.
    """
    return plan_cache.stash(plan if plan is not None else _plan(), algorithm=algorithm, account_id=account_id)


def _placing(monkeypatch, *, positions: dict | None = None, outcome: dict | None = None) -> dict:
    """Stub out everything downstream of the gates and capture the plan that reaches execute."""
    captured: dict = {}

    class _Book:
        def get_positions(self):
            return dict(positions or {})

    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=False))
    monkeypatch.setattr(mcp_server, "resolve_brokerage", lambda config: _Book())
    monkeypatch.setattr(mcp_server, "record_orders", lambda *a, **k: None)

    def fake_execute(plan, config, brokerage, **kwargs):
        captured["plan"] = plan
        return outcome if outcome is not None else {"status": "submitted", "order_results": []}

    monkeypatch.setattr(mcp_server, "execute_algorithm", fake_execute)
    return captured


def _shares(plan: AlgorithmPlan) -> dict[str, float]:
    return {intent.symbol: intent.value for intent in plan.intents if intent.kind == "shares"}


def test_the_plan_payload_carries_what_review_needs() -> None:
    """The agent reads this to form a judgement. It is no longer read back, so the only
    requirement is that nothing a reviewer needs is missing."""
    payload = mcp_server._plan_payload(_plan())

    assert payload["strategy"] == "rally_rotation"
    assert payload["intents"] == [{"symbol": "AAA", "kind": "weight", "value": 0.5}]
    assert payload["latest_prices"] == {"AAA": 100.0}
    assert payload["state"] == {"accrued": 12.0}


def test_desired_orders_reach_the_agent_for_review() -> None:
    """An order-book algorithm's proposal lives entirely in ``desired_orders`` -- ``intents``
    is empty for this shape -- so omitting it showed the agent an empty plan."""
    plan = AlgorithmPlan(
        strategy="options_flip",
        desired_orders=[
            DesiredOrder(
                key="GLD:target",
                request=OrderRequest(
                    symbol="GLD   260918C00380000", action="sell", quantity=1,
                    order_type="limit", limit_price=33.02, time_in_force="gtc",
                    asset_type="option", extra={"position_intent": "sell_to_close"},
                ),
                replace_tolerance=0.02,
            ),
        ],
    )
    payload = mcp_server._plan_payload(plan)

    assert payload["intents"] == []
    assert payload["desired_orders"][0]["key"] == "GLD:target"
    assert payload["desired_orders"][0]["request"]["limit_price"] == 33.02
    assert payload["desired_orders"][0]["request"]["time_in_force"] == "gtc"
    assert payload["desired_orders"][0]["replace_tolerance"] == 0.02


def test_signals_pass_through_whole_for_a_non_rally_shape() -> None:
    """Was a fixed whitelist of Rally Rotation's own fields (score/reason/...), which silently
    emptied an Options Flip signal -- its row carries ``checks``/``estimate``/``contract``
    instead, none of which that list named."""
    plan = AlgorithmPlan(
        strategy="options_flip",
        signals={"GLD": {"state": "held", "headline": "Holding GLD 380 call", "checks": [
            {"label": "Holding VWAP", "ok": True, "value": "above", "blocking": False},
        ], "estimate": {"direction": "call", "contract": "GLD   260918C00380000"}}},
    )
    payload = mcp_server._plan_payload(plan)
    assert payload["signals"]["GLD"]["headline"] == "Holding GLD 380 call"
    assert payload["signals"]["GLD"]["checks"][0]["label"] == "Holding VWAP"
    assert payload["signals"]["GLD"]["estimate"]["contract"] == "GLD   260918C00380000"


def test_get_algorithm_plan_hands_back_a_token_that_places_that_same_plan(monkeypatch) -> None:
    """The two-call seam: the agent reviews a payload and submits a token, and the plan that
    executes is the one this side kept -- never one rebuilt from what came back."""
    fake_server = _build(monkeypatch)
    captured = _placing(monkeypatch)
    monkeypatch.setattr(mcp_server, "run_algorithm", lambda algorithm, config: _plan())

    proposed = fake_server.get_algorithm_plan("rally_rotation")
    assert proposed["status"] == "ok"
    assert proposed["plan_token"]

    assert fake_server.place_orders(proposed["plan_token"])["status"] == "submitted"
    assert captured["plan"].target_weights == {"AAA": 0.5}
    # The opaque ledger survives because it never left: ``execute`` commits the state on the
    # plan it is given, and a round trip through the agent was where it used to be droppable.
    assert captured["plan"].state == {"accrued": 12.0}


def test_a_plan_token_is_single_use(monkeypatch) -> None:
    """A retried tool call must not submit the same batch of orders a second time."""
    fake_server = _build(monkeypatch)
    _placing(monkeypatch)
    token = _token()

    assert fake_server.place_orders(token)["status"] == "submitted"

    replayed = fake_server.place_orders(token)
    assert replayed["status"] == "refused"
    assert "already been used" in replayed["reason"]


def test_an_unknown_plan_token_is_refused(monkeypatch) -> None:
    fake_server = _build(monkeypatch)
    _placing(monkeypatch)

    result = fake_server.place_orders("not-a-real-token")

    assert result["status"] == "refused"
    assert "get_algorithm_plan" in result["reason"]


def test_an_expired_plan_is_refused_rather_than_submitted_stale(monkeypatch) -> None:
    """A plan commits the prices it was built with, so an old one submits stale limit prices."""
    fake_server = _build(monkeypatch)
    _placing(monkeypatch)
    token = plan_cache.stash(_plan(), algorithm="rally_rotation", account_id="paper", ttl_seconds=1)
    # Reach past the clock rather than sleeping: the expiry is a timestamp comparison.
    monkeypatch.setattr(plan_cache, "_now", lambda: datetime.now(timezone.utc) + timedelta(seconds=30))

    result = fake_server.place_orders(token)

    assert result["status"] == "refused"
    assert "get_algorithm_plan" in result["reason"]


def test_a_deployment_switched_off_after_planning_refuses_the_submission(monkeypatch) -> None:
    """The gap between reviewing and submitting is exactly where a kill decision lands."""
    controls = {"deployments": [_deployment()]}
    fake_server = _build(monkeypatch, controls["deployments"])
    _placing(monkeypatch)
    token = _token()

    # Switched off while the agent was reading the news.
    off = {"deployments": [_deployment(enabled=False)]}
    monkeypatch.setattr(mcp_server, "load_controls", lambda *a, **k: off)
    monkeypatch.setattr(controls_module, "load_controls", lambda *a, **k: off)

    result = fake_server.place_orders(token)

    assert result["status"] == "refused"
    assert "switched off" in result["reason"]


def test_a_plan_is_refused_if_its_algorithm_now_points_at_another_account(monkeypatch) -> None:
    """Quantities are sized against one book's holdings and equity; repointed, they describe an
    account these orders would no longer reach."""
    fake_server = _build(monkeypatch, [_deployment(account_id="schwab2")])
    _placing(monkeypatch)
    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=False, account_id="schwab2"))

    result = fake_server.place_orders(_token(account_id="paper"))

    assert result["status"] == "refused"
    assert "sized against paper" in result["reason"]


def test_skip_holds_a_position_instead_of_selling_it(monkeypatch) -> None:
    """The trap this exists to avoid: under ``target`` mode the intent list *is* the portfolio,
    so dropping a row targets zero -- a veto expressed as a deletion would sell the position the
    agent meant to leave alone."""
    fake_server = _build(monkeypatch)
    captured = _placing(monkeypatch, positions={"AAA": 30.0})

    result = fake_server.place_orders(_token(), [{"op": "skip", "symbol": "AAA"}])

    assert result["status"] == "submitted"
    # Pinned to what is held: target equals current, so the sizer plans no trade at all.
    assert _shares(captured["plan"]) == {"AAA": 30.0}
    assert result["applied_edits"] == [{"op": "skip", "symbol": "AAA", "effect": "held at 30 shares"}]


def test_skip_under_incremental_mode_drops_the_increment(monkeypatch) -> None:
    """Incremental intents are deltas and anything unlisted is genuinely left alone, so here the
    right spelling of "leave it" really is removal."""
    fake_server = _build(monkeypatch)
    captured = _placing(monkeypatch, positions={"AAA": 30.0})
    token = _token(_plan(mode="incremental"))

    result = fake_server.place_orders(token, [{"op": "skip", "symbol": "AAA"}])

    assert result["status"] == "submitted"
    assert captured["plan"].intents == []


def test_exit_closes_a_position_the_plan_wanted_to_keep(monkeypatch) -> None:
    fake_server = _build(monkeypatch)
    captured = _placing(monkeypatch, positions={"AAA": 30.0})

    result = fake_server.place_orders(_token(), [{"op": "exit", "symbol": "AAA"}])

    assert _shares(captured["plan"]) == {"AAA": 0.0}
    assert result["applied_edits"] == [{"op": "exit", "symbol": "AAA", "effect": "closed to zero"}]


def test_exit_under_incremental_mode_sells_what_is_held(monkeypatch) -> None:
    """An incremental target is ``current + value``, so reaching zero means a negative delta."""
    fake_server = _build(monkeypatch)
    captured = _placing(monkeypatch, positions={"AAA": 30.0})

    fake_server.place_orders(_token(_plan(mode="incremental")), [{"op": "exit", "symbol": "AAA"}])

    assert _shares(captured["plan"]) == {"AAA": -30.0}


def test_an_edit_cannot_change_a_size(monkeypatch) -> None:
    """The fence from SOUL.md: the agent judges, it does not size. There is no edit op that
    takes an amount, so a payload trying to smuggle one is refused on the op name."""
    fake_server = _build(monkeypatch)
    _placing(monkeypatch, positions={"AAA": 30.0})

    result = fake_server.place_orders(_token(), [{"op": "set_weight", "symbol": "AAA", "value": 0.35}])

    assert result["status"] == "refused"
    assert "Unknown edit op" in result["reason"]


def test_edits_are_refused_for_an_order_book_plan(monkeypatch) -> None:
    """Reconciliation cancels whatever is not in ``desired_orders``, so removing a leg there
    cancels a resting stop rather than declining an action. No safe partial edit exists."""
    fake_server = _build(monkeypatch, [_deployment(strategy="options_flip")])
    _placing(monkeypatch)
    plan = AlgorithmPlan(
        strategy="options_flip",
        desired_orders=[
            DesiredOrder(key="GLD:stop", request=OrderRequest(symbol="GLD", action="sell", quantity=1)),
        ],
    )

    result = fake_server.place_orders(_token(plan), [{"op": "skip", "symbol": "GLD"}])

    assert result["status"] == "refused"
    assert "cancels a resting order" in result["reason"]


def test_an_edit_naming_an_unrelated_symbol_is_refused(monkeypatch) -> None:
    """Almost always a mistyped ticker, and a silently ignored veto is the kind that gets
    noticed after the order fills."""
    fake_server = _build(monkeypatch)
    _placing(monkeypatch, positions={"AAA": 30.0})

    result = fake_server.place_orders(_token(), [{"op": "skip", "symbol": "ZZZ"}])

    assert result["status"] == "refused"
    assert "neither proposed by this plan nor held" in result["reason"]


def test_conflicting_edits_for_one_symbol_are_refused(monkeypatch) -> None:
    fake_server = _build(monkeypatch)
    _placing(monkeypatch, positions={"AAA": 30.0})

    result = fake_server.place_orders(
        _token(), [{"op": "skip", "symbol": "AAA"}, {"op": "exit", "symbol": "AAA"}]
    )

    assert result["status"] == "refused"
    assert "Conflicting edits" in result["reason"]


def test_placing_without_edits_reports_an_empty_edit_list(monkeypatch) -> None:
    """So a wrap can tell "the agent vetoed nothing" from "the field is missing"."""
    fake_server = _build(monkeypatch)
    _placing(monkeypatch)

    assert fake_server.place_orders(_token())["applied_edits"] == []


def test_place_orders_refuses_a_deployment_the_scheduler_drives(monkeypatch) -> None:
    """The invariant: one origin per enabled deployment, never both.

    A deployment with a cron is the scheduler's. Letting an agent submit for it too is
    two live origins on one algorithm, which is what this gate exists to prevent.
    """
    fake_server = _build(monkeypatch, [_deployment(cron="30 9 * * 1-5")])

    result = fake_server.place_orders(_token())

    assert result["status"] == "refused"
    assert "scheduler places its orders" in result["reason"].replace(", so the ", " ")


def test_place_orders_refuses_a_switched_off_deployment(monkeypatch) -> None:
    fake_server = _build(monkeypatch, [_deployment(enabled=False)])

    result = fake_server.place_orders(_token())

    assert result["status"] == "refused"
    assert "switched off" in result["reason"]


def test_place_orders_refuses_an_algorithm_with_no_deployment(monkeypatch) -> None:
    """Only ``dca`` is deployed, and the stashed plan is Rally Rotation's."""
    fake_server = _build(monkeypatch, [_deployment(algorithm="dca")])

    result = fake_server.place_orders(_token())

    assert result["status"] == "refused"
    assert "not deployed" in result["reason"]


def test_the_algorithm_alone_names_the_deployment(monkeypatch) -> None:
    """No disambiguating argument, because ambiguity is unrepresentable.

    One algorithm has at most one account, so the plan's own algorithm resolves the account it
    submits against. This used to take a second addressing argument precisely because two
    deployments could share a strategy, and guessing between them was guessing between a paper
    account and a real one.
    """
    fake_server = _build(monkeypatch, [_deployment(account_id="schwab")])
    # Stubbed, so it reports ``Config``'s own default account whatever it is handed; the stash
    # matches that, leaving the authorisation gate as the only thing under test here.
    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=True))

    result = fake_server.place_orders(_token())

    # Got past the authorisation gate on the algorithm alone, and stopped by the kill switch.
    assert result["status"] == "skipped"


def test_place_orders_uses_the_deployments_account_not_the_default(monkeypatch) -> None:
    """The bug this replaced: get_config() with no account_id resolves the *default* account,
    so an algorithm bound to a live account could have had its orders sent to a paper one."""
    fake_server = _build(monkeypatch, [_deployment(account_id="schwab2")])
    seen: dict = {}

    def fake_get_config(**kwargs):
        seen.update(kwargs)
        return Config(kill_switch=True)

    monkeypatch.setattr(mcp_server, "get_config", fake_get_config)
    fake_server.place_orders(_token())

    assert seen["account_id"] == "schwab2"


def test_get_algorithm_plan_runs_for_a_scheduled_deployment_but_says_it_cannot_trade(monkeypatch) -> None:
    """Computing a proposal is a read, like a backtest, so it is not gated -- but the agent is
    told plainly that acting on it will be refused."""
    fake_server = _build(monkeypatch, [_deployment(cron="30 9 * * 1-5")])
    monkeypatch.setattr(mcp_server, "get_config", lambda **kw: Config(kill_switch=True))

    result = fake_server.get_algorithm_plan("rally_rotation")

    assert result["can_place_orders"] is False
    assert result["status"] == "error"  # stopped by the kill switch, not by the deployment


def test_list_algorithms_reports_what_the_agent_may_drive(monkeypatch) -> None:
    fake_server = _build(
        monkeypatch,
        [
            _deployment(algorithm="rally_rotation"),
            _deployment(algorithm="bursty_dca", cron="30 9 * * 1-5"),
            _deployment(algorithm="options_flip", enabled=False),
        ],
    )

    rows = {row["algorithm"]: row for row in fake_server.list_algorithms()["algorithms"]}

    assert rows["rally_rotation"]["can_place_orders"] is True
    assert rows["rally_rotation"]["driven_by"] == "mcp"
    assert rows["bursty_dca"]["can_place_orders"] is False
    assert rows["bursty_dca"]["driven_by"] == "schedule"
    assert rows["options_flip"]["can_place_orders"] is False


def test_list_algorithms_reports_undeployed_ones_too(monkeypatch) -> None:
    """The registry is the list, not the config.

    An algorithm deployed nowhere used to be invisible over MCP, so an agent had no way to
    learn it existed -- and the ids are what every other algorithm tool takes.
    """
    fake_server = _build(monkeypatch, [_deployment(algorithm="rally_rotation")])

    rows = {row["algorithm"]: row for row in fake_server.list_algorithms()["algorithms"]}

    assert {"bursty_dca", "rally_rotation", "options_flip"} <= set(rows)
    assert rows["bursty_dca"]["deployed"] is False
    assert rows["bursty_dca"]["account_id"] == ""
    assert rows["bursty_dca"]["can_place_orders"] is False
    assert "not deployed" in rows["bursty_dca"]["reason"]


def test_one_deployment_is_driven_by_exactly_one_origin() -> None:
    """The scheduler and the MCP tools ask the same function, so the two can never both say yes."""
    for cron in ("*/15 9-15 * * 1-5", "0 11 * * 1-5", "30 9 * * 1", ""):
        deployment = _deployment(cron=cron)
        schedule_ok = not controls_module.deployment_refusal(deployment, controls_module.ORIGIN_SCHEDULE)
        mcp_ok = not controls_module.deployment_refusal(deployment, controls_module.ORIGIN_MCP)
        assert schedule_ok != mcp_ok, cron

    off = _deployment(enabled=False)
    assert controls_module.deployment_refusal(off, controls_module.ORIGIN_SCHEDULE)
    assert controls_module.deployment_refusal(off, controls_module.ORIGIN_MCP)


def test_account_positions_carries_both_halves_without_losing_an_error() -> None:
    """The balances and the computed figures are two reads, merged into one answer.

    A plain ``{**positions, **analytics}`` let the second dict's empty ``error`` erase the
    first's real one, so an unreachable broker reported null balances and said nothing about
    why -- the one case where an agent most needs to be told.
    """
    from src.api.payloads.accounts import _blank_analytics

    positions = {
        "account_id": "schwab3", "equity": None, "cash": None, "rows": [],
        "error": "Schwab API returned 401: token expired",
    }
    analytics = _blank_analytics("schwab3")

    carried = {
        key: value for key, value in analytics.items()
        if key not in ("error", "state", "computed_at", "account_id", "dividend_rows")
    }
    errors = [text for text in (positions.get("error"), analytics.get("error")) if text]
    merged = {**positions, **carried, "error": "; ".join(errors)}

    assert "401" in merged["error"], "a positions failure must survive the merge"
    # And the analytics keys are still carried, so the shape does not change with the failure.
    assert "realized_pl" in merged and "dividend_pl" in merged
    assert "state" not in merged, "a bare 'state' beside balances reads as the account's"


def _price_tool(monkeypatch, bars: list[tuple[str, float, int]] | None = None):
    """The tool with the bar store faked as ``[(stamp, close, interval_minutes), ...]``.

    The fake picks the nearest bar the way the real query's ``ORDER BY abs(...)`` does, so a
    test exercises the tool's contract rather than DuckDB's arithmetic.
    """
    import pandas as pd
    from src.data import duckdb_store

    rows = list(bars or [])

    def read_closest_bar(symbol, at, **kwargs):
        if not rows:
            return None
        target = pd.Timestamp(at)
        stamp, close, interval = min(
            rows, key=lambda row: abs((pd.Timestamp(row[0]) - target).total_seconds())
        )
        return {"timestamp": pd.Timestamp(stamp), "open": close, "close": close,
                "interval_minutes": interval}

    monkeypatch.setattr(mcp_server, "get_config", lambda *a, **k: Config())
    monkeypatch.setattr(duckdb_store, "read_closest_bar", read_closest_bar)
    return _build(monkeypatch).get_price


def test_price_answers_with_the_nearest_bar(monkeypatch) -> None:
    tool = _price_tool(monkeypatch, [
        ("2026-07-15T13:40:00+00:00", 120.7291, 5),
        ("2026-07-15T15:30:00+00:00", 118.63, 5),
    ])

    answer = tool("USO", "2026-07-15T15:35:00Z")

    assert answer["status"] == "ok"
    assert answer["price"] == 118.63
    assert answer["as_of"].startswith("2026-07-15T15:30")


def test_price_rounds_to_four_places(monkeypatch) -> None:
    tool = _price_tool(monkeypatch, [("2026-07-15T15:30:00+00:00", 120.72913456, 5)])

    assert tool("USO", "2026-07-15T15:30:00Z")["price"] == 120.7291


def test_a_time_of_day_gets_an_intraday_price_where_the_store_has_one(monkeypatch) -> None:
    """The point of searching both grids at once.

    A daily close is struck at the session's end, so answering "what was it at half past ten"
    with one hands back a price from hours after the moment asked about -- silently, since the
    number is plausible and nothing about it says it is six hours late.
    """
    tool = _price_tool(monkeypatch, [
        ("2026-07-15T13:40:00+00:00", 120.73, 5),
        ("2026-07-15T15:30:00+00:00", 118.63, 5),
        ("2026-07-15T21:00:00+00:00", 121.38, 1440),
    ])

    morning = tool("USO", "2026-07-15T13:45:00Z")
    afternoon = tool("USO", "2026-07-15T15:35:00Z")

    assert morning["price"] == 120.73
    assert afternoon["price"] == 118.63, "a different time is a different price"


def test_an_old_date_falls_through_to_the_daily_series(monkeypatch) -> None:
    """Intraday history is shallower than daily history, and the caller neither chooses nor
    needs to know which grid answered -- ``as_of`` says what was struck."""
    tool = _price_tool(monkeypatch, [("2025-09-15T20:00:00+00:00", 74.23, 1440)])

    answer = tool("USO", "2025-09-15T14:30:00Z")

    assert answer["status"] == "ok" and answer["price"] == 74.23
    assert answer["as_of"].startswith("2025-09-15")


def test_price_defaults_to_now(monkeypatch) -> None:
    from datetime import datetime as _dt, timezone as _tz

    recent = _dt.now(_tz.utc).isoformat()
    tool = _price_tool(monkeypatch, [("2020-01-02T00:00:00+00:00", 10.0, 1440), (recent, 155.49, 5)])

    assert tool("USO")["price"] == 155.49, "no timestamp means now, not the oldest bar"


def test_price_normalises_the_symbol(monkeypatch) -> None:
    tool = _price_tool(monkeypatch, [("2026-07-15T15:30:00+00:00", 601.0, 5)])

    assert tool("  uso ")["symbol"] == "USO"


def test_price_reports_an_unpriceable_symbol_rather_than_failing(monkeypatch) -> None:
    tool = _price_tool(monkeypatch, [])

    answer = tool("NOSUCH")

    assert answer["status"] == "error"
    assert answer["price"] is None, "never a zero someone might do arithmetic with"
    assert "NOSUCH" in answer["error"]


def test_price_refuses_an_empty_symbol_or_an_unreadable_date(monkeypatch) -> None:
    tool = _price_tool(monkeypatch, [("2026-07-15T15:30:00+00:00", 601.0, 5)])

    assert tool("")["status"] == "error"
    unreadable = tool("USO", "last Tuesday")
    assert unreadable["status"] == "error"
    assert "ISO" in unreadable["error"], "say what a readable date looks like"


def test_price_survives_a_dead_bar_store(monkeypatch) -> None:
    """Reported as itself, not as "no such symbol" -- the two call for different actions."""
    from src.data import duckdb_store

    def explode(*args, **kwargs):
        raise RuntimeError("bar store locked")

    monkeypatch.setattr(mcp_server, "get_config", lambda *a, **k: Config())
    monkeypatch.setattr(duckdb_store, "read_closest_bar", explode)

    answer = _build(monkeypatch).get_price("USO")

    assert answer["status"] == "error"
    assert "bar store locked" in answer["error"]
def _positions_tool(monkeypatch, *, positions: dict, analytics: dict):
    """The tool with both halves of the account read faked, so no test reaches a broker."""
    from src.api.payloads import accounts as accounts_module

    monkeypatch.setattr(accounts_module, "positions_payload", lambda *a, **k: positions)
    monkeypatch.setattr(accounts_module, "account_analytics_payload", lambda *a, **k: analytics)
    return _build(monkeypatch).get_account_positions


def test_positions_does_not_carry_the_dividend_rows(monkeypatch) -> None:
    """Forty distributions rode along on every call to report one number nobody read.

    The merge carried everything the analytics payload had except a named few, so
    ``dividend_rows`` -- each row a symbol, a date, an amount and the broker's own description
    string -- arrived on every account read. Nothing documents it and nothing uses it: the
    summed ``dividend_pl`` beside it is the whole of what this tool promises. The dashboard
    still renders the rows, reading the analytics payload directly rather than through here.
    """
    from src.api.payloads.accounts import _blank_analytics

    analytics = {
        **_blank_analytics("schwab1"),
        "dividend_pl": 41.2,
        "dividend_rows": [
            {"symbol": "SPYM", "date": "2026-03-20", "amount": 20.6,
             "description": "CASH DIVIDEND ON 40 SHARES AT 0.515 PER SHARE"},
        ] * 40,
    }
    answer = _positions_tool(
        monkeypatch,
        positions={"account_id": "schwab1", "equity": 100.0, "rows": [], "error": ""},
        analytics=analytics,
    )()

    assert "dividend_rows" not in answer
    assert answer["dividend_pl"] == 41.2, "the summed figure is the part that was ever wanted"


def test_positions_rounds_money_to_cents_and_ratios_finely_enough_to_survive(monkeypatch) -> None:
    """A ratio rounded like money is a day's move erased.

    ``day_pl_percent`` is a fraction: at two decimals a 0.13% day reads as ``0.0``, which is
    precisely the figure the brief leads with. Money rounds to the cent, ratios to four places.
    """
    from src.api.payloads.accounts import _blank_analytics

    answer = _positions_tool(
        monkeypatch,
        positions={
            "account_id": "alpaca1", "equity": 10525.339999999998, "cash": 1.005,
            "day_pl": 13.679999999999836, "day_pl_percent": 0.0013012345, "error": "",
            "rows": [{"symbol": "USO", "qty": 40.0, "unrealized_pl": 66.39599999999973,
                      "unrealized_plpc": 0.010789781591263647, "day_pl": None,
                      "day_pl_percent": None}],
        },
        analytics=_blank_analytics("alpaca1"),
    )()

    assert answer["equity"] == 10525.34
    assert answer["day_pl"] == 13.68
    assert answer["day_pl_percent"] == 0.0013, "a 0.13% day must not round away to nothing"
    row = answer["rows"][0]
    assert row["unrealized_pl"] == 66.4
    assert row["unrealized_plpc"] == 0.0108
    # Null is a claim this tool makes on purpose: the broker did not say where the session
    # started. Dropping the key would leave a reader unable to tell that from "flat".
    assert row["day_pl"] is None and "day_pl" in row


def _orders_tool(monkeypatch, rows: list[dict]):
    from src.api.payloads import accounts as accounts_module

    monkeypatch.setattr(
        accounts_module, "account_activity_payload",
        lambda *a, **k: {"account_id": "alpaca1", "rows": rows, "error": ""},
    )
    return _build(monkeypatch).get_account_orders


def test_orders_keep_a_refusal_and_drop_what_does_not_apply(monkeypatch) -> None:
    """The reason is the line the brief is built on; the empty prices are the weight.

    A market order has no limit and no stop, and an unfilled one no fill price. Sent as nulls
    they cost a reader three keys to learn what ``order_type`` and ``status`` already said. The
    broker's refusal is the opposite: it is the one thing a rejected order is read for, and it
    was not being carried at all.
    """
    answer = _orders_tool(monkeypatch, [
        {"symbol": "USO260916C00142000", "side": "sell", "status": "rejected", "qty": 2.0,
         "filled_qty": 0.0, "filled_avg_price": None, "order_type": "market",
         "limit_price": None, "stop_price": None, "submitted_at": "2026-09-15T14:31:02Z",
         "reason": "account not eligible to trade uncovered option contracts"},
        {"symbol": "SPYM", "side": "buy", "status": "filled", "qty": 2.0, "filled_qty": 2.0,
         "filled_avg_price": 89.739999999, "order_type": "limit", "limit_price": 89.75,
         "stop_price": None, "submitted_at": "2026-09-15T13:32:00Z", "reason": ""},
    ])()

    refused, filled = answer["rows"]
    assert refused["reason"].startswith("account not eligible")
    for absent in ("limit_price", "stop_price", "filled_avg_price"):
        assert absent not in refused, f"a market order has no {absent} to report"
    # Nobody refused the second one, so it says nothing about why.
    assert "reason" not in filled
    assert "stop_price" not in filled
    assert filled["limit_price"] == 89.75 and filled["filled_avg_price"] == 89.74
    # A real zero is a fact, not an absence: this order has genuinely filled nothing yet.
    assert refused["filled_qty"] == 0.0


def test_orders_surface_a_rejection_the_broker_never_recorded(monkeypatch) -> None:
    """The rows that were invisible, and the most important ones in the view.

    When ``submit_order`` raises -- "not eligible to trade uncovered option contracts", a halt,
    no buying power -- the broker never creates an order. It has no id and appears in no order
    feed, so reading the broker alone reports a clean session for an account whose every order
    was thrown out. Only the bot's journal witnessed it.
    """
    from src.api.payloads import accounts as accounts_module

    monkeypatch.setattr(accounts_module, "get_account_broker_type", lambda *a, **k: "schwab")
    monkeypatch.setattr(
        accounts_module, "_brokerage_activity",
        lambda *a, **k: {"rows": [{
            "symbol": "SPYM", "side": "buy", "status": "filled", "qty": 2.0, "filled_qty": 2.0,
            "filled_avg_price": 89.74, "order_type": "limit", "limit_price": 89.75,
            "stop_price": None, "submitted_at": "2026-09-15T13:32:00+00:00", "reason": "",
        }], "error": ""},
    )
    monkeypatch.setattr(accounts_module, "load_order_journal", lambda **k: [{
        "symbol": "USO260916C00142000", "side": "sell", "status": "rejected", "quantity": 2.0,
        "order_type": "market", "limit_price": 0.0, "stop_price": 0.0, "order_id": "",
        "submitted_at": "2026-09-15T14:31:02+00:00",
        "reason": "account not eligible to trade uncovered option contracts",
    }, {
        # Already at the broker, so the broker's own row is the better record of the two.
        "symbol": "SPYM", "side": "buy", "status": "rejected", "quantity": 2.0,
        "order_type": "limit", "order_id": "schwab-1001", "submitted_at": "2026-09-15T13:32:00+00:00",
        "reason": "should not appear -- this one has an id",
    }])

    payload = accounts_module.account_activity_payload("schwab1", limit=20)
    rows = payload["rows"]

    assert [row["symbol"] for row in rows] == ["USO260916C00142000", "SPYM"], "newest first"
    assert rows[0]["reason"].startswith("account not eligible")
    assert all(row["reason"] != "should not appear -- this one has an id" for row in rows)


def test_a_journal_rejection_reaches_the_mcp_tool_with_its_reason(monkeypatch) -> None:
    """End to end: the compaction keeps a real reason and still drops the empty ones."""
    from src.api.payloads import accounts as accounts_module

    monkeypatch.setattr(accounts_module, "get_account_broker_type", lambda *a, **k: "schwab")
    monkeypatch.setattr(accounts_module, "_brokerage_activity", lambda *a, **k: {"rows": [], "error": ""})
    monkeypatch.setattr(accounts_module, "load_order_journal", lambda **k: [{
        "symbol": "USO", "side": "buy", "status": "rejected", "quantity": 1.0,
        "order_type": "market", "order_id": "", "submitted_at": "2026-09-15T14:31:02+00:00",
        "reason": "insufficient buying power",
    }])

    row = _build(monkeypatch).get_account_orders("schwab1")["rows"][0]

    assert row["reason"] == "insufficient buying power"
    assert "limit_price" not in row, "a market order still drops what does not apply"


def test_list_algorithms_describes_each_one_in_a_line(monkeypatch) -> None:
    """The one-line form, not the explainer's paragraph.

    This lists every algorithm at once; ``rally_rotation``'s summary alone is 787 characters,
    more than the entire payload was before. A reader needs enough to say which algorithm a
    report is about, not how it sizes.
    """
    rows = _build(monkeypatch).list_algorithms()["algorithms"]
    by_id = {row["algorithm"]: row for row in rows}

    assert by_id["rally_rotation"]["description"], "every algorithm carries one"
    for row in rows:
        assert len(row["description"]) < 200, f'{row["algorithm"]} is a paragraph, not a line'


def _listing(monkeypatch, *, rows, headline=None, ready=True):
    from src.api.payloads import accounts as accounts_module

    monkeypatch.setattr(
        accounts_module, "accounts_payload",
        lambda *a, **k: {"default": "alpaca1", "rows": rows},
    )
    calls = []

    def fake_headline(account_id):
        calls.append(account_id)
        return dict(headline or {})

    monkeypatch.setattr(accounts_module, "account_headline", fake_headline)
    return _build(monkeypatch).list_accounts, calls


def test_list_accounts_carries_a_headline_per_account(monkeypatch) -> None:
    """So an agent can tell which accounts are worth a detail call before making any."""
    tool, _ = _listing(
        monkeypatch,
        rows=[{"id": "alpaca1", "label": "Alpaca Paper", "broker": "alpaca",
               "deployments": ["options_flip"], "credentials_ready": True}],
        headline={"equity": 10525.339999, "cash": 1.0, "day_pl": 560.12345,
                  "day_pl_percent": 0.0013012345, "total_pl": 1200.0,
                  "realized_pl": 41.2, "positions": 3, "orders_today": 4, "error": ""},
    )

    row = tool()["accounts"][0]

    assert row["label"] == "Alpaca Paper" and row["positions"] == 3 and row["orders_today"] == 4
    assert row["equity"] == 10525.34, "money rounded like every other tool rounds it"
    assert row["day_pl_percent"] == 0.0013, "a 0.13% day must not round away to nothing"
    assert "error" not in row, "an empty error is dropped rather than sent"


def test_list_accounts_does_not_ask_a_broker_it_has_no_keys_for(monkeypatch) -> None:
    """A guaranteed error per row tells the reader nothing ``credentials_ready`` did not."""
    tool, calls = _listing(
        monkeypatch,
        rows=[{"id": "schwab9", "label": "Unwired", "broker": "schwab",
               "deployments": [], "credentials_ready": False}],
    )

    row = tool()["accounts"][0]

    assert calls == [], "no broker read attempted"
    assert row["credentials_ready"] is False
    assert "equity" not in row


def test_a_headline_reports_nulls_and_an_error_rather_than_zeros(monkeypatch) -> None:
    """"We could not ask" and "nothing happened" must not look alike in a summary.

    Only one of the two is news, and a zero day P/L beside a zero order count reads as a quiet
    session rather than as a broker that never answered.
    """
    from src.api.payloads import accounts as accounts_module

    monkeypatch.setattr(accounts_module, "positions_payload", lambda *a, **k: {
        "equity": None, "cash": None, "day_pl": None, "day_pl_percent": None,
        "total_pl": None, "rows": [], "error": "Schwab API returned 401: token expired",
    })
    monkeypatch.setattr(accounts_module, "account_analytics_payload", lambda *a, **k: {"realized_pl": None})
    monkeypatch.setattr(accounts_module, "account_activity_payload", lambda *a, **k: {"rows": [], "error": ""})

    row = accounts_module.account_headline("schwab3")

    assert row["equity"] is None and row["day_pl"] is None
    assert "401" in row["error"]


def test_a_headline_survives_one_source_dying(monkeypatch) -> None:
    """Three independent reads. A broker slow on positions still contributes its order count."""
    from src.api.payloads import accounts as accounts_module

    def explode(*a, **k):
        raise RuntimeError("positions timed out")

    monkeypatch.setattr(accounts_module, "positions_payload", explode)
    monkeypatch.setattr(accounts_module, "account_analytics_payload", lambda *a, **k: {"realized_pl": 88.0})
    monkeypatch.setattr(accounts_module, "account_activity_payload", lambda *a, **k: {
        "rows": [{"symbol": "USO"}, {"symbol": "SPYM"}], "error": "",
    })

    row = accounts_module.account_headline("schwab1")

    assert row["orders_today"] == 2 and row["realized_pl"] == 88.0
    assert row["equity"] is None
    assert "positions timed out" in row["error"]
