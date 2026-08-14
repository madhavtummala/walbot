from __future__ import annotations

from pathlib import Path

from src.api.web_app import controls_payload, dca_payload, status_payload, universe_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_status_payload_redacts_secret_values() -> None:
    payload = status_payload()

    assert isinstance(payload["config"]["alpaca_api_key"], bool)
    assert isinstance(payload["config"]["alpaca_api_secret"], bool)
    assert isinstance(payload["config"]["alpha_vantage_api_key"], bool)


def test_universe_payload_returns_configured_rows() -> None:
    payload = universe_payload()

    assert payload["count"] > 0
    assert {"symbol", "name", "bucket", "tradable", "enabled"} <= set(payload["rows"][0])


def test_dca_payload_returns_plan_and_preview_shape() -> None:
    payload = dca_payload()

    assert "plan" in payload
    assert "available" in payload
    assert "preview" in payload
    assert {"max_item_amount", "buy", "sell"} <= set(payload["plan"])


def test_controls_payload_returns_switches() -> None:
    payload = controls_payload()

    assert {"algorithm_enabled", "options_trading_enabled"} <= set(payload["controls"])


def _assets():
    return (
        (PROJECT_ROOT / "web/static/app.js").read_text(encoding="utf-8"),
        (PROJECT_ROOT / "web/static/app.css").read_text(encoding="utf-8"),
        (PROJECT_ROOT / "web/index.html").read_text(encoding="utf-8"),
    )


def test_shell_is_a_sidebar_beside_scrolling_content() -> None:
    app_js, app_css, index_html = _assets()

    for element in ('id="sidebar"', 'id="algorithmNav"', 'id="watchlist"', 'id="content"', 'id="navToggle"'):
        assert element in index_html, element
    assert "function renderSidebar" in app_js
    assert ".shell {" in app_css
    # Fixed sidebar beside scrolling content.
    assert "grid-template-columns: 232px minmax(0, 1fr);" in app_css


def test_routing_gives_each_algorithm_a_full_page_with_lifecycle_tabs() -> None:
    app_js, _, _ = _assets()

    assert "function currentRoute" in app_js
    assert 'window.addEventListener("hashchange", render)' in app_js
    # Deploy is an action in the page header, not a section of its own.
    for tab in ("overview", "signals", "tune", "backtest"):
        assert f'id: "{tab}"' in app_js, tab
    assert 'id: "deploy"' not in app_js
    assert "renderDeployTab" not in app_js
    for view in ("renderOverviewTab", "renderTuneTab", "renderBacktestTab", "renderSignalsTab"):
        assert f"function {view}" in app_js, view
    assert app_js.index('id: "overview"') < app_js.index('id: "signals"') < app_js.index('id: "tune"')
    assert "const DEFAULT_TAB = TABS[0].id;" in app_js
    # An unknown tab falls back rather than rendering an empty page.
    assert "TABS.some((tab) => tab.id === parts[2]) ? parts[2] : DEFAULT_TAB" in app_js


def test_overview_explains_the_algorithm_and_lists_what_it_traded() -> None:
    """Description, facts, and the bot's own order journal -- no broker holdings."""
    app_js, _, _ = _assets()

    tab = app_js[app_js.index("function renderOverviewTab"):app_js.index("function renderSignalsTab")]
    assert "How it works" in tab
    assert "At a glance" in tab
    assert "strategy.horizon" in tab and "strategy.risk" in tab
    assert "Orders this algorithm placed" in tab
    assert "ensureAlgorithmActivity(strategy.key)" in tab
    # Blended broker P/L is not attributable to one algorithm, so it stays off this page.
    assert "day_pl" not in tab and "total_pl" not in tab


def test_one_algorithm_deploys_to_one_account_handled_in_the_header() -> None:
    app_js, app_css, _ = _assets()

    assert "function deploymentFor" in app_js
    assert 'id="deployTargetSelect"' in app_js
    assert ".deployControl {" in app_css
    # Every account is offered to every algorithm.
    assert "const options = accountRows();" in app_js


def test_controls_are_not_stretched_by_the_base_button_rule() -> None:
    _, app_css, _ = _assets()

    # The base button rule sets min-height: 42px; .ctl has to override it or icon buttons
    # render far taller than the selects beside them.
    assert "min-height: 30px;" in app_css
    assert ".ctl.powerButton,\n.ctl.removeBinding {" in app_css
    # Armed deployments must be visually distinct, not just aria-pressed.
    assert ".ctl.powerButton.on {" in app_css
    # Form tuning gets the full page width rather than a 74px value column.
    assert ".tuneBody .configField {" in app_css
    assert "grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));" in app_css


def test_sidebar_carries_a_configurable_watchlist() -> None:
    app_js, app_css, index_html = _assets()

    assert 'id="watchlist"' in index_html
    assert 'id="addWatchButton"' in index_html
    assert 'id="targetNav"' not in index_html
    assert 'id="globalStats"' not in index_html
    assert "function renderWatchlist" in app_js
    assert "function addWatchTicker" in app_js
    assert "function removeWatchTicker" in app_js
    assert "/api/watchlist" in app_js
    assert ".watchRow {" in app_css


def test_watchlist_round_trips_through_the_state_store() -> None:
    from src.api.api_payloads import DEFAULT_WATCHLIST, save_watchlist_payload, watchlist_payload
    from src.data.state_store import ephemeral_state

    with ephemeral_state():
        assert [row["symbol"] for row in watchlist_payload()["rows"]] == DEFAULT_WATCHLIST
        saved = save_watchlist_payload(["spy", " qqq ", "SPY", ""])
        # Upper-cased, de-duplicated, blanks dropped.
        assert saved["symbols"] == ["SPY", "QQQ"]

    with ephemeral_state():
        import pytest

        with pytest.raises(ValueError):
            save_watchlist_payload("SPY")


def test_broker_holdings_and_orders_live_on_the_account_page() -> None:
    """The broker reports per account and cannot attribute a fill to an algorithm.

    Hand-placed orders land in the same feed, so showing it under one algorithm would claim
    an attribution that does not exist.
    """
    app_js, app_css, index_html = _assets()

    assert '{ page: "account"' in app_js
    assert "function renderAccountPage" in app_js
    page = app_js[app_js.index("function renderAccountPage"):app_js.index("function accountPositionsTable")]
    assert "day_pl" in page and "total_pl" in page
    assert "including orders you placed yourself" in page
    # The algorithm page links out to the account rather than reproducing its numbers.
    assert '"#/account/' in app_js
    assert 'id="accountNav"' in index_html
    # Dense repeated rows render as tables, not stacked cards.
    for builder in ("function accountPositionsTable", "function accountOrdersTable"):
        assert builder in app_js, builder
    assert ".dataTable {" in app_css
    # Adding or deleting accounts is still config, not a dashboard action.
    for gone in ("deleteTarget", "addTarget"):
        assert gone not in app_js, gone


def test_every_unbounded_list_panel_scrolls_inside_its_card() -> None:
    """Positions, broker orders and the bot's own journal all grow without bound."""
    app_js, app_css, _ = _assets()

    assert app_js.count('class="tableWrap is-scroll"') == 3
    assert ".tableWrap.is-scroll {" in app_css
    # One shared cap, so signals and the tables cut off at the same height.
    assert "--panel-scroll: min(46vh, 420px);" in app_css
    assert app_css.count("max-height: var(--panel-scroll);") == 2
    # A scrolling table keeps its header visible.
    assert ".tableWrap.is-scroll thead th {" in app_css


def test_one_colour_rule_drives_algorithms_accounts_and_the_bot() -> None:
    """Green = on and on a clock. Orange = on but agent-driven. Dark = off.

    Three places used to decide this independently, and "deployed but paused" rendered amber
    -- indistinguishable from "the MCP agent is driving it".
    """
    app_js, _, _ = _assets()

    rule = app_js[app_js.index("function deploymentStatus"):app_js.index("function renderSidebar")]
    assert 'normalizeBindingFrequency(deployment.frequency) !== "mcp")) return "live"' in rule
    assert 'if (armed.length) return "idle"' in rule
    assert 'return "off"' in rule
    # Every consumer goes through it rather than re-deriving.
    assert app_js.count("deploymentStatus(") >= 4
    assert "deployments.length ? \"idle\" : \"off\"" not in app_js, "the old per-place reading is gone"


def test_the_footer_reports_every_binding_not_just_the_first() -> None:
    """One scheduler loop per deployment, so a single loop's state is not the bot's state."""
    app_js, _, _ = _assets()

    summary = app_js[app_js.index("function runtimeSummary"):]
    assert "Object.values(bot.bindings || {})" in summary
    assert "loops.filter((loop) => loop.running)" in summary
    # A binding parked on "mcp" is armed but deliberately unscheduled.
    assert 'normalizeBindingFrequency(binding.frequency) === "mcp"' in summary
    assert "state.bot?.algorithm" not in app_js, "the single-loop shortcut is what caused the bug"


def test_pages_stack_their_cards_through_one_spacing_rule() -> None:
    """The account page has no tabs, and drifted to a 0px gap where every other page has 14px."""
    app_js, app_css, _ = _assets()

    assert ".tabBody,\n.pageBody {" in app_css
    assert '<div class="pageBody">' in app_js


def test_algorithm_order_journal_is_recorded_by_the_bot_and_shown_per_algorithm() -> None:
    app_js, _, _ = _assets()

    assert "/api/algorithm-activity?strategy=" in app_js
    assert "function algorithmOrdersTable" in app_js
    assert "Orders this algorithm placed" in app_js
    live_runner = (PROJECT_ROOT / "src/execution/live_runner.py").read_text(encoding="utf-8")
    assert 'record_orders(strategy, config.account_id, outcome["order_results"])' in live_runner


def test_account_picker_is_not_a_link_and_has_no_delete() -> None:
    app_js, _, _ = _assets()

    header = app_js[app_js.index("const deployment = deploymentFor(strategy.key);"):app_js.index("content.innerHTML = `")]
    assert 'id="deployTargetSelect"' in header
    assert "href=" not in header
    assert "data-remove-binding" not in header
    assert "&times;" not in header
    # Changing the picker re-points the deployment rather than needing a deploy button.
    assert "async function setDeploymentAccount" in app_js


def test_backtest_tab_keeps_the_equity_and_order_breakdown() -> None:
    """Turnover, planned vs skipped, peak invested and max exposure -- not just the headline."""
    app_js, _, _ = _assets()

    tab = app_js[app_js.index("function renderBacktestTab"):app_js.index("function renderSignalsTab")]
    for builder in ("backtestCaptionText", "backtestEquityText", "backtestOrderText"):
        assert builder in tab, builder
    assert "cumulative turnover" in app_js
    assert "Ending equity =" in app_js
    assert "peak invested" in app_js
    assert "max exposure" in app_js


def test_backtest_period_is_selectable() -> None:
    app_js, _, _ = _assets()

    assert "const BACKTEST_PERIOD_CHOICES" in app_js
    assert 'id="backtestPeriodSelect"' in app_js
    # Switching period clears cached curves so the chart cannot show the wrong window.
    assert "configureBacktestPeriod(event.target.value);" in app_js


def test_watchlist_can_be_reordered_by_drag() -> None:
    app_js, app_css, _ = _assets()

    assert "function reorderWatchlist" in app_js
    assert "function wireWatchlistDrag" in app_js
    assert 'draggable="true"' in app_js
    # Firefox will not start a drag unless data is set on the transfer.
    assert 'event.dataTransfer.setData("text/plain"' in app_js
    assert ".watchRow.is-dragging {" in app_css


def test_tune_tab_renders_the_right_editor_per_algorithm() -> None:
    app_js, _, _ = _assets()

    # DCA's config is its budgets, so it gets the bubble board; everything else gets a form.
    assert "function renderDcaTuner" in app_js
    assert "function renderConfigForm" in app_js
    assert "if (isDca) renderDcaTuner(host, strategy);" in app_js
    assert "else renderConfigForm(host, strategy);" in app_js
    # Typed widgets, not a raw JSON box.
    assert "function configFieldKind" in app_js
    assert "function collectConfigValues" in app_js
    # Saving tuning invalidates the cached backtest, which is keyed on that tuning.
    assert "delete state.backtests[strategyKey];" in app_js


def test_secrets_never_travel_through_the_accounts_api() -> None:
    """A target references env var names; the key itself stays on the host."""
    from src.api.api_payloads import ACCOUNT_FIELDS, accounts_payload

    assert "api_key" not in ACCOUNT_FIELDS
    assert "api_secret" not in ACCOUNT_FIELDS
    assert {"api_key_env", "api_secret_env"} <= set(ACCOUNT_FIELDS)

    payload = accounts_payload()
    for row in payload["rows"]:
        assert "api_key" not in row
        assert "api_secret" not in row
        assert "credentials_ready" in row


def test_shell_is_mobile_adaptable() -> None:
    app_js, app_css, index_html = _assets()

    assert "width=device-width, initial-scale=1" in index_html
    for width in ("900px", "620px", "430px"):
        assert f"@media (max-width: {width})" in app_css, width
    # The sidebar becomes a drawer rather than stealing width from the content.
    assert "transform: translateX(-100%);" in app_css
    assert ".sidebar.is-open {" in app_css
    assert "function closeNavOnMobile" in app_js
    # 16px inputs stop iOS Safari zooming the viewport on focus.
    assert "font-size: 16px;" in app_css
    assert "overflow-x: hidden;" in app_css


def test_frontend_keeps_the_configured_backtest_period_and_chart() -> None:
    app_js, _, index_html = _assets()

    assert 'let BACKTEST_PERIOD = "4m";' in app_js
    assert "configureBacktestPeriod(statusPayload.config?.backtest_period)" in app_js
    assert "chart-crosshair" in app_js
    assert "backtestPositions(row.positions)" in app_js
    assert "renderUniverseProposalRows" in app_js
    assert "app.js?v=20260813-sidebar" in index_html


def test_options_and_carousel_stay_gone() -> None:
    app_js, app_css, index_html = _assets()

    for marker in ("optionsPowerToggle", "optionsDeck", "cardDeck", "deckArrow", "deckCard"):
        assert marker not in index_html, marker
        assert marker not in app_js, marker
        assert marker not in app_css, marker
    assert "OPTIONS_STRATEGIES" not in app_js
    assert "switchTab" not in app_js


def test_tune_tab_explains_the_algorithm_and_every_knob() -> None:
    app_js, app_css, _ = _assets()

    assert "function explainerCard" in app_js
    assert "How it decides" in app_js
    assert ".formula {" in app_css
    # Each field carries what the knob is and which direction to move it.
    assert "fieldWhat" in app_js and "fieldEffect" in app_js
    assert "renderConfigField(key, value, docs[key])" in app_js


def test_account_page_shows_broker_order_activity_beside_positions() -> None:
    app_js, _, _ = _assets()

    page = app_js[app_js.index("function renderAccountPage"):app_js.index("function accountPositionsTable")]
    assert "Recent orders" in page
    assert "Positions" in page
    assert "ensureActivity(account.id)" in page
    assert "/api/activity?account_id=" in app_js


def test_the_schwab_row_appears_for_a_configured_connector_not_a_completed_consent() -> None:
    """The row is the control that starts consent, so hiding it until connected is backwards."""
    app_js, _, _ = _assets()

    assert "auth?.connector_enabled || auth?.configured" in app_js
    # Clicking it without credentials explains what is missing instead of failing silently.
    connect = app_js[app_js.index("async function connectSchwab"):]
    assert "!state.schwabAuth.configured" in connect
    assert "showToast(state.schwabAuth.detail" in connect


def test_schwab_auth_payload_reports_whether_the_connector_is_wired_up() -> None:
    from src.api.api_payloads import schwab_auth_payload

    payload = schwab_auth_payload()

    assert "connector_enabled" in payload
    # connectors.yaml lists schwab in both ladders.
    assert payload["connector_enabled"] is True


def test_the_bot_pill_describes_the_algorithms_not_the_container() -> None:
    """Running an MCP server alongside the dashboard says nothing about what is switched on.

    The pill used to read "MCP mode" whenever the container was started with --mcp, even with
    every algorithm off. What runs is a per-binding fact: on with a frequency, or on and
    parked on "mcp" awaiting an external request.
    """
    app_js, _, _ = _assets()

    summary = app_js[app_js.index("function runtimeSummary"):]
    assert 'runtime_mode === "mcp"' not in summary
    assert "const armed = bindings().filter((binding) => binding.enabled);" in summary
    assert "const status = deploymentStatus(armed);" in summary
    assert '"Bot off"' in summary
    # The dot carries it; the words only repeated the colour, so they live in the tooltip.
    assert "escapeHtml(runtime.label)" not in app_js
    assert "escapeHtml(runtime.note)" not in app_js
