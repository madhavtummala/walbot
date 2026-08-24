from __future__ import annotations

from pathlib import Path

import re
import pandas as pd

from src.api.web_app import controls_payload, status_payload, universe_payload


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


def test_controls_payload_returns_switches() -> None:
    payload = controls_payload()

    assert {"algorithm_enabled"} <= set(payload["controls"])


def _assets():
    return (
        (PROJECT_ROOT / "web/static/app.js").read_text(encoding="utf-8"),
        (PROJECT_ROOT / "web/static/app.css").read_text(encoding="utf-8"),
        (PROJECT_ROOT / "web/index.html").read_text(encoding="utf-8"),
    )


def test_shell_is_a_sidebar_beside_scrolling_content() -> None:
    app_js, app_css, index_html = _assets()

    for element in ('id="sidebar"', 'id="algorithmNav"', 'id="content"', 'id="navToggle"'):
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
    """Positions, broker orders, received dividends and the bot's own journal all grow
    without bound."""
    app_js, app_css, _ = _assets()

    assert app_js.count('class="tableWrap is-scroll"') == 5
    assert ".tableWrap.is-scroll {" in app_css
    # One shared cap, so signals and the tables cut off at the same height.
    assert "--panel-scroll: min(46vh, 420px);" in app_css
    assert app_css.count("max-height: var(--panel-scroll);") == 1
    # A scrolling table keeps its header visible.
    assert ".tableWrap.is-scroll thead th {" in app_css


def test_one_colour_rule_drives_algorithms_accounts_and_the_bot() -> None:
    """Green = on and on a clock. Orange = on but agent-driven. Dark = off.

    Three places used to decide this independently, and "deployed but paused" rendered amber
    -- indistinguishable from "the MCP agent is driving it".
    """
    app_js, _, _ = _assets()

    rule = app_js[app_js.index("function deploymentStatus"):app_js.index("function renderSidebar")]
    assert "normalizeBindingCron(deployment.cron))) return \"live\"" in rule
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
    assert "!normalizeBindingCron(binding.cron)" in summary
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
    # A <select> keeps focus after its own change event, and render() defers while a control
    # is focused -- so without forcing, the cached payload for the newly chosen window landed
    # in state and was never painted. It only appeared once something else moved focus, which
    # made "Run backtest" look like the only way to change period.
    assert "render({ force: true });" in app_js


def test_a_deferred_repaint_is_not_silently_dropped() -> None:
    """render() skips while a control has focus; the skipped paint still has to happen.

    Otherwise any state that arrives during an interaction -- a cached backtest, a background
    poll -- is applied to state and never reaches the screen.
    """
    app_js, _, _ = _assets()

    assert "renderDeferred = true;" in app_js
    assert 'addEventListener("focusout"' in app_js


def test_tune_tab_renders_the_right_editor_per_algorithm() -> None:
    app_js, _, _ = _assets()

    # Bursty DCA's budgets are its plan and get the bubble board, but they are not all of its
    # config: its regime gate, scaling factor and cap knobs live in the config section like
    # every other algorithm's. It used to get the board *instead of* the parameter form, which
    # left those with no editor at all. It now gets both; everything else gets the form.
    assert "function renderDcaTuner" in app_js
    assert "function renderConfigForm" in app_js
    assert 'if (hasBudgets) renderDcaTuner($("#dcaBoard"), strategy);' in app_js
    assert 'renderConfigForm($("#tuneBody"), strategy);' in app_js
    # The save button is no longer suppressed, because there is now a form to save.
    assert 'hasBudgets ? "" : `<div class="cardActions">' not in app_js
    # Which editor to render is the algorithm's declaration, carried on the config payload --
    # not a list of "the DCA algorithms" kept here in the frontend.
    assert "DCA_ALGORITHM_KEYS" not in app_js
    assert 'state.algorithmConfigs[strategyKey]?.tune_editor === BUDGETS_EDITOR' in app_js
    # Typed widgets, not a raw JSON box.
    assert "function configFieldKind" in app_js
    assert "function collectConfigValues" in app_js
    # Saving tuning invalidates the cached backtest, which is keyed on that tuning.
    assert "delete state.backtests[strategyKey];" in app_js


def test_the_board_edits_the_algorithms_own_plan() -> None:
    """DCA is a normal algorithm with a custom editor, nothing more.

    The plan used to be a per-account section of its own, which had two consequences: ``dca``
    and ``bursty_dca`` shared one budget because the key had no room for the algorithm, and
    the board could be editing one account's plan while the views rendered another's. It is
    ordinary tuning now, so the page you are on decides what you are editing.
    """
    app_js, _, _ = _assets()

    assert "function planStrategyKey" in app_js
    assert "function currentPlan" in app_js
    assert "state.planStrategy = strategy.key;" in app_js
    # No DCA-shaped read or write path survives.
    assert "/api/dca" not in app_js
    assert "state.dca" not in app_js
    # The plan saves through the same endpoint as every other knob.
    save = app_js[app_js.index("async function savePlan"):app_js.index("function shadeColor")]
    assert '"/api/algorithm-config"' in save
    # It has its own editor, so it is not also rendered as a raw JSON field -- and the form's
    # save merges over the loaded config so leaving it out cannot delete it.
    assert "function isPlanField" in app_js
    assert "const merged = { ...(state.algorithmConfigs[strategyKey]?.config || {}), ...values };" in app_js


def test_one_algorithms_bubbles_are_never_written_into_anothers_plan() -> None:
    """``renderDca`` syncs the nodes into the plan before it rebuilds them from it.

    With one plan per account that was harmless, because the board only ever showed one. Now
    that each algorithm has its own, navigating from Bursty DCA to DCA synced the bubbles
    still on screen -- Bursty's budgets -- into DCA's freshly loaded plan, and the next save
    would have persisted them. The nodes record which plan built them so the sync can refuse.
    """
    app_js, _, _ = _assets()

    assert "state.nodesStrategy = planStrategyKey();" in app_js
    sync = app_js[app_js.index("function syncNodesToPlan"):app_js.index("function renderBoard")]
    assert "if (state.nodesStrategy !== planStrategyKey()) return;" in sync
    # And the board starts clean rather than animating the previous algorithm's bubbles.
    tuner = app_js[app_js.index("function renderDcaTuner"):app_js.index("function renderConfigForm")]
    assert "state.nodes = [];" in tuner


def test_signals_and_backtests_still_name_the_account_they_ran_for() -> None:
    """Accrual state stays per (algorithm, account), and the broker is per account, so a view
    still has to say which deployment it describes even though the plan no longer varies."""
    app_js, _, _ = _assets()

    assert "function accountForStrategy" in app_js
    assert "account_id=${encodeURIComponent(account)}" in app_js
    assert "account_id: accountForStrategy(strategyKey)," in app_js


def test_a_budget_can_be_typed_as_well_as_scrolled() -> None:
    """Scrolling is quick but imprecise: it only ever lands on a multiple of WHEEL_STEP, and
    reaching $1,600 from $25 is 63 notches. Selecting a bubble and typing sets it outright."""
    app_js, app_css, index_html = _assets()

    assert 'id="amountEntry"' in index_html
    assert "function showAmountEntry" in app_js
    assert "function commitAmountEntry" in app_js
    # A digit starts the edit and is carried in, because focusing an input mid-keydown does
    # not deliver the keystroke that caused it.
    assert 'event.key === "Enter" || /^[0-9]$/.test(event.key)' in app_js
    # A typed number means what it says, so it is not snapped to the scroll grid.
    assert "function setNodeAmount" in app_js
    assert "#amountEntry {" in app_css


def test_typing_a_budget_shows_it_in_the_bubble_not_in_a_box_over_it() -> None:
    """Same shape as naming a new bubble: the input is invisible but for its caret, and the
    bubble draws what is being typed. A visible box would show the same number twice, and
    would cover the bubble it belongs to.
    """
    app_js, app_css, _ = _assets()

    entry_css = app_css[app_css.index("#amountEntry {"):app_css.index("#amountEntry.show {")]
    assert "color: transparent;" in entry_css
    assert "background: transparent;" in entry_css
    assert "border: 0;" in entry_css
    assert "caret-color: transparent;" in entry_css
    # Only the caret becomes visible, and it sits on the amount label rather than the centre.
    assert "caret-color: var(--ink);" in app_css[app_css.index("#amountEntry.show {"):]
    assert "const AMOUNT_LABEL_DY = 14;" in app_js
    assert "(node.y + AMOUNT_LABEL_DY) * scaleY" in app_js
    # Each keystroke goes through the same write-through path scrolling uses, so the label,
    # the radius and the bucket total all follow the typing.
    assert "function previewAmountEntry" in app_js
    assert '$("#amountEntry")?.addEventListener("input", previewAmountEntry);' in app_js
    preview = app_js[app_js.index("function previewAmountEntry"):app_js.index("function hideAmountEntry")]
    assert "setNodeAmount(node, Number(digits || 0));" in preview
    # Because the preview is already in the plan, Escape has to be an edit of its own.
    assert "function cancelAmountEntry" in app_js
    assert "setNodeAmount(node, edit.originalAmount);" in app_js


def test_the_bubble_board_reports_whether_an_edit_was_saved() -> None:
    """The board has no save button -- it writes on every gesture -- so silence is ambiguous.

    A save that reached the server and one that failed looked identical, which is part of why
    a plan being written to the wrong account went unnoticed.
    """
    app_js, app_css, _ = _assets()

    assert "function planSaveStatusText" in app_js
    assert "function setPlanSaveStatus" in app_js
    assert 'setPlanSaveStatus("saving");' in app_js
    assert 'id="planSaveStatus"' in app_js
    assert ".saveStatus {" in app_css


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
    # That the assets are versioned, not which version they are. Pinning the literal string made
    # this test fail on every legitimate asset change and taught the reader to edit it without
    # looking -- which is worse than not asserting at all. Both files must carry the *same* stamp,
    # since a page that loads new CSS against old JS is the failure the query string exists to
    # prevent.
    css_version = re.search(r"app\.css\?v=([\w.-]+)", index_html)
    js_version = re.search(r"app\.js\?v=([\w.-]+)", index_html)
    assert css_version and js_version, "static assets must carry a cache-busting version"
    assert css_version.group(1) == js_version.group(1)


def test_the_backtest_chart_marks_where_the_strategy_traded() -> None:
    """The equity line alone cannot say. The replay emits a row for every step whether or not it
    ran, so a quiet stretch and a rotation that netted out are the same shape."""
    app_js, app_css, _ = _assets()

    # Read off the ``trades`` map the payload already carries -- no new backend plumbing.
    assert "backtestTrades(row)" in app_js
    assert "renderTradeMarkers(svg, rows" in app_js
    # Buys below the curve, sells above, and a step that did both is one amber mark rather than
    # two -- a rotation is a single decision, and drawing both sides left the reader to infer it.
    assert "is-buy" in app_js and "is-sell" in app_js and "is-rotation" in app_js
    assert ".trade-marker.is-buy { fill: var(--good); }" in app_css
    assert ".trade-marker.is-sell { fill: var(--bad); }" in app_css
    assert ".trade-marker.is-rotation { fill: #b45309; }" in app_css
    # Marks are drawn under the hover hitbox, so they must not swallow pointer events.
    assert "pointer-events: none;" in app_css
    # One mark per bucket, not per step: the intraday grid trades on most of its bars.
    assert "MIN_MARKER_SPACING" in app_js


def test_the_tooltip_reports_holdings_and_what_moved_on_one_line() -> None:
    """Two lists meant a rotation printed the same symbols twice and left them to be paired up."""
    app_js, _, _ = _assets()

    assert "function backtestHoldingLines" in app_js
    assert "backtestHoldingLines(row)" in app_js
    # Share counts, not a second dollar figure: "+1" beside $498 is a fact about the order.
    assert "function shareDelta" in app_js
    assert "row?.trade_shares" in app_js
    # A leg sold out entirely is absent from ``positions``, and is the most consequential line.
    assert "if (!held.has(symbol))" in app_js
    # The separate "Bought X $Y" block is what this replaced.
    assert "tradeTooltipLines" not in app_js


def test_backtest_rows_keep_their_time_of_day() -> None:
    """Truncating to a date collapsed every bar in a session onto one x-position.

    Options Flip's ~14,000 rows plotted at 179 of them, roughly 78 points deep, so the line was
    drawn through whichever arrived last and no mark could ever address a single bar.
    """
    from src.api.payloads.backtest import BACKTEST_CACHE_VERSION, _json_backtest_rows

    frame = pd.DataFrame(
        {"equity": [100.0, 101.0]},
        index=pd.DatetimeIndex(
            ["2026-05-20T14:30:00Z", "2026-05-20T15:00:00Z"], name="timestamp"
        ),
    )

    rows = _json_backtest_rows(frame)

    assert [row["timestamp"] for row in rows] == ["2026-05-20T14:30:00Z", "2026-05-20T15:00:00Z"]
    assert BACKTEST_CACHE_VERSION >= 10, "a payload shape change has to invalidate cached rows"


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
    every algorithm off. What runs is a per-binding fact: on with a cron, or on and
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
