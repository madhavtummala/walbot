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


def test_frontend_uses_configured_backtests_and_refresh_button() -> None:
    app_js = (PROJECT_ROOT / "web/static/app.js").read_text(encoding="utf-8")
    index_html = (PROJECT_ROOT / "web/index.html").read_text(encoding="utf-8")

    assert 'let BACKTEST_PERIOD = "4m";' in app_js
    assert "configureBacktestPeriod(statusPayload.config?.backtest_period)" in app_js
    assert "4M" in app_js
    assert "Ending equity =" in app_js
    assert "cash cap" not in app_js
    assert "${money(backtest.ending_equity)} end" not in app_js
    assert "cumulative turnover" in app_js
    assert "chart-crosshair" in app_js
    assert "backtestPositions(row.positions)" in app_js
    assert '`${symbol} : ${money(value)}`' in app_js
    assert "renderUniverseProposalRows" in app_js
    assert "renderUniverseReview" not in app_js
    assert "function renderSignalBacktestCard" in app_js
    assert "handleSignalCardClick" in app_js
    assert "Running" not in app_js
    assert "app.js?v=20260811-single-page" in index_html


def test_frontend_is_a_single_page_without_options() -> None:
    """One page: run controls, then DCA bubbles, then the algorithm picker."""
    app_js = (PROJECT_ROOT / "web/static/app.js").read_text(encoding="utf-8")
    index_html = (PROJECT_ROOT / "web/index.html").read_text(encoding="utf-8")

    # No tab chrome left anywhere.
    assert "tabPanel" not in index_html
    assert 'id="tabs"' not in index_html
    assert "switchTab" not in app_js

    # Options are gone from the dashboard entirely.
    for marker in ("optionsPowerToggle", "optionsDeck", "optionsTradingAccount", "optionsSignalCard"):
        assert marker not in index_html, marker
        assert marker not in app_js, marker
    assert "OPTIONS_STRATEGIES" not in app_js
    assert "renderOptionsPower" not in app_js

    # The four remaining algorithms all live in the one picker.
    for key in ("dca", "bursty_dca", "fast_momentum", "spy_rotation"):
        assert f'key: "{key}"' in app_js, key

    # Run controls sit above the bubble board, picker below it.
    assert index_html.index("algorithmRunPanel") < index_html.index("bubbleBoard")
    assert index_html.index("bubbleBoard") < index_html.index("algorithmDeck")
