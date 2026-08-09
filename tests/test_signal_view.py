from __future__ import annotations

from src.algorithms.base import signal_view_from_decision
from src.core.interfaces import AlgorithmContext, AlgorithmDecision, AlgorithmPlugin, SignalView


class _BarePlugin(AlgorithmPlugin):
    def generate_signals(self, context: AlgorithmContext):
        return []


def test_signal_view_from_decision_maps_signals_and_weights() -> None:
    decision = AlgorithmDecision(
        target_weights={"AAA": 0.5, "BBB": -0.25},
        signals={
            "AAA": {"signal": 1, "score": 2.0, "close": 10.0},
            "BBB": {"signal": -1, "score": 1.0},
            "CCC": {"signal": 0, "score": 0.1},
        },
    )

    view = signal_view_from_decision(decision, latest_prices={"CCC": 3.0})

    assert isinstance(view, SignalView)
    assert view.wired is True
    by_symbol = {row["symbol"]: row for row in view.leaders}
    assert by_symbol["AAA"]["side"] == "LONG"
    assert by_symbol["AAA"]["signal"] == "LONG"
    assert by_symbol["AAA"]["target_weight"] == 0.5
    assert by_symbol["BBB"]["side"] == "SHORT"
    assert by_symbol["CCC"]["side"] == "FLAT"
    assert by_symbol["CCC"]["close"] == 3.0  # filled from latest_prices
    # LONG ranks ahead of SHORT ahead of FLAT
    assert [row["symbol"] for row in view.leaders] == ["AAA", "BBB", "CCC"]
    assert view.summary == [
        {"label": "Long", "value": "1"},
        {"label": "Short", "value": "1"},
        {"label": "Universe", "value": "3"},
    ]


def test_every_plugin_exposes_a_signal_view() -> None:
    """The interface guarantees a signal_view on every algorithm; the default is 'not wired'."""
    view = _BarePlugin().signal_view(config=None)
    assert isinstance(view, SignalView)
    assert view.wired is False
