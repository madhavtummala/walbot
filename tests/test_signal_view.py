from __future__ import annotations

from src.algorithms.base import BaseAlgorithm, signal_view_from_plan
from src.core.interfaces import (
    ACTION_BLOCKED,
    ACTION_ENTER,
    ACTION_EXIT,
    ACTION_HOLD,
    ACTION_IDLE,
    AlgorithmPlan,
    SignalRow,
    SignalView,
    intents_from_weights,
)


def test_signal_view_from_plan_reads_weights_scores_and_gates() -> None:
    plan = AlgorithmPlan(
        intents=intents_from_weights({"AAA": 0.5, "BBB": 0.25}),
        latest_prices={"CCC": 3.0},
        signals={
            "AAA": {"action": ACTION_HOLD, "score": 2.0, "close": 10.0, "reason": "Rank 1 - held"},
            "BBB": {"action": ACTION_ENTER, "score": 1.0, "reason": "Rank 2 - opening"},
            "CCC": {"action": ACTION_BLOCKED, "score": 0.1, "reason": "Score below the floor",
                    "checks": [{"label": "Score floor", "ok": False, "value": "0.10",
                                "limit": ">= 0.50", "blocking": True}]},
        },
    )

    view = signal_view_from_plan(plan)

    assert isinstance(view, SignalView)
    assert all(isinstance(row, SignalRow) for row in view.rows)
    by_symbol = {row.symbol: row for row in view.rows}
    assert by_symbol["AAA"].action == ACTION_HOLD
    assert by_symbol["AAA"].headline == "Rank 1 - held"
    # Declared metrics, pre-formatted by the algorithm rather than sniffed by the deck.
    assert {"label": "Weight", "value": "50.0%"} in by_symbol["AAA"].metrics
    assert {"label": "Close", "value": "$10.00"} in by_symbol["AAA"].metrics
    # A symbol the signals did not price falls back to the plan's prices.
    assert {"label": "Close", "value": "$3.00"} in by_symbol["CCC"].metrics
    assert by_symbol["CCC"].checks[0].blocking is True


def test_rows_are_ordered_by_what_the_run_did_then_by_conviction() -> None:
    """Exits and entries first: the top of the list should be what this run changed."""
    plan = AlgorithmPlan(
        intents=intents_from_weights({"HELD": 0.4}),
        signals={
            "IDLE": {"action": ACTION_IDLE, "score": 5.0},
            "HELD": {"action": ACTION_HOLD, "score": 1.0},
            "GONE": {"action": ACTION_EXIT, "score": 0.2},
            "NEW": {"action": ACTION_ENTER, "score": 0.9},
            "STOPPED": {"action": ACTION_BLOCKED, "score": 4.0},
        },
    )

    assert [row.symbol for row in signal_view_from_plan(plan).rows] == [
        "GONE", "NEW", "HELD", "STOPPED", "IDLE"
    ]


def test_a_plan_that_proposes_nothing_still_renders() -> None:
    """The empty view is the honest one: no rows, and a universe of zero."""
    view = signal_view_from_plan(AlgorithmPlan())

    assert view.rows == []
    assert {"label": "Universe", "value": "0"} in view.summary
    assert {"label": "Held", "value": "0"} in view.summary
    # Entering/Exiting/Blocked earn a slot only when the run actually did one of them, so a
    # quiet session does not render a strip of zeroes.
    assert not any(item["label"] in {"Entering", "Exiting", "Blocked"} for item in view.summary)


def test_every_algorithm_inherits_a_signal_view() -> None:
    """The dashboard calls ``signal_view`` on whatever is registered, so the base defines it."""
    assert callable(BaseAlgorithm.signal_view)
