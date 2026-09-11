from __future__ import annotations

import pytest

from src.algorithms.explainers import EXPLAINERS, explainer_for
from src.api.api_payloads import algorithm_config_payload

ALGORITHMS = ("bursty_dca", "rally_rotation")


@pytest.mark.parametrize("algorithm_id", ALGORITHMS)
def test_every_algorithm_has_an_explanation(algorithm_id: str) -> None:
    entry = explainer_for(algorithm_id)
    assert entry["summary"], algorithm_id
    assert entry["formula"], algorithm_id


def _config_fields(algorithm_id: str) -> set[str]:
    """Knobs the algorithm actually reads, from its config dataclass.

    Not algorithms.yaml -- that holds only the tuned subset, so a knob left at its default
    would look like a documentation error.
    """
    from dataclasses import fields

    from src.algorithms.bursty_dca.config import PLAN_KEY
    from src.algorithms.bursty_dca.config import BurstyConfig
    from src.algorithms.rally_rotation.config import RallyRotationConfig
    from src.algorithms.registry import get_algorithm_class

    dataclasses = {
        "bursty_dca": BurstyConfig,
        "rally_rotation": RallyRotationConfig,
    }
    cls = dataclasses.get(algorithm_id)
    known = {field.name for field in fields(cls)} if cls else set()
    if getattr(get_algorithm_class(algorithm_id), "tune_editor", None) == "budgets":
        # The budgets are a knob the algorithm reads straight out of its config section rather
        # than through a dataclass, so a dataclass-only view calls the plan stale. Asked of the
        # class for the same reason the dashboard asks: it is the algorithm's own declaration.
        known.add(PLAN_KEY)
    return known


@pytest.mark.parametrize("algorithm_id", ALGORITHMS)
def test_documented_knobs_still_exist_on_the_algorithm(algorithm_id: str) -> None:
    """Docs live apart from the algorithms, so a renamed knob has to fail loudly here."""
    documented = {name for name in explainer_for(algorithm_id)["parameters"] if not name.startswith("__")}
    known = _config_fields(algorithm_id)
    if not documented or not known:
        return
    stale = documented - known
    assert not stale, f"{algorithm_id} documents knobs the algorithm does not read: {sorted(stale)}"


@pytest.mark.parametrize("algorithm_id", ALGORITHMS)
def test_every_saved_knob_is_documented(algorithm_id: str) -> None:
    saved = set(algorithm_config_payload(algorithm_id)["config"])
    documented = set(explainer_for(algorithm_id)["parameters"])
    undocumented = saved - documented
    assert not undocumented, f"{algorithm_id} has undocumented knobs: {sorted(undocumented)}"


def test_each_parameter_says_what_it_is_and_which_way_to_move_it() -> None:
    for algorithm_id, entry in EXPLAINERS.items():
        for name, doc in entry["parameters"].items():
            assert doc.get("what"), f"{algorithm_id}.{name} has no 'what'"
            assert doc.get("effect"), f"{algorithm_id}.{name} has no 'effect'"


def test_explanation_is_served_with_the_config() -> None:
    payload = algorithm_config_payload("bursty_dca")
    assert payload["explainer"]["formula"]
    assert "regime_ma_days" in payload["explainer"]["parameters"]


#: Names that look like knobs but are not. Two kinds: account-level settings the algorithms
#: describe but do not own, and *derived quantities* that appear in a formula as intermediate
#: terms -- ``pullback_bonus`` is computed from three knobs, it is not one.
#:
#: Keep this list short. Every entry is a place the check cannot help, so an addition should be
#: because the name genuinely is not a knob, never to quiet a real drift.
_PROSE_ALLOWANCES = {
    # account-level, not algorithm tuning
    "cash_buffer", "min_trade_dollars", "rebalance_threshold",
    # derived terms in formulas
    "pullback_bonus", "micro_return", "monthly_budget", "held_value", "elapsed_months",
    # accrual's own intermediate terms: elapsed time, and the smallest trade that can reach
    # the market. Both are computed, never configured.
    "hours_since_last_buy", "min_executable",
    # Bursty DCA's two sizing inputs: the reference the valuation z-score is measured against,
    # and the accrued balance expressed in months of budget. Both computed per run.
    "moving_average", "backlog_months",
}


@pytest.mark.parametrize("algorithm_id", ALGORITHMS)
def test_the_prose_does_not_describe_knobs_that_no_longer_exist(algorithm_id: str) -> None:
    """The summary and formula text drift silently; the parameter dict does not.

    ``test_documented_knobs_still_exist_on_the_algorithm`` only checks the keys, so an
    explainer can pass every other test here while its prose describes a completely different
    algorithm. Rally Rotation's did exactly that: long after the regime gate, the entry-timing
    signal and the volatility overlay were deleted, the summary still opened "a benchmark trend
    and breadth gate sets the regime", the formula still carried ``timing(i) = ...``, and the
    behaviour text still said it ran every 15 minutes and cited ``signal_refresh_minutes``,
    which no longer existed on the config at all.

    Anything shaped like a knob -- lower_snake_case with an underscore -- has to be a knob.
    """
    import re

    explainer = explainer_for(algorithm_id)
    known = _config_fields(algorithm_id)
    if not known:
        return

    prose = " ".join([
        str(explainer.get("summary", "")),
        " ".join(explainer.get("formula", [])),
        # Each knob's own description may name other knobs, and those go stale the same way.
        " ".join(f"{item.get('what', '')} {item.get('effect', '')}"
                 for item in explainer["parameters"].values() if isinstance(item, dict)),
    ])
    mentioned = {word for word in re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", prose)}
    stale = mentioned - known - _PROSE_ALLOWANCES

    assert not stale, (
        f"{algorithm_id} prose names knobs the algorithm does not read: {sorted(stale)}"
    )


def test_every_knob_says_which_way_to_turn_it() -> None:
    """A reader on the Tune page is deciding whether to raise or lower a number, so that is
    what the guidance has to answer.

    Twenty-two knobs described what the setting *was* without saying what moving it does --
    "Half the selection weight sits here", "Stops trivial orders" -- which is true and
    unactionable. Eight more spent their space on why the current default was chosen, which
    belongs in the source beside the value, not on a page someone opens to change it.
    """
    from src.algorithms.explainers import EXPLAINERS

    directional = (
        "higher", "lower", "longer", "shorter", "raise", "more", "less", "tighter", "wider",
        "above", "below", "0 ", "zero", "at 1", "on,", "off", "deeper", "easier", "harder",
        "one is", "whichever", "set it",
    )
    history = ("replaced", "earlier default", "used to", "previously", "the old ")

    for algorithm, entry in EXPLAINERS.items():
        for knob, doc in entry["parameters"].items():
            effect = doc["effect"].lower()
            assert any(word in effect for word in directional), (
                f"{algorithm}.{knob} does not say which way to turn it"
            )
            assert not any(word in effect for word in history), (
                f"{algorithm}.{knob} explains its own history rather than its effect"
            )
            assert len(doc["effect"].split()) <= 50, f"{algorithm}.{knob} runs long"
            assert len(doc["what"].split()) <= 30, f"{algorithm}.{knob}'s 'what' runs long"


def test_a_knob_shared_by_two_algorithms_describes_each_one() -> None:
    """``plan`` exists in both Bursty DCA and Options Flip and means different things -- a
    monthly budget in one, a per-position budget in the other.

    A bulk edit keyed on the knob name alone rewrote the first match twice, so DCA's monthly
    budget was briefly documented as buying option contracts. Guidance for a shared name has to
    be written per algorithm, and the giveaway is vocabulary from the wrong one.
    """
    from src.algorithms.explainers import EXPLAINERS

    vocabulary = {
        "bursty_dca": ("contract", "premium", "delta", "expiry"),
        "options_flip": ("monthly", "accrue", "backlog"),
    }
    for algorithm, foreign in vocabulary.items():
        for knob, doc in EXPLAINERS[algorithm]["parameters"].items():
            text = f"{doc['what']} {doc['effect']}".lower()
            for word in foreign:
                assert word not in text, (
                    f"{algorithm}.{knob} uses {word!r}, which belongs to a different algorithm"
                )
