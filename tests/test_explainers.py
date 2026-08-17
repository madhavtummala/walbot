from __future__ import annotations

import pytest

from src.algorithms.explainers import EXPLAINERS, explainer_for
from src.api.api_payloads import algorithm_config_payload

ALGORITHMS = ("dca", "bursty_dca", "fast_momentum", "dual_momentum", "spy_rotation")


@pytest.mark.parametrize("algorithm_id", ALGORITHMS)
def test_every_algorithm_has_an_explanation(algorithm_id: str) -> None:
    entry = explainer_for(algorithm_id)
    assert entry["summary"], algorithm_id
    assert entry["behavior"], algorithm_id
    assert entry["formula"], algorithm_id


def _config_fields(algorithm_id: str) -> set[str]:
    """Knobs the algorithm actually reads, from its config dataclass.

    Not algorithms.yaml -- that holds only the tuned subset, so a knob left at its default
    would look like a documentation error.
    """
    from dataclasses import fields

    from src.algorithms.dca import DCA_ALGORITHMS, PLAN_KEY
    from src.algorithms.dca.bursty import BurstyConfig
    from src.algorithms.dual_momentum import DualMomentumConfig
    from src.algorithms.fast_momentum import DefensiveMomentumConfig
    from src.algorithms.invest_spy import InvestSpyConfig

    dataclasses = {
        "bursty_dca": BurstyConfig,
        "fast_momentum": DefensiveMomentumConfig,
        "dual_momentum": DualMomentumConfig,
        "spy_rotation": InvestSpyConfig,
    }
    cls = dataclasses.get(algorithm_id)
    known = {field.name for field in fields(cls)} if cls else set()
    if algorithm_id in DCA_ALGORITHMS:
        # DCA's budgets are a knob the algorithm reads straight out of its config section
        # rather than through a dataclass, so a dataclass-only view calls the plan stale.
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
}


@pytest.mark.parametrize("algorithm_id", ALGORITHMS)
def test_the_prose_does_not_describe_knobs_that_no_longer_exist(algorithm_id: str) -> None:
    """The summary, formula and behaviour text drift silently; the parameter dict does not.

    ``test_documented_knobs_still_exist_on_the_algorithm`` only checks the keys, so an
    explainer can pass every other test here while its prose describes a completely different
    algorithm. Dual Momentum's did exactly that: long after the regime gate, the entry-timing
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
        str(explainer.get("behavior", "")),
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
