"""Structural properties, asserted rather than hoped for.

These are the invariants a refactor is easiest to lose quietly: nothing fails, nothing looks
wrong in a diff, and the damage only shows up as a mysterious import error months later or as
a module that cannot be tested in isolation.
"""

from __future__ import annotations

import ast
import os

import pytest

SRC = "src"


def _module_graph() -> tuple[dict[str, str], dict[str, set[str]]]:
    files: dict[str, str] = {}
    for root, dirs, names in os.walk(SRC):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            module = path[:-3].replace(os.sep, ".")
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            files[module] = path

    graph: dict[str, set[str]] = {}
    for module, path in files.items():
        try:
            tree = ast.parse(open(path).read())
        except SyntaxError:  # pragma: no cover - a parse failure is its own test failure
            continue
        edges: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                parts = module.split(".")
                parts = (
                    parts[: len(parts) - node.level + 1]
                    if files[module].endswith("__init__.py")
                    else parts[: -node.level]
                )
                target = ".".join(parts + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if not target.startswith(SRC):
                continue
            while target and target not in files:
                target = target.rsplit(".", 1)[0] if "." in target else ""
            if target and target != module:
                edges.add(target)
        graph[module] = edges
    return files, graph


def test_the_source_tree_has_no_import_cycles() -> None:
    """Six cycles existed before this was written, and every one was load-bearing.

    They were survivable only because the imports were deferred into function bodies -- which
    is the tell, not the fix: a module that cannot be imported at the top of another is a
    layering mistake wearing a workaround. The worst had ``core.config`` importing the
    algorithm registry, so loading configuration pulled in every algorithm.
    """
    _, graph = _module_graph()

    cycles: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()

    def walk(node: str, path: list[str], stack: set[str]) -> None:
        if node in stack:
            cycle = path[path.index(node):]
            key = tuple(sorted(cycle))
            if key not in seen:
                seen.add(key)
                cycles.append(cycle)
            return
        if node in path:
            return
        for nxt in sorted(graph.get(node, ())):
            walk(nxt, path + [node], stack | {node})

    for module in sorted(graph):
        walk(module, [], set())

    assert not cycles, "import cycles:\n" + "\n".join(
        "  " + " -> ".join(c + [c[0]]) for c in cycles
    )


def test_the_data_layer_does_not_import_an_algorithm() -> None:
    """Direction of dependency: algorithms read data, never the reverse.

    ``core.market_context`` used to import a concrete algorithm for a sentiment helper that was
    pure and generic, which made a core module depend on one concrete strategy.
    """
    _, graph = _module_graph()
    offenders = {
        module: sorted(t for t in targets if t.startswith("src.algorithms"))
        for module, targets in graph.items()
        if module.startswith(("src.data", "src.core.config"))
        and any(t.startswith("src.algorithms") and not t.endswith(".ids") for t in targets)
    }
    assert not offenders, f"data/config importing algorithms: {offenders}"


@pytest.mark.parametrize("package", ["src.connectors", "src.api.api_payloads"])
def test_importing_a_facade_stays_cheap(package: str) -> None:
    """A facade must not drag in every provider's third-party dependency."""
    import subprocess
    import sys

    probe = (
        f"import sys, {package}; "
        "print(int('yfinance' in sys.modules "
        "or any(m.startswith('alpaca.data') for m in sys.modules)))"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.stdout.strip() == "0", result.stdout + result.stderr


def test_a_package_facade_never_exports_two_classes_with_one_name() -> None:
    """A duplicated definition is worse than a duplicated line.

    Splitting ``config.py`` left ``UnknownAccountError`` defined in both ``defaults`` and
    ``accounts``. Both imported, both looked right, and the package facade exported one while
    the code that raises used the other -- so ``except UnknownAccountError`` silently stopped
    catching. Nothing failed until a test asserted the raise.

    Scoped to what a facade re-exports, because two algorithms each defining their own
    ``score_universe`` is not a clash -- nobody imports them through one name.
    """
    import importlib

    files, _ = _module_graph()
    packages = [m for m, p in files.items() if p.endswith("__init__.py")]

    clashes: dict[str, list[str]] = {}
    for package in packages:
        try:
            module = importlib.import_module(package)
        except Exception:  # pragma: no cover - import errors are other tests' business
            continue
        for name in getattr(module, "__all__", []) or []:
            exported = getattr(module, name, None)
            if not isinstance(exported, type):
                continue
            defined_in = [
                candidate
                for candidate, path in files.items()
                if candidate.startswith(package + ".")
                and any(
                    isinstance(node, ast.ClassDef) and node.name == name
                    for node in ast.parse(open(path).read()).body
                )
            ]
            if len(defined_in) > 1:
                clashes[f"{package}.{name}"] = sorted(defined_in)

    assert not clashes, f"exported name defined in more than one module: {clashes}"


#: The clock belongs at the edge. ``LiveContextSource`` reads it to stamp a context, the
#: scheduler reads it to decide when to run, and the freshness guard reads it to ask whether a
#: result is stale in real time. Everything else is handed the moment it is reasoning about.
CLOCK_EDGES = {
    "src.core.market_context",   # LiveContextSource.timestamp -- where "now" enters a context
    "src.core.bot_runtime",      # the scheduler, which is about wall-clock cadence
    "src.core.pipeline",         # _assert_fresh, a deliberately real-time staleness guard
    "src.algorithms.options.swing",  # the options runner, which has no replay path
}


def test_an_algorithm_never_reads_the_clock() -> None:
    """Anything shared by live trading and replay must take the moment as an argument.

    A backtest is the same code with the clock moved, so a ``datetime.now()`` or
    ``date.today()` buried in an algorithm silently means "today" for every simulated step.
    That is not a hypothetical: the drawdown breaker keyed its session on ``date.today()``,
    so one bad day latched it for an entire backtest and every later date proposed an all-cash
    book that read as a decision rather than as a bug.
    """
    files, _ = _module_graph()
    offenders: list[str] = []

    for module, path in sorted(files.items()):
        if not module.startswith("src.algorithms") or module in CLOCK_EDGES:
            continue
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value
            if not isinstance(owner, ast.Name):
                continue
            if (owner.id, node.func.attr) in {("datetime", "now"), ("date", "today")}:
                offenders.append(f"{module}:{node.lineno} reads {owner.id}.{node.func.attr}()")

    assert not offenders, (
        "these read a clock instead of taking the moment as an argument: " + "; ".join(offenders)
    )


def test_the_sweep_tool_measures_the_configuration_that_is_deployed() -> None:
    """A sweep's baseline has to be the running config, or every comparison is against a ghost.

    The first draft of ``tools/config_sweep.py`` restated the tuning by hand and disagreed with
    ``config/walbot.yaml`` on six keys, which would have made every "improvement" partly an
    artefact of the transcription.
    """
    from tools.config_sweep import _dual_baseline, dual_axes
    from src.common.config_utils import tuning_section
    from src.core.config import get_config

    for algorithm_id, baseline in (("rally_rotation", _dual_baseline()),):
        deployed = tuning_section(get_config(strategy_id=algorithm_id), algorithm_id)
        assert baseline == deployed, algorithm_id

    for axes in (dual_axes(),):
        labels = [label for label, _ in axes]
        assert labels[0] == "baseline"
        assert len(labels) == len(set(labels)), "two variants share a label, so one would be lost"
        # One factor at a time: every variant differs from the baseline in a bounded way, so a
        # result can be attributed to the axis it names.
        base = dict(axes[0][1])
        for label, tuning in axes[1:]:
            changed = {key for key in set(base) | set(tuning) if base.get(key) != tuning.get(key)}
            assert changed, f"{label} changes nothing"
            # Four rather than one, because a couple of axes are a single idea spelled across
            # several keys: a horizon mix is four weights, and a position count moves the entry
            # and exit ranks with it. Anything wider stops being attributable to its label.
            assert len(changed) <= 4, f"{label} moves {len(changed)} knobs at once: {sorted(changed)}"


def test_the_binding_frequency_vocabulary_is_declared_once() -> None:
    """One list of cadences, in one place.

    It used to be written out four times -- ``VALID_FREQUENCIES``, the normaliser five lines
    below it that re-listed rather than read it, ``bot_runtime._binding_frequency``, and
    ``bot_runtime._frequency_minutes`` -- each with its own fallback. Adding a cadence to the
    dashboard's list therefore produced a value the normaliser rewrote to ``1hr`` and the
    scheduler then timed as sixty minutes: accepted, silently renamed, wrongly clocked.
    """
    from src.api.controls import BINDING_FREQUENCIES

    files, _ = _module_graph()
    core = {"15m", "30m", "1hr"}
    offenders: list[str] = []

    for module, path in sorted(files.items()):
        if module == "src.api.controls":
            continue
        for node in ast.walk(ast.parse(open(path).read())):
            if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
                literals = {e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            elif isinstance(node, ast.Dict):
                literals = {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            else:
                continue
            if core <= literals:
                offenders.append(f"{module} re-declares the frequency vocabulary: {sorted(literals)}")

    assert not offenders, "\n".join(offenders) + "\n(import from src.api.controls instead)"


def test_the_dashboard_offers_exactly_the_cadences_the_backend_accepts() -> None:
    """The frontend keeps its own copy -- it cannot import Python -- so pin the two together.

    A cadence in the dropdown that the backend normalises away is a control that silently does
    nothing, and one missing from the dropdown is a config the user cannot reach.
    """
    import json
    import re as _re

    from src.api.controls import BINDING_FREQUENCIES

    source = open("web/static/app.js").read()
    # Every array literal in app.js that mentions "mcp" is a copy of this vocabulary.
    arrays = [
        json.loads(match.replace("'", '"'))
        for match in _re.findall(r"\[[^\[\]]*\"mcp\"[^\[\]]*\]", source)
    ]

    assert arrays, "app.js no longer declares the cadence list where this test can see it"
    for array in arrays:
        assert array == list(BINDING_FREQUENCIES), f"app.js has {array}, backend has {list(BINDING_FREQUENCIES)}"
