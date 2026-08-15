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

    ``core.market_context`` used to import ``fast_momentum`` for a sentiment helper that was
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
