from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE = PROJECT_ROOT / "tests" / "frontend" / "render_smoke.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_route_renders_without_throwing() -> None:
    """Walk the router in a DOM stub.

    String assertions cannot catch a render that throws -- a missing state key took out the
    whole dashboard once, because renderSidebar runs before anything paints.
    """
    result = subprocess.run(
        ["node", str(SMOKE)], capture_output=True, text=True, timeout=60, cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "hashchange listener registered: true" in result.stdout
    assert "no failures" in result.stdout
