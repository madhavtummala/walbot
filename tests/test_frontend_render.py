from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE = PROJECT_ROOT / "tests" / "frontend" / "render_smoke.js"

#: Belt and braces: strip escapes even if a future node ignores the no-colour request.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_every_route_renders_without_throwing() -> None:
    """Walk the router in a DOM stub.

    String assertions cannot catch a render that throws -- a missing state key took out the
    whole dashboard once, because renderSidebar runs before anything paints.
    """
    result = subprocess.run(
        ["node", str(SMOKE)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
        # Node colourises values printed through console.log when it thinks a terminal is
        # attached, so ``true`` arrives as ``\\x1b[33mtrue\\x1b[39m`` and a substring assertion
        # fails while every route actually rendered. Asking for plain output is more honest
        # than stripping escapes after the fact.
        env={**os.environ, "NO_COLOR": "1", "FORCE_COLOR": "0"},
    )
    stdout = _ANSI.sub("", result.stdout)
    assert result.returncode == 0, stdout + result.stderr
    assert "hashchange listener registered: true" in stdout
    assert "no failures" in stdout
