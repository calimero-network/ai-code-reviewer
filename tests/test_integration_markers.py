"""Live-API tests must stay opt-in.

Normal CI relies on pyproject's marker expression to exclude them; an unmarked
test under tests/integration, or a dropped expression, would spend real credit on
every push.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _collect(*marker_args: str) -> set[str]:
    """Ids under tests/integration, with the repo's addopts neutralised so output
    is parseable and the marker expression is the only variable."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            *marker_args,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {line for line in proc.stdout.splitlines() if "::" in line}


def _default_marker_expression() -> str:
    addopts = shlex.split(
        tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"][
            "addopts"
        ]
    )
    assert "-m" in addopts, "pyproject addopts no longer excludes live tests from the default run"
    return addopts[addopts.index("-m") + 1]


def test_every_live_test_carries_the_integration_marker():
    assert _collect() == _collect("-m", "integration")


def test_the_default_marker_expression_selects_no_live_tests():
    assert _collect("-m", _default_marker_expression()) == set()
