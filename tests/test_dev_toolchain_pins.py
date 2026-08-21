"""The tools that gate CI must be pinned exactly.

A floating ruff turned an upstream release into a repo-wide format failure on
branches that changed no code. Same exposure applies to mypy.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATING_TOOLS = ("ruff", "mypy")


def _dev_requirements() -> dict[str, str]:
    optional = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    return {
        m.group("name"): m.group("spec")
        for req in optional["dev"]
        if (m := re.fullmatch(r"(?P<name>[A-Za-z0-9._-]+)(?P<spec>.*)", req.strip()))
    }


def test_the_tools_that_gate_ci_are_pinned_exactly():
    dev = _dev_requirements()
    for tool in GATING_TOOLS:
        assert tool in dev, f"{tool} missing from the dev extra"
        assert dev[tool].startswith("=="), f"{tool} is {dev[tool]!r}, not an exact pin"


def test_ci_installs_the_linters_from_the_pinned_extra():
    """A bare `pip install ruff` in any job reintroduces the floating version the
    pin exists to prevent."""
    for workflow in (ROOT / ".github/workflows").glob("*.y*ml"):
        for line in workflow.read_text().splitlines():
            for tool in GATING_TOOLS:
                assert not re.search(rf"pip install\s+(-U\s+)?{tool}\b", line), (
                    f"{workflow.name}: {line.strip()}"
                )
