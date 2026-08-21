"""The skill files are the interface a session actually reads.

A skill naming a command that does not exist fails at the worst moment, so the
invocations it documents are checked against the CLI itself.
"""

from __future__ import annotations

import re
from pathlib import Path

from click.testing import CliRunner

from ai_reviewer.cli import cli

SKILL = Path(".claude/skills/review-contribution/SKILL.md")


def test_the_skill_exists_and_declares_itself():
    text = SKILL.read_text()

    assert text.startswith("---\n")
    assert "name: review-contribution" in text


def test_every_option_the_skill_names_exists_on_the_cli():
    """A skill that documents a flag the CLI does not have fails at the worst moment."""
    runner = CliRunner()
    help_text = "".join(
        runner.invoke(cli, [command, "--help"]).output for command in ("prompts", "publish")
    )
    named = set(re.findall(r"--[a-z][a-z-]+", SKILL.read_text()))

    assert named, "the skill names no options at all - the regex or the skill is wrong"
    missing = sorted(option for option in named if option not in help_text)
    assert not missing, f"skill names options the CLI does not have: {missing}"


def test_the_skill_states_its_safety_contract():
    """This path posts on someone else's pull request under the user's identity. The
    promises that it never edits their code and never approves are the contract."""
    text = SKILL.read_text()

    assert "never edits the contributor's code" in text
    assert "Approve the pull request" in text
    assert "Never offer to repair anything here." in text
