"""The reviewers' brief must carry the pull request's own words.

Without this a "PR review" cannot answer the most valuable question a reviewer
answers: does this diff do what the pull request says it does.
"""

from __future__ import annotations

import asyncio
import subprocess

import pytest

from ai_reviewer.context.local_source import PRMeta
from ai_reviewer.review import build_agent_prompts


def _git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@e.st")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n")
    return tmp_path


def test_the_brief_contains_the_pull_request_title_and_body(repo):
    meta = PRMeta(
        repo="acme/widget",
        number=42,
        title="fix(api): let a group add a member by account name",
        body="Members could only be added by public key.",
    )

    built = asyncio.run(build_agent_prompts(root=str(repo), num_agents=1, pr_meta=meta))

    prompt = next(iter(built.values()))["prompt"]
    assert "fix(api): let a group add a member by account name" in prompt
    assert "Members could only be added by public key." in prompt


def test_without_a_pull_request_the_brief_is_unchanged(repo):
    """The brief's PR metadata block comes from build_local_pr, not from the
    ReviewContext, so this is the placeholder a reviewer actually sees."""
    built = asyncio.run(build_agent_prompts(root=str(repo), num_agents=1))

    prompt = next(iter(built.values()))["prompt"]
    assert "Local review of the working tree" in prompt
