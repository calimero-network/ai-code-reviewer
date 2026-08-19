"""Working-tree review: the same pipeline with no pull request behind it."""

from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import ai_reviewer.review as rev
from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.models.review import AgentReview

CFG = AnthropicApiConfig(api_key="sk-test")


def _git(repo, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@e.st")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-qm", "init")
    return tmp_path


def _stub_agents(findings=None):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    review = AgentReview(
        agent_id="security-reviewer-0",
        agent_type="security-reviewer",
        focus_areas=[],
        findings=findings or [],
        summary="ok",
        review_time_ms=1,
    )
    return client, review


@pytest.mark.asyncio
async def test_reviews_uncommitted_changes_without_a_pull_request(repo):
    (repo / "a.py").write_text("x = 2\n")
    client, agent_review = _stub_agents()

    with (
        patch.object(rev, "AnthropicClient", return_value=client),
        patch.object(rev, "_prepare_shared_context", new=AsyncMock(return_value=([], [], set()))),
        patch.object(rev, "_run_agent_safe", new=AsyncMock(return_value=agent_review)) as ran,
    ):
        result = await rev.review_local(root=str(repo), anthropic_cfg=CFG, num_agents=1)

    assert ran.await_count == 1
    assert result.pr_number == 0
    assert result.repo == repo.name


@pytest.mark.asyncio
async def test_a_clean_tree_spends_nothing(repo):
    """No diff means no agents and no plan quota burned."""
    with (
        patch.object(rev, "AnthropicClient") as factory,
        patch.object(rev, "_run_agent_safe", new=AsyncMock()) as ran,
    ):
        result = await rev.review_local(root=str(repo), anthropic_cfg=CFG, num_agents=1)

    assert ran.await_count == 0
    assert factory.call_count == 0
    assert result.findings == []


@pytest.mark.asyncio
async def test_staged_mode_reviews_the_index_only(repo):
    (repo / "a.py").write_text("x = 2\n")
    _git(repo, "add", "a.py")
    (repo / "b.py").write_text("unstaged = True\n")
    _git(repo, "add", "-N", "b.py")
    client, agent_review = _stub_agents()

    captured: dict = {}

    async def capture(**kwargs):
        captured.update(kwargs)
        return [], [], set()

    with (
        patch.object(rev, "AnthropicClient", return_value=client),
        patch.object(rev, "_prepare_shared_context", new=capture),
        patch.object(rev, "_run_agent_safe", new=AsyncMock(return_value=agent_review)),
    ):
        await rev.review_local(root=str(repo), anthropic_cfg=CFG, num_agents=1, staged=True)

    assert "b.py" not in captured["changed_file_contents"]
    assert "a.py" in captured["changed_file_contents"]


class TestReviewCommand:
    """`ai-reviewer review` — working-tree review with no PR argument."""

    def _invoke(self, *args):
        from click.testing import CliRunner

        import ai_reviewer.cli as cli

        empty = rev.aggregate_findings([], "r", 0)
        with patch.object(cli, "review_local", new=AsyncMock(return_value=empty)) as spy:
            result = CliRunner().invoke(cli.cli, ["review", *args])
        return result, spy

    def test_defaults_to_the_working_tree(self):
        result, spy = self._invoke()

        assert result.exit_code == 0, result.output
        assert spy.await_args.kwargs["staged"] is False
        assert spy.await_args.kwargs["base"] is None

    def test_staged_flag_reviews_the_index(self):
        _result, spy = self._invoke("--staged")

        assert spy.await_args.kwargs["staged"] is True

    def test_base_flag_reviews_a_branch_range(self):
        _result, spy = self._invoke("--base", "main")

        assert spy.await_args.kwargs["base"] == "main"

    def test_json_output_is_machine_readable(self):
        import json

        result, _spy = self._invoke("--output", "json")

        assert json.loads(result.output)["findings"] == []

    def test_runs_without_an_anthropic_config_section(self):
        """The claude-code engine needs no API key, so a bare config still works."""
        from click.testing import CliRunner

        import ai_reviewer.cli as cli
        from ai_reviewer.config import Config, GitHubConfig

        bare = Config(anthropic=None, github=GitHubConfig(token=""), agents=[])
        empty = rev.aggregate_findings([], "r", 0)
        with (
            patch.object(cli, "load_config", return_value=bare),
            patch.object(cli, "review_local", new=AsyncMock(return_value=empty)) as spy,
        ):
            result = CliRunner().invoke(cli.cli, ["review"])

        assert result.exit_code == 0, result.output
        assert spy.await_args.kwargs["anthropic_cfg"].api_key == ""


@pytest.mark.asyncio
async def test_ignored_paths_are_not_sent_to_reviewers(repo):
    """review_local must apply the repo's ignore list, as review_pr does."""
    (repo / ".ai-reviewer.yaml").write_text('version: 1\nignore:\n  - "generated/**"\n')
    (repo / "generated").mkdir()
    (repo / "generated" / "api.py").write_text("noise = 1\n")
    (repo / "a.py").write_text("x = 2\n")
    client, agent_review = _stub_agents()

    captured: dict = {}

    async def capture(**kwargs):
        captured.update(kwargs)
        return [], [], set()

    with (
        patch.object(rev, "AnthropicClient", return_value=client),
        patch.object(rev, "_prepare_shared_context", new=capture),
        patch.object(rev, "_run_agent_safe", new=AsyncMock(return_value=agent_review)),
    ):
        await rev.review_local(root=str(repo), anthropic_cfg=CFG, num_agents=1)

    assert "generated/api.py" not in captured["changed_file_contents"]
    assert "generated/api.py" not in captured["diff"]
    assert "a.py" in captured["changed_file_contents"]


@pytest.mark.asyncio
async def test_a_lone_local_reviewer_is_told_to_cover_every_perspective(repo):
    """One reviewer's role prompt is one perspective; nobody else covers the rest."""
    # Large enough that agent scaling does not collapse the multi-agent case to one.
    for name in ("a.py", "b.py", "c.py", "d.py"):
        (repo / name).write_text("".join(f"x{i} = {i}\n" for i in range(200)))

    with patch.object(rev, "_prepare_shared_context", new=AsyncMock(return_value=([], [], set()))):
        one = await rev.build_agent_prompts(root=str(repo), num_agents=1)
        three = await rev.build_agent_prompts(root=str(repo), num_agents=3)

    assert len(three) == 3

    assert all(rev._SOLE_REVIEWER_INSTRUCTION in s["prompt"] for s in one.values())
    assert not any(rev._SOLE_REVIEWER_INSTRUCTION in s["prompt"] for s in three.values())
