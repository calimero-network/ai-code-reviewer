"""The two commands that bracket the reviewer subagents."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from ai_reviewer.cli import cli


def test_pr_and_staged_are_mutually_exclusive(tmp_path):
    result = CliRunner().invoke(
        cli,
        ["prompts", "--out", str(tmp_path), "--pr", "acme/widget#42", "--staged"],
    )

    assert result.exit_code != 0
    assert "--pr cannot be combined" in result.output


def test_pr_writes_a_target_file_and_the_briefs(tmp_path):
    from ai_reviewer.context.pr_checkout import PreparedPR

    out = tmp_path / "session"
    prepared = PreparedPR(
        repo="acme/widget",
        number=42,
        title="fix: the thing",
        clone=str(tmp_path / "clone"),
        root=str(out / "wt"),
        base_sha="b" * 40,
        head_sha="h" * 40,
    )
    pull = MagicMock()
    pull.title = "fix: the thing"
    pull.body = "Because."
    pull.base.ref = "main"

    with (
        patch("ai_reviewer.cli.GitHubClient") as client,
        patch("ai_reviewer.cli.resolve_clone", return_value=tmp_path / "clone"),
        patch("ai_reviewer.cli.create_pr_worktree", return_value=prepared) as create,
        patch("ai_reviewer.cli.github_token", return_value="t"),
        patch(
            "ai_reviewer.cli.build_agent_prompts",
            new_callable=AsyncMock,
            return_value={"security-reviewer": {"model": "claude-sonnet-5", "prompt": "brief"}},
        ) as build,
    ):
        client.return_value.get_pull_request.return_value = pull
        result = CliRunner().invoke(
            cli, ["prompts", "--out", str(out), "--pr", "acme/widget#42"], catch_exceptions=False
        )

    assert result.exit_code == 0
    assert json.loads((out / "target.json").read_text())["number"] == 42
    assert (out / "security-reviewer.md").read_text() == "brief"
    assert "security-reviewer\tclaude-sonnet-5" in result.output
    assert create.call_args.args[3] == "main"
    assert build.call_args.kwargs["root"] == prepared.root
    assert build.call_args.kwargs["base"] == "b" * 40
    assert build.call_args.kwargs["pr_meta"].title == "fix: the thing"


def test_pr_and_base_are_mutually_exclusive(tmp_path):
    result = CliRunner().invoke(
        cli,
        ["prompts", "--out", str(tmp_path), "--pr", "acme/widget#42", "--base", "main"],
    )

    assert result.exit_code != 0
    assert "--pr cannot be combined" in result.output


def test_a_failure_after_the_worktree_exists_removes_it(tmp_path):
    from ai_reviewer.context.pr_checkout import PreparedPR

    out = tmp_path / "session"
    prepared = PreparedPR(
        repo="acme/widget",
        number=42,
        title="",
        clone=str(tmp_path / "clone"),
        root=str(out / "wt"),
        base_sha="b" * 40,
        head_sha="h" * 40,
    )
    pull = MagicMock()
    pull.title = "t"
    pull.body = ""
    pull.base.ref = "main"

    with (
        patch("ai_reviewer.cli.GitHubClient") as client,
        patch("ai_reviewer.cli.resolve_clone", return_value=tmp_path / "clone"),
        patch("ai_reviewer.cli.create_pr_worktree", return_value=prepared),
        patch("ai_reviewer.cli.github_token", return_value="t"),
        patch("ai_reviewer.cli.build_agent_prompts", side_effect=RuntimeError("boom")),
        patch("ai_reviewer.cli.remove_pr_worktree") as remove,
    ):
        client.return_value.get_pull_request.return_value = pull
        result = CliRunner().invoke(cli, ["prompts", "--out", str(out), "--pr", "acme/widget#42"])

    assert result.exit_code == 1
    remove.assert_called_once_with(prepared)


def test_a_failure_writing_target_json_after_the_worktree_exists_removes_it(tmp_path):
    from ai_reviewer.context.pr_checkout import PreparedPR

    out = tmp_path / "session"
    prepared = PreparedPR(
        repo="acme/widget",
        number=42,
        title="",
        clone=str(tmp_path / "clone"),
        root=str(out / "wt"),
        base_sha="b" * 40,
        head_sha="h" * 40,
    )
    prepared.write = MagicMock(side_effect=OSError("disk full"))
    pull = MagicMock()
    pull.title = "t"
    pull.body = ""
    pull.base.ref = "main"

    with (
        patch("ai_reviewer.cli.GitHubClient") as client,
        patch("ai_reviewer.cli.resolve_clone", return_value=tmp_path / "clone"),
        patch("ai_reviewer.cli.create_pr_worktree", return_value=prepared),
        patch("ai_reviewer.cli.github_token", return_value="t"),
        patch(
            "ai_reviewer.cli.build_agent_prompts",
            new_callable=AsyncMock,
            return_value={"security-reviewer": {"model": "m", "prompt": "brief"}},
        ),
        patch("ai_reviewer.cli.remove_pr_worktree") as remove,
    ):
        client.return_value.get_pull_request.return_value = pull
        result = CliRunner().invoke(cli, ["prompts", "--out", str(out), "--pr", "acme/widget#42"])

    assert result.exit_code == 1
    remove.assert_called_once_with(prepared)


def test_the_output_directory_is_not_created_when_the_build_fails(tmp_path):
    out = tmp_path / "would-be-created"

    with patch(
        "ai_reviewer.cli.build_agent_prompts",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        result = CliRunner().invoke(cli, ["prompts", "--out", str(out)])

    assert result.exit_code == 1
    assert not out.exists()


def test_github_token_falls_back_to_the_gh_cli():
    from ai_reviewer.cli import github_token
    from ai_reviewer.config import GitHubConfig, load_config

    config = load_config(None)
    config.github = GitHubConfig(token="")

    with patch("ai_reviewer.cli.subprocess.run") as run:
        run.return_value = MagicMock(stdout="gho_fromgh\n", returncode=0)
        assert github_token(config) == "gho_fromgh"


def test_github_token_explains_itself_when_there_is_none():
    import click

    from ai_reviewer.cli import github_token
    from ai_reviewer.config import GitHubConfig, load_config

    config = load_config(None)
    config.github = GitHubConfig(token="")

    with patch("ai_reviewer.cli.subprocess.run") as run:
        run.return_value = MagicMock(stdout="", returncode=1)
        with pytest.raises(click.ClickException, match="gh auth login"):
            github_token(config)


def test_github_token_explains_itself_when_gh_is_not_installed(monkeypatch):
    import click

    from ai_reviewer.cli import github_token
    from ai_reviewer.config import GitHubConfig, load_config

    config = load_config(None)
    config.github = GitHubConfig(token="")
    monkeypatch.setenv("PATH", "")

    with pytest.raises(click.ClickException, match="gh auth login"):
        github_token(config)
