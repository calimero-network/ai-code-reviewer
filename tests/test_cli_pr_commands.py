"""The two commands that bracket the reviewer subagents."""

from __future__ import annotations

import json
from pathlib import Path
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
    # The brief lines are parsed by the caller, so stdout carries those and nothing
    # else; everything describing the preparation belongs on stderr.
    assert result.stdout.splitlines() == [
        f"security-reviewer\tclaude-sonnet-5\t{out / 'security-reviewer.md'}"
    ]
    assert "Preparing acme/widget#42" in result.stderr
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


def _session(tmp_path) -> Path:
    from ai_reviewer.context.pr_checkout import PreparedPR

    out = tmp_path / "session"
    (out / "out").mkdir(parents=True)
    (out / "wt").mkdir()
    PreparedPR(
        repo="acme/widget",
        number=42,
        clone=str(tmp_path / "clone"),
        root=str(out / "wt"),
        base_sha="b" * 40,
        head_sha="h" * 40,
    ).write(out / "target.json")
    (out / "out" / "security-reviewer.json").write_text('{"findings": [], "summary": "ok"}')
    return out


@pytest.fixture
def local_scope():
    """publish measures the reviewed diff to size the finding cap; the worktree
    in these tests is a stub directory, so the measurement is stubbed with it."""
    with (
        patch("ai_reviewer.context.local_source.local_diff", return_value="diff") as diff,
        patch("ai_reviewer.context.local_source.changed_files", return_value={}),
        patch(
            "ai_reviewer.context.local_source.build_local_context",
            return_value=MagicMock(additions=10, deletions=2),
        ) as build,
    ):
        yield diff, build


def test_publish_consolidates_posts_and_removes_the_worktree(tmp_path, local_scope):
    from ai_reviewer.github.publish import PublishResult

    local_diff, build_local_context = local_scope
    out = _session(tmp_path)
    result_obj = PublishResult(
        posted=True, action="COMMENT", inline_comments=2, resolved=0, skipped=False, body="b"
    )

    with (
        patch("ai_reviewer.cli.GitHubClient"),
        patch("ai_reviewer.cli.github_token", return_value="t"),
        patch("ai_reviewer.cli.consolidate_agent_findings") as consolidate,
        patch("ai_reviewer.cli.publish_review", return_value=result_obj) as publish,
        patch("ai_reviewer.cli.remove_pr_worktree") as remove,
    ):
        consolidate.return_value = MagicMock(findings=[], summary="ok", agent_count=1)
        result = CliRunner().invoke(cli, ["publish", str(out)], catch_exceptions=False)

    assert result.exit_code == 0
    assert consolidate.call_args.kwargs["repo"] == "acme/widget"
    assert publish.call_args.kwargs["allow_approve"] is False
    remove.assert_called_once()

    # Regression guard: a swap to os.getcwd() would still pass every other
    # assertion here but would silently review the wrong tree.
    root = str(out / "wt")
    assert local_diff.call_args.args[0] == root
    assert local_diff.call_args.kwargs["base"] == "b" * 40
    assert build_local_context.call_args.args[0] == root


def test_publish_removes_the_worktree_even_when_posting_fails(
    tmp_path,
    local_scope,  # noqa: ARG001 - present so the diff measurement is stubbed
):
    out = _session(tmp_path)

    with (
        patch("ai_reviewer.cli.GitHubClient"),
        patch("ai_reviewer.cli.github_token", return_value="t"),
        patch("ai_reviewer.cli.consolidate_agent_findings") as consolidate,
        patch("ai_reviewer.cli.publish_review", side_effect=RuntimeError("boom")),
        patch("ai_reviewer.cli.remove_pr_worktree") as remove,
    ):
        consolidate.return_value = MagicMock(findings=[], summary="ok", agent_count=1)
        result = CliRunner().invoke(cli, ["publish", str(out)])

    assert result.exit_code == 1
    remove.assert_called_once()


def test_publish_removes_the_worktree_when_a_click_exception_interrupts_the_try(
    tmp_path,
    local_scope,  # noqa: ARG001 - present so the diff measurement is stubbed
):
    """An expired gh session raises inside the try (github_token), not before it -
    cleanup must not depend on which exception type interrupted the post."""
    import click

    out = _session(tmp_path)

    with (
        patch("ai_reviewer.cli.GitHubClient"),
        patch("ai_reviewer.cli.github_token", side_effect=click.ClickException("no token")),
        patch("ai_reviewer.cli.consolidate_agent_findings") as consolidate,
        patch("ai_reviewer.cli.remove_pr_worktree") as remove,
    ):
        consolidate.return_value = MagicMock(findings=[], summary="ok", agent_count=1)
        result = CliRunner().invoke(cli, ["publish", str(out)])

    assert result.exit_code != 0
    remove.assert_called_once()


def test_publish_dry_run_keeps_the_worktree_for_another_attempt(
    tmp_path,
    local_scope,  # noqa: ARG001 - present so the diff measurement is stubbed
):
    from ai_reviewer.github.publish import PublishResult

    out = _session(tmp_path)
    result_obj = PublishResult(
        posted=False, action="", inline_comments=0, resolved=0, skipped=False, body="body text"
    )

    with (
        patch("ai_reviewer.cli.GitHubClient"),
        patch("ai_reviewer.cli.github_token", return_value="t"),
        patch("ai_reviewer.cli.consolidate_agent_findings") as consolidate,
        patch("ai_reviewer.cli.publish_review", return_value=result_obj),
        patch("ai_reviewer.cli.remove_pr_worktree") as remove,
    ):
        consolidate.return_value = MagicMock(findings=[], summary="ok", agent_count=1)
        result = CliRunner().invoke(cli, ["publish", str(out), "--dry-run"])

    assert result.exit_code == 0
    assert "body text" in result.output
    remove.assert_not_called()


def test_publish_needs_a_prepared_session(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    result = CliRunner().invoke(cli, ["publish", str(empty)])

    assert result.exit_code != 0
    assert "target.json" in result.output


def test_publish_needs_at_least_one_agent_result(tmp_path):
    out = _session(tmp_path)
    (out / "out" / "security-reviewer.json").unlink()

    with patch("ai_reviewer.cli.remove_pr_worktree") as remove:
        result = CliRunner().invoke(cli, ["publish", str(out)])

    assert result.exit_code != 0
    assert "no agent findings" in result.output
    # The worktree exists once target.json is read, so the one cleanup rule
    # applies here too, not only once findings have been loaded.
    remove.assert_called_once()


class TestExtraReviewerUsersWiring:
    """config.github.extra_reviewer_users must reach the GitHubClient (was dead config).

    Without it a repo's bot identity is not an AI reviewer here, so every comment
    that bot already posted is invisible and gets posted a second time.
    """

    @staticmethod
    def _config(tmp_path) -> str:
        path = tmp_path / "cfg.yaml"
        path.write_text('github:\n  token: t\n  extra_reviewer_users:\n    - "meroreviewer[bot]"\n')
        return str(path)

    def test_prompts_pr_passes_extra_reviewer_users(self, tmp_path):
        from ai_reviewer.context.pr_checkout import PreparedPR

        out = tmp_path / "session"
        prepared = PreparedPR(
            repo="acme/widget",
            number=42,
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
            patch(
                "ai_reviewer.cli.build_agent_prompts",
                new_callable=AsyncMock,
                return_value={"security-reviewer": {"model": "m", "prompt": "brief"}},
            ),
        ):
            client.return_value.get_pull_request.return_value = pull
            result = CliRunner().invoke(
                cli,
                [
                    "prompts",
                    "--out",
                    str(out),
                    "--pr",
                    "acme/widget#42",
                    "--config",
                    self._config(tmp_path),
                ],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        assert client.call_args.kwargs["extra_reviewer_users"] == ["meroreviewer[bot]"]

    def test_publish_passes_extra_reviewer_users(self, tmp_path, local_scope):  # noqa: ARG002
        from ai_reviewer.github.publish import PublishResult

        out = _session(tmp_path)
        result_obj = PublishResult(
            posted=True, action="COMMENT", inline_comments=0, resolved=0, skipped=False, body=""
        )

        with (
            patch("ai_reviewer.cli.GitHubClient") as client,
            patch("ai_reviewer.cli.consolidate_agent_findings") as consolidate,
            patch("ai_reviewer.cli.publish_review", return_value=result_obj),
            patch("ai_reviewer.cli.remove_pr_worktree"),
        ):
            consolidate.return_value = MagicMock(findings=[], summary="ok", agent_count=1)
            result = CliRunner().invoke(
                cli,
                ["publish", str(out), "--config", self._config(tmp_path)],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        assert client.call_args.kwargs["extra_reviewer_users"] == ["meroreviewer[bot]"]


def test_a_worktree_that_cannot_be_prepared_reports_the_reason(tmp_path):
    """The likeliest real failure of this command, and it sat outside the try."""
    out = tmp_path / "session"
    pull = MagicMock()
    pull.title = "t"
    pull.body = ""
    pull.base.ref = "main"

    with (
        patch("ai_reviewer.cli.GitHubClient") as client,
        patch("ai_reviewer.cli.resolve_clone", return_value=tmp_path / "clone"),
        patch(
            "ai_reviewer.cli.create_pr_worktree",
            side_effect=RuntimeError(
                "git fetch --no-tags origin: fatal: couldn't find remote ref refs/pull/42/head"
            ),
        ),
        patch("ai_reviewer.cli.github_token", return_value="t"),
        patch("ai_reviewer.cli.remove_pr_worktree") as remove,
    ):
        client.return_value.get_pull_request.return_value = pull
        result = CliRunner().invoke(cli, ["prompts", "--out", str(out), "--pr", "acme/widget#42"])

    assert result.exit_code == 1
    assert "couldn't find remote ref" in result.stderr
    # There is no worktree to release when the command that creates one failed.
    remove.assert_not_called()


@pytest.mark.parametrize("command", ["prompts", "publish"])
def test_the_help_text_carries_no_unrendered_markup(command):
    """Click prints docstrings as they are written, so RST literals show up raw."""
    result = CliRunner().invoke(cli, [command, "--help"])

    assert "``" not in result.output


# Long enough that Rich's 80-column default would break it across lines.
_LONG_PR_URL = (
    "https://github.com/some-really-long-organisation-name/"
    "an-equally-long-repository-name/pull/12345"
)


def test_publish_prints_where_the_review_landed(tmp_path, local_scope):  # noqa: ARG001
    """The skill quotes this link verbatim, so it must survive unwrapped."""
    from ai_reviewer.github.publish import PublishResult

    out = _session(tmp_path)
    posted = PublishResult(
        posted=True, action="COMMENT", inline_comments=2, resolved=0, skipped=False, body=""
    )

    with (
        patch("ai_reviewer.cli.GitHubClient") as client,
        patch("ai_reviewer.cli.github_token", return_value="t"),
        patch("ai_reviewer.cli.consolidate_agent_findings") as consolidate,
        patch("ai_reviewer.cli.publish_review", return_value=posted),
        patch("ai_reviewer.cli.remove_pr_worktree"),
    ):
        consolidate.return_value = MagicMock(findings=[], summary="ok", agent_count=1)
        client.return_value.get_pull_request.return_value.html_url = _LONG_PR_URL
        result = CliRunner().invoke(cli, ["publish", str(out)], catch_exceptions=False)

    assert _LONG_PR_URL in result.output


def test_publish_prints_no_link_when_nothing_was_posted(tmp_path, local_scope):  # noqa: ARG001
    from ai_reviewer.github.publish import PublishResult

    out = _session(tmp_path)
    skipped = PublishResult(
        posted=False, action="", inline_comments=0, resolved=0, skipped=True, body=""
    )

    with (
        patch("ai_reviewer.cli.GitHubClient") as client,
        patch("ai_reviewer.cli.github_token", return_value="t"),
        patch("ai_reviewer.cli.consolidate_agent_findings") as consolidate,
        patch("ai_reviewer.cli.publish_review", return_value=skipped),
        patch("ai_reviewer.cli.remove_pr_worktree"),
    ):
        consolidate.return_value = MagicMock(findings=[], summary="ok", agent_count=1)
        client.return_value.get_pull_request.return_value.html_url = _LONG_PR_URL
        result = CliRunner().invoke(cli, ["publish", str(out)], catch_exceptions=False)

    assert _LONG_PR_URL not in result.output
