"""Tests for CLI commands."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner


class TestCLI:
    """Tests for CLI commands."""

    def test_cli_help(self):
        """Test that CLI shows help."""
        from ai_reviewer.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "AI Code Reviewer" in result.output or "review" in result.output

    def test_review_pr_command(self):
        """Test review-pr command."""
        from ai_reviewer.cli import cli

        runner = CliRunner()

        with patch("ai_reviewer.cli.review_pr_async", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = MagicMock(
                findings=[],
                summary="No issues",
                agent_count=3,
            )

            runner.invoke(
                cli,
                ["review-pr", "test-org/test-repo", "42"],
                catch_exceptions=False,
            )

            # Should call review function
            mock_review.assert_called_once()
            call_args = mock_review.call_args
            assert call_args.kwargs["repo"] == "test-org/test-repo"
            assert call_args.kwargs["pr_number"] == 42

    def test_review_pr_dry_run(self):
        """Test review-pr with dry-run flag."""
        from ai_reviewer.cli import cli

        runner = CliRunner()

        with patch("ai_reviewer.cli.review_pr_async", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = MagicMock(
                findings=[],
                summary="No issues",
                agent_count=3,
            )

            runner.invoke(
                cli,
                ["review-pr", "test-org/test-repo", "42", "--dry-run"],
            )

            # Should not post to GitHub in dry-run mode
            if mock_review.called:
                call_args = mock_review.call_args
                assert call_args.kwargs.get("dry_run", False) is True

    def test_config_validate_command(self):
        """Test config validate command."""
        from ai_reviewer.cli import cli

        runner = CliRunner()

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
        ):
            mock_load.return_value = MagicMock()

            result = runner.invoke(cli, ["config", "validate"])

            assert result.exit_code == 0 or "valid" in result.output.lower()

    def test_config_validate_invalid(self):
        """Test config validate with invalid config."""
        from ai_reviewer.cli import cli

        runner = CliRunner()

        with patch("ai_reviewer.cli.load_config") as mock_load:
            mock_load.side_effect = ValueError("Missing required field: cursor.api_key")

            result = runner.invoke(cli, ["config", "validate"])

            # Should report error
            assert result.exit_code != 0 or "error" in result.output.lower()

    def test_serve_command_starts_server(self):
        """Test that serve command starts the webhook server."""
        from ai_reviewer.cli import cli

        runner = CliRunner()

        with (
            patch("ai_reviewer.cli.uvicorn") as mock_uvicorn,
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
        ):
            mock_load.return_value = MagicMock()
            runner.invoke(
                cli,
                ["serve", "--port", "9000", "--host", "127.0.0.1"],
                catch_exceptions=False,
            )

            mock_uvicorn.run.assert_called_once()
            call_args = mock_uvicorn.run.call_args
            assert call_args.kwargs["port"] == 9000
            assert call_args.kwargs["host"] == "127.0.0.1"


class TestUpdateDocsCLI:
    """Tests for the update-docs command."""

    def test_update_docs_help(self):
        from ai_reviewer.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["update-docs", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output
        assert "REPO" in result.output

    def test_update_docs_dry_run_no_mapping(self):
        """When run_doc_update returns skipped, CLI exits cleanly."""
        from ai_reviewer.cli import cli
        from ai_reviewer.docs.updater import DocUpdateResult

        runner = CliRunner()
        skipped_result = DocUpdateResult(
            skipped=True, skip_reason="no doc-relevant changes detected"
        )
        with (
            patch("ai_reviewer.cli.load_config") as mock_cfg,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient"),
            patch("ai_reviewer.cli.run_doc_update", new=AsyncMock(return_value=skipped_result)),
        ):
            mock_cfg.return_value.anthropic.api_key = "sk-test"
            mock_cfg.return_value.anthropic = MagicMock(api_key="sk-test")
            mock_cfg.return_value.github.token = "ghp_test"
            mock_cfg.return_value.doc_generation = MagicMock()

            result = runner.invoke(cli, ["update-docs", "org/repo", "42", "--dry-run"])

        assert result.exit_code == 0
        assert "no doc-relevant changes detected" in result.output


class TestDocGenerationSettings:
    """Tests for DocGenerationSettings config."""

    def test_defaults(self):
        from ai_reviewer.config import DocGenerationSettings

        s = DocGenerationSettings()
        assert s.enabled is False
        assert s.model == "claude-haiku-4-5-20251001"
        assert s.max_files == 15
        assert "docs/" in s.static_docs_dirs
        assert "docs-static/" in s.static_docs_dirs
        assert s.pr_draft is True
        assert "automated-docs" in s.pr_labels

    def test_parsed_from_config(self):
        from ai_reviewer.config import _parse_config

        cfg = _parse_config(
            {
                "anthropic": {"api_key": "sk-test"},
                "github": {"token": "ghp_test"},
                "doc_generation": {
                    "enabled": True,
                    "model": "claude-haiku-4-5-20251001",
                    "max_files": 3,
                },
            }
        )
        assert cfg.doc_generation.enabled is True
        assert cfg.doc_generation.model == "claude-haiku-4-5-20251001"
        assert cfg.doc_generation.max_files == 3

    def test_disabled_by_default_in_parsed_config(self):
        from ai_reviewer.config import _parse_config

        cfg = _parse_config({"anthropic": {"api_key": "sk-test"}, "github": {"token": "ghp_test"}})
        assert cfg.doc_generation.enabled is False


class TestAllAgentsFailedDryRun:
    """The all-agents-failed notice must respect --dry-run (self-review catch)."""

    def _run(self, dry_run: bool):
        import asyncio
        from datetime import datetime

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        review = ConsolidatedReview(
            id="r-fail",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="Agent failed: boom",
            agent_count=2,
            review_quality_score=0.0,
            total_review_time_ms=1000,
            failed_agents=["security-reviewer", "logic-reviewer"],
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "abc123"

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.run_review", return_value=review),
            patch("ai_reviewer.cli.GitHubClient") as mock_gh_cls,
        ):
            mock_load.return_value = MagicMock()
            mock_gh = MagicMock()
            mock_gh_cls.return_value = mock_gh
            mock_gh.get_pull_request.return_value = mock_pr
            mock_gh.get_review_metadata.return_value = None

            with pytest.raises(SystemExit):
                asyncio.run(
                    review_pr_async(
                        repo="test/repo",
                        pr_number=42,
                        output="github",
                        dry_run=dry_run,
                    )
                )
            return mock_gh

    def test_dry_run_does_not_post(self):
        gh = self._run(dry_run=True)
        gh.post_review.assert_not_called()

    def test_wet_run_posts_notice(self):
        gh = self._run(dry_run=False)
        gh.post_review.assert_called_once()
        body = gh.post_review.call_args.args[1]
        assert "Review could not complete" in body
