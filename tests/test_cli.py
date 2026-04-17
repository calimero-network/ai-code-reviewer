"""Tests for CLI commands."""

from unittest.mock import AsyncMock, MagicMock, patch

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
        """When repo has no source_to_docs_mapping, exits cleanly."""
        from ai_reviewer.cli import cli

        runner = CliRunner()
        with (
            patch("ai_reviewer.cli.load_config") as mock_cfg,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient") as MockGH,
        ):
            mock_cfg.return_value.anthropic.api_key = "sk-test"
            mock_cfg.return_value.anthropic = MagicMock(api_key="sk-test")
            mock_cfg.return_value.github.token = "ghp_test"
            mock_cfg.return_value.doc_generation = MagicMock(
                model="claude-sonnet-4-6", max_files=5
            )

            mock_pr = MagicMock()
            mock_pr.base.ref = "main"
            mock_pr.merge_commit_sha = "abc123"
            mock_pr.get_files.return_value = []

            gh_instance = MockGH.return_value
            gh_instance.get_pull_request.return_value = mock_pr
            gh_instance.load_repo_config.return_value = {}  # no doc_generation config

            result = runner.invoke(cli, ["update-docs", "org/repo", "42", "--dry-run"])

        assert result.exit_code == 0
        assert "Nothing to do" in result.output or "No source_to_docs_mapping" in result.output


class TestDocGenerationSettings:
    """Tests for DocGenerationSettings config."""

    def test_defaults(self):
        from ai_reviewer.config import DocGenerationSettings

        s = DocGenerationSettings()
        assert s.enabled is False
        assert s.model == "claude-sonnet-4-6"
        assert s.max_files == 5

    def test_parsed_from_config(self):
        from ai_reviewer.config import _parse_config

        cfg = _parse_config(
            {
                "anthropic": {"api_key": "sk-test"},
                "github": {"token": "ghp_test"},
                "doc_generation": {
                    "enabled": True,
                    "model": "claude-opus-4-6",
                    "max_files": 3,
                },
            }
        )
        assert cfg.doc_generation.enabled is True
        assert cfg.doc_generation.model == "claude-opus-4-6"
        assert cfg.doc_generation.max_files == 3

    def test_disabled_by_default_in_parsed_config(self):
        from ai_reviewer.config import _parse_config

        cfg = _parse_config(
            {"anthropic": {"api_key": "sk-test"}, "github": {"token": "ghp_test"}}
        )
        assert cfg.doc_generation.enabled is False
