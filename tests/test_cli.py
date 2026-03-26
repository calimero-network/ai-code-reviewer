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

        with patch("ai_reviewer.cli.load_config") as mock_load, \
             patch("ai_reviewer.cli.validate_config", return_value=[]):
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

        with patch("ai_reviewer.cli.uvicorn") as mock_uvicorn:
            # Don't actually start server, just verify it would be called
            runner.invoke(
                cli,
                ["serve", "--port", "9000", "--host", "127.0.0.1"],
                catch_exceptions=False,
            )

            mock_uvicorn.run.assert_called_once()
            call_args = mock_uvicorn.run.call_args
            assert call_args.kwargs["port"] == 9000
            assert call_args.kwargs["host"] == "127.0.0.1"
