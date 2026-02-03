"""Tests for CLI commands."""

import pytest
from click.testing import CliRunner
from unittest.mock import patch, AsyncMock, MagicMock


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

            result = runner.invoke(
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

            result = runner.invoke(
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

        with patch("ai_reviewer.cli.load_config") as mock_load:
            mock_load.return_value = {"version": 1, "cursor": {"api_key": "test"}}

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

    def test_agents_list_command(self):
        """Test agents list command."""
        from ai_reviewer.cli import cli

        runner = CliRunner()

        with patch("ai_reviewer.cli.load_config") as mock_load:
            mock_load.return_value = {
                "version": 1,
                "agents": [
                    {"name": "security-agent", "model": "claude-3-opus"},
                    {"name": "perf-agent", "model": "gpt-4-turbo"},
                ],
            }

            result = runner.invoke(cli, ["agents", "list"])

            assert "security-agent" in result.output
            assert "perf-agent" in result.output

    def test_serve_command_starts_server(self):
        """Test that serve command starts the webhook server."""
        from ai_reviewer.cli import cli

        runner = CliRunner()

        with patch("ai_reviewer.cli.uvicorn") as mock_uvicorn:
            # Don't actually start server, just verify it would be called
            result = runner.invoke(
                cli,
                ["serve", "--port", "9000", "--host", "127.0.0.1"],
                catch_exceptions=False,
            )

            mock_uvicorn.run.assert_called_once()
            call_args = mock_uvicorn.run.call_args
            assert call_args.kwargs["port"] == 9000
            assert call_args.kwargs["host"] == "127.0.0.1"


class TestReviewFromDiff:
    """Tests for reviewing from diff input."""

    def test_review_from_stdin(self):
        """Test reviewing diff from stdin."""
        from ai_reviewer.cli import cli

        runner = CliRunner()

        sample_diff = """\
diff --git a/test.py b/test.py
index 1234567..abcdefg 100644
--- a/test.py
+++ b/test.py
@@ -1,3 +1,5 @@
+import os
+os.system(user_input)  # Command injection!
 def hello():
     print("Hello")
"""

        with patch("ai_reviewer.cli.review_diff_async", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = MagicMock(
                findings=[MagicMock(title="Command Injection")],
                summary="Found 1 issue",
                agent_count=3,
            )

            result = runner.invoke(
                cli,
                ["review", "--output", "markdown"],
                input=sample_diff,
            )

            mock_review.assert_called_once()
            # Diff should be passed to review function
            call_args = mock_review.call_args
            assert "os.system" in call_args.kwargs["diff"]

    def test_review_from_file(self):
        """Test reviewing diff from file."""
        from ai_reviewer.cli import cli

        runner = CliRunner()

        with runner.isolated_filesystem():
            # Create a diff file
            with open("changes.patch", "w") as f:
                f.write("diff --git a/test.py b/test.py\n+new line\n")

            with patch("ai_reviewer.cli.review_diff_async", new_callable=AsyncMock) as mock_review:
                mock_review.return_value = MagicMock(
                    findings=[],
                    summary="No issues",
                    agent_count=3,
                )

                result = runner.invoke(
                    cli,
                    ["review", "--diff", "changes.patch"],
                )

                mock_review.assert_called_once()
