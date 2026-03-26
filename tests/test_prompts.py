"""Tests for repo config loading, convention loading, and prompt wiring (Task 4)."""

from unittest.mock import MagicMock, patch

import pytest
from github.GithubException import GithubException

from ai_reviewer.github.client import GitHubClient


class TestLoadRepoConfig:
    """Tests for GitHubClient.load_repo_config()."""

    def _make_client(self):
        with patch("ai_reviewer.github.client.Github"):
            return GitHubClient(token="test-token")

    def test_valid_yaml_returns_parsed_dict(self):
        client = self._make_client()
        yaml_content = b"ignore:\n  - '*.generated.py'\ncustom_rules:\n  - No raw SQL\n"
        mock_content = MagicMock()
        mock_content.decoded_content = yaml_content

        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = mock_content
        client._gh.get_repo.return_value = mock_repo

        result = client.load_repo_config("owner/repo", ref="abc123")
        assert result == {"ignore": ["*.generated.py"], "custom_rules": ["No raw SQL"]}
        mock_repo.get_contents.assert_called_once_with(".ai-reviewer.yaml", ref="abc123")

    def test_missing_file_returns_none(self):
        client = self._make_client()
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = GithubException(
            status=404, data={"message": "Not Found"}, headers={}
        )
        client._gh.get_repo.return_value = mock_repo

        result = client.load_repo_config("owner/repo", ref="abc123")
        assert result is None

    def test_invalid_yaml_returns_none(self):
        client = self._make_client()
        mock_content = MagicMock()
        mock_content.decoded_content = b": : : not valid yaml [[["

        mock_repo = MagicMock()
        mock_repo.get_contents.return_value = mock_content
        client._gh.get_repo.return_value = mock_repo

        result = client.load_repo_config("owner/repo", ref="abc123")
        assert result is None

    def test_403_raises_permission_error(self):
        client = self._make_client()
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = GithubException(
            status=403, data={"message": "Forbidden"}, headers={}
        )
        client._gh.get_repo.return_value = mock_repo

        with pytest.raises(PermissionError):
            client.load_repo_config("owner/repo", ref="abc123")


class TestLoadRepoConventions:
    """Tests for GitHubClient.load_repo_conventions()."""

    def _make_client(self):
        with patch("ai_reviewer.github.client.Github"):
            return GitHubClient(token="test-token")

    def test_concatenates_multiple_convention_files(self):
        client = self._make_client()

        def fake_get_contents(path, ref=None):  # noqa: ARG001
            files = {
                "AGENTS.md": b"# Agents rules\nBe concise.",
                "CLAUDE.md": b"# Claude rules\nNo fluff.",
                "CONTRIBUTING.md": b"# Contributing\nFork first.",
            }
            if path in files:
                mock = MagicMock()
                mock.decoded_content = files[path]
                return mock
            raise GithubException(status=404, data={"message": "Not Found"}, headers={})

        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = fake_get_contents
        client._gh.get_repo.return_value = mock_repo

        result = client.load_repo_conventions("owner/repo", ref="abc123")
        assert result is not None
        assert "Agents rules" in result
        assert "Claude rules" in result
        assert "Contributing" in result
        assert result.index("Agents rules") < result.index("Claude rules")
        assert result.index("Claude rules") < result.index("Contributing")

    def test_missing_files_skipped_without_error(self):
        client = self._make_client()

        def fake_get_contents(path, ref=None):  # noqa: ARG001
            if path == "CONTRIBUTING.md":
                mock = MagicMock()
                mock.decoded_content = b"# Contributing\nFork first."
                return mock
            raise GithubException(status=404, data={"message": "Not Found"}, headers={})

        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = fake_get_contents
        client._gh.get_repo.return_value = mock_repo

        result = client.load_repo_conventions("owner/repo", ref="abc123")
        assert result is not None
        assert "Contributing" in result

    def test_all_missing_returns_none(self):
        client = self._make_client()
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = GithubException(
            status=404, data={"message": "Not Found"}, headers={}
        )
        client._gh.get_repo.return_value = mock_repo

        result = client.load_repo_conventions("owner/repo", ref="abc123")
        assert result is None

    def test_individual_file_truncated(self):
        client = self._make_client()
        long_content = b"x" * 20000

        def fake_get_contents(path, ref=None):  # noqa: ARG001
            if path == "AGENTS.md":
                mock = MagicMock()
                mock.decoded_content = long_content
                return mock
            raise GithubException(status=404, data={"message": "Not Found"}, headers={})

        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = fake_get_contents
        client._gh.get_repo.return_value = mock_repo

        result = client.load_repo_conventions("owner/repo", ref="abc123")
        assert result is not None
        assert len(result) <= 10000 + 200  # per-file limit + header overhead


class TestIgnoreFiltering:
    """Tests for ignore-pattern filtering of diff and file contents."""

    def test_filter_files_by_ignore_patterns(self):
        from ai_reviewer.review import filter_by_ignore_patterns

        files = {
            "src/main.py": "code",
            "generated/output.generated.py": "gen code",
            "docs/README.md": "readme",
        }
        patterns = ["*.generated.py", "docs/*"]
        result = filter_by_ignore_patterns(files, patterns)
        assert "src/main.py" in result
        assert "generated/output.generated.py" not in result
        assert "docs/README.md" not in result

    def test_filter_diff_by_ignore_patterns(self):
        from ai_reviewer.review import filter_diff_by_ignore_patterns

        diff = (
            "diff --git a/src/main.py b/src/main.py\n"
            "--- a/src/main.py\n"
            "+++ b/src/main.py\n"
            "@@ -1,3 +1,4 @@\n"
            "+new line\n"
            "\n"
            "diff --git a/gen/out.generated.py b/gen/out.generated.py\n"
            "--- a/gen/out.generated.py\n"
            "+++ b/gen/out.generated.py\n"
            "@@ -1,2 +1,3 @@\n"
            "+generated\n"
        )
        patterns = ["*.generated.py"]
        result = filter_diff_by_ignore_patterns(diff, patterns)
        assert "src/main.py" in result
        assert "out.generated.py" not in result

    def test_empty_patterns_returns_unchanged(self):
        from ai_reviewer.review import filter_by_ignore_patterns, filter_diff_by_ignore_patterns

        files = {"a.py": "code", "b.py": "code"}
        assert filter_by_ignore_patterns(files, []) == files

        diff = "diff --git a/a.py b/a.py\n+++ b/a.py\n"
        assert filter_diff_by_ignore_patterns(diff, []) == diff


class TestPromptConventionsAndRules:
    """Tests for conventions/custom_rules sections in get_base_prompt."""

    def test_conventions_section_appended_when_present(self):
        from ai_reviewer.models.context import ReviewContext
        from ai_reviewer.review import get_base_prompt

        ctx = ReviewContext(
            repo_name="test/repo",
            pr_number=1,
            pr_title="Test",
            pr_description="",
            base_branch="main",
            head_branch="feat",
            author="dev",
            changed_files_count=1,
            additions=10,
            deletions=2,
            conventions="Be concise. No raw SQL.",
        )
        prompt = get_base_prompt(ctx, "diff text", {})
        assert "Repository Conventions" in prompt
        assert "Be concise. No raw SQL." in prompt

    def test_custom_rules_section_appended_when_present(self):
        from ai_reviewer.models.context import ReviewContext
        from ai_reviewer.review import get_base_prompt

        ctx = ReviewContext(
            repo_name="test/repo",
            pr_number=1,
            pr_title="Test",
            pr_description="",
            base_branch="main",
            head_branch="feat",
            author="dev",
            changed_files_count=1,
            additions=10,
            deletions=2,
            repo_config={"custom_rules": ["No raw SQL", "Always use type hints"]},
        )
        prompt = get_base_prompt(ctx, "diff text", {})
        assert "Repository-Specific Rules" in prompt
        assert "No raw SQL" in prompt
        assert "Always use type hints" in prompt

    def test_no_sections_when_fields_are_none(self):
        from ai_reviewer.models.context import ReviewContext
        from ai_reviewer.review import get_base_prompt

        ctx = ReviewContext(
            repo_name="test/repo",
            pr_number=1,
            pr_title="Test",
            pr_description="",
            base_branch="main",
            head_branch="feat",
            author="dev",
            changed_files_count=1,
            additions=10,
            deletions=2,
        )
        prompt = get_base_prompt(ctx, "diff text", {})
        assert "Repository Conventions" not in prompt
        assert "Repository-Specific Rules" not in prompt
