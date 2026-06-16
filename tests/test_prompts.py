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


class TestClassifyPR:
    """Tests for classify_pr() — PR type + size classification."""

    def test_trivial_code_pr(self):
        from ai_reviewer.review import classify_pr

        pr_type, pr_size = classify_pr(["src/main.py"], additions=20, deletions=10)
        assert pr_type == "code"
        assert pr_size == "trivial"

    def test_small_code_pr(self):
        from ai_reviewer.review import classify_pr

        pr_type, pr_size = classify_pr(["src/main.py"], additions=100, deletions=50)
        assert pr_type == "code"
        assert pr_size == "small"

    def test_medium_code_pr(self):
        from ai_reviewer.review import classify_pr

        pr_type, pr_size = classify_pr(
            ["src/main.py", "src/utils.py"], additions=400, deletions=200
        )
        assert pr_type == "code"
        assert pr_size == "medium"

    def test_large_code_pr(self):
        from ai_reviewer.review import classify_pr

        pr_type, pr_size = classify_pr(["src/main.py"], additions=800, deletions=500)
        assert pr_type == "code"
        assert pr_size == "large"

    def test_docs_only_pr(self):
        from ai_reviewer.review import classify_pr

        pr_type, pr_size = classify_pr(
            ["docs/README.md", "CHANGELOG.md"], additions=50, deletions=10
        )
        assert pr_type == "docs"
        assert pr_size == "small"

    def test_ci_only_pr(self):
        from ai_reviewer.review import classify_pr

        pr_type, pr_size = classify_pr([".github/workflows/ci.yml"], additions=10, deletions=5)
        assert pr_type == "ci"
        assert pr_size == "trivial"

    def test_boundary_trivial_small(self):
        """Exactly 50 total lines should be 'small', not 'trivial'."""
        from ai_reviewer.review import classify_pr

        _, size = classify_pr(["a.py"], additions=30, deletions=20)
        assert size == "small"

    def test_boundary_small_medium(self):
        """Exactly 200 total lines should be 'medium', not 'small'."""
        from ai_reviewer.review import classify_pr

        _, size = classify_pr(["a.py"], additions=120, deletions=80)
        assert size == "medium"

    def test_boundary_medium_large(self):
        """Exactly 1000 total lines should be 'large', not 'medium'."""
        from ai_reviewer.review import classify_pr

        _, size = classify_pr(["a.py"], additions=600, deletions=400)
        assert size == "large"

    def test_empty_paths_defaults_to_code(self):
        from ai_reviewer.review import classify_pr

        pr_type, _ = classify_pr([], additions=10, deletions=5)
        assert pr_type == "code"

    def test_zero_lines_is_trivial(self):
        from ai_reviewer.review import classify_pr

        _, pr_size = classify_pr(["a.py"], additions=0, deletions=0)
        assert pr_size == "trivial"


class TestLanguageRules:
    """Tests for language-specific review rules."""

    def test_python_rules(self):
        """Python language returns Python-specific rules."""
        from ai_reviewer.review import get_language_rules

        rules = get_language_rules(["Python"])
        assert "mutable default" in rules.lower()
        assert "type hint" in rules.lower()
        assert "__init__" in rules or "context manager" in rules.lower()

    def test_rust_rules(self):
        """Rust language returns Rust-specific rules."""
        from ai_reviewer.review import get_language_rules

        rules = get_language_rules(["Rust"])
        assert "unwrap" in rules.lower()
        assert "unsafe" in rules.lower()
        assert "clone" in rules.lower() or "lifetime" in rules.lower()
        # High-severity correctness/security priorities added from the research pass
        assert "send" in rules.lower() and "sync" in rules.lower()
        assert "deadlock" in rules.lower()
        assert "semver" in rules.lower()
        assert "supply chain" in rules.lower()

    def test_javascript_rules(self):
        """JavaScript language returns JS-specific rules."""
        from ai_reviewer.review import get_language_rules

        rules = get_language_rules(["JavaScript"])
        assert "prototype" in rules.lower()
        assert "===" in rules
        assert "async" in rules.lower() or "promise" in rules.lower()

    def test_typescript_rules(self):
        """TypeScript language returns TS-specific rules."""
        from ai_reviewer.review import get_language_rules

        rules = get_language_rules(["TypeScript"])
        assert "any" in rules.lower()
        assert "===" in rules
        assert "promise" in rules.lower() or "async" in rules.lower()
        assert "type assertion" in rules.lower() or "type guard" in rules.lower()

    def test_unknown_language_returns_empty(self):
        """Unknown language returns empty string."""
        from ai_reviewer.review import get_language_rules

        rules = get_language_rules(["BrainFuck"])
        assert rules == ""

    def test_multiple_languages(self):
        """Multiple languages returns combined rules."""
        from ai_reviewer.review import get_language_rules

        rules = get_language_rules(["Python", "Rust", "JavaScript"])
        assert "mutable default" in rules.lower()
        assert "unwrap" in rules.lower()
        assert "prototype" in rules.lower()
