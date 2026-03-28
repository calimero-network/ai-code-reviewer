"""Tests for documentation analysis (Tier 1 zero-config and Tier 2 configured)."""

from __future__ import annotations

import pytest

from ai_reviewer.docs.analyzer import (
    ARCHITECTURE_DIRS,
    CONVENTION_FILES,
    DocAnalyzer,
    DocSuggestion,
    format_doc_comment,
    is_architecture_impacting,
)


class TestIsArchitectureImpacting:
    """Tests for the architecture-impact heuristic."""

    def test_new_top_level_directory(self):
        paths = ["newpkg/main.py"]
        status = {"newpkg/main.py": "added"}
        assert is_architecture_impacting(paths, status)

    def test_removed_top_level_directory(self):
        paths = ["oldpkg/util.py"]
        status = {"oldpkg/util.py": "removed"}
        assert is_architecture_impacting(paths, status)

    def test_manifest_file_pyproject(self):
        paths = ["pyproject.toml"]
        status = {"pyproject.toml": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_manifest_file_package_json(self):
        paths = ["package.json"]
        status = {"package.json": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_manifest_file_cargo_toml(self):
        paths = ["Cargo.toml"]
        status = {"Cargo.toml": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_ci_workflow_file(self):
        paths = [".github/workflows/ci.yml"]
        status = {".github/workflows/ci.yml": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_gitlab_ci(self):
        paths = [".gitlab-ci.yml"]
        status = {".gitlab-ci.yml": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_jenkinsfile(self):
        paths = ["Jenkinsfile"]
        status = {"Jenkinsfile": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_entry_point_main(self):
        paths = ["src/main.py"]
        status = {"src/main.py": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_entry_point_cli(self):
        paths = ["cli.ts"]
        status = {"cli.ts": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_entry_point_index(self):
        paths = ["frontend/index.js"]
        status = {"frontend/index.js": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_dockerfile(self):
        paths = ["Dockerfile"]
        status = {"Dockerfile": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_docker_compose(self):
        paths = ["docker-compose.yaml"]
        status = {"docker-compose.yaml": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_terraform_file(self):
        paths = ["infra/main.tf"]
        status = {"infra/main.tf": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_cloudbuild(self):
        paths = ["cloudbuild.yaml"]
        status = {"cloudbuild.yaml": "modified"}
        assert is_architecture_impacting(paths, status)

    def test_routine_bugfix_not_impacting(self):
        paths = ["src/utils/helpers.py"]
        status = {"src/utils/helpers.py": "modified"}
        assert not is_architecture_impacting(paths, status)

    def test_only_markdown_not_impacting(self):
        paths = ["README.md", "CHANGELOG.md"]
        status = {"README.md": "modified", "CHANGELOG.md": "modified"}
        assert not is_architecture_impacting(paths, status)

    def test_nested_file_changes_not_impacting(self):
        paths = ["src/models/user.py", "src/models/post.py"]
        status = {
            "src/models/user.py": "modified",
            "src/models/post.py": "modified",
        }
        assert not is_architecture_impacting(paths, status)

    def test_empty_changeset(self):
        assert not is_architecture_impacting([], {})

    def test_top_level_file_added_no_directory(self):
        """A top-level file (no slash) should not count as a new directory."""
        paths = ["setup.cfg"]
        status = {"setup.cfg": "added"}
        assert not is_architecture_impacting(paths, status)


class TestCheckArchitectureFolder:
    """Tier 1: architecture folder existence checks."""

    def test_no_architecture_folder_emits_high_priority(self):
        analyzer = DocAnalyzer(
            changed_paths=["src/foo.py"],
            changed_paths_with_status={"src/foo.py": "modified"},
            existing_repo_paths=set(),
        )
        suggestions = analyzer.check_architecture_folder()
        assert len(suggestions) == 1
        assert suggestions[0].priority == "high"
        assert "architecture/" in suggestions[0].file

    def test_architecture_dir_present(self):
        analyzer = DocAnalyzer(
            changed_paths=["src/foo.py"],
            changed_paths_with_status={"src/foo.py": "modified"},
            existing_repo_paths={"architecture/"},
        )
        assert analyzer.check_architecture_folder() == []

    def test_docs_dir_present(self):
        analyzer = DocAnalyzer(
            changed_paths=["src/foo.py"],
            changed_paths_with_status={"src/foo.py": "modified"},
            existing_repo_paths={"docs/"},
        )
        assert analyzer.check_architecture_folder() == []

    def test_doc_dir_present(self):
        analyzer = DocAnalyzer(
            changed_paths=["src/foo.py"],
            changed_paths_with_status={"src/foo.py": "modified"},
            existing_repo_paths={"doc/"},
        )
        assert analyzer.check_architecture_folder() == []


class TestCheckConventionFiles:
    """Tier 1: convention file freshness checks."""

    def test_impacting_pr_with_stale_claude_md(self):
        """Architecture-impacting PR + CLAUDE.md exists but not changed -> suggestion."""
        analyzer = DocAnalyzer(
            changed_paths=["newpkg/main.py"],
            changed_paths_with_status={"newpkg/main.py": "added"},
            existing_repo_paths={"CLAUDE.md", "docs/"},
        )
        suggestions = analyzer.check_convention_files()
        assert len(suggestions) == 1
        assert suggestions[0].file == "CLAUDE.md"
        assert "was not updated" in suggestions[0].reason

    def test_impacting_pr_with_updated_agents_md(self):
        """Architecture-impacting PR + AGENTS.md exists and IS changed -> no suggestion."""
        analyzer = DocAnalyzer(
            changed_paths=["newpkg/main.py", "AGENTS.md"],
            changed_paths_with_status={
                "newpkg/main.py": "added",
                "AGENTS.md": "modified",
            },
            existing_repo_paths={"AGENTS.md", "docs/"},
        )
        suggestions = analyzer.check_convention_files()
        assert not any(s.file == "AGENTS.md" for s in suggestions)

    def test_non_impacting_pr_with_convention_files_silent(self):
        """Non-impacting PR + convention files exist -> zero suggestions."""
        analyzer = DocAnalyzer(
            changed_paths=["src/utils/helpers.py"],
            changed_paths_with_status={"src/utils/helpers.py": "modified"},
            existing_repo_paths={"CLAUDE.md", "AGENTS.md", "CONTRIBUTING.md", "docs/"},
        )
        assert analyzer.check_convention_files() == []

    def test_convention_file_not_in_repo(self):
        """Convention file doesn't exist in repo -> no suggestion even if PR is impacting."""
        analyzer = DocAnalyzer(
            changed_paths=["newpkg/main.py"],
            changed_paths_with_status={"newpkg/main.py": "added"},
            existing_repo_paths={"docs/"},
        )
        assert analyzer.check_convention_files() == []

    def test_multiple_stale_convention_files(self):
        """Multiple convention files exist and none updated -> one suggestion per file."""
        analyzer = DocAnalyzer(
            changed_paths=["Dockerfile"],
            changed_paths_with_status={"Dockerfile": "modified"},
            existing_repo_paths={
                "CLAUDE.md",
                "AGENTS.md",
                "CONTRIBUTING.md",
                "docs/",
            },
        )
        suggestions = analyzer.check_convention_files()
        suggested_files = {s.file for s in suggestions}
        assert "CLAUDE.md" in suggested_files
        assert "AGENTS.md" in suggested_files
        assert "CONTRIBUTING.md" in suggested_files

    def test_dockerfile_addition_triggers_impact(self):
        """PR adds new Dockerfile + CLAUDE.md exists but not updated -> suggestion."""
        analyzer = DocAnalyzer(
            changed_paths=["Dockerfile.prod"],
            changed_paths_with_status={"Dockerfile.prod": "added"},
            existing_repo_paths={"CLAUDE.md", "docs/"},
        )
        suggestions = analyzer.check_convention_files()
        assert len(suggestions) == 1
        assert suggestions[0].file == "CLAUDE.md"


class TestCheckSourceToDocsMapping:
    """Tier 2: explicit source-to-docs mapping from .ai-reviewer.yaml."""

    def test_matching_files_without_doc_changes(self):
        doc_config = {
            "source_to_docs_mapping": {
                "src/api/*": ["docs/api.md"],
                "src/models/*": ["docs/models.md"],
            }
        }
        analyzer = DocAnalyzer(
            changed_paths=["src/api/routes.py"],
            changed_paths_with_status={"src/api/routes.py": "modified"},
            existing_repo_paths={"docs/"},
            doc_config=doc_config,
        )
        suggestions = analyzer.check_source_to_docs_mapping()
        assert len(suggestions) == 1
        assert suggestions[0].file == "docs/api.md"
        assert "source_to_docs_mapping" in suggestions[0].reason

    def test_matching_files_with_doc_changes(self):
        doc_config = {
            "source_to_docs_mapping": {
                "src/api/*": ["docs/api.md"],
            }
        }
        analyzer = DocAnalyzer(
            changed_paths=["src/api/routes.py", "docs/api.md"],
            changed_paths_with_status={
                "src/api/routes.py": "modified",
                "docs/api.md": "modified",
            },
            existing_repo_paths={"docs/"},
            doc_config=doc_config,
        )
        assert analyzer.check_source_to_docs_mapping() == []

    def test_no_matching_files(self):
        doc_config = {
            "source_to_docs_mapping": {
                "src/api/*": ["docs/api.md"],
            }
        }
        analyzer = DocAnalyzer(
            changed_paths=["src/utils/helpers.py"],
            changed_paths_with_status={"src/utils/helpers.py": "modified"},
            existing_repo_paths={"docs/"},
            doc_config=doc_config,
        )
        assert analyzer.check_source_to_docs_mapping() == []

    def test_no_doc_config(self):
        analyzer = DocAnalyzer(
            changed_paths=["src/api/routes.py"],
            changed_paths_with_status={"src/api/routes.py": "modified"},
            existing_repo_paths={"docs/"},
            doc_config=None,
        )
        assert analyzer.check_source_to_docs_mapping() == []

    def test_empty_mapping(self):
        doc_config = {"source_to_docs_mapping": {}}
        analyzer = DocAnalyzer(
            changed_paths=["src/api/routes.py"],
            changed_paths_with_status={"src/api/routes.py": "modified"},
            existing_repo_paths={"docs/"},
            doc_config=doc_config,
        )
        assert analyzer.check_source_to_docs_mapping() == []

    def test_multiple_targets_for_one_glob(self):
        doc_config = {
            "source_to_docs_mapping": {
                "src/core/*": ["docs/core.md", "docs/architecture.md"],
            }
        }
        analyzer = DocAnalyzer(
            changed_paths=["src/core/engine.py"],
            changed_paths_with_status={"src/core/engine.py": "modified"},
            existing_repo_paths={"docs/"},
            doc_config=doc_config,
        )
        suggestions = analyzer.check_source_to_docs_mapping()
        target_files = {s.file for s in suggestions}
        assert target_files == {"docs/core.md", "docs/architecture.md"}

    def test_multiple_sources_same_target_deduped(self):
        """Two globs matching different changed files point to the same doc target."""
        doc_config = {
            "source_to_docs_mapping": {
                "src/api/*": ["docs/api.md"],
                "src/routes/*": ["docs/api.md"],
            }
        }
        analyzer = DocAnalyzer(
            changed_paths=["src/api/v1.py", "src/routes/main.py"],
            changed_paths_with_status={
                "src/api/v1.py": "modified",
                "src/routes/main.py": "modified",
            },
            existing_repo_paths={"docs/"},
            doc_config=doc_config,
        )
        suggestions = analyzer.check_source_to_docs_mapping()
        assert len(suggestions) == 1
        assert suggestions[0].file == "docs/api.md"


class TestDocAnalyzerRun:
    """Integration tests for DocAnalyzer.run()."""

    def test_enabled_false_skips_entirely(self):
        doc_config = {"enabled": False}
        analyzer = DocAnalyzer(
            changed_paths=["newpkg/main.py"],
            changed_paths_with_status={"newpkg/main.py": "added"},
            existing_repo_paths=set(),
            doc_config=doc_config,
        )
        assert analyzer.run() == []

    def test_enabled_true_runs_checks(self):
        doc_config = {"enabled": True}
        analyzer = DocAnalyzer(
            changed_paths=["newpkg/main.py"],
            changed_paths_with_status={"newpkg/main.py": "added"},
            existing_repo_paths=set(),
            doc_config=doc_config,
        )
        suggestions = analyzer.run()
        assert len(suggestions) >= 1

    def test_no_doc_config_runs_tier1(self):
        analyzer = DocAnalyzer(
            changed_paths=["newpkg/main.py"],
            changed_paths_with_status={"newpkg/main.py": "added"},
            existing_repo_paths={"CLAUDE.md"},
            doc_config=None,
        )
        suggestions = analyzer.run()
        assert any(s.file == "architecture/" for s in suggestions)
        assert any(s.file == "CLAUDE.md" for s in suggestions)

    def test_deduplication_by_file(self):
        """If architecture check and convention check both target the same file, deduplicate."""
        doc_config = {
            "source_to_docs_mapping": {
                "src/*": ["architecture/"],
            }
        }
        analyzer = DocAnalyzer(
            changed_paths=["src/main.py"],
            changed_paths_with_status={"src/main.py": "added"},
            existing_repo_paths=set(),
            doc_config=doc_config,
        )
        suggestions = analyzer.run()
        file_counts = {}
        for s in suggestions:
            file_counts[s.file] = file_counts.get(s.file, 0) + 1
        assert all(c == 1 for c in file_counts.values()), (
            f"Duplicate files in suggestions: {file_counts}"
        )

    def test_high_priority_sorted_first(self):
        analyzer = DocAnalyzer(
            changed_paths=["newpkg/main.py"],
            changed_paths_with_status={"newpkg/main.py": "added"},
            existing_repo_paths={"CLAUDE.md"},
            doc_config=None,
        )
        suggestions = analyzer.run()
        assert len(suggestions) >= 2
        assert suggestions[0].priority == "high"

    def test_routine_pr_zero_config_silent(self):
        """Non-impacting PR with zero config -> only architecture folder check matters."""
        analyzer = DocAnalyzer(
            changed_paths=["src/utils/helpers.py"],
            changed_paths_with_status={"src/utils/helpers.py": "modified"},
            existing_repo_paths={"docs/", "CLAUDE.md", "AGENTS.md"},
            doc_config=None,
        )
        suggestions = analyzer.run()
        assert suggestions == []

    def test_tier2_runs_alongside_tier1(self):
        doc_config = {
            "source_to_docs_mapping": {
                "src/api/*": ["docs/api-reference.md"],
            }
        }
        analyzer = DocAnalyzer(
            changed_paths=["src/api/routes.py", "newpkg/init.py"],
            changed_paths_with_status={
                "src/api/routes.py": "modified",
                "newpkg/init.py": "added",
            },
            existing_repo_paths={"CLAUDE.md"},
            doc_config=doc_config,
        )
        suggestions = analyzer.run()
        files = {s.file for s in suggestions}
        assert "architecture/" in files
        assert "CLAUDE.md" in files
        assert "docs/api-reference.md" in files


class TestFormatDocComment:
    """Tests for format_doc_comment()."""

    def test_produces_marker_and_header(self):
        marker = "<!-- AI-CODE-REVIEWER-DOC-BOT -->"
        suggestions = [
            DocSuggestion(file="CLAUDE.md", reason="Needs update", priority="normal"),
        ]
        result = format_doc_comment(suggestions, marker)
        assert result.startswith(marker)
        assert "## Documentation Review" in result

    def test_high_priority_red_circle(self):
        marker = "<!-- TEST -->"
        suggestions = [
            DocSuggestion(
                file="architecture/",
                reason="No architecture folder",
                priority="high",
            ),
        ]
        result = format_doc_comment(suggestions, marker)
        assert "\U0001f534" in result
        assert "architecture/" in result

    def test_normal_priority_yellow_circle(self):
        marker = "<!-- TEST -->"
        suggestions = [
            DocSuggestion(file="CLAUDE.md", reason="Needs update", priority="normal"),
        ]
        result = format_doc_comment(suggestions, marker)
        assert "\U0001f7e1" in result

    def test_empty_suggestions_all_current(self):
        marker = "<!-- TEST -->"
        result = format_doc_comment([], marker)
        assert marker in result
        assert "All documentation looks current" in result

    def test_multiple_suggestions_listed(self):
        marker = "<!-- TEST -->"
        suggestions = [
            DocSuggestion(file="CLAUDE.md", reason="Reason A"),
            DocSuggestion(file="AGENTS.md", reason="Reason B"),
        ]
        result = format_doc_comment(suggestions, marker)
        assert "CLAUDE.md" in result
        assert "AGENTS.md" in result
        assert result.count("\n- ") == 2

    def test_format_preserves_reason_text(self):
        marker = "<!-- TEST -->"
        reason = "Custom reason with `backticks` and details."
        suggestions = [DocSuggestion(file="foo.md", reason=reason)]
        result = format_doc_comment(suggestions, marker)
        assert reason in result


class TestDocSuggestionDataclass:
    """Basic tests for the DocSuggestion dataclass."""

    def test_default_priority_is_normal(self):
        s = DocSuggestion(file="x.md", reason="test")
        assert s.priority == "normal"

    def test_frozen(self):
        s = DocSuggestion(file="x.md", reason="test")
        with pytest.raises(AttributeError):
            s.file = "y.md"  # type: ignore[misc]

    def test_equality(self):
        a = DocSuggestion(file="x.md", reason="r", priority="high")
        b = DocSuggestion(file="x.md", reason="r", priority="high")
        assert a == b

    def test_hash_for_set_membership(self):
        a = DocSuggestion(file="x.md", reason="r")
        b = DocSuggestion(file="x.md", reason="r")
        assert {a, b} == {a}
