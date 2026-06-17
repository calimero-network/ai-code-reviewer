"""Tests for documentation analysis (Tier 1 zero-config and Tier 2 configured)."""

from __future__ import annotations

import pytest

from ai_reviewer.docs.analyzer import (
    DocAnalyzer,
    DocSuggestion,
    _apply_html_patches,
    format_doc_comment,
    is_architecture_impacting,
)


class TestIsArchitectureImpacting:
    """Tests for the architecture-impact heuristic."""

    def test_new_top_level_directory(self):
        paths = ["newpkg/main.py"]
        status = {"newpkg/main.py": "added"}
        assert is_architecture_impacting(paths, status, existing_repo_paths=set())

    def test_adding_file_to_existing_dir_not_impacting(self):
        """Adding a file to an already-existing top-level dir is NOT a new dir."""
        paths = ["src/new_module.py"]
        status = {"src/new_module.py": "added"}
        existing = {"src/"}
        assert not is_architecture_impacting(paths, status, existing_repo_paths=existing)

    def test_removed_top_level_directory(self):
        paths = ["oldpkg/util.py"]
        status = {"oldpkg/util.py": "removed"}
        assert is_architecture_impacting(paths, status, existing_repo_paths=set())

    def test_removing_file_from_existing_dir_not_impacting(self):
        """Removing one file from a dir that still exists is NOT a removed dir."""
        paths = ["oldpkg/util.py"]
        status = {"oldpkg/util.py": "removed"}
        existing = {"oldpkg/"}
        assert not is_architecture_impacting(paths, status, existing_repo_paths=existing)

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

    def test_custom_architecture_dirs(self):
        """User-configured architecture_dirs are respected."""
        analyzer = DocAnalyzer(
            changed_paths=["src/foo.py"],
            changed_paths_with_status={"src/foo.py": "modified"},
            existing_repo_paths={"design/"},
            architecture_dirs=["design/", "specs/"],
        )
        assert analyzer.check_architecture_folder() == []

    def test_custom_architecture_dirs_missing(self):
        analyzer = DocAnalyzer(
            changed_paths=["src/foo.py"],
            changed_paths_with_status={"src/foo.py": "modified"},
            existing_repo_paths=set(),
            architecture_dirs=["design/"],
        )
        suggestions = analyzer.check_architecture_folder()
        assert len(suggestions) == 1
        assert suggestions[0].priority == "high"


class TestCheckConventionFiles:
    """Tier 1: convention file freshness checks."""

    def test_impacting_pr_with_stale_claude_md(self):
        """Architecture-impacting PR + CLAUDE.md exists but not changed -> suggestion."""
        analyzer = DocAnalyzer(
            changed_paths=["pyproject.toml"],
            changed_paths_with_status={"pyproject.toml": "modified"},
            existing_repo_paths={"CLAUDE.md", "docs/"},
        )
        suggestions = analyzer.check_convention_files()
        assert len(suggestions) == 1
        assert suggestions[0].file == "CLAUDE.md"
        assert "was not updated" in suggestions[0].reason

    def test_impacting_pr_with_updated_agents_md(self):
        """Architecture-impacting PR + AGENTS.md exists and IS changed -> no suggestion."""
        analyzer = DocAnalyzer(
            changed_paths=["pyproject.toml", "AGENTS.md"],
            changed_paths_with_status={
                "pyproject.toml": "modified",
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
            changed_paths=["pyproject.toml"],
            changed_paths_with_status={"pyproject.toml": "modified"},
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

    def test_custom_convention_files(self):
        """User-configured convention_files are respected."""
        analyzer = DocAnalyzer(
            changed_paths=["Dockerfile"],
            changed_paths_with_status={"Dockerfile": "modified"},
            existing_repo_paths={"CUSTOM_RULES.md", "docs/"},
            convention_files=["CUSTOM_RULES.md"],
        )
        suggestions = analyzer.check_convention_files()
        assert len(suggestions) == 1
        assert suggestions[0].file == "CUSTOM_RULES.md"


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
            changed_paths=["pyproject.toml"],
            changed_paths_with_status={"pyproject.toml": "modified"},
            existing_repo_paths=set(),
            doc_config=doc_config,
        )
        suggestions = analyzer.run()
        assert len(suggestions) >= 1

    def test_no_doc_config_runs_tier1(self):
        analyzer = DocAnalyzer(
            changed_paths=["pyproject.toml"],
            changed_paths_with_status={"pyproject.toml": "modified"},
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
        file_counts: dict[str, int] = {}
        for s in suggestions:
            file_counts[s.file] = file_counts.get(s.file, 0) + 1
        assert all(c == 1 for c in file_counts.values()), (
            f"Duplicate files in suggestions: {file_counts}"
        )

    def test_high_priority_sorted_first(self):
        analyzer = DocAnalyzer(
            changed_paths=["pyproject.toml"],
            changed_paths_with_status={"pyproject.toml": "modified"},
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
            changed_paths=["src/api/routes.py", "pyproject.toml"],
            changed_paths_with_status={
                "src/api/routes.py": "modified",
                "pyproject.toml": "modified",
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


class TestDocDraft:
    """Tests for the DocDraft dataclass."""

    def test_fields(self):
        from ai_reviewer.docs.analyzer import DocDraft, DocSuggestion

        s = DocSuggestion(file="README.md", reason="needs update")
        d = DocDraft(suggestion=s, updated_content="# Updated\nfoo")
        assert d.suggestion is s
        assert d.updated_content == "# Updated\nfoo"
        assert d.error is None

    def test_error_field(self):
        from ai_reviewer.docs.analyzer import DocDraft, DocSuggestion

        s = DocSuggestion(file="README.md", reason="needs update")
        d = DocDraft(suggestion=s, updated_content="", error="file not found")
        assert d.error == "file not found"
        assert d.updated_content == ""


class TestGenerateDocDrafts:
    """Tests for generate_doc_drafts() — mocks AnthropicClient and GitHubClient."""

    @pytest.mark.asyncio
    async def test_skips_directory_suggestions(self):
        """Suggestions with a trailing '/' (directories) are skipped."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_reviewer.config import AnthropicApiConfig
        from ai_reviewer.docs.analyzer import DocSuggestion, generate_doc_drafts

        dir_suggestion = DocSuggestion(file="architecture/", reason="missing folder")
        mock_gh = MagicMock()

        with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            cfg = AnthropicApiConfig(api_key="sk-test")
            drafts = await generate_doc_drafts(
                suggestions=[dir_suggestion],
                diff="some diff",
                repo_name="org/repo",
                ref="abc123",
                anthropic_cfg=cfg,
                gh=mock_gh,
            )

        assert drafts == []
        mock_instance.run_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_generates_draft_for_file_suggestion(self):
        """A file suggestion produces a DocDraft with updated_content."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_reviewer.config import AnthropicApiConfig
        from ai_reviewer.docs.analyzer import DocSuggestion, generate_doc_drafts

        suggestion = DocSuggestion(file="README.md", reason="cli.py changed")
        mock_file_content = MagicMock()
        mock_file_content.decoded_content = b"# Old README\n"

        mock_gh = MagicMock()
        mock_gh.get_file_contents.return_value = mock_file_content

        with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.run_completion = AsyncMock(return_value="# Updated README\n")
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            cfg = AnthropicApiConfig(api_key="sk-test")
            drafts = await generate_doc_drafts(
                suggestions=[suggestion],
                diff="diff --git a/src/cli.py...",
                repo_name="org/repo",
                ref="abc123",
                anthropic_cfg=cfg,
                gh=mock_gh,
            )

        assert len(drafts) == 1
        assert drafts[0].updated_content == "# Updated README"  # strip() removes trailing \n
        assert drafts[0].error is None
        assert drafts[0].suggestion is suggestion

    @pytest.mark.asyncio
    async def test_handles_file_fetch_error(self):
        """If fetching the current file fails, draft has error set."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_reviewer.config import AnthropicApiConfig
        from ai_reviewer.docs.analyzer import DocSuggestion, generate_doc_drafts

        suggestion = DocSuggestion(file="docs/api.md", reason="api changed")
        mock_gh = MagicMock()
        mock_gh.get_file_contents.side_effect = Exception("404 not found")

        with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            cfg = AnthropicApiConfig(api_key="sk-test")
            drafts = await generate_doc_drafts(
                suggestions=[suggestion],
                diff="diff",
                repo_name="org/repo",
                ref="abc123",
                anthropic_cfg=cfg,
                gh=mock_gh,
            )

        assert len(drafts) == 1
        assert drafts[0].error == "404 not found"
        assert drafts[0].updated_content == ""
        mock_instance.run_completion.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_max_files(self):
        """max_files caps how many suggestions are processed."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_reviewer.config import AnthropicApiConfig
        from ai_reviewer.docs.analyzer import DocSuggestion, generate_doc_drafts

        suggestions = [DocSuggestion(file=f"docs/file{i}.md", reason="changed") for i in range(5)]
        mock_file_content = MagicMock()
        mock_file_content.decoded_content = b"content"

        mock_gh = MagicMock()
        mock_gh.get_file_contents.return_value = mock_file_content

        with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.run_completion = AsyncMock(return_value="updated")
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            cfg = AnthropicApiConfig(api_key="sk-test")
            drafts = await generate_doc_drafts(
                suggestions=suggestions,
                diff="diff",
                repo_name="org/repo",
                ref="abc123",
                anthropic_cfg=cfg,
                gh=mock_gh,
                max_files=2,
            )

        assert len(drafts) == 2
        assert mock_instance.run_completion.call_count == 2


class TestIsNoUpdateResponse:
    """Tests for _is_no_update_response — the sentinel parser.

    Regression for calimero-network/core#2296: bare-equality detection
    failed when the model appended a justification, and the entire response
    (sentinel + reasoning) was written to disk as the new HTML.
    """

    def test_bare_sentinel(self):
        from ai_reviewer.docs.analyzer import _is_no_update_response

        assert _is_no_update_response("NO_UPDATE_NEEDED")

    def test_sentinel_with_trailing_explanation(self):
        # Exact shape from core#2296.
        from ai_reviewer.docs.analyzer import _is_no_update_response

        response = (
            "NO_UPDATE_NEEDED\n"
            "\n"
            "The code diff shows changes to .github/workflows/doc-update.yaml, "
            "which is a GitHub Actions workflow configuration file."
        )
        assert _is_no_update_response(response)

    def test_sentinel_with_leading_blank_lines(self):
        from ai_reviewer.docs.analyzer import _is_no_update_response

        assert _is_no_update_response("\n\n  NO_UPDATE_NEEDED  \n")

    def test_sentinel_inside_markdown_fence(self):
        from ai_reviewer.docs.analyzer import _is_no_update_response

        assert _is_no_update_response("```\nNO_UPDATE_NEEDED\n```")

    def test_html_response_is_not_a_no_update(self):
        from ai_reviewer.docs.analyzer import _is_no_update_response

        assert not _is_no_update_response("<!DOCTYPE html>\n<html>...")

    def test_sentinel_buried_in_html_is_not_a_no_update(self):
        # The model returned real HTML that happens to mention the token —
        # e.g. inside a <pre> block. Should round-trip as an update, not be
        # discarded.
        from ai_reviewer.docs.analyzer import _is_no_update_response

        html = "<!DOCTYPE html>\n<pre>NO_UPDATE_NEEDED is the sentinel</pre>"
        assert not _is_no_update_response(html)

    def test_empty_response(self):
        from ai_reviewer.docs.analyzer import _is_no_update_response

        assert not _is_no_update_response("")


class TestApplyHtmlPatches:
    """Tests for _apply_html_patches — surgical FIND/REPLACE on HTML files."""

    def test_basic_replacement(self):
        original = "<div>old value</div>"
        response = (
            "<<<FIND\n<div>old value</div>\nFIND>>>\n<<<REPLACE\n<div>new value</div>\nREPLACE>>>"
        )
        assert _apply_html_patches(original, response) == "<div>new value</div>"

    def test_multiple_patches(self):
        original = "<a>foo</a><b>bar</b>"
        response = (
            "<<<FIND\n<a>foo</a>\nFIND>>>\n<<<REPLACE\n<a>FOO</a>\nREPLACE>>>\n"
            "<<<FIND\n<b>bar</b>\nFIND>>>\n<<<REPLACE\n<b>BAR</b>\nREPLACE>>>"
        )
        assert _apply_html_patches(original, response) == "<a>FOO</a><b>BAR</b>"

    def test_find_not_found_returns_none(self):
        original = "<div>actual content</div>"
        response = (
            "<<<FIND\n<div>different content</div>\nFIND>>>\n<<<REPLACE\n<div>x</div>\nREPLACE>>>"
        )
        assert _apply_html_patches(original, response) is None

    def test_malformed_no_blocks_returns_none(self):
        assert _apply_html_patches("<html>x</html>", "plain prose response") is None

    def test_whitespace_normalized_fallback(self):
        original = "<div>  value  </div>"
        # FIND has trailing spaces stripped — normalized match should still work
        response = "<<<FIND\n<div>  value\nFIND>>>\n<<<REPLACE\n<div>new\nREPLACE>>>"
        assert _apply_html_patches(original, response) is not None

    def test_leading_indentation_mismatch_applies(self):
        """The model often copies FIND with different leading indentation than
        the source — the whitespace-insensitive fallback should still apply it,
        and leave the rest of the document untouched."""
        original = "<ul>\n    <li>store: the storage crate</li>\n    <li>keep me</li>\n</ul>"
        # FIND lost the leading 4-space indent + reflowed the line.
        response = (
            "<<<FIND\n<li>store: the storage crate</li>\nFIND>>>\n"
            "<<<REPLACE\n<li>store: the CRDT storage crate</li>\nREPLACE>>>"
        )
        out = _apply_html_patches(original, response)
        assert out is not None
        assert "the CRDT storage crate" in out
        # Untouched lines + surrounding indentation preserved exactly.
        assert "    <li>keep me</li>" in out
        assert out.startswith("<ul>\n    <li>")

    def test_internal_whitespace_run_difference_applies(self):
        original = "<p>tools   crate   docs</p>"  # multiple spaces in source
        response = (
            "<<<FIND\n<p>tools crate docs</p>\nFIND>>>\n<<<REPLACE\n<p>updated</p>\nREPLACE>>>"
        )
        assert _apply_html_patches(original, response) == "<p>updated</p>"

    def test_crlf_markers_tolerated(self):
        original = "<div>old</div>"
        response = (
            "<<<FIND\r\n<div>old</div>\r\nFIND>>>\r\n<<<REPLACE\r\n<div>new</div>\r\nREPLACE>>>"
        )
        assert _apply_html_patches(original, response) == "<div>new</div>"

    def test_truly_absent_find_still_returns_none(self):
        original = "<div>actual content</div>"
        response = "<<<FIND\n<div>totally unrelated text</div>\nFIND>>>\n<<<REPLACE\n<div>x</div>\nREPLACE>>>"
        assert _apply_html_patches(original, response) is None

    def test_ambiguous_fuzzy_match_returns_none(self):
        """When the exact FIND misses but the whitespace-relaxed pattern matches
        more than once, refuse rather than patch the wrong occurrence."""
        # Same fragment twice; FIND's internal spacing differs from the source so
        # the exact path misses and the fuzzy path sees two candidates.
        original = "<td>store   crate</td>\n<td>store   crate</td>"
        response = "<<<FIND\n<td>store crate</td>\nFIND>>>\n<<<REPLACE\n<td>x</td>\nREPLACE>>>"
        assert _apply_html_patches(original, response) is None


class TestGenerateDocDraftsHtmlSentinel:
    """End-to-end tests that NO_UPDATE_NEEDED variants are filtered correctly."""

    @pytest.mark.asyncio
    async def test_html_no_update_sentinel_with_reasoning_filters_draft(self):
        """The exact PR #2296 failure: sentinel + trailing reasoning must
        produce no draft, not a draft whose updated_content is the model's
        explanation."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_reviewer.config import AnthropicApiConfig
        from ai_reviewer.docs.analyzer import DocSuggestion, generate_doc_drafts

        suggestion = DocSuggestion(
            file="architecture/app-lifecycle.html",
            reason="ci workflow change — scanning for updates",
        )
        mock_file_content = MagicMock()
        mock_file_content.decoded_content = b"<!DOCTYPE html>\n<html>...</html>"

        mock_gh = MagicMock()
        mock_gh.get_file_contents.return_value = mock_file_content

        model_response = (
            "NO_UPDATE_NEEDED\n\n"
            "The code diff shows changes to .github/workflows/doc-update.yaml, "
            "which is a GitHub Actions workflow configuration file."
        )

        with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.run_completion = AsyncMock(return_value=model_response)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            cfg = AnthropicApiConfig(api_key="sk-test")
            drafts = await generate_doc_drafts(
                suggestions=[suggestion],
                diff="diff",
                repo_name="org/repo",
                ref="abc123",
                anthropic_cfg=cfg,
                gh=mock_gh,
            )

        assert drafts == []

    @pytest.mark.asyncio
    async def test_html_non_html_response_marked_failed(self):
        """If the model forgets the sentinel and just explains in prose,
        treat as a failed draft rather than overwriting the file."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_reviewer.config import AnthropicApiConfig
        from ai_reviewer.docs.analyzer import DocSuggestion, generate_doc_drafts

        suggestion = DocSuggestion(file="architecture/index.html", reason="ci change")
        mock_file_content = MagicMock()
        mock_file_content.decoded_content = b"<!DOCTYPE html>\n<html>...</html>"

        mock_gh = MagicMock()
        mock_gh.get_file_contents.return_value = mock_file_content

        with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.run_completion = AsyncMock(
                return_value="No update needed; the workflow file is unrelated to this page."
            )
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            cfg = AnthropicApiConfig(api_key="sk-test")
            drafts = await generate_doc_drafts(
                suggestions=[suggestion],
                diff="diff",
                repo_name="org/repo",
                ref="abc123",
                anthropic_cfg=cfg,
                gh=mock_gh,
            )

        assert len(drafts) == 1
        assert drafts[0].updated_content == ""
        assert drafts[0].error is not None
        assert "could not apply HTML patches" in drafts[0].error

    @pytest.mark.asyncio
    async def test_html_real_update_passes_through(self):
        """A genuine FIND/REPLACE patch response is applied correctly."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_reviewer.config import AnthropicApiConfig
        from ai_reviewer.docs.analyzer import DocSuggestion, generate_doc_drafts

        suggestion = DocSuggestion(file="architecture/concepts.html", reason="api change")
        mock_file_content = MagicMock()
        mock_file_content.decoded_content = b"<!DOCTYPE html>\n<html>old</html>"

        mock_gh = MagicMock()
        mock_gh.get_file_contents.return_value = mock_file_content

        patch_response = (
            "<<<FIND\n<html>old</html>\nFIND>>>\n<<<REPLACE\n<html>new</html>\nREPLACE>>>"
        )
        expected_html = "<!DOCTYPE html>\n<html>new</html>"

        with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.run_completion = AsyncMock(return_value=patch_response)
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            cfg = AnthropicApiConfig(api_key="sk-test")
            drafts = await generate_doc_drafts(
                suggestions=[suggestion],
                diff="diff",
                repo_name="org/repo",
                ref="abc123",
                anthropic_cfg=cfg,
                gh=mock_gh,
            )

        assert len(drafts) == 1
        assert drafts[0].updated_content == expected_html
        assert drafts[0].error is None

    @pytest.mark.asyncio
    async def test_markdown_files_unaffected_by_sentinel_logic(self):
        """Sentinel handling is HTML-only; for .md the response is taken as-is.

        This guards against accidentally extending the sentinel filter to
        Markdown, where the prompt does not opt-in to NO_UPDATE_NEEDED.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from ai_reviewer.config import AnthropicApiConfig
        from ai_reviewer.docs.analyzer import DocSuggestion, generate_doc_drafts

        suggestion = DocSuggestion(file="docs/api.md", reason="api change")
        mock_file_content = MagicMock()
        mock_file_content.decoded_content = b"# Old"

        mock_gh = MagicMock()
        mock_gh.get_file_contents.return_value = mock_file_content

        with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.run_completion = AsyncMock(return_value="NO_UPDATE_NEEDED")
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            cfg = AnthropicApiConfig(api_key="sk-test")
            drafts = await generate_doc_drafts(
                suggestions=[suggestion],
                diff="diff",
                repo_name="org/repo",
                ref="abc123",
                anthropic_cfg=cfg,
                gh=mock_gh,
            )

        # Markdown drafts pass through whatever the model returned.
        assert len(drafts) == 1
        assert drafts[0].updated_content == "NO_UPDATE_NEEDED"


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
