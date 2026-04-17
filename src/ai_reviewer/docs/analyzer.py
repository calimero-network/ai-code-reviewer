"""Rule-based documentation analysis for PRs.

Two-tier design:
  Tier 1 (zero-config): probes for architecture folders and convention files,
    emits suggestions only when the PR is architecture-impacting.
  Tier 2 (configured): additionally checks explicit source_to_docs_mapping
    from the repo's .ai-reviewer.yaml documentation section.
"""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DocSuggestion:
    file: str
    reason: str
    priority: str = "normal"


DEFAULT_CONVENTION_FILES = [
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    ".cursor/rules/README.md",
]

DEFAULT_ARCHITECTURE_DIRS = ["architecture/", "docs/", "doc/"]

_MANIFEST_FILES = {
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Gemfile",
    "build.gradle",
    "pom.xml",
    "CMakeLists.txt",
}

_CI_PATTERNS = [
    ".github/workflows/*",
    ".gitlab-ci.yml",
    "Jenkinsfile",
]

_ENTRY_POINT_BASENAMES = {"main", "cli", "app", "index", "server"}

_INFRA_PATTERNS = [
    "Dockerfile*",
    "docker-compose*",
    "*.tf",
    "cloudbuild.yaml",
]


def _matches_any_glob(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _is_entry_point(path: str) -> bool:
    basename = os.path.basename(path)
    name, _ = os.path.splitext(basename)
    return name in _ENTRY_POINT_BASENAMES


def _has_new_top_level_dir(
    changed_paths_with_status: dict[str, str],
    existing_repo_paths: set[str],
) -> bool:
    """True if any added file introduces a genuinely new top-level directory.

    Uses *existing_repo_paths* (probed from the repo) to know which
    directories already exist, avoiding false positives when a PR only adds
    files to an existing directory without modifying other files there.
    """
    known_dirs: set[str] = set()
    for p in existing_repo_paths:
        stripped = p.rstrip("/")
        if "/" not in stripped:
            known_dirs.add(stripped)

    for path, status in changed_paths_with_status.items():
        parts = path.split("/")
        if len(parts) < 2:
            continue
        top = parts[0]
        if status != "added":
            known_dirs.add(top)

    for path, status in changed_paths_with_status.items():
        parts = path.split("/")
        if len(parts) < 2:
            continue
        if status == "added" and parts[0] not in known_dirs:
            return True
    return False


def _has_removed_top_level_dir(
    changed_paths_with_status: dict[str, str],
    existing_repo_paths: set[str],
) -> bool:
    """True if a top-level dir appears only with 'removed' status in the diff
    AND is not known to still exist in the repo via *existing_repo_paths*.
    """
    known_dirs: set[str] = set()
    for p in existing_repo_paths:
        stripped = p.rstrip("/")
        if "/" not in stripped:
            known_dirs.add(stripped)

    dir_statuses: dict[str, set[str]] = {}
    for path, status in changed_paths_with_status.items():
        parts = path.split("/")
        if len(parts) < 2:
            continue
        dir_statuses.setdefault(parts[0], set()).add(status)

    return any(
        statuses == {"removed"} and top_dir not in known_dirs
        for top_dir, statuses in dir_statuses.items()
    )


def is_architecture_impacting(
    changed_paths: list[str],
    changed_paths_with_status: dict[str, str],
    existing_repo_paths: set[str] | None = None,
) -> bool:
    repo_paths = existing_repo_paths or set()
    if _has_new_top_level_dir(changed_paths_with_status, repo_paths):
        return True
    if _has_removed_top_level_dir(changed_paths_with_status, repo_paths):
        return True
    for path in changed_paths:
        basename = os.path.basename(path)
        if basename in _MANIFEST_FILES:
            return True
        if _matches_any_glob(path, _CI_PATTERNS):
            return True
        if _is_entry_point(path):
            return True
        if _matches_any_glob(basename, _INFRA_PATTERNS):
            return True
    return False


class DocAnalyzer:
    """Analyzes a PR for missing documentation updates."""

    def __init__(
        self,
        changed_paths: list[str],
        changed_paths_with_status: dict[str, str],
        existing_repo_paths: set[str],
        doc_config: dict | None = None,
        architecture_dirs: list[str] | None = None,
        convention_files: list[str] | None = None,
    ) -> None:
        self.changed_paths = changed_paths
        self.changed_paths_with_status = changed_paths_with_status
        self.existing_repo_paths = existing_repo_paths
        self.doc_config = doc_config
        self.architecture_dirs = architecture_dirs or DEFAULT_ARCHITECTURE_DIRS
        self.convention_files = convention_files or DEFAULT_CONVENTION_FILES

    def check_architecture_folder(self) -> list[DocSuggestion]:
        for d in self.architecture_dirs:
            if d in self.existing_repo_paths:
                return []
        return [
            DocSuggestion(
                file="architecture/",
                reason=(
                    "This repository has no architecture documentation folder "
                    "(e.g. `architecture/`, `docs/`, or `doc/`). "
                    "Consider adding one to document high-level design decisions."
                ),
                priority="high",
            )
        ]

    def check_convention_files(self) -> list[DocSuggestion]:
        if not is_architecture_impacting(
            self.changed_paths,
            self.changed_paths_with_status,
            self.existing_repo_paths,
        ):
            return []

        suggestions: list[DocSuggestion] = []
        changed_set = set(self.changed_paths)
        for conv_file in self.convention_files:
            if conv_file in self.existing_repo_paths and conv_file not in changed_set:
                suggestions.append(
                    DocSuggestion(
                        file=conv_file,
                        reason=(
                            f"`{conv_file}` exists but was not updated — consider "
                            "updating it to reflect the architecture changes in this PR."
                        ),
                    )
                )
        return suggestions

    def check_source_to_docs_mapping(self) -> list[DocSuggestion]:
        if self.doc_config is None:
            return []
        mapping: dict[str, list[str]] = self.doc_config.get("source_to_docs_mapping", {})
        if not mapping:
            return []

        changed_set = set(self.changed_paths)
        unupdated_targets: dict[str, str] = {}

        for glob_pattern, doc_targets in mapping.items():
            matched_sources = [p for p in self.changed_paths if fnmatch.fnmatch(p, glob_pattern)]
            if not matched_sources:
                continue
            for target in doc_targets:
                if target not in changed_set and target not in unupdated_targets:
                    unupdated_targets[target] = glob_pattern

        return [
            DocSuggestion(
                file=target,
                reason=(
                    f"Files matching `{pattern}` were changed but `{target}` "
                    "was not updated (per `source_to_docs_mapping`)."
                ),
            )
            for target, pattern in sorted(unupdated_targets.items())
        ]

    def run(self) -> list[DocSuggestion]:
        if self.doc_config is not None and not self.doc_config.get("enabled", True):
            return []

        suggestions: list[DocSuggestion] = []
        suggestions.extend(self.check_architecture_folder())
        suggestions.extend(self.check_convention_files())
        suggestions.extend(self.check_source_to_docs_mapping())

        seen: set[str] = set()
        deduped: list[DocSuggestion] = []
        for s in suggestions:
            if s.file not in seen:
                seen.add(s.file)
                deduped.append(s)

        deduped.sort(key=lambda s: (0 if s.priority == "high" else 1, s.file))
        return deduped


@dataclass
class DocDraft:
    """AI-generated full-file update for a stale documentation file."""

    suggestion: DocSuggestion
    updated_content: str
    error: str | None = None


_DOC_DRAFT_SYSTEM = """\
You are a technical writer updating documentation after a code change.
Given the current content of a documentation file and a code diff, rewrite the
COMPLETE file incorporating all necessary updates.

Rules:
- Return the ENTIRE updated file, not just the changed sections.
- Only change sections that are actually affected by the diff.
- Preserve all sections that don't need updating, word for word.
- Be precise and concise. No marketing language.
- Do not add any commentary before or after the file content — just the file.
"""

_MAX_DIFF_CHARS = 4000
_MAX_DOC_CHARS = 8000


async def generate_doc_drafts(
    suggestions: list[DocSuggestion],
    diff: str,
    repo_name: str,
    ref: str,
    anthropic_cfg: object,
    gh: object,
    model: str = "claude-sonnet-4-6",
    max_files: int = 5,
) -> list[DocDraft]:
    """Generate AI-drafted full-file updates for stale documentation files.

    Only processes suggestions that map to real files (produced by
    ``check_source_to_docs_mapping``).  Architecture/convention suggestions
    without a concrete file target are skipped.

    Returns one ``DocDraft`` per processed suggestion with the complete updated
    file content ready for committing.
    """
    from ai_reviewer.agents.anthropic_client import AnthropicClient  # local to avoid circular

    drafts: list[DocDraft] = []
    truncated_diff = diff[:_MAX_DIFF_CHARS] + ("…" if len(diff) > _MAX_DIFF_CHARS else "")

    candidates = [s for s in suggestions if not s.file.endswith("/")][:max_files]

    async with AnthropicClient(anthropic_cfg) as client:  # type: ignore[arg-type]
        for suggestion in candidates:
            try:
                raw = gh.get_file_contents(repo_name, suggestion.file, ref)  # type: ignore[union-attr]
                if isinstance(raw, list):
                    drafts.append(
                        DocDraft(
                            suggestion=suggestion,
                            updated_content="",
                            error="file resolved to multiple entries",
                        )
                    )
                    continue
                current_content = raw.decoded_content.decode("utf-8", errors="replace")
            except Exception as exc:
                drafts.append(DocDraft(suggestion=suggestion, updated_content="", error=str(exc)))
                continue

            truncated_doc = current_content[:_MAX_DOC_CHARS] + (
                "\n…(truncated)" if len(current_content) > _MAX_DOC_CHARS else ""
            )

            user_prompt = (
                f"## Documentation File: {suggestion.file}\n\n"
                f"{truncated_doc}\n\n"
                f"## Code Diff\n\n{truncated_diff}\n\n"
                f"## Why This File Needs Updating\n\n{suggestion.reason}\n\n"
                "Return the complete updated file content."
            )

            try:
                result = await client.run_completion(
                    model=model,
                    system=_DOC_DRAFT_SYSTEM,
                    user=user_prompt,
                    max_tokens=8192,
                )
                drafts.append(DocDraft(suggestion=suggestion, updated_content=result.strip()))
            except Exception as exc:
                drafts.append(DocDraft(suggestion=suggestion, updated_content="", error=str(exc)))

    return drafts


def format_doc_comment(suggestions: list[DocSuggestion], marker: str) -> str:
    lines = [marker, "", "## Documentation Review", ""]
    if not suggestions:
        lines.append("All documentation looks current — no updates needed for this PR.")
        return "\n".join(lines)

    lines.append("The following documentation may need updates based on the changes in this PR:")
    lines.append("")
    for s in suggestions:
        icon = "\U0001f534" if s.priority == "high" else "\U0001f7e1"
        lines.append(f"- {icon} **{s.file}**: {s.reason}")
    return "\n".join(lines)
