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


CONVENTION_FILES = [
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    ".cursor/rules/README.md",
]

ARCHITECTURE_DIRS = ["architecture/", "docs/", "doc/"]

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
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
    return False


def _is_entry_point(path: str) -> bool:
    basename = os.path.basename(path)
    name, _ = os.path.splitext(basename)
    return name in _ENTRY_POINT_BASENAMES


def _has_new_top_level_dir(changed_paths_with_status: dict[str, str]) -> bool:
    """True if any added file introduces a previously-unseen top-level directory."""
    existing_dirs: set[str] = set()
    added_dirs: set[str] = set()
    for path, status in changed_paths_with_status.items():
        parts = path.split("/")
        if len(parts) < 2:
            continue
        top = parts[0]
        if status == "added":
            added_dirs.add(top)
        else:
            existing_dirs.add(top)
    return bool(added_dirs - existing_dirs)


def _has_removed_top_level_dir(changed_paths_with_status: dict[str, str]) -> bool:
    """True if every file under some top-level dir was removed."""
    dir_statuses: dict[str, set[str]] = {}
    for path, status in changed_paths_with_status.items():
        parts = path.split("/")
        if len(parts) < 2:
            continue
        dir_statuses.setdefault(parts[0], set()).add(status)
    return any(statuses == {"removed"} for statuses in dir_statuses.values())


def is_architecture_impacting(
    changed_paths: list[str],
    changed_paths_with_status: dict[str, str],
) -> bool:
    if _has_new_top_level_dir(changed_paths_with_status):
        return True
    if _has_removed_top_level_dir(changed_paths_with_status):
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
    ) -> None:
        self.changed_paths = changed_paths
        self.changed_paths_with_status = changed_paths_with_status
        self.existing_repo_paths = existing_repo_paths
        self.doc_config = doc_config

    def check_architecture_folder(self) -> list[DocSuggestion]:
        for d in ARCHITECTURE_DIRS:
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
            self.changed_paths, self.changed_paths_with_status
        ):
            return []

        suggestions: list[DocSuggestion] = []
        changed_set = set(self.changed_paths)
        for conv_file in CONVENTION_FILES:
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
        mapping: dict[str, list[str]] = self.doc_config.get(
            "source_to_docs_mapping", {}
        )
        if not mapping:
            return []

        changed_set = set(self.changed_paths)
        unupdated_targets: dict[str, str] = {}

        for glob_pattern, doc_targets in mapping.items():
            matched_sources = [
                p for p in self.changed_paths if fnmatch.fnmatch(p, glob_pattern)
            ]
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


def format_doc_comment(suggestions: list[DocSuggestion], marker: str) -> str:
    lines = [marker, "", "## Documentation Review", ""]
    if not suggestions:
        lines.append(
            "All documentation looks current — no updates needed for this PR."
        )
        return "\n".join(lines)

    lines.append(
        "The following documentation may need updates based on the changes in this PR:"
    )
    lines.append("")
    for s in suggestions:
        icon = "\U0001f534" if s.priority == "high" else "\U0001f7e1"
        lines.append(f"- {icon} **{s.file}**: {s.reason}")
    return "\n".join(lines)
