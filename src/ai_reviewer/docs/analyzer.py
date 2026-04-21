"""Rule-based documentation analysis for PRs.

Two-tier design:
  Tier 1 (zero-config): probes for architecture folders and convention files,
    emits suggestions only when the PR is architecture-impacting.
  Tier 2 (configured): additionally checks explicit source_to_docs_mapping
    from the repo's .ai-reviewer.yaml documentation section.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re as _re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig
    from ai_reviewer.github.client import GitHubClient


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

DEFAULT_ARCHITECTURE_DIRS = ["architecture/", "docs/", "doc/", "docs-static/"]

# Directories that may contain static HTML docs (GitHub Pages sites).
DEFAULT_STATIC_DOC_DIRS: list[str] = ["architecture/", "docs/", "docs-static/"]

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
        static_docs_dirs: list[str] | None = None,
    ) -> None:
        self.changed_paths = changed_paths
        self.changed_paths_with_status = changed_paths_with_status
        self.existing_repo_paths = existing_repo_paths
        self.doc_config = doc_config
        self.architecture_dirs = architecture_dirs or DEFAULT_ARCHITECTURE_DIRS
        self.convention_files = convention_files or DEFAULT_CONVENTION_FILES
        # static_docs_dirs: explicit override or fall through to doc_config then default
        if static_docs_dirs is not None:
            self.static_docs_dirs = static_docs_dirs
        elif doc_config is not None:
            self.static_docs_dirs = doc_config.get("static_docs_dirs", DEFAULT_STATIC_DOC_DIRS)
        else:
            self.static_docs_dirs = DEFAULT_STATIC_DOC_DIRS

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
        # Track (glob_pattern, any_source_deleted) per unupdated target
        unupdated_targets: dict[str, tuple[str, bool]] = {}

        for glob_pattern, doc_targets in mapping.items():
            matched_sources = [p for p in self.changed_paths if fnmatch.fnmatch(p, glob_pattern)]
            if not matched_sources:
                continue
            any_deleted = any(
                self.changed_paths_with_status.get(p) == "removed" for p in matched_sources
            )
            for target in doc_targets:
                if target not in changed_set and target not in unupdated_targets:
                    unupdated_targets[target] = (glob_pattern, any_deleted)

        suggestions = []
        for target, (pattern, deleted) in sorted(unupdated_targets.items()):
            if deleted:
                reason = (
                    f"Files matching `{pattern}` were **deleted** but `{target}` "
                    "was not updated — references to removed code may now be stale "
                    "(per `source_to_docs_mapping`)."
                )
            else:
                reason = (
                    f"Files matching `{pattern}` were changed but `{target}` "
                    "was not updated (per `source_to_docs_mapping`)."
                )
            suggestions.append(DocSuggestion(file=target, reason=reason))
        return suggestions

    def check_static_html_docs(
        self, already_covered: set[str] | None = None
    ) -> list[DocSuggestion]:
        """Flag static HTML doc directories when architecture-impacting changes occur.

        Only emits suggestions when the PR touches something architecture-relevant
        AND a configured static docs directory actually exists in the repo.

        *already_covered* should be the set of file paths already flagged by
        :meth:`check_source_to_docs_mapping` so that :meth:`run` can pass the
        pre-computed result and avoid calling the mapping check twice.
        """
        if not self.static_docs_dirs:
            return []
        if not is_architecture_impacting(
            self.changed_paths,
            self.changed_paths_with_status,
            self.existing_repo_paths,
        ):
            return []

        covered = (
            already_covered
            if already_covered is not None
            else {s.file for s in self.check_source_to_docs_mapping()}
        )
        suggestions = []
        for d in self.static_docs_dirs:
            normalized = d if d.endswith("/") else d + "/"
            # Only flag dirs that exist in the repo and aren't already flagged by the mapping
            if normalized in self.existing_repo_paths and normalized not in covered:
                suggestions.append(
                    DocSuggestion(
                        file=normalized,
                        reason=(
                            f"Static HTML docs in `{normalized}` may need updating — "
                            "architecture-impacting changes detected. "
                            "On merge, `update-docs` will scan this directory and open a PR "
                            "if any pages need to change."
                        ),
                    )
                )
        return suggestions

    def run(self) -> list[DocSuggestion]:
        if self.doc_config is not None and not self.doc_config.get("enabled", True):
            return []

        suggestions: list[DocSuggestion] = []
        suggestions.extend(self.check_architecture_folder())
        suggestions.extend(self.check_convention_files())
        mapping = self.check_source_to_docs_mapping()
        suggestions.extend(mapping)
        suggestions.extend(self.check_static_html_docs(already_covered={s.file for s in mapping}))

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

_DOC_DRAFT_SYSTEM_HTML = """\
You are a technical writer updating a static HTML documentation page after a code change.
Given the current HTML file and a code diff, decide whether the page needs updating.

If NO update is needed, return exactly this single token and nothing else:
NO_UPDATE_NEEDED

If an update IS needed, return the COMPLETE updated HTML file.

Rules for HTML updates:
- Preserve ALL HTML structure, tags, attributes, CSS classes, inline styles, and scripts exactly.
- Only update human-readable text content that is made inaccurate by the code change.
- Do not reformat, re-indent, or restructure the HTML.
- Do not escape or unescape entities that were already correct.
- Do not add commentary before or after the HTML — return only the file or NO_UPDATE_NEEDED.
"""

# Sentinel returned by Claude when an HTML page does not need updating.
_NO_UPDATE_SENTINEL = "NO_UPDATE_NEEDED"

# ~4K chars ≈ ~1K tokens — keeps prompt cost low while providing enough context.
_MAX_DIFF_CHARS = 4000
# ~8K chars ≈ ~2K tokens — sufficient for most Markdown docs.
_MAX_DOC_CHARS = 8000
# HTML docs are larger due to markup; allow more context for accurate updates.
_MAX_DOC_CHARS_HTML = 20_000


def _strip_html_tags(html: str) -> str:
    """Strip HTML markup to plain text for relevance comparison."""
    text = _re.sub(r"<[^>]+>", " ", html)
    text = _re.sub(r"\s+", " ", text)
    return text.strip()


async def generate_doc_drafts(
    suggestions: list[DocSuggestion],
    diff: str,
    repo_name: str,
    ref: str,
    anthropic_cfg: AnthropicApiConfig,
    gh: GitHubClient,
    model: str = "claude-sonnet-4-6",
    max_files: int = 15,
) -> list[DocDraft]:
    """Generate AI-drafted full-file updates for stale documentation files.

    Processes suggestions that map to real files.  Directory-level suggestions
    (path ending with ``/``) are skipped — those are handled by scanning the
    directory for HTML files in the caller.

    For HTML files, uses a separate prompt that allows Claude to return
    ``NO_UPDATE_NEEDED`` when the page content is already accurate.

    Files are processed concurrently (up to 5 at a time) to reduce wall-clock
    time when many pages need updating.

    Returns one ``DocDraft`` per processed suggestion with the complete updated
    file content ready for committing.  Drafts where ``error`` is set or
    ``updated_content`` is empty should be discarded by the caller.
    """
    from ai_reviewer.agents.anthropic_client import AnthropicClient  # local to avoid circular

    truncated_diff = diff[:_MAX_DIFF_CHARS] + ("…" if len(diff) > _MAX_DIFF_CHARS else "")
    candidates = [s for s in suggestions if not s.file.endswith("/")][:max_files]

    async with AnthropicClient(anthropic_cfg) as client:
        semaphore = asyncio.Semaphore(5)

        async def _process_one(suggestion: DocSuggestion) -> DocDraft | None:
            async with semaphore:
                is_html = suggestion.file.lower().endswith(".html")
                max_chars = _MAX_DOC_CHARS_HTML if is_html else _MAX_DOC_CHARS
                system_prompt = _DOC_DRAFT_SYSTEM_HTML if is_html else _DOC_DRAFT_SYSTEM

                try:
                    raw = gh.get_file_contents(repo_name, suggestion.file, ref)
                    if isinstance(raw, list):
                        return DocDraft(
                            suggestion=suggestion,
                            updated_content="",
                            error="file resolved to multiple entries",
                        )
                    current_content = raw.decoded_content.decode("utf-8", errors="replace")
                except Exception as exc:
                    return DocDraft(suggestion=suggestion, updated_content="", error=str(exc))

                truncated_doc = current_content[:max_chars] + (
                    "\n…(truncated)" if len(current_content) > max_chars else ""
                )

                if is_html:
                    plain_text = _strip_html_tags(current_content)[:2000]
                    user_prompt = (
                        f"## HTML File: {suggestion.file}\n\n"
                        f"### Raw HTML\n\n{truncated_doc}\n\n"
                        f"### Plain-text content (for relevance check)\n\n{plain_text}\n\n"
                        f"## Code Diff\n\n{truncated_diff}\n\n"
                        f"## Why This File May Need Updating\n\n{suggestion.reason}\n\n"
                        "Return NO_UPDATE_NEEDED if unchanged, or the complete updated HTML file."
                    )
                else:
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
                        system=system_prompt,
                        user=user_prompt,
                        max_tokens=8192,
                    )
                    content = result.strip()
                    if content == _NO_UPDATE_SENTINEL:
                        return None  # Claude says no update needed
                    return DocDraft(suggestion=suggestion, updated_content=content)
                except Exception as exc:
                    return DocDraft(suggestion=suggestion, updated_content="", error=str(exc))

        raw_results = await asyncio.gather(*[_process_one(s) for s in candidates])

    return [r for r in raw_results if r is not None]


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
