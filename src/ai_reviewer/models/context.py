"""Review context models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class RepoSource(Protocol):
    """Repository reads the context builder and tool registry need.

    Satisfied by GitHubClient (Contents API) and by LocalGitSource (working
    tree), which is why these two methods are the whole seam.
    """

    def get_file_contents(self, repo: str, path: str, ref: str) -> Any: ...

    def get_tree(self, repo: str, sha: str, recursive: bool = True) -> Any: ...


@dataclass
class ReviewContext:
    """Context provided to agents for informed reviews."""

    repo_name: str
    pr_number: int
    pr_title: str
    pr_description: str
    base_branch: str
    head_branch: str
    author: str
    changed_files_count: int
    additions: int
    deletions: int
    labels: list[str] = field(default_factory=list)
    repo_languages: list[str] = field(default_factory=list)
    custom_instructions: str | None = None
    repo_config: dict[str, Any] | None = None
    conventions: str | None = None

    def to_prompt_context(self) -> str:
        """Format context for inclusion in agent prompts."""
        return f"""## Pull Request Context
- Repository: {self.repo_name}
- PR #{self.pr_number}: {self.pr_title}
- Author: {self.author}
- Branch: {self.head_branch} → {self.base_branch}
- Changes: +{self.additions} / -{self.deletions} in {self.changed_files_count} files
- Languages: {", ".join(self.repo_languages) if self.repo_languages else "Unknown"}
- Labels: {", ".join(self.labels) if self.labels else "None"}

## PR Description
{self.pr_description or "No description provided."}
"""
