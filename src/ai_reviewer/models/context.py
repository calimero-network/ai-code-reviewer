"""Review context models."""

from dataclasses import dataclass, field
from typing import Optional


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
    custom_instructions: Optional[str] = None

    def to_prompt_context(self) -> str:
        """Format context for inclusion in agent prompts."""
        return f"""## Pull Request Context
- Repository: {self.repo_name}
- PR #{self.pr_number}: {self.pr_title}
- Author: {self.author}
- Branch: {self.head_branch} → {self.base_branch}
- Changes: +{self.additions} / -{self.deletions} in {self.changed_files_count} files
- Languages: {', '.join(self.repo_languages) if self.repo_languages else 'Unknown'}
- Labels: {', '.join(self.labels) if self.labels else 'None'}

## PR Description
{self.pr_description or 'No description provided.'}
"""
