"""Review result models."""

from dataclasses import dataclass, field
from datetime import datetime

from ai_reviewer.models.findings import Category, ConsolidatedFinding, ReviewFinding, Severity


@dataclass
class AgentReview:
    """Complete review from a single agent."""

    agent_id: str
    agent_type: str  # "claude", "gpt4", etc.
    focus_areas: list[str]
    findings: list[ReviewFinding]
    summary: str
    review_time_ms: int

    @property
    def findings_count(self) -> int:
        """Total number of findings."""
        return len(self.findings)

    @property
    def critical_count(self) -> int:
        """Number of critical findings."""
        return sum(1 for f in self.findings if f.severity == Severity.CRITICAL)


@dataclass
class ConsolidatedReview:
    """Final aggregated review output."""

    id: str
    created_at: datetime
    repo: str
    pr_number: int

    # Results
    findings: list[ConsolidatedFinding]
    summary: str

    # Metadata
    agent_count: int
    review_quality_score: float  # How confident we are in this review
    total_review_time_ms: int

    # Optional: original reviews for transparency
    agent_reviews: list[AgentReview] = field(default_factory=list)

    # Track failed agents
    failed_agents: list[str] = field(default_factory=list)

    @property
    def findings_by_severity(self) -> dict[Severity, int]:
        """Count findings by severity level."""
        counts: dict[Severity, int] = dict.fromkeys(Severity, 0)
        for finding in self.findings:
            counts[finding.severity] += 1
        return counts

    @property
    def findings_by_category(self) -> dict[Category, int]:
        """Count findings by category."""
        counts: dict[Category, int] = dict.fromkeys(Category, 0)
        for finding in self.findings:
            counts[finding.category] += 1
        return counts

    @property
    def has_critical_issues(self) -> bool:
        """Check if review has any critical findings."""
        return any(f.severity == Severity.CRITICAL for f in self.findings)

    @property
    def has_blocking_issues(self) -> bool:
        """Check if review has issues that should block merge."""
        return self.has_critical_issues

    @property
    def all_agents_failed(self) -> bool:
        """Check if all agents failed to produce a review."""
        return len(self.failed_agents) == self.agent_count and self.agent_count > 0
