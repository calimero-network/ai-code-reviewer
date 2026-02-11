"""Finding models for code review results."""

from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    """Severity levels for findings. Canonical semantics for prompts and formatting.

    - CRITICAL: Must fix before merge (security bugs or data corruption risks only).
    - WARNING: Should fix; other serious correctness or maintainability issues.
    - SUGGESTION: Consider; improves code health. Optional but recommended.
    - NITPICK: Optional polish or style; prompt instructs to prefix title with "Nit: ".
    """

    CRITICAL = "critical"  # Security or data corruption only
    WARNING = "warning"  # Should fix, potential issues
    SUGGESTION = "suggestion"  # Nice to have improvements
    NITPICK = "nitpick"  # Style/formatting only; use "Nit: " prefix in title


class Category(Enum):
    """Categories for review findings."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    LOGIC = "logic"
    STYLE = "style"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DOCUMENTATION = "documentation"


@dataclass
class ReviewFinding:
    """A single finding from an agent's review."""

    file_path: str
    line_start: int
    line_end: int | None
    severity: Severity
    category: Category
    title: str
    description: str
    suggested_fix: str | None
    confidence: float  # 0.0 - 1.0

    def __post_init__(self) -> None:
        """Validate finding data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.line_start < 1:
            raise ValueError(f"line_start must be >= 1, got {self.line_start}")
        if self.line_end is not None and self.line_end < self.line_start:
            raise ValueError(
                f"line_end ({self.line_end}) must be >= line_start ({self.line_start})"
            )


@dataclass
class ConsolidatedFinding:
    """A finding that has been merged from multiple agents."""

    id: str
    file_path: str
    line_start: int
    line_end: int | None
    severity: Severity
    category: Category
    title: str
    description: str
    suggested_fix: str | None

    # Consensus metadata
    consensus_score: float  # 0.0 - 1.0 (% of agents that found this)
    agreeing_agents: list[str]
    confidence: float  # Average confidence across agents

    # Source tracking
    original_findings: list[ReviewFinding] = field(default_factory=list)

    @property
    def priority_score(self) -> float:
        """Compute priority based on severity and consensus."""
        severity_weights = {
            Severity.CRITICAL: 1.0,
            Severity.WARNING: 0.6,
            Severity.SUGGESTION: 0.3,
            Severity.NITPICK: 0.1,
        }
        return severity_weights[self.severity] * self.consensus_score * self.confidence
