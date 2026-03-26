"""Finding models for code review results."""

import re
from dataclasses import dataclass, field
from enum import Enum

_FUZZY_WORD_RE = re.compile(r"\b\w{4,}\b")


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
    def finding_hash(self) -> str:
        """Deterministic 12-char hash for deduplication across review runs.

        Key uses normalized title (lowercase+strip) and excludes severity so the
        hash stays stable when AI-generated titles vary in casing/whitespace or
        when severity is re-assessed between runs.
        """
        import hashlib

        normalized_title = self.title.lower().strip()
        key = f"{self.file_path or ''}:{self.line_start or 0}:{normalized_title}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @property
    def finding_hash_fuzzy(self) -> str | None:
        """Fuzzy hash ignoring line number and category for cross-run matching.

        Uses file_path + sorted title keywords (4+ chars) so the hash stays
        stable when a finding drifts lines or gets recategorized between runs.
        Returns None when file_path or title is empty (mirrors PreviousComment).
        """
        import hashlib

        if not self.file_path or not self.title:
            return None
        words = sorted(set(_FUZZY_WORD_RE.findall(self.title.lower())))
        word_key = ":".join(words[:5]) if words else self.title.lower().strip()
        key = f"{self.file_path}:{word_key}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

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
