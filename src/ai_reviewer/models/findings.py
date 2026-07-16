"""Finding models for code review results."""

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum

_FUZZY_WORD_RE = re.compile(r"\b\w{4,}\b")
_FUZZY_SYMBOL_RE = re.compile(r"`([\w.]+)`")
_LINE_BUCKET_SIZE = 20


def compute_fuzzy_hash(
    file_path: str,
    title: str,
    category: str | None = None,
    line_start: int | None = None,
) -> str | None:
    """Substance-stable 12-char SHA256 key for cross-run fuzzy matching.

    Keys on substance rather than exact title wording so a re-worded re-raise
    of the same issue hashes the same. The symbol component is up to 3
    backtick-quoted identifiers from the title (sorted); when the title has no
    backticked identifiers it falls back to sorted 4+ char title keywords so
    prose titles still hash meaningfully.

    ``category`` and ``line_start`` are optional and only fold into the key when
    supplied (line_bucket = line_start // 20, stable under small drift). Callers
    that lack them - e.g. previous comments parsed from posted text - simply omit
    them, and two callers that both omit them produce identical hashes.

    Returns None when file_path or title is empty.
    """
    if not file_path or not title:
        return None
    idents = sorted(set(_FUZZY_SYMBOL_RE.findall(title)))[:3]
    if idents:
        symbol_key = ":".join(idents)
    else:
        words = sorted(set(_FUZZY_WORD_RE.findall(title.lower())))
        symbol_key = ":".join(words[:5]) if words else title.lower().strip()

    parts = [file_path]
    cat_key = (category or "").lower().strip()
    if cat_key:
        parts.append(cat_key)
    if line_start:
        parts.append(str(line_start // _LINE_BUCKET_SIZE))
    parts.append(symbol_key)
    key = ":".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


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
    # Exact replacement source for line_start..line_end (whole lines, no diff
    # syntax). Null when the fix is non-local; only prose lives in suggested_fix.
    suggested_replacement: str | None = None

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

    # Exact replacement source for line_start..line_end, carried from the
    # representative finding. fix_validated flips true only after fix_check
    # confirms it applies cleanly and stays syntactically valid.
    suggested_replacement: str | None = None
    fix_validated: bool = False

    # Source tracking
    original_findings: list[ReviewFinding] = field(default_factory=list)

    @property
    def finding_hash(self) -> str:
        """Deterministic 12-char hash for deduplication across review runs.

        Key uses normalized title (lowercase+strip) and excludes severity so the
        hash stays stable when AI-generated titles vary in casing/whitespace or
        when severity is re-assessed between runs.
        """
        normalized_title = self.title.lower().strip()
        key = f"{self.file_path or ''}:{self.line_start or 0}:{normalized_title}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @property
    def finding_hash_fuzzy(self) -> str | None:
        """Substance-stable fuzzy hash for cross-run matching.

        Keys on file_path + title symbols (see compute_fuzzy_hash) so a re-worded
        re-raise of the same issue matches across runs. category/line_start are
        deliberately omitted: the matching side (PreviousComment, parsed from
        posted text) cannot recover them, and both sides must hash identically.
        Returns None when file_path or title is empty (mirrors PreviousComment).
        """
        return compute_fuzzy_hash(self.file_path, self.title)

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
