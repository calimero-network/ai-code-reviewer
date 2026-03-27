"""Tests for the Phase 2 composite quality scoring formula."""

from ai_reviewer.models.findings import (
    Category,
    ConsolidatedFinding,
    Severity,
)
from ai_reviewer.review import compute_quality_score


def _make_finding(
    severity: Severity = Severity.WARNING,
    confidence: float = 0.9,
) -> ConsolidatedFinding:
    return ConsolidatedFinding(
        id="test-1",
        file_path="src/auth.py",
        line_start=10,
        line_end=None,
        severity=severity,
        category=Category.SECURITY,
        title="Issue",
        description="desc",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=["a"],
        confidence=confidence,
    )


class TestComputeQualityScore:
    """Tests for the new composite quality scoring formula."""

    def test_clean_review_scores_high(self):
        """No findings should score 0.85-0.95."""
        score, _ = compute_quality_score([], agent_count=3, total_lines=100)
        assert 0.85 <= score <= 0.95

    def test_clean_review_scales_with_agents(self):
        """More agents on clean review = higher confidence."""
        s1, _ = compute_quality_score([], agent_count=1, total_lines=100)
        s3, _ = compute_quality_score([], agent_count=3, total_lines=100)
        assert s3 > s1

    def test_critical_finding_penalizes_heavily(self):
        """A critical finding should significantly lower the score."""
        findings = [_make_finding(severity=Severity.CRITICAL)]
        score, _ = compute_quality_score(findings, agent_count=3, total_lines=500)
        assert score < 0.85

    def test_nitpick_barely_penalizes(self):
        """A single nitpick should barely affect the score."""
        findings = [_make_finding(severity=Severity.NITPICK)]
        score, _ = compute_quality_score(findings, agent_count=3, total_lines=500)
        assert score > 0.90

    def test_density_normalized_by_pr_size(self):
        """Same findings in a large PR should score higher than in a small PR."""
        findings = [_make_finding(severity=Severity.WARNING) for _ in range(3)]
        small_pr, _ = compute_quality_score(findings, agent_count=3, total_lines=50)
        large_pr, _ = compute_quality_score(findings, agent_count=3, total_lines=5000)
        assert large_pr > small_pr

    def test_score_never_negative(self):
        """Score should never go below 0.0 even with many critical findings."""
        findings = [_make_finding(severity=Severity.CRITICAL) for _ in range(10)]
        score, _ = compute_quality_score(findings, agent_count=3, total_lines=50)
        assert score >= 0.0

    def test_zero_agents_no_negative_bonus(self):
        """Zero agents should not produce a negative bonus (score must be >= 0.85)."""
        score, breakdown = compute_quality_score([], agent_count=0, total_lines=100)
        assert score >= 0.85
        assert breakdown.agent_factor >= 0.0

    def test_score_capped_at_095(self):
        """Clean review score should never exceed 0.95."""
        score, _ = compute_quality_score([], agent_count=10, total_lines=10)
        assert score <= 0.95
