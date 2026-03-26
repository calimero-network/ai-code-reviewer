"""Tests for data models."""

import re
from datetime import datetime


class TestReviewFinding:
    """Tests for ReviewFinding model."""

    def test_finding_creation(self):
        """Test creating a basic finding."""
        # Will test once models are implemented
        from ai_reviewer.models.findings import Category, ReviewFinding, Severity

        finding = ReviewFinding(
            file_path="auth/login.py",
            line_start=15,
            line_end=18,
            severity=Severity.CRITICAL,
            category=Category.SECURITY,
            title="SQL Injection Vulnerability",
            description="User input directly interpolated into SQL query",
            suggested_fix="Use parameterized queries",
            confidence=0.95,
        )

        assert finding.file_path == "auth/login.py"
        assert finding.severity == Severity.CRITICAL
        assert finding.confidence == 0.95

    def test_finding_without_suggested_fix(self):
        """Test finding without a suggested fix."""
        from ai_reviewer.models.findings import Category, ReviewFinding, Severity

        finding = ReviewFinding(
            file_path="utils/helper.py",
            line_start=10,
            line_end=None,
            severity=Severity.SUGGESTION,
            category=Category.STYLE,
            title="Missing type hints",
            description="Function parameters lack type annotations",
            suggested_fix=None,
            confidence=0.8,
        )

        assert finding.suggested_fix is None
        assert finding.line_end is None


class TestAgentReview:
    """Tests for AgentReview model."""

    def test_agent_review_creation(self):
        """Test creating an agent review."""
        from ai_reviewer.models.findings import Category, ReviewFinding, Severity
        from ai_reviewer.models.review import AgentReview

        findings = [
            ReviewFinding(
                file_path="test.py",
                line_start=1,
                line_end=5,
                severity=Severity.WARNING,
                category=Category.PERFORMANCE,
                title="Inefficient loop",
                description="O(n²) complexity",
                suggested_fix=None,
                confidence=0.85,
            )
        ]

        review = AgentReview(
            agent_id="claude-security-1",
            agent_type="claude",
            focus_areas=["security", "architecture"],
            findings=findings,
            summary="Found 1 performance issue",
            review_time_ms=1500,
        )

        assert review.agent_id == "claude-security-1"
        assert len(review.findings) == 1
        assert review.review_time_ms == 1500


class TestConsolidatedReview:
    """Tests for ConsolidatedReview model."""

    def test_consolidated_review_statistics(self):
        """Test that consolidated review computes statistics correctly."""
        from ai_reviewer.models.findings import (
            Category,
            ConsolidatedFinding,
            Severity,
        )
        from ai_reviewer.models.review import ConsolidatedReview

        findings = [
            ConsolidatedFinding(
                id="f1",
                file_path="auth.py",
                line_start=10,
                line_end=15,
                severity=Severity.CRITICAL,
                category=Category.SECURITY,
                title="SQL Injection",
                description="Vulnerable query",
                suggested_fix="Use params",
                consensus_score=1.0,
                agreeing_agents=["agent1", "agent2", "agent3"],
                confidence=0.95,
            ),
            ConsolidatedFinding(
                id="f2",
                file_path="utils.py",
                line_start=20,
                line_end=25,
                severity=Severity.WARNING,
                category=Category.PERFORMANCE,
                title="Slow loop",
                description="O(n²)",
                suggested_fix=None,
                consensus_score=0.67,
                agreeing_agents=["agent1", "agent2"],
                confidence=0.8,
            ),
        ]

        review = ConsolidatedReview(
            id="review-123",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=findings,
            summary="Found issues",
            agent_count=3,
            review_quality_score=0.87,
            total_review_time_ms=3000,
        )

        assert review.agent_count == 3
        assert len(review.findings) == 2
        # First finding has full consensus
        assert review.findings[0].consensus_score == 1.0


def test_finding_hash_is_stable_and_deterministic():
    """finding_hash must be stable across calls and determined by content."""
    from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity

    f = ConsolidatedFinding(
        id="f1",
        file_path="auth.py",
        line_start=42,
        line_end=None,
        severity=Severity.CRITICAL,
        category=Category.SECURITY,
        title="SQL injection",
        description="User input used in query",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=["security-agent"],
        confidence=1.0,
    )
    # Stable across calls
    assert f.finding_hash == f.finding_hash
    # 12 chars
    assert len(f.finding_hash) == 12
    # Deterministic — same file_path/line_start/title (normalized) = same hash regardless of severity
    f2 = ConsolidatedFinding(
        id="f2",
        file_path="auth.py",
        line_start=42,
        line_end=None,
        severity=Severity.WARNING,  # different severity — hash must still match
        category=Category.PERFORMANCE,
        title="SQL INJECTION",  # different casing — hash must still match after normalize
        description="Different description",
        suggested_fix=None,
        consensus_score=0.5,
        agreeing_agents=[],
        confidence=0.5,
    )
    assert f.finding_hash == f2.finding_hash  # title normalized, severity excluded


def test_finding_hash_differs_for_different_content():
    """Different file/line/title/severity should produce different hashes."""
    from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity

    f1 = ConsolidatedFinding(
        id="f1",
        file_path="a.py",
        line_start=1,
        line_end=None,
        severity=Severity.CRITICAL,
        category=Category.SECURITY,
        title="Issue A",
        description="",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=[],
        confidence=1.0,
    )
    f2 = ConsolidatedFinding(
        id="f2",
        file_path="b.py",
        line_start=2,
        line_end=None,
        severity=Severity.WARNING,
        category=Category.PERFORMANCE,
        title="Issue B",
        description="",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=[],
        confidence=1.0,
    )
    assert f1.finding_hash != f2.finding_hash


def _make_consolidated_finding(
    file_path: str = "src/auth.py",
    line_start: int = 10,
    title: str = "SQL Injection Vulnerability",
    severity: "Severity | None" = None,
    category: "Category | None" = None,
) -> "ConsolidatedFinding":
    from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity

    return ConsolidatedFinding(
        id="test-1",
        file_path=file_path,
        line_start=line_start,
        line_end=None,
        severity=severity or Severity.WARNING,
        category=category or Category.SECURITY,
        title=title,
        description="desc",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=["a"],
        confidence=0.9,
    )


class TestFindingHashFuzzy:
    """Tests for the fuzzy hash property on ConsolidatedFinding."""

    def test_fuzzy_hash_is_12_chars_hex(self):
        """Fuzzy hash returns a 12-character hex string."""
        f = _make_consolidated_finding()
        assert re.fullmatch(r"[a-f0-9]{12}", f.finding_hash_fuzzy)

    def test_fuzzy_hash_ignores_line_number(self):
        """Same file+title at different lines produce same fuzzy hash."""
        f1 = _make_consolidated_finding(line_start=10)
        f2 = _make_consolidated_finding(line_start=50)
        assert f1.finding_hash_fuzzy == f2.finding_hash_fuzzy

    def test_fuzzy_hash_differs_from_primary(self):
        """Fuzzy and primary hashes are different values."""
        f = _make_consolidated_finding()
        assert f.finding_hash != f.finding_hash_fuzzy

    def test_fuzzy_hash_ignores_minor_title_variation(self):
        """Fuzzy hash matches when titles share the same keywords."""
        f1 = _make_consolidated_finding(title="SQL Injection Vulnerability Found")
        f2 = _make_consolidated_finding(title="Found SQL Injection Vulnerability")
        assert f1.finding_hash_fuzzy == f2.finding_hash_fuzzy

    def test_fuzzy_hash_stable_across_category_changes(self):
        """Fuzzy hash is same regardless of category (not included)."""
        from ai_reviewer.models.findings import Category

        f1 = _make_consolidated_finding(category=Category.SECURITY)
        f2 = _make_consolidated_finding(category=Category.PERFORMANCE)
        assert f1.finding_hash_fuzzy == f2.finding_hash_fuzzy

    def test_fuzzy_hash_differs_for_different_file(self):
        """Different files produce different fuzzy hashes."""
        f1 = _make_consolidated_finding(file_path="a.py")
        f2 = _make_consolidated_finding(file_path="b.py")
        assert f1.finding_hash_fuzzy != f2.finding_hash_fuzzy

    def test_primary_hash_unchanged(self):
        """Existing finding_hash behavior is preserved."""
        f1 = _make_consolidated_finding(line_start=10)
        f2 = _make_consolidated_finding(line_start=50)
        assert f1.finding_hash != f2.finding_hash  # Primary IS line-sensitive
