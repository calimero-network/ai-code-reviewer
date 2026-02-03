"""Tests for data models."""

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
