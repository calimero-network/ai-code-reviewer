"""Tests for the review aggregator."""

import pytest
from datetime import datetime


class TestReviewAggregator:
    """Tests for ReviewAggregator."""

    def test_deduplicates_identical_findings(self):
        """Test that identical findings from multiple agents are merged."""
        from ai_reviewer.orchestrator.aggregator import ReviewAggregator
        from ai_reviewer.models.review import AgentReview
        from ai_reviewer.models.findings import ReviewFinding, Severity, Category

        # Same finding from two agents
        finding_1 = ReviewFinding(
            file_path="auth/login.py",
            line_start=15,
            line_end=16,
            severity=Severity.CRITICAL,
            category=Category.SECURITY,
            title="SQL Injection",
            description="User input in query",
            suggested_fix="Use params",
            confidence=0.95,
        )

        finding_2 = ReviewFinding(
            file_path="auth/login.py",
            line_start=15,
            line_end=16,
            severity=Severity.CRITICAL,
            category=Category.SECURITY,
            title="SQL Injection Vulnerability",  # Slightly different title
            description="Direct string interpolation in SQL",
            suggested_fix="Use parameterized queries",
            confidence=0.9,
        )

        review_1 = AgentReview(
            agent_id="agent-1",
            agent_type="claude",
            focus_areas=["security"],
            findings=[finding_1],
            summary="Found SQL injection",
            review_time_ms=1000,
        )

        review_2 = AgentReview(
            agent_id="agent-2",
            agent_type="gpt4",
            focus_areas=["security"],
            findings=[finding_2],
            summary="Found SQL injection",
            review_time_ms=1200,
        )

        aggregator = ReviewAggregator()
        consolidated = aggregator.aggregate([review_1, review_2])

        # Should merge into single finding with consensus score
        assert len(consolidated.findings) == 1
        assert consolidated.findings[0].consensus_score == 1.0  # Both agreed
        assert len(consolidated.findings[0].agreeing_agents) == 2

    def test_keeps_unique_findings_separate(self):
        """Test that different findings remain separate."""
        from ai_reviewer.orchestrator.aggregator import ReviewAggregator
        from ai_reviewer.models.review import AgentReview
        from ai_reviewer.models.findings import ReviewFinding, Severity, Category

        finding_security = ReviewFinding(
            file_path="auth/login.py",
            line_start=15,
            line_end=16,
            severity=Severity.CRITICAL,
            category=Category.SECURITY,
            title="SQL Injection",
            description="Security issue",
            suggested_fix=None,
            confidence=0.95,
        )

        finding_performance = ReviewFinding(
            file_path="utils/processor.py",
            line_start=30,
            line_end=40,
            severity=Severity.WARNING,
            category=Category.PERFORMANCE,
            title="O(n²) Loop",
            description="Performance issue",
            suggested_fix=None,
            confidence=0.85,
        )

        review_1 = AgentReview(
            agent_id="agent-1",
            agent_type="claude",
            focus_areas=["security"],
            findings=[finding_security],
            summary="Security issue",
            review_time_ms=1000,
        )

        review_2 = AgentReview(
            agent_id="agent-2",
            agent_type="gpt4",
            focus_areas=["performance"],
            findings=[finding_performance],
            summary="Performance issue",
            review_time_ms=1200,
        )

        aggregator = ReviewAggregator()
        consolidated = aggregator.aggregate([review_1, review_2])

        # Should have 2 separate findings
        assert len(consolidated.findings) == 2
        # Each found by only 1 agent
        assert all(f.consensus_score == 0.5 for f in consolidated.findings)

    def test_priority_ranking(self):
        """Test that findings are ranked by severity × consensus."""
        from ai_reviewer.orchestrator.aggregator import ReviewAggregator
        from ai_reviewer.models.review import AgentReview
        from ai_reviewer.models.findings import ReviewFinding, Severity, Category

        # Critical finding with partial consensus
        critical_partial = ReviewFinding(
            file_path="a.py",
            line_start=1,
            line_end=2,
            severity=Severity.CRITICAL,
            category=Category.SECURITY,
            title="Critical Issue",
            description="Critical but only one agent found it",
            suggested_fix=None,
            confidence=0.9,
        )

        # Warning with full consensus
        warning_full = ReviewFinding(
            file_path="b.py",
            line_start=10,
            line_end=20,
            severity=Severity.WARNING,
            category=Category.PERFORMANCE,
            title="Warning Issue",
            description="Warning but both agents agree",
            suggested_fix=None,
            confidence=0.85,
        )

        review_1 = AgentReview(
            agent_id="agent-1",
            agent_type="claude",
            focus_areas=["security"],
            findings=[critical_partial, warning_full],
            summary="Found issues",
            review_time_ms=1000,
        )

        review_2 = AgentReview(
            agent_id="agent-2",
            agent_type="gpt4",
            focus_areas=["performance"],
            findings=[warning_full],  # Only found the warning
            summary="Found warning",
            review_time_ms=1200,
        )

        aggregator = ReviewAggregator()
        consolidated = aggregator.aggregate([review_1, review_2])

        # Critical with partial consensus should still rank higher
        # because severity weight (1.0) × 0.5 consensus > warning weight (0.6) × 1.0
        assert consolidated.findings[0].severity == Severity.CRITICAL

    def test_computes_review_quality_score(self):
        """Test that overall review quality score is computed."""
        from ai_reviewer.orchestrator.aggregator import ReviewAggregator
        from ai_reviewer.models.review import AgentReview
        from ai_reviewer.models.findings import ReviewFinding, Severity, Category

        finding = ReviewFinding(
            file_path="test.py",
            line_start=1,
            line_end=5,
            severity=Severity.WARNING,
            category=Category.LOGIC,
            title="Issue",
            description="Test",
            suggested_fix=None,
            confidence=0.9,
        )

        # All 3 agents find same issue - high quality review
        reviews = [
            AgentReview(
                agent_id=f"agent-{i}",
                agent_type="claude",
                focus_areas=["logic"],
                findings=[finding],
                summary="Found issue",
                review_time_ms=1000,
            )
            for i in range(3)
        ]

        aggregator = ReviewAggregator()
        consolidated = aggregator.aggregate(reviews)

        # High consensus should result in high quality score
        assert consolidated.review_quality_score >= 0.8

    def test_handles_empty_reviews(self):
        """Test aggregating reviews with no findings."""
        from ai_reviewer.orchestrator.aggregator import ReviewAggregator
        from ai_reviewer.models.review import AgentReview

        review_1 = AgentReview(
            agent_id="agent-1",
            agent_type="claude",
            focus_areas=["security"],
            findings=[],
            summary="No issues found",
            review_time_ms=1000,
        )

        review_2 = AgentReview(
            agent_id="agent-2",
            agent_type="gpt4",
            focus_areas=["performance"],
            findings=[],
            summary="LGTM",
            review_time_ms=800,
        )

        aggregator = ReviewAggregator()
        consolidated = aggregator.aggregate([review_1, review_2])

        assert len(consolidated.findings) == 0
        assert consolidated.agent_count == 2
        # Clean review should have good quality score
        assert consolidated.review_quality_score >= 0.9
