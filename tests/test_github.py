"""Tests for GitHub integration."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock


class TestGitHubClient:
    """Tests for GitHub API client."""

    def test_extracts_pr_diff(self):
        """Test extracting diff from a PR."""
        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()
        mock_pr.get_files.return_value = [
            MagicMock(
                filename="auth/login.py",
                patch="@@ -10,6 +10,12 @@\n+new code",
                status="modified",
                additions=6,
                deletions=0,
            )
        ]

        with patch("ai_reviewer.github.client.Github") as mock_github:
            client = GitHubClient(token="test-token")
            diff = client.get_pr_diff(mock_pr)

            assert "auth/login.py" in diff
            assert "+new code" in diff

    def test_builds_review_context(self):
        """Test building review context from PR."""
        from ai_reviewer.github.client import GitHubClient
        from ai_reviewer.models.context import ReviewContext

        mock_pr = MagicMock()
        mock_pr.number = 42
        mock_pr.title = "Add authentication"
        mock_pr.body = "This PR adds auth"
        mock_pr.base.ref = "main"
        mock_pr.head.ref = "feature/auth"
        mock_pr.user.login = "testuser"
        mock_pr.additions = 100
        mock_pr.deletions = 10
        mock_pr.changed_files = 5
        mock_pr.get_labels.return_value = [MagicMock(name="enhancement")]

        mock_repo = MagicMock()
        mock_repo.full_name = "test-org/test-repo"
        mock_repo.get_languages.return_value = {"Python": 1000, "JavaScript": 500}

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            context = client.build_review_context(mock_pr, mock_repo)

            assert context.pr_number == 42
            assert context.pr_title == "Add authentication"
            assert context.author == "testuser"
            assert "Python" in context.repo_languages


class TestGitHubPRHandler:
    """Tests for PR event handling."""

    @pytest.mark.asyncio
    async def test_handles_pr_opened_event(self):
        """Test handling PR opened webhook event."""
        from ai_reviewer.github.webhook import handle_pr_event, PREvent

        event = PREvent(
            repo="test-org/test-repo",
            pr_number=42,
            action="opened",
        )

        with patch("ai_reviewer.github.webhook.review_pr", new_callable=AsyncMock) as mock_review:
            await handle_pr_event(event)
            mock_review.assert_called_once_with(
                repo="test-org/test-repo",
                pr_number=42,
            )

    @pytest.mark.asyncio
    async def test_ignores_irrelevant_actions(self):
        """Test that irrelevant PR actions are ignored."""
        from ai_reviewer.github.webhook import handle_pr_event, PREvent

        event = PREvent(
            repo="test-org/test-repo",
            pr_number=42,
            action="labeled",  # Not a review trigger
        )

        with patch("ai_reviewer.github.webhook.review_pr", new_callable=AsyncMock) as mock_review:
            await handle_pr_event(event)
            mock_review.assert_not_called()


class TestReviewFormatter:
    """Tests for GitHub comment formatting."""

    def test_formats_critical_findings(self):
        """Test formatting critical findings for GitHub."""
        from ai_reviewer.github.formatter import GitHubFormatter
        from ai_reviewer.models.review import ConsolidatedReview
        from ai_reviewer.models.findings import ConsolidatedFinding, Severity, Category
        from datetime import datetime

        findings = [
            ConsolidatedFinding(
                id="f1",
                file_path="auth/login.py",
                line_start=15,
                line_end=18,
                severity=Severity.CRITICAL,
                category=Category.SECURITY,
                title="SQL Injection",
                description="User input in query",
                suggested_fix="Use parameterized queries",
                consensus_score=1.0,
                agreeing_agents=["agent-1", "agent-2", "agent-3"],
                confidence=0.95,
            )
        ]

        review = ConsolidatedReview(
            id="review-123",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=findings,
            summary="Found 1 critical issue",
            agent_count=3,
            review_quality_score=0.95,
            total_review_time_ms=3000,
        )

        formatter = GitHubFormatter()
        comment = formatter.format_review(review)

        # Should include critical emoji/indicator
        assert "🔴" in comment or "Critical" in comment
        # Should show consensus
        assert "3/3" in comment or "100%" in comment
        # Should include the finding
        assert "SQL Injection" in comment
        # Should include file reference
        assert "auth/login.py" in comment

    def test_formats_empty_review(self):
        """Test formatting review with no findings."""
        from ai_reviewer.github.formatter import GitHubFormatter
        from ai_reviewer.models.review import ConsolidatedReview
        from datetime import datetime

        review = ConsolidatedReview(
            id="review-123",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="No issues found",
            agent_count=3,
            review_quality_score=0.98,
            total_review_time_ms=2500,
        )

        formatter = GitHubFormatter()
        comment = formatter.format_review(review)

        # Should indicate clean review
        assert "No issues" in comment or "LGTM" in comment or "✅" in comment

    def test_determines_review_action(self):
        """Test determining GitHub review action based on findings."""
        from ai_reviewer.github.formatter import GitHubFormatter
        from ai_reviewer.models.review import ConsolidatedReview
        from ai_reviewer.models.findings import ConsolidatedFinding, Severity, Category
        from datetime import datetime

        formatter = GitHubFormatter()

        # Review with critical issues should request changes
        critical_review = ConsolidatedReview(
            id="review-1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[
                ConsolidatedFinding(
                    id="f1",
                    file_path="test.py",
                    line_start=1,
                    line_end=5,
                    severity=Severity.CRITICAL,
                    category=Category.SECURITY,
                    title="Critical Issue",
                    description="Bad",
                    suggested_fix=None,
                    consensus_score=1.0,
                    agreeing_agents=["a1"],
                    confidence=0.9,
                )
            ],
            summary="Critical issue",
            agent_count=1,
            review_quality_score=0.8,
            total_review_time_ms=1000,
        )
        assert formatter.get_review_action(critical_review) == "REQUEST_CHANGES"

        # Clean review should approve (with allow_approve=True, the default)
        clean_review = ConsolidatedReview(
            id="review-2",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="Clean",
            agent_count=3,
            review_quality_score=0.95,
            total_review_time_ms=2000,
        )
        assert formatter.get_review_action(clean_review) == "APPROVE"
        assert formatter.get_review_action(clean_review, allow_approve=True) == "APPROVE"

        # Clean review with allow_approve=False should COMMENT (used in GitHub Actions)
        assert formatter.get_review_action(clean_review, allow_approve=False) == "COMMENT"

        # Critical review always returns REQUEST_CHANGES regardless of allow_approve
        assert formatter.get_review_action(critical_review, allow_approve=False) == "REQUEST_CHANGES"
