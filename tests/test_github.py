"""Tests for GitHub integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            diff = client.get_pr_diff(mock_pr)

            assert "auth/login.py" in diff
            assert "+new code" in diff

    def test_builds_review_context(self):
        """Test building review context from PR."""
        from ai_reviewer.github.client import GitHubClient

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
        from ai_reviewer.github.webhook import PREvent, handle_pr_event

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
        from ai_reviewer.github.webhook import PREvent, handle_pr_event

        event = PREvent(
            repo="test-org/test-repo",
            pr_number=42,
            action="labeled",  # Not a review trigger
        )

        with patch("ai_reviewer.github.webhook.review_pr", new_callable=AsyncMock) as mock_review:
            await handle_pr_event(event)
            mock_review.assert_not_called()


class TestResolveFixedComments:
    """Tests for duplicate resolved comment prevention."""

    def test_get_resolved_comment_ids_finds_existing_resolved(self):
        """Test that _get_resolved_comment_ids finds already-resolved comments."""
        from ai_reviewer.github.client import GitHubClient

        # Create mock comments - one is a resolved reply
        mock_original_comment = MagicMock()
        mock_original_comment.id = 100
        mock_original_comment.body = "🔴 **SQL Injection**\n\nBad query"
        mock_original_comment.user.login = "github-actions[bot]"
        mock_original_comment.in_reply_to_id = None

        mock_resolved_reply = MagicMock()
        mock_resolved_reply.id = 200
        mock_resolved_reply.body = "✅ **Resolved** - This issue has been addressed in the latest changes."
        mock_resolved_reply.user.login = "github-actions[bot]"
        mock_resolved_reply.in_reply_to_id = 100  # Reply to comment 100

        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = [mock_original_comment, mock_resolved_reply]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._current_user_login = "github-actions[bot]"
            resolved_ids = client._get_resolved_comment_ids(mock_pr)

            assert 100 in resolved_ids
            assert len(resolved_ids) == 1

    def test_get_resolved_comment_ids_ignores_other_users(self):
        """Test that resolved comments from other users are not counted."""
        from ai_reviewer.github.client import GitHubClient

        # Create a resolved reply from a different user
        mock_resolved_reply = MagicMock()
        mock_resolved_reply.id = 200
        mock_resolved_reply.body = "✅ **Resolved** - Fixed!"
        mock_resolved_reply.user.login = "random-user"  # Not the bot
        mock_resolved_reply.in_reply_to_id = 100

        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = [mock_resolved_reply]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._current_user_login = "github-actions[bot]"
            resolved_ids = client._get_resolved_comment_ids(mock_pr)

            # Should not include comment 100 since the resolver was a different user
            assert 100 not in resolved_ids
            assert len(resolved_ids) == 0

    def test_get_resolved_comment_ids_handles_none_reply_to(self):
        """Test handling of comments without in_reply_to_id."""
        from ai_reviewer.github.client import GitHubClient

        # Comment with Resolved text but no in_reply_to_id
        mock_comment = MagicMock()
        mock_comment.id = 100
        mock_comment.body = "✅ **Resolved** - Fixed!"
        mock_comment.user.login = "github-actions[bot]"
        mock_comment.in_reply_to_id = None

        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = [mock_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._current_user_login = "github-actions[bot]"
            resolved_ids = client._get_resolved_comment_ids(mock_pr)

            # Should be empty since comment has no in_reply_to_id
            assert len(resolved_ids) == 0

    def test_get_resolved_comment_ids_handles_none_user(self):
        """Test handling of comments from deleted users."""
        from ai_reviewer.github.client import GitHubClient

        # Comment with Resolved text but user is None (deleted user)
        mock_comment = MagicMock()
        mock_comment.id = 100
        mock_comment.body = "✅ **Resolved** - Fixed!"
        mock_comment.user = None  # Deleted user
        mock_comment.in_reply_to_id = 50

        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = [mock_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._current_user_login = "github-actions[bot]"
            resolved_ids = client._get_resolved_comment_ids(mock_pr)

            # Should be empty since comment.user is None
            assert len(resolved_ids) == 0

    def test_get_resolved_comment_ids_handles_none_login(self):
        """Test handling of comments where user exists but login is None."""
        from ai_reviewer.github.client import GitHubClient

        # Comment with Resolved text but user.login is None
        mock_comment = MagicMock()
        mock_comment.id = 100
        mock_comment.body = "✅ **Resolved** - Fixed!"
        mock_comment.user.login = None  # Login is None
        mock_comment.in_reply_to_id = 50

        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = [mock_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._current_user_login = "github-actions[bot]"
            resolved_ids = client._get_resolved_comment_ids(mock_pr)

            # Should be empty since comment.user.login is None
            assert len(resolved_ids) == 0

    def test_resolve_fixed_comments_skips_already_resolved(self):
        """Test that resolve_fixed_comments skips already-resolved comments."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment, ReviewDelta

        # Create a fixed finding
        fixed_comment = PreviousComment(
            id=100,
            file_path="test.py",
            line=10,
            title="SQL Injection",
            severity="critical",
            body="🔴 **SQL Injection**",
        )

        delta = ReviewDelta(fixed_findings=[fixed_comment])

        # Mock existing resolved reply
        mock_resolved_reply = MagicMock()
        mock_resolved_reply.id = 200
        mock_resolved_reply.body = "✅ **Resolved** - This issue has been addressed in the latest changes."
        mock_resolved_reply.user.login = "github-actions[bot]"
        mock_resolved_reply.in_reply_to_id = 100  # Already resolved

        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = [mock_resolved_reply]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._current_user_login = "github-actions[bot]"
            resolved_count = client.resolve_fixed_comments(mock_pr, delta)

            # Should skip the comment since it's already resolved
            assert resolved_count == 0
            # Should NOT call create_review_comment_reply
            mock_pr.create_review_comment_reply.assert_not_called()

    def test_resolve_fixed_comments_posts_for_new_fixes(self):
        """Test that resolve_fixed_comments posts for newly fixed issues."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment, ReviewDelta

        # Create a fixed finding
        fixed_comment = PreviousComment(
            id=100,
            file_path="test.py",
            line=10,
            title="SQL Injection",
            severity="critical",
            body="🔴 **SQL Injection**",
        )

        delta = ReviewDelta(fixed_findings=[fixed_comment])

        # No existing resolved comments
        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = []
        mock_pr.get_review_comment.return_value = MagicMock()

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._current_user_login = "github-actions[bot]"
            resolved_count = client.resolve_fixed_comments(mock_pr, delta)

            # Should post a resolved reply
            assert resolved_count == 1
            mock_pr.create_review_comment_reply.assert_called_once_with(
                comment_id=100,
                body="✅ **Resolved** - This issue has been addressed in the latest changes.",
            )

    def test_get_current_user_login_caches_result(self):
        """Test that _get_current_user_login caches the API response."""
        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github") as mock_github:
            mock_github.return_value.get_user.return_value.login = "test-bot"
            client = GitHubClient(token="test-token")

            # First call should fetch from API
            login1 = client._get_current_user_login()
            assert login1 == "test-bot"

            # Second call should use cached value, not call API again
            login2 = client._get_current_user_login()
            assert login2 == "test-bot"

            # get_user should only be called once due to caching
            assert mock_github.return_value.get_user.call_count == 1

    def test_get_current_user_login_caches_failure(self):
        """Test that _get_current_user_login caches API failures."""
        from github.GithubException import GithubException

        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github") as mock_github:
            mock_github.return_value.get_user.side_effect = GithubException(401, "Unauthorized", None)
            client = GitHubClient(token="test-token")

            # First call should fail and return None
            login1 = client._get_current_user_login()
            assert login1 is None

            # Second call should also return None without calling API again
            login2 = client._get_current_user_login()
            assert login2 is None

            # get_user should only be called once - failure is cached
            assert mock_github.return_value.get_user.call_count == 1

    def test_get_allowed_users_caches_result(self):
        """Test that _get_allowed_users caches the set."""
        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._current_user_login = "test-bot"

            # First call should build the set
            users1 = client._get_allowed_users()
            assert "test-bot" in users1
            assert "github-actions[bot]" in users1

            # Second call should return cached set (same object)
            users2 = client._get_allowed_users()
            assert users1 is users2  # Same object, not rebuilt

    def test_resolve_fixed_comments_avoids_redundant_api_calls(self):
        """Test that resolve_fixed_comments fetches comments only once."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment, ReviewDelta

        # Create a fixed finding
        fixed_comment = PreviousComment(
            id=100,
            file_path="test.py",
            line=10,
            title="SQL Injection",
            severity="critical",
            body="🔴 **SQL Injection**",
        )

        delta = ReviewDelta(fixed_findings=[fixed_comment])

        # No existing resolved comments
        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = []
        mock_pr.get_review_comment.return_value = MagicMock()

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._current_user_login = "github-actions[bot]"
            client.resolve_fixed_comments(mock_pr, delta)

            # get_review_comments should only be called once
            assert mock_pr.get_review_comments.call_count == 1


class TestReviewFormatter:
    """Tests for GitHub comment formatting."""

    def test_formats_critical_findings(self):
        """Test formatting critical findings for GitHub."""
        from datetime import datetime

        from ai_reviewer.github.formatter import GitHubFormatter
        from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity
        from ai_reviewer.models.review import ConsolidatedReview

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
        from datetime import datetime

        from ai_reviewer.github.formatter import GitHubFormatter
        from ai_reviewer.models.review import ConsolidatedReview

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
        from datetime import datetime

        from ai_reviewer.github.formatter import GitHubFormatter
        from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity
        from ai_reviewer.models.review import ConsolidatedReview

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

        # Critical review with allow_approve=True returns REQUEST_CHANGES
        assert formatter.get_review_action(critical_review, allow_approve=True) == "REQUEST_CHANGES"

        # Critical review with allow_approve=False returns COMMENT (GitHub Actions can't block merges)
        # This is intentional - REQUEST_CHANGES blocks merging and Actions can't approve to unblock
        assert formatter.get_review_action(critical_review, allow_approve=False) == "COMMENT"
