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


class TestResolveFixedComments:
    """Tests for duplicate detection and resolved comment handling."""

    def test_get_resolved_comment_ids_handles_notset(self):
        """Test that NotSet in_reply_to_id is handled gracefully."""
        from github.GithubObject import NotSet

        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()
        mock_comment = MagicMock()
        mock_comment.body = "✅ **Resolved** - This issue has been addressed"
        mock_comment.in_reply_to_id = NotSet
        mock_comment.user.login = "github-actions[bot]"
        mock_pr.get_review_comments.return_value = [mock_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            resolved = client._get_resolved_comment_ids(mock_pr)

        assert len(resolved) == 0  # Should not crash, should return empty

    def test_get_resolved_comment_ids_handles_none(self):
        """Test that None in_reply_to_id is handled gracefully."""
        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()
        mock_comment = MagicMock()
        mock_comment.body = "✅ **Resolved** - This issue has been addressed"
        mock_comment.in_reply_to_id = None
        mock_comment.user.login = "github-actions[bot]"
        mock_pr.get_review_comments.return_value = [mock_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            resolved = client._get_resolved_comment_ids(mock_pr)

        assert len(resolved) == 0  # None is not valid, should be skipped

    def test_get_resolved_comment_ids_handles_zero(self):
        """Test that 0 in_reply_to_id is handled gracefully."""
        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()
        mock_comment = MagicMock()
        mock_comment.body = "✅ **Resolved** - This issue has been addressed"
        mock_comment.in_reply_to_id = 0
        mock_comment.user.login = "github-actions[bot]"
        mock_pr.get_review_comments.return_value = [mock_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            resolved = client._get_resolved_comment_ids(mock_pr)

        assert len(resolved) == 0  # 0 is not valid, should be skipped

    def test_get_resolved_comment_ids_valid_reply(self):
        """Test that valid resolved replies are detected."""
        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()
        mock_comment = MagicMock()
        mock_comment.body = "✅ **Resolved** - This issue has been addressed"
        mock_comment.in_reply_to_id = 12345
        mock_comment.user.login = "github-actions[bot]"
        mock_pr.get_review_comments.return_value = [mock_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            resolved = client._get_resolved_comment_ids(mock_pr)

        assert 12345 in resolved

    def test_get_resolved_comment_ids_filters_by_user(self):
        """Test that resolved replies from unknown users are ignored."""
        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()
        mock_comment = MagicMock()
        mock_comment.body = "✅ **Resolved** - This issue has been addressed"
        mock_comment.in_reply_to_id = 12345
        mock_comment.user.login = "random-user"  # Not in AI_REVIEWER_USERS
        mock_pr.get_review_comments.return_value = [mock_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            resolved = client._get_resolved_comment_ids(mock_pr)

        assert len(resolved) == 0  # Unknown user, should be ignored

    def test_get_resolved_comment_ids_handles_none_user(self):
        """Test that comments with None user are handled gracefully."""
        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()
        mock_comment = MagicMock()
        mock_comment.body = "✅ **Resolved** - This issue has been addressed"
        mock_comment.in_reply_to_id = 12345
        mock_comment.user = None
        mock_pr.get_review_comments.return_value = [mock_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            resolved = client._get_resolved_comment_ids(mock_pr)

        assert len(resolved) == 0

    def test_get_resolved_comment_ids_handles_none_login(self):
        """Test that comments with None login are handled gracefully."""
        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()
        mock_comment = MagicMock()
        mock_comment.body = "✅ **Resolved** - This issue has been addressed"
        mock_comment.in_reply_to_id = 12345
        mock_comment.user.login = None
        mock_pr.get_review_comments.return_value = [mock_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            resolved = client._get_resolved_comment_ids(mock_pr)

        assert len(resolved) == 0

    def test_get_current_user_login_caches_result(self):
        """Test that current user login is cached."""
        from ai_reviewer.github.client import GitHubClient

        mock_gh = MagicMock()
        mock_gh.get_user.return_value.login = "test-user"

        with patch("ai_reviewer.github.client.Github", return_value=mock_gh):
            client = GitHubClient(token="test-token")

            # First call fetches
            login1 = client._get_current_user_login()
            # Second call uses cache
            login2 = client._get_current_user_login()

            assert login1 == "test-user"
            assert login2 == "test-user"
            # Should only call API once
            assert mock_gh.get_user.call_count == 1

    def test_get_current_user_login_caches_failure(self):
        """Test that failed user login fetch is also cached."""
        from ai_reviewer.github.client import GitHubClient

        mock_gh = MagicMock()
        mock_gh.get_user.side_effect = Exception("API error")

        with patch("ai_reviewer.github.client.Github", return_value=mock_gh):
            client = GitHubClient(token="test-token")

            # First call fails and caches failure
            login1 = client._get_current_user_login()
            # Second call returns cached failure
            login2 = client._get_current_user_login()

            assert login1 is None
            assert login2 is None
            # Should only call API once (failure is cached)
            assert mock_gh.get_user.call_count == 1

    def test_get_allowed_users_caches_result(self):
        """Test that allowed users set is cached."""
        from ai_reviewer.github.client import GitHubClient

        mock_gh = MagicMock()
        mock_gh.get_user.return_value.login = "test-user"

        with patch("ai_reviewer.github.client.Github", return_value=mock_gh):
            client = GitHubClient(token="test-token")

            users1 = client._get_allowed_users()
            users2 = client._get_allowed_users()

            assert "test-user" in users1
            assert "github-actions[bot]" in users1
            assert users1 is users2  # Same cached object

    def test_resolve_fixed_comments_avoids_redundant_api_calls(self):
        """Test that resolve_fixed_comments fetches comments only once."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment, ReviewDelta

        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = []
        mock_pr.base.repo.full_name = "test/repo"
        mock_pr.number = 1

        delta = ReviewDelta(
            fixed_findings=[
                PreviousComment(
                    id=123,
                    file_path="test.py",
                    line=10,
                    title="Test",
                    severity="warning",
                    body="test",
                )
            ]
        )

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")

            # This should fetch comments once and pass to helper
            client.resolve_fixed_comments(mock_pr, delta)

            # get_review_comments should be called exactly once (not twice)
            assert mock_pr.get_review_comments.call_count == 1

    def test_get_previous_review_comments_excludes_resolved(self):
        """Test that 'Resolved' replies are not treated as findings."""
        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()

        # Original finding comment
        original_comment = MagicMock()
        original_comment.body = "🔴 **SQL Injection**\n\nUser input in query"
        original_comment.user.login = "github-actions[bot]"
        original_comment.id = 100
        original_comment.path = "test.py"
        original_comment.line = 10
        original_comment.original_line = 10

        # Our "Resolved" reply - should be excluded
        resolved_reply = MagicMock()
        resolved_reply.body = "✅ **Resolved** - This issue has been addressed in the latest changes."
        resolved_reply.user.login = "github-actions[bot]"
        resolved_reply.id = 101
        resolved_reply.path = "test.py"
        resolved_reply.line = 10
        resolved_reply.original_line = 10

        mock_pr.get_review_comments.return_value = [original_comment, resolved_reply]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            comments = client.get_previous_review_comments(mock_pr)

        # Should only return the original finding, not the "Resolved" reply
        assert len(comments) == 1
        assert comments[0].id == 100
        assert "SQL Injection" in comments[0].title

    def test_resolve_thread_for_comment_calls_graphql(self):
        """Test that thread resolution uses GraphQL."""
        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")

            # Mock the GraphQL methods
            client._get_thread_id_for_comment = MagicMock(return_value="thread_123")
            client._resolve_review_thread = MagicMock(return_value=True)

            result = client._resolve_thread_for_comment("test/repo", 1, 456)

            assert result is True
            client._get_thread_id_for_comment.assert_called_once_with("test/repo", 1, 456)
            client._resolve_review_thread.assert_called_once_with("thread_123")
