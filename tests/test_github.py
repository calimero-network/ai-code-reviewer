"""Tests for GitHub integration."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestGitHubClient:
    """Tests for GitHub API client."""

    def test_403_raises_permission_error_immediately(self):
        """403 from a method using _raise_if_forbidden must surface as PermissionError."""
        from github.GithubException import GithubException

        from ai_reviewer.github.client import GitHubClient

        mock_file = MagicMock(filename="foo.py", status="modified")
        mock_repo = MagicMock()
        mock_repo.get_contents.side_effect = GithubException(
            status=403, data={"message": "Forbidden"}, headers={}
        )

        mock_pr = MagicMock()
        mock_pr.get_files.return_value = [mock_file]
        mock_pr.base.repo = mock_repo

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            with pytest.raises(PermissionError):
                client.get_changed_files(mock_pr)
            assert mock_repo.get_contents.call_count == 1

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

    def test_extra_reviewer_users_included_in_allowed_users(self):
        """extra_reviewer_users passed to GitHubClient are included in allowed set."""
        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github") as mock_gh:
            mock_gh.return_value.get_user.return_value.login = "my-bot"
            client = GitHubClient(
                token="t", extra_reviewer_users=["custom-bot[bot]", "ci-reviewer"]
            )
            allowed = client._get_allowed_users()
        assert "custom-bot[bot]" in allowed
        assert "ci-reviewer" in allowed
        assert "github-actions[bot]" in allowed  # default still present

    def test_default_allowlist_unchanged_without_extra_users(self):
        """Default allowlist is unchanged when no extra_reviewer_users provided."""
        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github") as mock_gh:
            mock_gh.return_value.get_user.return_value.login = "bot"
            client = GitHubClient(token="t")
            allowed = client._get_allowed_users()
        assert "github-actions[bot]" in allowed

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
        from ai_reviewer.github import webhook
        from ai_reviewer.github.webhook import PREvent, handle_pr_event

        mock_handler = AsyncMock()
        event = PREvent(repo="test-org/test-repo", pr_number=42, action="opened")

        with patch.object(webhook, "_review_handler", mock_handler):
            await handle_pr_event(event)
            mock_handler.assert_called_once_with(repo="test-org/test-repo", pr_number=42)

    @pytest.mark.asyncio
    async def test_ignores_irrelevant_actions(self):
        """Test that irrelevant PR actions are ignored."""
        from ai_reviewer.github import webhook
        from ai_reviewer.github.webhook import PREvent, handle_pr_event

        mock_handler = AsyncMock()
        event = PREvent(repo="test-org/test-repo", pr_number=42, action="labeled")

        with patch.object(webhook, "_review_handler", mock_handler):
            await handle_pr_event(event)
            mock_handler.assert_not_called()

    @pytest.mark.asyncio
    async def test_ai_review_comment_triggers_review(self):
        """Posting '/ai-review' as a PR comment should trigger a review."""
        from ai_reviewer.github import webhook
        from ai_reviewer.github.webhook import _handle_issue_comment_event

        mock_handler = AsyncMock()
        payload = {
            "action": "created",
            "comment": {"body": "/ai-review", "user": {"login": "contributor"}},
            "issue": {"number": 42, "pull_request": {"url": "https://..."}},
            "repository": {"full_name": "owner/repo"},
        }

        with patch.object(webhook, "_review_handler", mock_handler):
            await _handle_issue_comment_event(payload)
            mock_handler.assert_called_once_with(repo="owner/repo", pr_number=42)

    @pytest.mark.asyncio
    async def test_ai_review_comment_ignored_on_plain_issue(self):
        """'/ai-review' on a plain issue (not PR) must be ignored."""
        from ai_reviewer.github import webhook
        from ai_reviewer.github.webhook import _handle_issue_comment_event

        mock_handler = AsyncMock()
        payload = {
            "action": "created",
            "comment": {"body": "/ai-review"},
            "issue": {"number": 99},  # No pull_request key
            "repository": {"full_name": "owner/repo"},
        }

        with patch.object(webhook, "_review_handler", mock_handler):
            await _handle_issue_comment_event(payload)
            mock_handler.assert_not_called()


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

    def test_format_review_compact_is_short_with_inline_hint(self):
        """Compact format used when posting inline comments: short body + 'See inline'."""
        from datetime import datetime

        from ai_reviewer.github.formatter import GitHubFormatter
        from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity
        from ai_reviewer.models.review import ConsolidatedReview

        review = ConsolidatedReview(
            id="review-123",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[
                ConsolidatedFinding(
                    id="f1",
                    file_path="auth.py",
                    line_start=10,
                    line_end=12,
                    severity=Severity.WARNING,
                    category=Category.SECURITY,
                    title="Missing validation",
                    description="Add input check",
                    suggested_fix=None,
                    consensus_score=0.66,
                    agreeing_agents=["agent-1", "agent-2"],
                    confidence=0.8,
                ),
            ],
            summary="One warning",
            agent_count=3,
            review_quality_score=0.7,
            total_review_time_ms=2000,
        )

        formatter = GitHubFormatter()
        compact = formatter.format_review_compact(review)

        assert "See inline comments" in compact
        assert "🟡" in compact or "warnings" in compact
        # No full finding description in body (lives inline only)
        assert "Add input check" not in compact

    def test_format_review_compact_empty_is_one_line_lgtm(self):
        """Compact format with no findings is one-line LGTM."""
        from datetime import datetime

        from ai_reviewer.github.formatter import GitHubFormatter
        from ai_reviewer.models.review import ConsolidatedReview

        review = ConsolidatedReview(
            id="review-123",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="No issues",
            agent_count=1,
            review_quality_score=0.95,
            total_review_time_ms=1000,
        )
        formatter = GitHubFormatter()
        compact = formatter.format_review_compact(review)
        assert "LGTM" in compact
        assert "No issues" in compact or "✅" in compact

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

        # LGTM-with-comments: only nits/suggestions (no critical/warning) → COMMENT (don't block author)
        nits_only_review = ConsolidatedReview(
            id="review-nits",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[
                ConsolidatedFinding(
                    id="f1",
                    file_path="test.py",
                    line_start=1,
                    line_end=2,
                    severity=Severity.SUGGESTION,
                    category=Category.STYLE,
                    title="Consider renaming",
                    description="Optional",
                    suggested_fix=None,
                    consensus_score=0.5,
                    agreeing_agents=["a1"],
                    confidence=0.8,
                ),
                ConsolidatedFinding(
                    id="f2",
                    file_path="test.py",
                    line_start=3,
                    line_end=3,
                    severity=Severity.NITPICK,
                    category=Category.STYLE,
                    title="Nit: trailing space",
                    description="Style",
                    suggested_fix=None,
                    consensus_score=0.3,
                    agreeing_agents=["a1"],
                    confidence=0.7,
                ),
            ],
            summary="Minor suggestions",
            agent_count=2,
            review_quality_score=0.9,
            total_review_time_ms=1500,
        )
        assert formatter.get_review_action(nits_only_review) == "COMMENT"
        assert formatter.get_review_action(nits_only_review, allow_approve=True) == "COMMENT"

        # Only warnings (no critical) → COMMENT (we don't block on warnings)
        warnings_only_review = ConsolidatedReview(
            id="review-warn",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[
                ConsolidatedFinding(
                    id="f1",
                    file_path="test.py",
                    line_start=1,
                    line_end=1,
                    severity=Severity.WARNING,
                    category=Category.LOGIC,
                    title="Edge case",
                    description="Consider handling",
                    suggested_fix=None,
                    consensus_score=1.0,
                    agreeing_agents=["a1"],
                    confidence=0.85,
                ),
            ],
            summary="One warning",
            agent_count=1,
            review_quality_score=0.85,
            total_review_time_ms=1000,
        )
        assert formatter.get_review_action(warnings_only_review) == "COMMENT"


class TestResolveFixedComments:
    """Tests for duplicate detection and resolved comment handling."""

    def test_get_resolved_comment_ids_handles_notset(self):
        """Test that NotSet in_reply_to_id is handled gracefully."""
        from github.GithubObject import NotSet

        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()
        mock_comment = MagicMock()
        mock_comment.body = "✅ **No longer detected** - This issue was not re-detected after the latest changes."
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
        mock_comment.body = "✅ **No longer detected** - This issue was not re-detected after the latest changes."
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
        mock_comment.body = "✅ **No longer detected** - This issue was not re-detected after the latest changes."
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
        mock_comment.body = "✅ **No longer detected** - This issue was not re-detected after the latest changes."
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
        mock_comment.body = "✅ **No longer detected** - This issue was not re-detected after the latest changes."
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
        mock_comment.body = "✅ **No longer detected** - This issue was not re-detected after the latest changes."
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
        mock_comment.body = "✅ **No longer detected** - This issue was not re-detected after the latest changes."
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

    def test_get_allowed_users_fallback_when_user_fetch_fails(self):
        """Test that allowed users still works when user login fetch fails."""
        from ai_reviewer.github.client import GitHubClient

        mock_gh = MagicMock()
        mock_gh.get_user.side_effect = Exception("API error")

        with patch("ai_reviewer.github.client.Github", return_value=mock_gh):
            client = GitHubClient(token="test-token")

            users = client._get_allowed_users()

            # Should still include the known bot users
            assert "github-actions[bot]" in users
            # Should NOT include None or empty string
            assert None not in users
            assert "" not in users

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
        resolved_reply.body = (
            "✅ **No longer detected** - This issue was not re-detected after the latest changes."
        )
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

    def test_get_previous_review_comments_excludes_human_comments(self):
        """Test that human comments (even with AI-like format) are never processed."""
        from ai_reviewer.github.client import GitHubClient

        mock_pr = MagicMock()

        # Human comment with emoji/format that looks like ours - should be excluded
        human_comment = MagicMock()
        human_comment.body = "🔴 **Critical** - Consider fixing this"
        human_comment.user.login = "human-dev"
        human_comment.id = 200
        human_comment.path = "src/foo.py"
        human_comment.line = 5
        human_comment.original_line = 5

        mock_pr.get_review_comments.return_value = [human_comment]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            comments = client.get_previous_review_comments(mock_pr)

        # Must not treat human comments as ours - no reply "Resolved" to them
        assert len(comments) == 0

    def test_compute_review_delta_fixes_when_file_removed(self):
        """Test that we mark as fixed when the commented file is no longer in the diff."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment

        mock_pr = MagicMock()
        # Only bar.py in diff - foo.py was removed
        mock_file = MagicMock()
        mock_file.filename = "src/bar.py"
        mock_file.patch = "@@ -1,3 +1,3 @@\n context\n-old\n+new\n context"
        mock_pr.get_files.return_value = [mock_file]

        # Previous comment on removed file - should be fixed
        prev_removed = PreviousComment(
            id=1,
            file_path="src/removed.py",
            line=5,
            title="Bug in removed",
            severity="suggestion",
            body="💡 **Bug in removed**\n\nIssue",
        )

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client.get_previous_review_comments = MagicMock(return_value=[prev_removed])

            delta = client.compute_review_delta(mock_pr, [])

        # File removed from diff → fixed
        assert len(delta.fixed_findings) == 1
        assert delta.fixed_findings[0].file_path == "src/removed.py"

    def test_compute_review_delta_fixes_when_deleted_file_has_no_patch(self):
        """Test that we mark as fixed when the file was deleted but has no patch (binary/large)."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment

        mock_pr = MagicMock()
        # Deleted file in diff - no patch (binary or large file)
        mock_file = MagicMock()
        mock_file.filename = "src/deleted.bin"
        mock_file.patch = None  # No patch for binary/large/deleted files
        mock_file.status = "removed"
        mock_pr.get_files.return_value = [mock_file]

        prev_deleted = PreviousComment(
            id=1,
            file_path="src/deleted.bin",
            line=5,
            title="Issue in binary file",
            severity="warning",
            body="🟡 **Issue in binary file**\n\nIssue",
        )

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client.get_previous_review_comments = MagicMock(return_value=[prev_deleted])

            delta = client.compute_review_delta(mock_pr, [])

        # File deleted (status=removed), no patch → fixed
        assert len(delta.fixed_findings) == 1
        assert delta.fixed_findings[0].file_path == "src/deleted.bin"

    def test_compute_review_delta_fixes_when_line_modified(self):
        """Test that we mark as fixed when the commented line was modified."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment

        mock_pr = MagicMock()
        # File still in diff with changes around line 10
        mock_file = MagicMock()
        mock_file.filename = "src/foo.py"
        mock_file.patch = (
            "@@ -8,7 +8,7 @@\n context\n context\n-old line 10\n+new line 10\n context\n context"
        )
        mock_pr.get_files.return_value = [mock_file]

        # Previous comment on line 10 - line was modified, so should be fixed
        prev_modified = PreviousComment(
            id=1,
            file_path="src/foo.py",
            line=10,
            title="Bug on line 10",
            severity="warning",
            body="🟡 **Bug on line 10**\n\nIssue",
        )

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client.get_previous_review_comments = MagicMock(return_value=[prev_modified])

            # AI didn't find the issue again → fixed
            delta = client.compute_review_delta(mock_pr, [])

        assert len(delta.fixed_findings) == 1
        assert delta.fixed_findings[0].file_path == "src/foo.py"

    def test_compute_review_delta_not_fixed_when_line_unmodified(self):
        """Test that we don't mark as fixed when the commented line wasn't touched."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment

        mock_pr = MagicMock()
        # File in diff but changes are far from line 100
        mock_file = MagicMock()
        mock_file.filename = "src/foo.py"
        mock_file.patch = "@@ -1,3 +1,4 @@\n context\n+added line\n context\n context"
        mock_pr.get_files.return_value = [mock_file]

        # Previous comment on line 100 - far from the changes
        prev_untouched = PreviousComment(
            id=1,
            file_path="src/foo.py",
            line=100,
            title="Bug on line 100",
            severity="warning",
            body="🟡 **Bug on line 100**\n\nIssue",
        )

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client.get_previous_review_comments = MagicMock(return_value=[prev_untouched])

            delta = client.compute_review_delta(mock_pr, [])

        # Line wasn't modified → NOT fixed (might still be an issue)
        assert len(delta.fixed_findings) == 0

    def test_parse_modified_lines_extracts_added_lines(self):
        """Test that _parse_modified_lines correctly extracts modified line numbers."""
        from ai_reviewer.github.client import GitHubClient

        # Diff patch with changes at lines 10-11
        diff_patch = """@@ -8,6 +8,7 @@
 context line 8
 context line 9
-old line 10
+new line 10
+added line 11
 context line 12
 context line 13"""

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")

            modified = client._parse_modified_lines(diff_patch)

            # Lines 10 and 11 should be marked as modified
            assert 10 in modified
            assert 11 in modified
            # Line 8 (context) should not be in modified
            assert 8 not in modified

    def test_is_line_in_modified_range_with_tolerance(self):
        """Test that _is_line_in_modified_range respects tolerance."""
        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")

            modified_lines = {10, 11, 12}

            # Line 10 is exactly modified
            assert client._is_line_in_modified_range(10, modified_lines) is True

            # Line 13 is within tolerance (3) of line 12
            assert client._is_line_in_modified_range(13, modified_lines, tolerance=3) is True

            # Line 7 is within tolerance (3) of line 10
            assert client._is_line_in_modified_range(7, modified_lines, tolerance=3) is True

            # Line 20 is outside tolerance
            assert client._is_line_in_modified_range(20, modified_lines, tolerance=3) is False

    def test_resolve_fixed_comments_skips_when_already_replied(self):
        """resolve_fixed_comments does not post a second Resolved reply."""
        from ai_reviewer.github.client import (
            GitHubClient,
            PreviousComment,
            ReviewDelta,
        )

        mock_pr = MagicMock()
        mock_pr.base.repo.full_name = "test/repo"
        mock_pr.number = 1

        # Comment 100 is the original finding; comment 101 is our existing Resolved reply
        original_comment = MagicMock()
        original_comment.id = 100
        original_comment.body = "🔴 **Bug**"
        original_comment.in_reply_to_id = None
        original_comment.user.login = "github-actions[bot]"

        resolved_reply = MagicMock()
        resolved_reply.id = 101
        resolved_reply.body = (
            "✅ **No longer detected** - This issue was not re-detected after the latest changes."
        )
        resolved_reply.in_reply_to_id = 100
        resolved_reply.user.login = "github-actions[bot]"

        mock_pr.get_review_comments.return_value = [original_comment, resolved_reply]

        delta = ReviewDelta(
            fixed_findings=[
                PreviousComment(
                    id=100,
                    file_path="src/foo.py",
                    line=10,
                    title="Bug",
                    severity="warning",
                    body="🔴 **Bug**",
                )
            ]
        )

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._fetch_thread_mapping = MagicMock(return_value={})

            count = client.resolve_fixed_comments(mock_pr, delta)

        # Should not post again
        assert count == 0
        mock_pr.create_review_comment_reply.assert_not_called()

    def test_resolve_thread_for_comment_uses_mapping(self):
        """Test that thread resolution uses pre-fetched mapping."""
        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")

            # Mock the resolve method
            client._resolve_review_thread = MagicMock(return_value=True)

            # Pre-built mapping from comment_id to thread_id
            thread_mapping = {456: "thread_123", 789: "thread_456"}

            result = client._resolve_thread_for_comment(456, thread_mapping)

            assert result is True
            client._resolve_review_thread.assert_called_once_with("thread_123")

    def test_resolve_thread_for_comment_missing_in_mapping(self):
        """Test that missing comment in mapping returns False."""
        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client._resolve_review_thread = MagicMock(return_value=True)

            thread_mapping = {456: "thread_123"}  # 999 not in mapping

            result = client._resolve_thread_for_comment(999, thread_mapping)

            assert result is False
            client._resolve_review_thread.assert_not_called()

    def test_fetch_thread_mapping_respects_max_pages(self):
        """Test that thread mapping fetch respects max page limit."""
        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")

            # Mock GraphQL to always return hasNextPage=True (infinite loop scenario)
            def mock_graphql(_query, _variables=None):
                return {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "pageInfo": {"hasNextPage": True, "endCursor": "cursor"},
                                "nodes": [
                                    {
                                        "id": "thread_1",
                                        "isResolved": False,
                                        "comments": {"nodes": [{"databaseId": 123}]},
                                    }
                                ],
                            }
                        }
                    }
                }

            client._graphql_request = MagicMock(side_effect=mock_graphql)

            # This should NOT loop forever due to _MAX_GRAPHQL_PAGES
            result = client._fetch_thread_mapping("test/repo", 1)

            # Should have called GraphQL exactly _MAX_GRAPHQL_PAGES times
            assert client._graphql_request.call_count == client._MAX_GRAPHQL_PAGES
            # Should still return the mapping it built
            assert 123 in result

    def test_fixed_findings_deduplicated_by_id(self):
        """compute_review_delta must deduplicate fixed_findings by comment ID."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment

        # Build a comment that appears in a file NOT in the PR diff —
        # that path marks it as fixed in compute_review_delta.
        stale_comment = PreviousComment(
            id=42, file_path="deleted.py", line=5, title="Old issue", severity="Warning", body="b"
        )

        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = []

        # get_previous_review_comments returns the same comment twice (simulating
        # a scenario where the same comment was stored/retrieved twice)
        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")

        with patch.object(
            client,
            "get_previous_review_comments",
            return_value=[stale_comment, stale_comment],  # duplicate
        ):
            mock_file = MagicMock()
            mock_file.filename = "other.py"  # deleted.py is NOT in the diff → triggers fixed
            mock_file.status = "modified"
            mock_file.patch = "@@ -1,3 +1,3 @@\n-old\n+new"
            mock_pr.get_files.return_value = [mock_file]

            delta = client.compute_review_delta(mock_pr, current_findings=[])

        # Even though the same comment appeared twice in previous_comments,
        # fixed_findings must be deduplicated to exactly 1 entry
        assert len(delta.fixed_findings) == 1
        assert delta.fixed_findings[0].id == 42

    def test_spoofed_resolved_comment_not_counted(self):
        """A 'Resolved' comment from a non-bot user must not be counted."""
        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
        mock_comment = MagicMock()
        mock_comment.body = "✅ **Resolved** - fake"
        mock_comment.in_reply_to_id = 999
        mock_comment.user.login = "malicious-user"
        mock_pr = MagicMock()
        mock_pr.get_review_comments.return_value = [mock_comment]
        with patch.object(client, "_get_allowed_users", return_value={"github-actions[bot]"}):
            resolved = client._get_resolved_comment_ids(mock_pr)
        assert 999 not in resolved

    def test_graphql_errors_not_logged_verbatim_at_warning(self, caplog):
        """Raw GraphQL error details must not appear in WARNING-level logs."""
        import logging

        from ai_reviewer.github.client import GitHubClient

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"errors": [{"message": "secret internal detail"}]}
        with patch("requests.post", return_value=mock_resp), caplog.at_level(logging.WARNING):
            result = client._graphql_request("{ viewer { login } }")
        assert result is None
        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("secret internal detail" in m for m in warning_msgs)


class TestApplyCommentLimits:
    """Tests for the apply_comment_limits utility function."""

    @staticmethod
    def _make_finding(
        file_path: str, severity: str, confidence: float = 0.9, consensus: float = 1.0
    ):
        from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity

        sev = Severity(severity)
        return ConsolidatedFinding(
            id=f"{file_path}-{severity}-{confidence}",
            file_path=file_path,
            line_start=1,
            line_end=None,
            severity=sev,
            category=Category.LOGIC,
            title=f"Issue in {file_path}",
            description="desc",
            suggested_fix=None,
            consensus_score=consensus,
            agreeing_agents=["a1"],
            confidence=confidence,
        )

    def test_respects_max_total(self):
        """apply_comment_limits caps total findings."""
        from ai_reviewer.github.client import apply_comment_limits

        findings = [self._make_finding(f"file{i}.py", "warning") for i in range(20)]
        result = apply_comment_limits(findings, max_total=5, max_per_file=100)
        assert len(result) == 5

    def test_respects_max_per_file(self):
        """apply_comment_limits caps findings per file."""
        from ai_reviewer.github.client import apply_comment_limits

        findings = [
            self._make_finding("same.py", "warning", confidence=0.9 - i * 0.01) for i in range(10)
        ]
        result = apply_comment_limits(findings, max_total=100, max_per_file=3)
        assert len(result) == 3
        assert all(f.file_path == "same.py" for f in result)

    def test_sorts_by_priority_descending(self):
        """Higher-priority findings are kept over lower-priority ones."""
        from ai_reviewer.github.client import apply_comment_limits

        critical = self._make_finding("a.py", "critical", confidence=0.95)
        nitpick = self._make_finding("b.py", "nitpick", confidence=0.5)
        result = apply_comment_limits([nitpick, critical], max_total=1, max_per_file=10)
        assert len(result) == 1
        assert result[0].severity.value == "critical"

    def test_per_file_limit_distributes_across_files(self):
        """Per-file cap lets findings from other files through."""
        from ai_reviewer.github.client import apply_comment_limits

        findings = [
            self._make_finding("a.py", "warning", confidence=0.9 - i * 0.01) for i in range(5)
        ] + [self._make_finding("b.py", "warning", confidence=0.85)]
        result = apply_comment_limits(findings, max_total=100, max_per_file=2)
        a_count = sum(1 for f in result if f.file_path == "a.py")
        b_count = sum(1 for f in result if f.file_path == "b.py")
        assert a_count == 2
        assert b_count == 1

    def test_empty_input(self):
        """Empty findings list returns empty."""
        from ai_reviewer.github.client import apply_comment_limits

        assert apply_comment_limits([], max_total=10, max_per_file=5) == []

    def test_defaults_match_config_defaults(self):
        """Default arguments match OutputSettings defaults (50 total, 10 per file)."""
        from ai_reviewer.github.client import apply_comment_limits

        findings = [self._make_finding(f"f{i}.py", "warning") for i in range(60)]
        result = apply_comment_limits(findings)
        assert len(result) == 50

    def test_does_not_mutate_input(self):
        """Input list is not modified."""
        from ai_reviewer.github.client import apply_comment_limits

        findings = [self._make_finding(f"f{i}.py", "warning") for i in range(5)]
        original_order = [f.id for f in findings]
        apply_comment_limits(findings, max_total=2, max_per_file=10)
        assert [f.id for f in findings] == original_order


def _make_finding_for_delta(
    file_path: str = "src/auth.py",
    line_start: int = 10,
    title: str = "SQL Injection Vulnerability",
):
    from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity

    return ConsolidatedFinding(
        id="test-1",
        file_path=file_path,
        line_start=line_start,
        line_end=None,
        severity=Severity.WARNING,
        category=Category.SECURITY,
        title=title,
        description="desc",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=["a"],
        confidence=0.9,
    )


class TestPreviousCommentFuzzyHash:
    """Tests for the fuzzy hash property on PreviousComment."""

    def test_fuzzy_hash_matches_finding_fuzzy_hash(self):
        """PreviousComment fuzzy hash matches ConsolidatedFinding fuzzy hash for same content."""
        from ai_reviewer.github.client import PreviousComment

        finding = _make_finding_for_delta(
            file_path="src/auth.py", title="SQL Injection Vulnerability"
        )
        comment = PreviousComment(
            id=1,
            file_path="src/auth.py",
            line=10,
            title="SQL Injection Vulnerability",
            severity="warning",
            body="body",
        )
        assert comment.finding_hash_fuzzy == finding.finding_hash_fuzzy

    def test_fuzzy_hash_none_when_no_file_path(self):
        """Returns None when file_path is empty."""
        from ai_reviewer.github.client import PreviousComment

        comment = PreviousComment(
            id=1, file_path="", line=10, title="Issue", severity="warning", body="body"
        )
        assert comment.finding_hash_fuzzy is None

    def test_fuzzy_hash_none_when_no_title(self):
        """Returns None when title is empty."""
        from ai_reviewer.github.client import PreviousComment

        comment = PreviousComment(
            id=1, file_path="src/auth.py", line=10, title="", severity="warning", body="body"
        )
        assert comment.finding_hash_fuzzy is None

    def test_fuzzy_hash_ignores_line_drift(self):
        """Same file+title at different lines produce same fuzzy hash."""
        from ai_reviewer.github.client import PreviousComment

        c1 = PreviousComment(
            id=1,
            file_path="src/auth.py",
            line=10,
            title="SQL Injection Vulnerability",
            severity="warning",
            body="body",
        )
        c2 = PreviousComment(
            id=2,
            file_path="src/auth.py",
            line=50,
            title="SQL Injection Vulnerability",
            severity="warning",
            body="body",
        )
        assert c1.finding_hash_fuzzy == c2.finding_hash_fuzzy


class TestComputeReviewDeltaFuzzyMatching:
    """Tests for multi-tier matching in compute_review_delta()."""

    def test_fuzzy_match_catches_line_drifted_finding(self):
        """A finding that moved lines is matched via fuzzy hash (not new)."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment

        prev_comment = PreviousComment(
            id=1,
            file_path="src/auth.py",
            line=10,
            title="SQL Injection Vulnerability",
            severity="warning",
            body="🟡 **SQL Injection Vulnerability**\n\ndesc\n\n<!-- ai-reviewer-id: aabbccddee11 -->",
            finding_hash="aabbccddee11",
        )

        current_finding = _make_finding_for_delta(
            file_path="src/auth.py",
            line_start=25,
            title="SQL Injection Vulnerability",
        )

        mock_pr = MagicMock()
        mock_file = MagicMock()
        mock_file.filename = "src/auth.py"
        mock_file.patch = "@@ -1,3 +1,3 @@\n-old\n+new"
        mock_file.status = "modified"
        mock_pr.get_files.return_value = [mock_file]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client.get_previous_review_comments = MagicMock(return_value=[prev_comment])

            delta = client.compute_review_delta(mock_pr, [current_finding])

        assert len(delta.open_findings) == 1
        assert len(delta.new_findings) == 0

    def test_strict_hash_takes_priority_over_fuzzy(self):
        """When strict hash matches, fuzzy hash is not consulted."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment

        finding = _make_finding_for_delta(file_path="src/auth.py", line_start=10)
        strict_hash = finding.finding_hash

        prev_comment = PreviousComment(
            id=1,
            file_path="src/auth.py",
            line=10,
            title="SQL Injection Vulnerability",
            severity="warning",
            body=f"body\n\n<!-- ai-reviewer-id: {strict_hash} -->",
            finding_hash=strict_hash,
        )

        mock_pr = MagicMock()
        mock_file = MagicMock()
        mock_file.filename = "src/auth.py"
        mock_file.patch = "@@ -1,3 +1,3 @@\n-old\n+new"
        mock_file.status = "modified"
        mock_pr.get_files.return_value = [mock_file]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client.get_previous_review_comments = MagicMock(return_value=[prev_comment])

            delta = client.compute_review_delta(mock_pr, [finding])

        assert len(delta.open_findings) == 1
        assert len(delta.new_findings) == 0

    def test_title_fallback_still_works(self):
        """Legacy title+line matching still works when neither hash tier matches.

        The PreviousComment has no embedded strict hash and its fuzzy hash is
        patched to None, so only the title+line fallback (tier 3) can match.
        """
        from ai_reviewer.github.client import GitHubClient, PreviousComment

        prev_comment = PreviousComment(
            id=1,
            file_path="src/auth.py",
            line=10,
            title="SQL Injection Vulnerability",
            severity="warning",
            body="🟡 **SQL Injection Vulnerability**\n\ndesc",
            finding_hash=None,
        )

        current_finding = _make_finding_for_delta(
            file_path="src/auth.py",
            line_start=10,
            title="SQL Injection Vulnerability",
        )

        mock_pr = MagicMock()
        mock_file = MagicMock()
        mock_file.filename = "src/auth.py"
        mock_file.patch = "@@ -1,3 +1,3 @@\n-old\n+new"
        mock_file.status = "modified"
        mock_pr.get_files.return_value = [mock_file]

        with (
            patch("ai_reviewer.github.client.Github"),
            patch.object(
                PreviousComment,
                "finding_hash_fuzzy",
                new_callable=lambda: property(lambda _self: None),
            ),
        ):
            client = GitHubClient(token="test-token")
            client.get_previous_review_comments = MagicMock(return_value=[prev_comment])

            delta = client.compute_review_delta(mock_pr, [current_finding])

        assert len(delta.open_findings) == 1
        assert len(delta.new_findings) == 0

    def test_truly_new_finding_not_matched(self):
        """A genuinely new finding (different file+title) is classified as new."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment

        prev_comment = PreviousComment(
            id=1,
            file_path="src/auth.py",
            line=10,
            title="SQL Injection Vulnerability",
            severity="warning",
            body="body",
            finding_hash="aabbccddee11",
        )

        new_finding = _make_finding_for_delta(
            file_path="src/utils.py",
            line_start=5,
            title="Buffer Overflow Risk",
        )

        mock_pr = MagicMock()
        mock_file = MagicMock()
        mock_file.filename = "src/utils.py"
        mock_file.patch = "@@ -1,3 +1,3 @@\n-old\n+new"
        mock_file.status = "modified"
        mock_pr.get_files.return_value = [mock_file]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            client.get_previous_review_comments = MagicMock(return_value=[prev_comment])

            delta = client.compute_review_delta(mock_pr, [new_finding])

        assert len(delta.new_findings) == 1
        assert len(delta.open_findings) == 0
