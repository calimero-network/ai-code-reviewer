"""Tests for convergence detection (Task 7) and severity stabilization (Task 8).

Convergence means "no delta churn" — the issue set hasn't changed since the
last review.  This is distinct from ``all_issues_resolved`` (PR is clean).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_reviewer.github.client import (
    PreviousComment,
    ReviewDelta,
    has_converged,
    should_skip_review,
    stabilize_severity,
)
from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity


def _finding(
    severity: Severity = Severity.WARNING,
    title: str = "Issue",
    file_path: str = "src/foo.py",
    line_start: int = 10,
) -> ConsolidatedFinding:
    return ConsolidatedFinding(
        id="f1",
        file_path=file_path,
        line_start=line_start,
        line_end=None,
        severity=severity,
        category=Category.LOGIC,
        title=title,
        description="desc",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=["a1"],
        confidence=0.9,
    )


def _prev_comment(
    id: int = 1,
    severity: str = "warning",
    title: str = "Issue",
) -> PreviousComment:
    return PreviousComment(
        id=id,
        file_path="src/foo.py",
        line=10,
        title=title,
        severity=severity,
        body="body",
    )


class TestHasConverged:
    """Unit tests for has_converged()."""

    def test_converged_when_only_open_findings(self):
        delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[],
            open_findings=[_finding()],
            previous_comments=[_prev_comment()],
        )
        assert has_converged(delta) is True

    def test_not_converged_when_new_findings(self):
        delta = ReviewDelta(
            new_findings=[_finding()],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )
        assert has_converged(delta) is False

    def test_not_converged_when_fixed_findings(self):
        delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )
        assert has_converged(delta) is False

    def test_converged_when_empty_delta_with_previous(self):
        """Empty delta (no open, no new, no fixed) with previous comments is converged."""
        delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )
        assert has_converged(delta) is True

    def test_not_converged_when_both_new_and_fixed(self):
        delta = ReviewDelta(
            new_findings=[_finding()],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )
        assert has_converged(delta) is False


class TestShouldSkipReview:
    """Unit tests for should_skip_review()."""

    def test_never_skips_first_review(self):
        delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[],
            open_findings=[_finding()],
            previous_comments=[],
        )
        assert should_skip_review(review_count=1, delta=delta) is False

    def test_never_skips_review_count_zero(self):
        delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[],
        )
        assert should_skip_review(review_count=0, delta=delta) is False

    def test_skips_converged_second_review(self):
        delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[],
            open_findings=[_finding()],
            previous_comments=[_prev_comment()],
        )
        assert should_skip_review(review_count=2, delta=delta) is True

    def test_does_not_skip_second_review_with_new_findings(self):
        delta = ReviewDelta(
            new_findings=[_finding()],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )
        assert should_skip_review(review_count=2, delta=delta) is False

    def test_does_not_skip_second_review_with_fixed_findings(self):
        delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )
        assert should_skip_review(review_count=2, delta=delta) is False

    def test_skips_third_review_with_only_new_nitpicks(self):
        delta = ReviewDelta(
            new_findings=[_finding(severity=Severity.NITPICK, title="Nit: trailing space")],
            fixed_findings=[],
            open_findings=[_finding()],
            previous_comments=[_prev_comment()],
        )
        assert should_skip_review(review_count=3, delta=delta) is True

    def test_does_not_skip_second_review_with_only_new_nitpicks(self):
        """Nitpick-only skip requires review_count >= 3."""
        delta = ReviewDelta(
            new_findings=[_finding(severity=Severity.NITPICK, title="Nit: trailing space")],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )
        assert should_skip_review(review_count=2, delta=delta) is False

    def test_does_not_skip_third_review_with_non_nitpick_new_findings(self):
        delta = ReviewDelta(
            new_findings=[_finding(severity=Severity.WARNING, title="Real issue")],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )
        assert should_skip_review(review_count=3, delta=delta) is False

    def test_does_not_skip_third_review_with_mixed_new_findings(self):
        """If any new finding is not a nitpick, don't skip."""
        delta = ReviewDelta(
            new_findings=[
                _finding(severity=Severity.NITPICK, title="Nit: style"),
                _finding(severity=Severity.WARNING, title="Real bug"),
            ],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )
        assert should_skip_review(review_count=3, delta=delta) is False

    def test_skips_high_review_count_when_converged(self):
        delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[],
            open_findings=[_finding()],
            previous_comments=[_prev_comment()],
        )
        assert should_skip_review(review_count=10, delta=delta) is True


class TestCLIConvergenceGate:
    """Integration tests for the CLI convergence skip gate."""

    def test_cli_skips_posting_when_converged(self):
        """CLI should skip posting when convergence is detected."""
        from click.testing import CliRunner

        from ai_reviewer.cli import cli

        runner = CliRunner()

        with patch("ai_reviewer.cli.review_pr_async", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = MagicMock(
                findings=[],
                summary="No issues",
                agent_count=3,
            )

            runner.invoke(
                cli,
                ["review-pr", "test-org/test-repo", "42"],
                catch_exceptions=False,
            )

            mock_review.assert_called_once()
            call_kwargs = mock_review.call_args.kwargs
            assert "force_review" in call_kwargs
            assert call_kwargs["force_review"] is False

    def test_cli_force_review_flag_passes_through(self):
        """--force-review flag should pass force_review=True to review_pr_async."""
        from click.testing import CliRunner

        from ai_reviewer.cli import cli

        runner = CliRunner()

        with patch("ai_reviewer.cli.review_pr_async", new_callable=AsyncMock) as mock_review:
            mock_review.return_value = MagicMock(
                findings=[],
                summary="No issues",
                agent_count=3,
            )

            runner.invoke(
                cli,
                ["review-pr", "test-org/test-repo", "42", "--force-review"],
                catch_exceptions=False,
            )

            mock_review.assert_called_once()
            call_kwargs = mock_review.call_args.kwargs
            assert call_kwargs["force_review"] is True

    def test_cli_skips_posting_on_convergence(self):
        """When convergence is detected and --force-review is not set, posting is skipped."""
        from datetime import datetime

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        review = ConsolidatedReview(
            id="r1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[_finding()],
            summary="One issue",
            agent_count=1,
            review_quality_score=0.9,
            total_review_time_ms=1000,
        )

        converged_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[],
            open_findings=[_finding()],
            previous_comments=[_prev_comment(id=i) for i in range(5)],
        )

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.review_pr_with_cursor_agent", return_value=review),
            patch("ai_reviewer.cli.GitHubClient") as mock_gh_cls,
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            mock_gh = MagicMock()
            mock_gh_cls.return_value = mock_gh
            mock_gh.compute_review_delta.return_value = converged_delta

            import asyncio

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    force_review=False,
                )
            )

            mock_gh.post_review.assert_not_called()
            mock_gh.resolve_fixed_comments.assert_not_called()
            mock_gh.post_inline_comments.assert_not_called()

    def test_cli_posts_when_force_review_overrides_convergence(self):
        """When --force-review is set, posting proceeds even if converged."""
        from datetime import datetime

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        review = ConsolidatedReview(
            id="r1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[_finding()],
            summary="One issue",
            agent_count=1,
            review_quality_score=0.9,
            total_review_time_ms=1000,
        )

        converged_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[],
            open_findings=[_finding()],
            previous_comments=[_prev_comment(id=i) for i in range(5)],
        )

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.review_pr_with_cursor_agent", return_value=review),
            patch("ai_reviewer.cli.GitHubClient") as mock_gh_cls,
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            mock_gh = MagicMock()
            mock_gh_cls.return_value = mock_gh
            mock_gh.compute_review_delta.return_value = converged_delta

            import asyncio

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    force_review=True,
                )
            )

            mock_gh.post_review.assert_called_once()


class TestSeverityStabilization:
    """Unit tests for stabilize_severity() (Task 8)."""

    def test_same_severity_returns_current(self):
        assert (
            stabilize_severity(Severity.WARNING, Severity.WARNING, review_count=3)
            == Severity.WARNING
        )

    def test_upgrade_always_allowed(self):
        assert (
            stabilize_severity(Severity.CRITICAL, Severity.WARNING, review_count=5)
            == Severity.CRITICAL
        )
        assert (
            stabilize_severity(Severity.WARNING, Severity.SUGGESTION, review_count=3)
            == Severity.WARNING
        )

    def test_downgrade_blocked_after_stabilization(self):
        assert (
            stabilize_severity(Severity.SUGGESTION, Severity.WARNING, review_count=2)
            == Severity.WARNING
        )
        assert (
            stabilize_severity(Severity.NITPICK, Severity.WARNING, review_count=3)
            == Severity.WARNING
        )

    def test_downgrade_allowed_on_first_review(self):
        assert (
            stabilize_severity(Severity.SUGGESTION, Severity.WARNING, review_count=1)
            == Severity.SUGGESTION
        )

    def test_downgrade_from_critical_blocked_after_stabilization(self):
        assert (
            stabilize_severity(Severity.WARNING, Severity.CRITICAL, review_count=2)
            == Severity.CRITICAL
        )
        assert (
            stabilize_severity(Severity.NITPICK, Severity.CRITICAL, review_count=4)
            == Severity.CRITICAL
        )

    def test_downgrade_from_critical_allowed_on_first_review(self):
        assert (
            stabilize_severity(Severity.WARNING, Severity.CRITICAL, review_count=1)
            == Severity.WARNING
        )


class TestWebhookConvergenceGate:
    """Tests for convergence skip logic in the webhook handler."""

    @pytest.mark.asyncio
    async def test_webhook_skips_posting_when_converged(self):
        """Webhook default_review_handler skips posting on convergence."""
        from datetime import datetime

        from ai_reviewer.models.review import ConsolidatedReview

        review = ConsolidatedReview(
            id="r1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[_finding()],
            summary="One issue",
            agent_count=1,
            review_quality_score=0.9,
            total_review_time_ms=1000,
        )

        converged_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[],
            open_findings=[_finding()],
            previous_comments=[_prev_comment(id=i) for i in range(5)],
        )

        mock_gh = MagicMock()
        mock_gh.compute_review_delta.return_value = converged_delta

        with (
            patch.dict(
                "os.environ",
                {"CURSOR_API_KEY": "test", "GITHUB_TOKEN": "test"},
            ),
            patch("ai_reviewer.config.load_config"),
            patch(
                "ai_reviewer.review.review_pr_with_cursor_agent",
                return_value=review,
            ),
            patch("ai_reviewer.github.client.GitHubClient", return_value=mock_gh),
            patch("ai_reviewer.github.formatter.GitHubFormatter"),
        ):
            from ai_reviewer.github import webhook
            from ai_reviewer.github.webhook import _setup_default_review_handler

            _setup_default_review_handler()

            handler = webhook._review_handler
            assert handler is not None
            await handler(repo="test/repo", pr_number=42)

            mock_gh.post_review.assert_not_called()
            mock_gh.resolve_fixed_comments.assert_not_called()
            mock_gh.post_inline_comments.assert_not_called()

    @pytest.mark.asyncio
    async def test_webhook_posts_when_not_converged(self):
        """Webhook posts normally when delta has new findings."""
        from datetime import datetime

        from ai_reviewer.models.review import ConsolidatedReview

        new_f = _finding(title="New bug")

        review = ConsolidatedReview(
            id="r1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[new_f],
            summary="One issue",
            agent_count=1,
            review_quality_score=0.9,
            total_review_time_ms=1000,
        )

        non_converged_delta = ReviewDelta(
            new_findings=[new_f],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        mock_gh = MagicMock()
        mock_gh.compute_review_delta.return_value = non_converged_delta

        with (
            patch.dict(
                "os.environ",
                {"CURSOR_API_KEY": "test", "GITHUB_TOKEN": "test"},
            ),
            patch("ai_reviewer.config.load_config"),
            patch(
                "ai_reviewer.review.review_pr_with_cursor_agent",
                return_value=review,
            ),
            patch("ai_reviewer.github.client.GitHubClient", return_value=mock_gh),
            patch("ai_reviewer.github.formatter.GitHubFormatter"),
        ):
            from ai_reviewer.github import webhook
            from ai_reviewer.github.webhook import _setup_default_review_handler

            _setup_default_review_handler()

            handler = webhook._review_handler
            assert handler is not None
            await handler(repo="test/repo", pr_number=42)

            mock_gh.post_review.assert_called_once()
