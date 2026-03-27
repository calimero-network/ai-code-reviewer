"""Tests for convergence detection (Task 7) and severity stabilization (Task 8).

Convergence means "no delta churn" — the issue set hasn't changed since the
last review.  This is distinct from ``all_issues_resolved`` (PR is clean).
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_reviewer.github.client import (
    PreviousComment,
    ReviewDelta,
    ReviewMeta,
    SkipReason,
    compute_findings_hash,
    has_converged,
    lgtm_placeholder_review,
    should_skip_before_agents,
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

        mock_pr = MagicMock()
        mock_pr.head.sha = "abc123"

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
            mock_gh.get_pull_request.return_value = mock_pr
            mock_gh.get_review_metadata.return_value = None
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

        mock_pr = MagicMock()
        mock_pr.head.sha = "abc123"

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
            mock_gh.get_pull_request.return_value = mock_pr
            mock_gh.get_review_metadata.return_value = None
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

        mock_pr = MagicMock()
        mock_pr.head.sha = "abc123"
        mock_pr.get_labels.return_value = []

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.compute_review_delta.return_value = converged_delta
        mock_gh.get_review_metadata.return_value = None

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

        mock_pr = MagicMock()
        mock_pr.head.sha = "abc123"
        mock_pr.get_labels.return_value = []

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.compute_review_delta.return_value = non_converged_delta
        mock_gh.get_review_metadata.return_value = None

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


class TestReviewMetaParsing:
    """Tests for ReviewMeta parsing from comment bodies."""

    def test_parse_valid_json(self):
        body = (
            "Some review text\n"
            '<!-- ai-reviewer-meta: {"commit_sha":"abc123","review_count":3,'
            '"timestamp":"2026-03-27T12:00:00Z","findings_hash":"deadbeef"} -->'
        )
        meta = ReviewMeta.parse(body)
        assert meta is not None
        assert meta.commit_sha == "abc123"
        assert meta.review_count == 3
        assert meta.timestamp == "2026-03-27T12:00:00Z"
        assert meta.findings_hash == "deadbeef"

    def test_parse_missing_tag(self):
        body = "Just a normal review comment with no metadata."
        assert ReviewMeta.parse(body) is None

    def test_parse_malformed_json(self):
        body = "<!-- ai-reviewer-meta: {not valid json} -->"
        assert ReviewMeta.parse(body) is None

    def test_parse_missing_fields(self):
        body = '<!-- ai-reviewer-meta: {"commit_sha":"abc"} -->'
        assert ReviewMeta.parse(body) is None

    def test_parse_extra_fields_ignored(self):
        body = (
            '<!-- ai-reviewer-meta: {"commit_sha":"abc","review_count":1,'
            '"timestamp":"2026-01-01T00:00:00Z","findings_hash":"ff","extra":"ignored"} -->'
        )
        meta = ReviewMeta.parse(body)
        assert meta is not None
        assert meta.commit_sha == "abc"

    def test_to_html_comment_roundtrip(self):
        original = ReviewMeta(
            commit_sha="abc123def456",
            review_count=5,
            timestamp="2026-03-27T12:00:00+00:00",
            findings_hash="deadbeef12345678",
        )
        html = original.to_html_comment()
        parsed = ReviewMeta.parse(html)
        assert parsed is not None
        assert parsed.commit_sha == original.commit_sha
        assert parsed.review_count == original.review_count
        assert parsed.timestamp == original.timestamp
        assert parsed.findings_hash == original.findings_hash

    def test_build_creates_valid_meta(self):
        meta = ReviewMeta.build(
            commit_sha="abc123",
            review_count=2,
            finding_hashes=["hash1", "hash2"],
        )
        assert meta.commit_sha == "abc123"
        assert meta.review_count == 2
        assert meta.findings_hash == compute_findings_hash(["hash1", "hash2"])
        datetime.fromisoformat(meta.timestamp)


class TestShouldSkipBeforeAgents:
    """Tests for the pre-agent convergence check."""

    def test_returns_none_when_force_review(self):
        meta = ReviewMeta(
            commit_sha="abc123",
            review_count=3,
            timestamp=datetime.now(UTC).isoformat(),
            findings_hash="ff",
        )
        assert should_skip_before_agents(meta, "abc123", force_review=True) is None

    def test_returns_none_when_no_metadata(self):
        assert should_skip_before_agents(None, "abc123") is None

    def test_already_reviewed_same_sha(self):
        meta = ReviewMeta(
            commit_sha="abc123",
            review_count=2,
            timestamp="2026-01-01T00:00:00Z",
            findings_hash="ff",
        )
        result = should_skip_before_agents(meta, "abc123")
        assert result == SkipReason.ALREADY_REVIEWED

    def test_different_sha_proceeds(self):
        meta = ReviewMeta(
            commit_sha="abc123",
            review_count=2,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )
        assert should_skip_before_agents(meta, "def456") is None

    def test_invalid_timestamp_does_not_skip(self):
        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=2,
            timestamp="not-a-timestamp",
            findings_hash="ff",
        )
        assert should_skip_before_agents(meta, "new_sha") is None


class TestLgtmFastPathVacuous:
    """LGTM fast path must not fire when there are no previous comments."""

    def test_no_previous_comments_returns_none(self):
        """When no previous inline comments exist, LGTM should not trigger."""
        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=5,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )
        gh = MagicMock()
        gh.compute_review_delta.return_value = ReviewDelta(previous_comments=[])

        from ai_reviewer.github.client import GitHubClient

        result = GitHubClient.check_lgtm_fast_path(gh, MagicMock(), meta)
        assert result is None

    def test_with_previous_comments_all_resolved(self):
        """When previous comments exist and all resolved, LGTM should trigger."""
        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )
        prev = PreviousComment(
            id=1, file_path="f.py", line=1, title="Bug", severity="warning", body="x"
        )
        delta = ReviewDelta(
            previous_comments=[prev],
            fixed_findings=[prev],
            open_findings=[],
            new_findings=[],
        )
        gh = MagicMock()
        gh.compute_review_delta.return_value = delta

        from ai_reviewer.github.client import GitHubClient

        result = GitHubClient.check_lgtm_fast_path(gh, MagicMock(), meta)
        assert result is not None
        assert result.all_issues_resolved


class TestLgtmPlaceholderReview:
    """Tests for the synthetic LGTM review helper."""

    def test_created_at_is_timezone_aware_utc(self):
        review = lgtm_placeholder_review("test/repo", 42)

        assert review.created_at.tzinfo is UTC
        assert review.created_at.utcoffset() == timedelta(0)


class TestFindingsHash:
    """Tests for compute_findings_hash and findings_hash comparison."""

    def test_deterministic(self):
        h1 = compute_findings_hash(["a", "b", "c"])
        h2 = compute_findings_hash(["a", "b", "c"])
        assert h1 == h2

    def test_order_independent(self):
        h1 = compute_findings_hash(["c", "a", "b"])
        h2 = compute_findings_hash(["a", "b", "c"])
        assert h1 == h2

    def test_different_inputs_differ(self):
        h1 = compute_findings_hash(["a", "b"])
        h2 = compute_findings_hash(["a", "c"])
        assert h1 != h2

    def test_empty_list(self):
        h = compute_findings_hash([])
        assert isinstance(h, str)
        assert len(h) == 16


class TestMetadataRoundTrip:
    """End-to-end: embed metadata in formatter, parse back in client."""

    def test_formatter_embeds_and_client_parses(self):
        from ai_reviewer.github.formatter import GitHubFormatter
        from ai_reviewer.models.review import ConsolidatedReview

        review = ConsolidatedReview(
            id="r1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="Clean",
            agent_count=1,
            review_quality_score=0.95,
            total_review_time_ms=1000,
        )
        meta = ReviewMeta.build(
            commit_sha="abc123def",
            review_count=3,
            finding_hashes=["h1", "h2"],
        )

        formatter = GitHubFormatter()
        body = formatter.format_review(review, meta=meta)

        parsed = ReviewMeta.parse(body)
        assert parsed is not None
        assert parsed.commit_sha == "abc123def"
        assert parsed.review_count == 3
        assert parsed.findings_hash == meta.findings_hash

    def test_compact_format_embeds_metadata(self):
        from ai_reviewer.github.formatter import GitHubFormatter
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
        meta = ReviewMeta.build(
            commit_sha="sha456",
            review_count=2,
            finding_hashes=["h1"],
        )

        formatter = GitHubFormatter()
        body = formatter.format_review_compact(review, meta=meta)

        parsed = ReviewMeta.parse(body)
        assert parsed is not None
        assert parsed.commit_sha == "sha456"

    def test_delta_compact_format_embeds_metadata(self):
        from ai_reviewer.github.formatter import GitHubFormatter
        from ai_reviewer.models.review import ConsolidatedReview

        review = ConsolidatedReview(
            id="r1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="Clean",
            agent_count=1,
            review_quality_score=0.95,
            total_review_time_ms=1000,
        )
        delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )
        meta = ReviewMeta.build(
            commit_sha="sha789",
            review_count=4,
            finding_hashes=[],
        )

        formatter = GitHubFormatter()
        body = formatter.format_review_with_delta_compact(review, delta, meta=meta)

        parsed = ReviewMeta.parse(body)
        assert parsed is not None
        assert parsed.commit_sha == "sha789"
        assert parsed.review_count == 4


class TestReviewCountFromMetadata:
    """Tests that accurate review count from metadata is used over heuristic."""

    def test_metadata_count_used_when_available(self):
        """When metadata exists, review_count = meta.review_count + 1."""
        from ai_reviewer.github.client import estimate_review_count

        meta = ReviewMeta(
            commit_sha="abc",
            review_count=5,
            timestamp="2026-01-01T00:00:00Z",
            findings_hash="ff",
        )
        delta = ReviewDelta(
            previous_comments=[_prev_comment(id=i) for i in range(20)],
        )
        heuristic_count = estimate_review_count(delta)
        meta_count = meta.review_count + 1

        assert meta_count == 6
        assert heuristic_count != meta_count

    def test_heuristic_fallback_for_legacy_comments(self):
        """When no metadata, estimate_review_count is used."""
        from ai_reviewer.github.client import estimate_review_count

        delta = ReviewDelta(previous_comments=[])
        assert estimate_review_count(delta) == 1

        delta_with_comments = ReviewDelta(
            previous_comments=[_prev_comment(id=i) for i in range(3)],
        )
        assert estimate_review_count(delta_with_comments) >= 2


class TestComputeReviewDeltaWithReviewCount:
    """Tests that compute_review_delta uses explicit review_count for stabilization."""

    def test_explicit_review_count_used_for_stabilization(self):
        """Passing review_count to compute_review_delta affects severity stabilization."""
        from ai_reviewer.github.client import GitHubClient, PreviousComment
        from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity

        prev_comment = PreviousComment(
            id=1,
            file_path="src/auth.py",
            line=10,
            title="SQL Injection Vulnerability",
            severity="warning",
            body="🟡 **SQL Injection Vulnerability**\n\n<!-- ai-reviewer-id: aabbccddee11 -->",
            finding_hash="aabbccddee11",
        )

        current_finding = ConsolidatedFinding(
            id="test-1",
            file_path="src/auth.py",
            line_start=10,
            line_end=None,
            severity=Severity.SUGGESTION,
            category=Category.SECURITY,
            title="SQL Injection Vulnerability",
            description="desc",
            suggested_fix=None,
            consensus_score=1.0,
            agreeing_agents=["a"],
            confidence=0.9,
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

            delta = client.compute_review_delta(mock_pr, [current_finding], review_count=5)

        assert len(delta.open_findings) == 1
        assert delta.open_findings[0].severity == Severity.WARNING

    def test_none_review_count_falls_back_to_heuristic(self):
        """When review_count is None, estimate_review_count heuristic is used.

        With 1 previous comment, estimate_review_count returns max(2, 1//3+1) = 2,
        so severity downgrade is blocked (same as explicit review_count >= 2).
        """
        from ai_reviewer.github.client import GitHubClient, PreviousComment
        from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity

        prev_comment = PreviousComment(
            id=1,
            file_path="src/auth.py",
            line=10,
            title="SQL Injection Vulnerability",
            severity="warning",
            body="🟡 **SQL Injection Vulnerability**\n\n<!-- ai-reviewer-id: aabbccddee11 -->",
            finding_hash="aabbccddee11",
        )

        current_finding = ConsolidatedFinding(
            id="test-1",
            file_path="src/auth.py",
            line_start=10,
            line_end=None,
            severity=Severity.SUGGESTION,
            category=Category.SECURITY,
            title="SQL Injection Vulnerability",
            description="desc",
            suggested_fix=None,
            consensus_score=1.0,
            agreeing_agents=["a"],
            confidence=0.9,
        )

        mock_pr = MagicMock()
        mock_file = MagicMock()
        mock_file.filename = "src/auth.py"
        mock_file.patch = "@@ -1,3 +1,3 @@\n-old\n+new"
        mock_file.status = "modified"
        mock_pr.get_files.return_value = [mock_file]

        with patch("ai_reviewer.github.client.Github"):
            client = GitHubClient(token="test-token")
            # 1 previous comment → estimate_review_count returns 2 → downgrade blocked
            client.get_previous_review_comments = MagicMock(return_value=[prev_comment])

            delta = client.compute_review_delta(mock_pr, [current_finding], review_count=None)

        assert len(delta.open_findings) == 1
        # Heuristic returns 2 for 1 previous comment, so downgrade is blocked
        assert delta.open_findings[0].severity == Severity.WARNING


class TestLgtmLightweightRecheck:
    """Tests for the lightweight 1-agent re-check before posting LGTM.

    The LGTM candidate gate (check_lgtm_fast_path) only detects *candidates*.
    A single-agent re-check must confirm zero findings before LGTM is posted.
    """

    def test_cli_lgtm_candidate_triggers_recheck(self):
        """CLI runs a 1-agent re-check when LGTM candidate is detected."""
        import asyncio

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )

        lgtm_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        recheck_review = ConsolidatedReview(
            id="recheck",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="Clean",
            agent_count=1,
            review_quality_score=1.0,
            total_review_time_ms=500,
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta
        mock_gh.check_lgtm_fast_path.return_value = lgtm_delta

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient", return_value=mock_gh),
            patch(
                "ai_reviewer.cli.review_pr_with_cursor_agent",
                return_value=recheck_review,
            ) as mock_agent,
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    min_validation_agreement=0.75,
                    force_review=False,
                )
            )

            mock_agent.assert_called_once()
            call_kwargs = mock_agent.call_args.kwargs
            assert call_kwargs["num_agents"] == 1
            assert call_kwargs["enable_cross_review"] is False
            assert call_kwargs["min_validation_agreement"] == pytest.approx(0.75)

    def test_cli_lgtm_recheck_zero_findings_posts_lgtm(self):
        """When re-check returns zero findings, LGTM review is posted."""
        import asyncio

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )

        lgtm_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        recheck_review = ConsolidatedReview(
            id="recheck",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="Clean",
            agent_count=1,
            review_quality_score=1.0,
            total_review_time_ms=500,
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta
        mock_gh.check_lgtm_fast_path.return_value = lgtm_delta

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient", return_value=mock_gh),
            patch(
                "ai_reviewer.cli.review_pr_with_cursor_agent",
                return_value=recheck_review,
            ),
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    force_review=False,
                )
            )

            mock_gh.post_review.assert_called_once()
            mock_gh.resolve_fixed_comments.assert_called_once()

    def test_cli_lgtm_recheck_with_findings_falls_back_to_normal(self):
        """When re-check finds issues, fall back to normal review posting."""
        import asyncio

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )

        lgtm_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        recheck_finding = _finding(title="Missed bug")
        recheck_review = ConsolidatedReview(
            id="recheck",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[recheck_finding],
            summary="One issue",
            agent_count=1,
            review_quality_score=0.8,
            total_review_time_ms=500,
        )

        normal_delta = ReviewDelta(
            new_findings=[recheck_finding],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta
        mock_gh.check_lgtm_fast_path.return_value = lgtm_delta
        mock_gh.compute_review_delta.return_value = normal_delta

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient", return_value=mock_gh),
            patch(
                "ai_reviewer.cli.review_pr_with_cursor_agent",
                return_value=recheck_review,
            ) as mock_agent,
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    force_review=False,
                )
            )

            mock_agent.assert_called_once()
            mock_gh.post_review.assert_called_once()

    def test_cli_lgtm_recheck_error_falls_back_to_normal(self):
        """When re-check fails, CLI falls back to the normal review flow."""
        import asyncio

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )

        lgtm_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        normal_finding = _finding(title="Missed bug")
        normal_review = ConsolidatedReview(
            id="normal",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[normal_finding],
            summary="One issue",
            agent_count=1,
            review_quality_score=0.8,
            total_review_time_ms=500,
        )

        normal_delta = ReviewDelta(
            new_findings=[normal_finding],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta
        mock_gh.check_lgtm_fast_path.return_value = lgtm_delta
        mock_gh.compute_review_delta.return_value = normal_delta

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient", return_value=mock_gh),
            patch(
                "ai_reviewer.cli.review_pr_with_cursor_agent",
                new_callable=AsyncMock,
                side_effect=[RuntimeError("recheck failed"), normal_review],
            ) as mock_agent,
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    force_review=False,
                )
            )

            assert mock_agent.await_count == 2
            mock_gh.post_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_lgtm_candidate_triggers_recheck(self):
        """Webhook runs a 1-agent re-check when LGTM candidate is detected."""
        from ai_reviewer.models.review import ConsolidatedReview

        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )

        lgtm_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        recheck_review = ConsolidatedReview(
            id="recheck",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="Clean",
            agent_count=1,
            review_quality_score=1.0,
            total_review_time_ms=500,
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"
        mock_pr.get_labels.return_value = []

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta
        mock_gh.check_lgtm_fast_path.return_value = lgtm_delta

        with (
            patch.dict(
                "os.environ",
                {
                    "CURSOR_API_KEY": "test",
                    "GITHUB_TOKEN": "test",
                    "MIN_VALIDATION_AGREEMENT": "0.8",
                },
            ),
            patch("ai_reviewer.config.load_config"),
            patch(
                "ai_reviewer.review.review_pr_with_cursor_agent",
                return_value=recheck_review,
            ) as mock_agent,
            patch("ai_reviewer.github.client.GitHubClient", return_value=mock_gh),
            patch("ai_reviewer.github.formatter.GitHubFormatter"),
        ):
            from ai_reviewer.github import webhook
            from ai_reviewer.github.webhook import _setup_default_review_handler

            _setup_default_review_handler()

            handler = webhook._review_handler
            assert handler is not None
            await handler(repo="test/repo", pr_number=42)

            mock_agent.assert_called_once()
            call_kwargs = mock_agent.call_args.kwargs
            assert call_kwargs["num_agents"] == 1
            assert call_kwargs["enable_cross_review"] is False
            assert call_kwargs["min_validation_agreement"] == pytest.approx(0.8)
            mock_gh.post_review.assert_called_once()
            mock_gh.resolve_fixed_comments.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_lgtm_recheck_with_findings_falls_back(self):
        """Webhook falls back to normal review when re-check finds issues."""
        from ai_reviewer.models.review import ConsolidatedReview

        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )

        lgtm_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        recheck_finding = _finding(title="Missed bug")
        recheck_review = ConsolidatedReview(
            id="recheck",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[recheck_finding],
            summary="One issue",
            agent_count=1,
            review_quality_score=0.8,
            total_review_time_ms=500,
        )

        normal_delta = ReviewDelta(
            new_findings=[recheck_finding],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"
        mock_pr.get_labels.return_value = []

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta
        mock_gh.check_lgtm_fast_path.return_value = lgtm_delta
        mock_gh.compute_review_delta.return_value = normal_delta

        with (
            patch.dict(
                "os.environ",
                {"CURSOR_API_KEY": "test", "GITHUB_TOKEN": "test"},
            ),
            patch("ai_reviewer.config.load_config"),
            patch(
                "ai_reviewer.review.review_pr_with_cursor_agent",
                return_value=recheck_review,
            ) as mock_agent,
            patch("ai_reviewer.github.client.GitHubClient", return_value=mock_gh),
            patch("ai_reviewer.github.formatter.GitHubFormatter"),
        ):
            from ai_reviewer.github import webhook
            from ai_reviewer.github.webhook import _setup_default_review_handler

            _setup_default_review_handler()

            handler = webhook._review_handler
            assert handler is not None
            await handler(repo="test/repo", pr_number=42)

            mock_agent.assert_called_once()
            mock_gh.post_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_lgtm_recheck_error_falls_back(self):
        """Webhook falls back to normal review when re-check errors."""
        from ai_reviewer.models.review import ConsolidatedReview

        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )

        lgtm_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        normal_finding = _finding(title="Missed bug")
        normal_review = ConsolidatedReview(
            id="normal",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[normal_finding],
            summary="One issue",
            agent_count=1,
            review_quality_score=0.8,
            total_review_time_ms=500,
        )

        normal_delta = ReviewDelta(
            new_findings=[normal_finding],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"
        mock_pr.get_labels.return_value = []

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta
        mock_gh.check_lgtm_fast_path.return_value = lgtm_delta
        mock_gh.compute_review_delta.return_value = normal_delta

        with (
            patch.dict(
                "os.environ",
                {"CURSOR_API_KEY": "test", "GITHUB_TOKEN": "test"},
            ),
            patch("ai_reviewer.config.load_config"),
            patch(
                "ai_reviewer.review.review_pr_with_cursor_agent",
                new_callable=AsyncMock,
                side_effect=[RuntimeError("recheck failed"), normal_review],
            ) as mock_agent,
            patch("ai_reviewer.github.client.GitHubClient", return_value=mock_gh),
            patch("ai_reviewer.github.formatter.GitHubFormatter"),
        ):
            from ai_reviewer.github import webhook
            from ai_reviewer.github.webhook import _setup_default_review_handler

            _setup_default_review_handler()

            handler = webhook._review_handler
            assert handler is not None
            await handler(repo="test/repo", pr_number=42)

            assert mock_agent.await_count == 2
            mock_gh.post_review.assert_called_once()

    def test_already_reviewed_skip_unchanged(self):
        """Existing already_reviewed skip behavior remains unchanged."""
        meta = ReviewMeta(
            commit_sha="abc123",
            review_count=2,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )
        result = should_skip_before_agents(meta, "abc123")
        assert result == SkipReason.ALREADY_REVIEWED

    def test_candidate_gate_does_not_directly_post(self):
        """check_lgtm_fast_path returning a delta does not mean LGTM is posted.

        The caller must run a re-check first.  This test verifies the gate
        only returns a candidate, not a final decision.
        """
        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )
        prev = PreviousComment(
            id=1, file_path="f.py", line=1, title="Bug", severity="warning", body="x"
        )
        delta = ReviewDelta(
            previous_comments=[prev],
            fixed_findings=[prev],
            open_findings=[],
            new_findings=[],
        )
        gh = MagicMock()
        gh.compute_review_delta.return_value = delta

        from ai_reviewer.github.client import GitHubClient

        result = GitHubClient.check_lgtm_fast_path(gh, MagicMock(), meta)
        assert result is not None
        assert result.all_issues_resolved


class TestCLIPreAgentChecks:
    """Integration tests for pre-agent checks in the CLI."""

    def test_cli_compact_delta_body_uses_postable_new_finding_count(self):
        """Compact delta body counts only inline findings that will actually be posted."""
        import asyncio

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        findings = [_finding(title=f"Issue {i}", line_start=10 + i) for i in range(4)]
        postable_findings = findings[:2]
        review = ConsolidatedReview(
            id="r1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=findings,
            summary="Four issues",
            agent_count=1,
            review_quality_score=0.9,
            total_review_time_ms=1000,
        )
        delta = ReviewDelta(
            new_findings=findings,
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = None
        mock_gh.compute_review_delta.return_value = delta
        mock_gh.get_postable_inline_findings.return_value = postable_findings

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient", return_value=mock_gh),
            patch("ai_reviewer.cli.review_pr_with_cursor_agent", return_value=review),
        ):
            mock_config = MagicMock()
            mock_config.cursor.api_key = "cursor-token"
            mock_config.cursor.base_url = "https://cursor.example"
            mock_config.cursor.timeout_seconds = 30
            mock_config.github.token = "github-token"
            mock_config.output.max_total_findings = 50
            mock_config.output.max_findings_per_file = 10
            mock_load.return_value = mock_config

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    force_review=False,
                )
            )

            post_args = mock_gh.post_review.call_args
            assert post_args is not None
            assert "🆕 2 new" in post_args.args[1]
            assert "🆕 4 new" not in post_args.args[1]
            assert post_args.kwargs["inline_findings"] == postable_findings

    def test_cli_skips_on_already_reviewed(self):
        """CLI returns early when commit SHA was already reviewed."""
        import asyncio

        from ai_reviewer.cli import review_pr_async

        meta = ReviewMeta(
            commit_sha="abc123",
            review_count=2,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "abc123"

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient", return_value=mock_gh),
            patch("ai_reviewer.cli.review_pr_with_cursor_agent") as mock_agent,
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    force_review=False,
                )
            )

            mock_agent.assert_not_called()

    def test_cli_lgtm_fast_path_runs_recheck_then_posts(self):
        """CLI uses LGTM fast path with 1-agent re-check when all issues resolved."""
        import asyncio

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )

        lgtm_delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[_prev_comment()],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        recheck_review = ConsolidatedReview(
            id="recheck",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="Clean",
            agent_count=1,
            review_quality_score=1.0,
            total_review_time_ms=500,
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta
        mock_gh.check_lgtm_fast_path.return_value = lgtm_delta

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient", return_value=mock_gh),
            patch(
                "ai_reviewer.cli.review_pr_with_cursor_agent",
                return_value=recheck_review,
            ) as mock_agent,
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    force_review=False,
                )
            )

            mock_agent.assert_called_once()
            call_kwargs = mock_agent.call_args.kwargs
            assert call_kwargs["num_agents"] == 1
            assert call_kwargs["enable_cross_review"] is False
            mock_gh.post_review.assert_called_once()

    def test_cli_lgtm_fast_path_not_triggered_when_issues_remain(self):
        """LGTM fast path does not trigger when check_lgtm_fast_path returns None."""
        import asyncio

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        meta = ReviewMeta(
            commit_sha="old_sha",
            review_count=3,
            timestamp="2020-01-01T00:00:00Z",
            findings_hash="ff",
        )

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

        delta = ReviewDelta(
            new_findings=[_finding()],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta
        mock_gh.check_lgtm_fast_path.return_value = None
        mock_gh.compute_review_delta.return_value = delta

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient", return_value=mock_gh),
            patch("ai_reviewer.cli.review_pr_with_cursor_agent", return_value=review),
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    force_review=False,
                )
            )

            mock_gh.post_review.assert_called_once()

    def test_cli_force_review_bypasses_pre_agent_checks(self):
        """--force-review bypasses both pre-agent skip and LGTM fast path."""
        import asyncio

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        meta = ReviewMeta(
            commit_sha="abc123",
            review_count=5,
            timestamp=datetime.now(UTC).isoformat(),
            findings_hash="ff",
        )

        review = ConsolidatedReview(
            id="r1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="Clean",
            agent_count=1,
            review_quality_score=0.95,
            total_review_time_ms=1000,
        )

        delta = ReviewDelta(
            new_findings=[],
            fixed_findings=[],
            open_findings=[],
            previous_comments=[],
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "abc123"

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = meta
        mock_gh.compute_review_delta.return_value = delta

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient", return_value=mock_gh),
            patch("ai_reviewer.cli.review_pr_with_cursor_agent", return_value=review),
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="github",
                    force_review=True,
                )
            )

            mock_gh.post_review.assert_called_once()

    def test_cli_json_output_skips_pre_agent_checks(self):
        """JSON output mode does not perform pre-agent checks."""
        import asyncio

        from ai_reviewer.cli import review_pr_async
        from ai_reviewer.models.review import ConsolidatedReview

        review = ConsolidatedReview(
            id="r1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=[],
            summary="Clean",
            agent_count=1,
            review_quality_score=0.95,
            total_review_time_ms=1000,
        )

        with (
            patch("ai_reviewer.cli.load_config") as mock_load,
            patch("ai_reviewer.cli.validate_config", return_value=[]),
            patch("ai_reviewer.cli.GitHubClient") as mock_gh_cls,
            patch("ai_reviewer.cli.review_pr_with_cursor_agent", return_value=review),
        ):
            mock_config = MagicMock()
            mock_load.return_value = mock_config

            asyncio.run(
                review_pr_async(
                    repo="test/repo",
                    pr_number=42,
                    output="json",
                )
            )

            mock_gh_cls.assert_not_called()


class TestWebhookCompactSummaryCounts:
    """Tests that webhook compact summaries match postable inline comments."""

    @pytest.mark.asyncio
    async def test_webhook_compact_delta_body_uses_postable_new_finding_count(self):
        """Webhook compact delta body counts only inline findings that will be posted."""
        from ai_reviewer.models.review import ConsolidatedReview

        findings = [_finding(title=f"Issue {i}", line_start=10 + i) for i in range(4)]
        postable_findings = findings[:2]
        review = ConsolidatedReview(
            id="r1",
            created_at=datetime.now(),
            repo="test/repo",
            pr_number=42,
            findings=findings,
            summary="Four issues",
            agent_count=1,
            review_quality_score=0.9,
            total_review_time_ms=1000,
        )
        delta = ReviewDelta(
            new_findings=findings,
            fixed_findings=[],
            open_findings=[],
            previous_comments=[_prev_comment()],
        )

        mock_pr = MagicMock()
        mock_pr.head.sha = "new_sha"
        mock_pr.get_labels.return_value = []

        mock_gh = MagicMock()
        mock_gh.get_pull_request.return_value = mock_pr
        mock_gh.get_review_metadata.return_value = None
        mock_gh.compute_review_delta.return_value = delta
        mock_gh.get_postable_inline_findings.return_value = postable_findings

        with (
            patch.dict(
                "os.environ",
                {
                    "CURSOR_API_KEY": "test",
                    "GITHUB_TOKEN": "test",
                },
            ),
            patch("ai_reviewer.config.load_config"),
            patch(
                "ai_reviewer.review.review_pr_with_cursor_agent",
                return_value=review,
            ),
            patch("ai_reviewer.github.client.GitHubClient", return_value=mock_gh),
        ):
            from ai_reviewer.github import webhook
            from ai_reviewer.github.webhook import _setup_default_review_handler

            _setup_default_review_handler()

            handler = webhook._review_handler
            assert handler is not None
            await handler(repo="test/repo", pr_number=42)

            post_args = mock_gh.post_review.call_args
            assert post_args is not None
            assert "🆕 2 new" in post_args.args[1]
            assert "🆕 4 new" not in post_args.args[1]
            assert post_args.kwargs["inline_findings"] == postable_findings
