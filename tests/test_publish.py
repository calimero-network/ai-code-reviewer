"""Posting a consolidated review to a pull request.

The client is faked rather than mocked piecemeal: these tests are about which
branch runs and what payload comes out, not about PyGithub.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ai_reviewer.config import Config, GitHubConfig, load_config
from ai_reviewer.github.client import PreviousComment, ReviewDelta, ReviewMeta
from ai_reviewer.github.publish import publish_review
from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity
from ai_reviewer.models.review import ConsolidatedReview


def _finding(title: str = "Unchecked index", line: int = 214) -> ConsolidatedFinding:
    return ConsolidatedFinding(
        id="f1",
        file_path="src/admin.rs",
        line_start=line,
        line_end=None,
        severity=Severity.CRITICAL,
        category=Category.LOGIC,
        title=title,
        description="Member index used before the account lookup is checked.",
        suggested_fix=None,
        consensus_score=1.0,
        agreeing_agents=["security-reviewer", "logic-reviewer"],
        confidence=0.9,
    )


def _prev_comment(title: str = "Unchecked index", line: int = 214) -> PreviousComment:
    return PreviousComment(
        id=1,
        file_path="src/admin.rs",
        line=line,
        title=title,
        severity="critical",
        body="body",
    )


def _review(findings=None) -> ConsolidatedReview:
    return ConsolidatedReview(
        id="r1",
        created_at=datetime.now(),
        repo="acme/widget",
        pr_number=42,
        findings=findings if findings is not None else [_finding()],
        summary="1 issue",
        agent_count=2,
        review_quality_score=0.9,
        total_review_time_ms=1000,
    )


@pytest.fixture
def config() -> Config:
    cfg = load_config(None)
    cfg.github = GitHubConfig(token="t")
    return cfg


@pytest.fixture
def gh():
    client = MagicMock()
    client.compute_review_delta.return_value = ReviewDelta(
        new_findings=[_finding()], open_findings=[], fixed_findings=[], previous_comments=[]
    )
    client.get_postable_inline_findings.return_value = [_finding()]
    client.post_review.return_value = 1
    client.resolve_fixed_comments.return_value = 0
    return client


@pytest.fixture
def pr():
    p = MagicMock()
    p.number = 42
    p.head.sha = "h" * 40
    return p


def test_a_first_review_is_posted_with_its_inline_comments(gh, pr, config):
    """With nothing posted before, every finding is inline-eligible - the delta's
    new_findings are not the source, so the review carries one the delta lacks."""
    review = _review([_finding(), _finding("Second issue", 300)])

    result = publish_review(
        gh=gh,
        pr=pr,
        review=review,
        config=config,
        meta=None,
        reviewer_name="AI Code Reviewer",
        force_review=False,
        dry_run=False,
        allow_approve=False,
    )

    assert result.posted is True
    assert result.inline_comments == 1
    gh.post_review.assert_called_once()
    assert gh.post_review.call_args.args[2] == "COMMENT"
    assert gh.get_postable_inline_findings.call_args.kwargs["inline_findings"] == review.findings


def test_a_dry_run_posts_nothing_but_returns_the_body(gh, pr, config):
    result = publish_review(
        gh=gh,
        pr=pr,
        review=_review(),
        config=config,
        meta=None,
        reviewer_name="AI Code Reviewer",
        force_review=False,
        dry_run=True,
        allow_approve=False,
    )

    assert result.posted is False
    assert "Unchecked index" in result.body
    gh.post_review.assert_not_called()


def test_unchanged_findings_are_not_posted_again(gh, pr, config):
    """The gate that keeps 'post immediately' from becoming comment spam."""
    previous = _prev_comment()
    gh.compute_review_delta.return_value = ReviewDelta(
        new_findings=[], open_findings=[_finding()], fixed_findings=[], previous_comments=[previous]
    )
    meta = ReviewMeta.build(commit_sha="h" * 40, review_count=2, finding_hashes=[])
    said: list[str] = []

    result = publish_review(
        gh=gh,
        pr=pr,
        review=_review(),
        config=config,
        meta=meta,
        reviewer_name="AI Code Reviewer",
        force_review=False,
        dry_run=False,
        allow_approve=False,
        emit=said.append,
    )

    assert result.skipped is True
    assert result.posted is False
    assert any("Findings unchanged since last review - skipping post" in line for line in said)
    gh.post_review.assert_not_called()


def test_force_review_overrides_the_convergence_gate(gh, pr, config):
    previous = _prev_comment()
    gh.compute_review_delta.return_value = ReviewDelta(
        new_findings=[], open_findings=[_finding()], fixed_findings=[], previous_comments=[previous]
    )
    meta = ReviewMeta.build(commit_sha="h" * 40, review_count=2, finding_hashes=[])

    result = publish_review(
        gh=gh,
        pr=pr,
        review=_review(),
        config=config,
        meta=meta,
        reviewer_name="AI Code Reviewer",
        force_review=True,
        dry_run=False,
        allow_approve=False,
    )

    assert result.skipped is False
    gh.post_review.assert_called_once()
    # Once something has been posted before, only the new findings may go inline.
    assert gh.get_postable_inline_findings.call_args.kwargs["inline_findings"] == []


def test_approve_is_never_used_when_it_is_not_allowed(gh, pr, config):
    """This path posts under a person's identity; it must not approve for them."""
    gh.compute_review_delta.return_value = ReviewDelta(
        new_findings=[], open_findings=[], fixed_findings=[], previous_comments=[]
    )
    gh.get_postable_inline_findings.return_value = []
    config.review_policy.auto_approve_if_no_findings = True

    result = publish_review(
        gh=gh,
        pr=pr,
        review=_review(findings=[]),
        config=config,
        meta=None,
        reviewer_name="AI Code Reviewer",
        force_review=False,
        dry_run=False,
        allow_approve=False,
    )

    assert result.action == "COMMENT"


def test_fixed_findings_get_their_comments_resolved(gh, pr, config):
    """fixed_findings holds PreviousComment, not ConsolidatedFinding."""
    previous = _prev_comment(title="Old")
    delta = ReviewDelta(
        new_findings=[_finding("New issue", 300)],
        open_findings=[],
        fixed_findings=[previous],
        previous_comments=[previous],
    )
    gh.compute_review_delta.return_value = delta
    gh.resolve_fixed_comments.return_value = 1

    result = publish_review(
        gh=gh,
        pr=pr,
        review=_review([_finding("New issue", 300)]),
        config=config,
        meta=None,
        reviewer_name="AI Code Reviewer",
        force_review=False,
        dry_run=False,
        allow_approve=False,
    )

    assert result.resolved == 1
    gh.resolve_fixed_comments.assert_called_once_with(pr, delta)
