"""Posting a consolidated review to a pull request.

Lifted out of the CLI so the API-billed ``review-pr`` and the subagent-driven
``publish`` apply the same delta tracking, convergence gate and inline-comment
limits rather than two approximations of them. The webhook handler still carries
its own copy of this pipeline and does not go through here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from github.PullRequest import PullRequest

from ai_reviewer.config import Config
from ai_reviewer.github.client import (
    GitHubClient,
    ReviewMeta,
    estimate_review_count,
    is_convergence_all_clear,
    should_skip_review,
)
from ai_reviewer.github.formatter import GitHubFormatter
from ai_reviewer.models.review import ConsolidatedReview


@dataclass
class PublishResult:
    """What happened on GitHub, for the caller to report."""

    posted: bool
    action: str
    inline_comments: int
    resolved: int
    skipped: bool
    body: str


def publish_review(
    *,
    gh: GitHubClient,
    pr: PullRequest,
    review: ConsolidatedReview,
    config: Config,
    meta: ReviewMeta | None,
    reviewer_name: str,
    force_review: bool,
    dry_run: bool,
    allow_approve: bool,
    emit: Callable[[str], None] | None = None,
) -> PublishResult:
    """Post *review* to *pr*, or explain why it was not posted."""
    say = emit or (lambda _message: None)
    formatter = GitHubFormatter(reviewer_name)
    current_sha = pr.head.sha

    say("🔄 Checking for previous review comments...")
    meta_review_count = (meta.review_count + 1) if meta is not None else None
    delta = gh.compute_review_delta(pr, review.findings, review_count=meta_review_count)

    if delta.previous_comments:
        say(
            f"   Found {len(delta.previous_comments)} previous comments: "
            f"[green]{len(delta.fixed_findings)} fixed[/green], "
            f"[yellow]{len(delta.open_findings)} open[/yellow], "
            f"[cyan]{len(delta.new_findings)} new[/cyan]"
        )
    else:
        say("   No previous review comments found (first run)")

    review_count = (
        meta_review_count if meta_review_count is not None else estimate_review_count(delta)
    )

    if delta.previous_comments and not force_review:
        if should_skip_review(review_count, delta):
            say(
                "[dim]⏭️  Findings unchanged since last review - skipping post "
                "(use --force-review to override)[/dim]"
            )
            return PublishResult(
                posted=False, action="", inline_comments=0, resolved=0, skipped=True, body=""
            )
    elif force_review and delta.previous_comments:
        say("[dim]⚡ --force-review: bypassing convergence check[/dim]")

    new_meta = ReviewMeta.build(
        commit_sha=current_sha,
        review_count=review_count,
        finding_hashes=[f.finding_hash for f in review.findings],
    )
    all_clear = is_convergence_all_clear(review, delta)

    if dry_run:
        say("\n[yellow]Dry run - not posting to GitHub[/yellow]")
        if all_clear:
            body = formatter.format_all_clear(review, delta, meta=new_meta)
        elif delta.previous_comments:
            body = formatter.format_review_with_delta(review, delta, meta=new_meta)
        else:
            body = formatter.format_review(review, meta=new_meta)
        return PublishResult(
            posted=False, action="", inline_comments=0, resolved=0, skipped=False, body=body
        )

    max_total = config.output.max_total_findings
    max_per_file = config.output.max_findings_per_file

    if all_clear:
        body = formatter.format_all_clear(review, delta, meta=new_meta)
        auto_approve = config.review_policy.auto_approve_if_no_findings
        action = (
            "APPROVE"
            if (allow_approve and auto_approve and not review.failed_agents)
            else "COMMENT"
        )
        postable_inline_findings = []
    else:
        candidate_inline_findings = (
            delta.new_findings if delta.previous_comments else review.findings
        )
        postable_inline_findings = gh.get_postable_inline_findings(
            pr,
            inline_findings=candidate_inline_findings,
            max_total=max_total,
            max_per_file=max_per_file,
        )
        use_compact_body = len(postable_inline_findings) > 0

        if delta.previous_comments:
            body = (
                formatter.format_review_with_delta_compact(
                    review, delta, meta=new_meta, inline_new_findings=postable_inline_findings
                )
                if use_compact_body
                else formatter.format_review_with_delta(review, delta, meta=new_meta)
            )
            action = formatter.get_review_action_with_delta(review, delta, allow_approve)
        else:
            body = (
                formatter.format_review_compact(
                    review, meta=new_meta, inline_findings=postable_inline_findings
                )
                if use_compact_body
                else formatter.format_review(review, meta=new_meta)
            )
            action = formatter.get_review_action(review, allow_approve=allow_approve)

    posted = gh.post_review(pr, body, action, inline_findings=postable_inline_findings or None)
    say(f"📝 Posted review to GitHub ({action}, {posted} inline comments)")

    resolved = 0
    if delta.fixed_findings:
        say(f"✅ Marking {len(delta.fixed_findings)} fixed issues as resolved...")
        resolved = gh.resolve_fixed_comments(pr, delta)
        say(f"   Resolved {resolved} comments")

    if delta.all_issues_resolved:
        say("\n[green]🎉 All issues resolved! Ready to merge.[/green]")
    elif delta.previous_comments:
        open_count = len(delta.open_findings) + len(delta.new_findings)
        say(f"\n[yellow]⚠️  {open_count} issues remaining[/yellow]")

    return PublishResult(
        posted=True,
        action=action,
        inline_comments=posted,
        resolved=resolved,
        skipped=False,
        body=body,
    )
