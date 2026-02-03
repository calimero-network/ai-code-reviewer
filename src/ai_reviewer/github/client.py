"""GitHub API client for PR operations."""

import logging
import re
from dataclasses import dataclass, field

from github import Github
from github.GithubException import GithubException
from github.PullRequest import PullRequest
from github.PullRequestComment import PullRequestComment
from github.Repository import Repository

from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import ConsolidatedFinding
from ai_reviewer.models.review import ConsolidatedReview

logger = logging.getLogger(__name__)


@dataclass
class GitHubConfig:
    """Configuration for GitHub client."""

    token: str
    base_url: str | None = None  # For GitHub Enterprise


@dataclass
class PreviousComment:
    """Represents a previous review comment from the AI reviewer."""

    id: int
    file_path: str
    line: int
    title: str
    severity: str  # Extracted from emoji
    body: str
    is_resolved: bool = False


@dataclass
class ReviewDelta:
    """Tracks changes between review runs."""

    new_findings: list[ConsolidatedFinding] = field(default_factory=list)
    fixed_findings: list[PreviousComment] = field(default_factory=list)
    open_findings: list[ConsolidatedFinding] = field(default_factory=list)
    previous_comments: list[PreviousComment] = field(default_factory=list)

    @property
    def all_issues_resolved(self) -> bool:
        """Check if all previously found issues are now resolved."""
        return len(self.open_findings) == 0 and len(self.new_findings) == 0

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes from the previous review."""
        return len(self.new_findings) > 0 or len(self.fixed_findings) > 0


class GitHubClient:
    """Client for GitHub API operations."""

    def __init__(self, token: str, base_url: str | None = None) -> None:
        """Initialize the GitHub client.

        Args:
            token: GitHub personal access token or app token
            base_url: Optional base URL for GitHub Enterprise
        """
        if base_url:
            self._gh = Github(token, base_url=base_url)
        else:
            self._gh = Github(token)

    def get_repo(self, repo_name: str) -> Repository:
        """Get a repository by name.

        Args:
            repo_name: Repository in "owner/name" format

        Returns:
            Repository object
        """
        return self._gh.get_repo(repo_name)

    def get_pull_request(self, repo_name: str, pr_number: int) -> PullRequest:
        """Get a pull request.

        Args:
            repo_name: Repository in "owner/name" format
            pr_number: Pull request number

        Returns:
            PullRequest object
        """
        repo = self.get_repo(repo_name)
        return repo.get_pull(pr_number)

    def get_pr_diff(self, pr: PullRequest) -> str:
        """Get the unified diff for a PR.

        Args:
            pr: Pull request object

        Returns:
            Unified diff string
        """
        files = pr.get_files()
        diff_parts = []

        for file in files:
            if file.patch:
                diff_parts.append(f"diff --git a/{file.filename} b/{file.filename}")
                diff_parts.append(f"--- a/{file.filename}")
                diff_parts.append(f"+++ b/{file.filename}")
                diff_parts.append(file.patch)
                diff_parts.append("")

        return "\n".join(diff_parts)

    def get_changed_files(self, pr: PullRequest) -> dict[str, str]:
        """Get the contents of changed files.

        Args:
            pr: Pull request object

        Returns:
            Dict mapping file paths to their contents
        """
        files = {}
        repo = pr.base.repo

        for file in pr.get_files():
            if file.status == "removed":
                continue

            try:
                content = repo.get_contents(file.filename, ref=pr.head.sha)
                if hasattr(content, "decoded_content"):
                    files[file.filename] = content.decoded_content.decode("utf-8")
            except Exception as e:
                logger.warning(f"Could not fetch {file.filename}: {e}")

        return files

    def build_review_context(self, pr: PullRequest, repo: Repository) -> ReviewContext:
        """Build review context from a PR.

        Args:
            pr: Pull request object
            repo: Repository object

        Returns:
            ReviewContext with PR information
        """
        labels = [label.name for label in pr.get_labels()]
        languages = list(repo.get_languages().keys())

        return ReviewContext(
            repo_name=repo.full_name,
            pr_number=pr.number,
            pr_title=pr.title,
            pr_description=pr.body or "",
            base_branch=pr.base.ref,
            head_branch=pr.head.ref,
            author=pr.user.login,
            changed_files_count=pr.changed_files,
            additions=pr.additions,
            deletions=pr.deletions,
            labels=labels,
            repo_languages=languages,
        )

    def _dismiss_pending_reviews(self, pr: PullRequest) -> bool:
        """Dismiss any pending reviews from the current user.

        Args:
            pr: Pull request object

        Returns:
            True if a pending review was dismissed
        """
        try:
            current_user = self._gh.get_user().login
            reviews = pr.get_reviews()

            for review in reviews:
                # Find pending reviews from the current user
                if review.user.login == current_user and review.state == "PENDING":
                    logger.info(f"Dismissing pending review {review.id} from {current_user}")
                    # Submit the pending review as a comment to clear it
                    review.dismiss("Superseded by new AI review")
                    return True
        except Exception as e:
            logger.warning(f"Could not dismiss pending reviews: {e}")

        return False

    def post_review(
        self,
        pr: PullRequest,
        review: ConsolidatedReview,  # noqa: ARG002
        body: str,
        event: str = "COMMENT",
    ) -> None:
        """Post a review to a PR.

        Args:
            pr: Pull request to review
            review: Consolidated review data (kept for API compatibility)
            body: Review body text
            event: Review event type (APPROVE, REQUEST_CHANGES, COMMENT)
        """
        logger.info(f"Posting review to PR #{pr.number}: {event}")

        try:
            pr.create_review(body=body, event=event)
        except GithubException as e:
            if e.status == 422 and "pending review" in str(e.data).lower():
                logger.warning("User has a pending review, falling back to issue comment")
                # Fall back to posting as a regular comment
                self._post_as_comment(pr, body)
            else:
                raise

    def _post_as_comment(self, pr: PullRequest, body: str) -> None:
        """Post review as a regular issue comment (fallback).

        Args:
            pr: Pull request object
            body: Comment body
        """
        # Add a note that this is posted as a comment due to pending review
        comment_body = (
            f"⚠️ *Posted as comment because you have a pending review on this PR.*\n\n---\n\n{body}"
        )
        pr.create_issue_comment(comment_body)
        logger.info(f"Posted review as issue comment on PR #{pr.number}")

    def post_inline_comments(
        self,
        pr: PullRequest,
        review: ConsolidatedReview,
    ) -> int:
        """Post inline comments for each finding.

        Args:
            pr: Pull request
            review: Consolidated review with findings

        Returns:
            Number of successfully posted comments
        """
        # Get the head commit for inline comments
        head_commit = pr.get_commits().reversed[0]
        posted_count = 0

        for finding in review.findings[:10]:  # Limit inline comments
            try:
                # Build comment body with emoji for severity
                severity_emoji = {
                    "critical": "🔴",
                    "warning": "🟡",
                    "suggestion": "💡",
                    "nitpick": "📝",
                }.get(finding.severity.value, "ℹ️")

                comment_body = f"{severity_emoji} **{finding.title}**\n\n{finding.description}"
                if finding.suggested_fix:
                    comment_body += f"\n\n**Suggested fix:**\n```\n{finding.suggested_fix}\n```"

                # Use create_review_comment for inline comments on the diff
                pr.create_review_comment(
                    body=comment_body,
                    commit=head_commit,
                    path=finding.file_path,
                    line=finding.line_start,
                )
                posted_count += 1
                logger.debug(f"Posted inline comment on {finding.file_path}:{finding.line_start}")
            except Exception as e:
                # Inline comments can fail if the line isn't in the diff
                logger.warning(
                    f"Could not post inline comment on {finding.file_path}:{finding.line_start}: {e}"
                )

        return posted_count

    def get_previous_review_comments(self, pr: PullRequest) -> list[PreviousComment]:
        """Get all previous review comments from this AI reviewer.

        Args:
            pr: Pull request object

        Returns:
            List of previous comments from this reviewer
        """
        comments: list[PreviousComment] = []
        current_user = self._gh.get_user().login

        # Get review comments (inline comments on code)
        for comment in pr.get_review_comments():
            if comment.user.login != current_user:
                continue

            # Parse the comment to extract title and severity
            parsed = self._parse_review_comment(comment)
            if parsed:
                comments.append(parsed)

        logger.info(f"Found {len(comments)} previous review comments from {current_user}")
        return comments

    def _parse_review_comment(self, comment: PullRequestComment) -> PreviousComment | None:
        """Parse a review comment to extract structured data.

        Args:
            comment: GitHub review comment

        Returns:
            Parsed comment or None if not parseable
        """
        body = comment.body

        # Extract severity from emoji
        severity_map = {
            "🔴": "critical",
            "🟡": "warning",
            "💡": "suggestion",
            "📝": "nitpick",
        }

        severity = "unknown"
        for emoji, sev in severity_map.items():
            if emoji in body:
                severity = sev
                break

        # Extract title from **Title** pattern
        title_match = re.search(r"\*\*([^*]+)\*\*", body)
        title = title_match.group(1) if title_match else "Unknown Issue"

        return PreviousComment(
            id=comment.id,
            file_path=comment.path,
            line=comment.line or comment.original_line or 0,
            title=title,
            severity=severity,
            body=body,
            is_resolved=False,  # GitHub API doesn't expose this directly
        )

    def compute_review_delta(
        self,
        pr: PullRequest,
        current_findings: list[ConsolidatedFinding],
    ) -> ReviewDelta:
        """Compare current findings with previous comments to compute delta.

        Args:
            pr: Pull request object
            current_findings: Current review findings

        Returns:
            ReviewDelta showing new, fixed, and open issues
        """
        previous_comments = self.get_previous_review_comments(pr)
        delta = ReviewDelta(previous_comments=previous_comments)

        # Create a lookup for previous comments by (file, line, title_normalized)
        previous_lookup: dict[tuple[str, int, str], PreviousComment] = {}
        for comment in previous_comments:
            key = (comment.file_path, comment.line, self._normalize_title(comment.title))
            previous_lookup[key] = comment

        # Track which previous comments are still open
        matched_previous: set[int] = set()

        for finding in current_findings:
            key = (finding.file_path, finding.line_start, self._normalize_title(finding.title))

            if key in previous_lookup:
                # This finding was already reported - it's still OPEN
                delta.open_findings.append(finding)
                matched_previous.add(previous_lookup[key].id)
            else:
                # This is a NEW finding
                delta.new_findings.append(finding)

        # Any previous comments not matched are FIXED
        for comment in previous_comments:
            if comment.id not in matched_previous:
                delta.fixed_findings.append(comment)

        logger.info(
            f"Review delta: {len(delta.new_findings)} new, "
            f"{len(delta.fixed_findings)} fixed, "
            f"{len(delta.open_findings)} open"
        )

        return delta

    def _normalize_title(self, title: str) -> str:
        """Normalize a title for comparison.

        Args:
            title: Original title

        Returns:
            Normalized title (lowercase, stripped)
        """
        return title.lower().strip()

    def resolve_fixed_comments(self, pr: PullRequest, delta: ReviewDelta) -> int:
        """Mark fixed issues as resolved by replying to them.

        Args:
            pr: Pull request object
            delta: Review delta with fixed findings

        Returns:
            Number of comments marked as resolved
        """
        resolved_count = 0

        for fixed in delta.fixed_findings:
            try:
                # Find the comment and reply to it
                comment = pr.get_review_comment(fixed.id)
                comment.create_reaction("hooray")  # 🎉 reaction

                # Post a reply indicating it's fixed
                pr.create_review_comment_reply(
                    comment_id=fixed.id,
                    body="✅ **Resolved** - This issue has been addressed in the latest changes.",
                )
                resolved_count += 1
                logger.debug(f"Marked comment {fixed.id} as resolved")
            except Exception as e:
                logger.warning(f"Could not resolve comment {fixed.id}: {e}")

        return resolved_count
