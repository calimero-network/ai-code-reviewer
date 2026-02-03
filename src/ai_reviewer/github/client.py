"""GitHub API client for PR operations."""

import logging
from dataclasses import dataclass
from typing import Any

from github import Github
from github.GithubException import GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository

from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.review import ConsolidatedReview

logger = logging.getLogger(__name__)


@dataclass
class GitHubConfig:
    """Configuration for GitHub client."""

    token: str
    base_url: str | None = None  # For GitHub Enterprise


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
        review: ConsolidatedReview,
        body: str,
        event: str = "COMMENT",
    ) -> None:
        """Post a review to a PR.

        Args:
            pr: Pull request to review
            review: Consolidated review data
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
            "⚠️ *Posted as comment because you have a pending review on this PR.*\n\n"
            "---\n\n"
            f"{body}"
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
                logger.warning(f"Could not post inline comment on {finding.file_path}:{finding.line_start}: {e}")

        return posted_count
