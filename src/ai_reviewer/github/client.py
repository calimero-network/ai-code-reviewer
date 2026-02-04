"""GitHub API client for PR operations."""

import contextlib
import logging
import re
from dataclasses import dataclass, field

import requests
from github import Github
from github.GithubException import GithubException
from github.PullRequest import PullRequest
from github.PullRequestComment import PullRequestComment
from github.Repository import Repository

from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import ConsolidatedFinding
from ai_reviewer.models.review import ConsolidatedReview

logger = logging.getLogger(__name__)

# GitHub GraphQL API endpoint
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"


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


class GitHubClient:
    """Client for GitHub API operations."""

    def __init__(self, token: str, base_url: str | None = None) -> None:
        """Initialize the GitHub client.

        Args:
            token: GitHub personal access token or app token
            base_url: Optional base URL for GitHub Enterprise
        """
        self._token = token
        self._base_url = base_url
        if base_url:
            self._gh = Github(token, base_url=base_url)
        else:
            self._gh = Github(token)
        # Cache for current user login (lazily populated)
        # None = not fetched yet, "" = fetch failed, other = actual login
        self._current_user_login: str | None = None
        # Cache for allowed users set (lazily populated after user login fetch)
        self._allowed_users_cache: set[str] | None = None

    def _get_current_user_login(self) -> str | None:
        """Get the current authenticated user's login, with caching.

        Returns:
            The current user's login or None if unable to fetch
        """
        if self._current_user_login is None:
            try:
                self._current_user_login = self._gh.get_user().login
            except GithubException as e:
                logger.warning(f"Could not fetch current user: {e}")
                # Cache the failure to avoid repeated API calls
                self._current_user_login = ""
        # Return None if fetch failed (empty string)
        return self._current_user_login if self._current_user_login else None

    def _get_allowed_users(self) -> set[str]:
        """Get the set of allowed AI reviewer users, with caching.

        Returns:
            Set of usernames that are considered AI reviewers
        """
        if self._allowed_users_cache is not None:
            return self._allowed_users_cache

        current_user = self._get_current_user_login()
        if current_user:
            self._allowed_users_cache = self.AI_REVIEWER_USERS | {current_user}
        else:
            self._allowed_users_cache = self.AI_REVIEWER_USERS.copy()
        return self._allowed_users_cache

    def _graphql_request(self, query: str, variables: dict | None = None) -> dict | None:
        """Make a GraphQL request to GitHub API.

        Args:
            query: GraphQL query or mutation
            variables: Optional variables for the query

        Returns:
            Response data or None if request failed
        """
        graphql_url = GITHUB_GRAPHQL_URL
        if self._base_url:
            # For GitHub Enterprise, construct GraphQL endpoint
            # PyGithub uses REST API URL (e.g., https://github.example.com/api/v3)
            # but GraphQL endpoint is at /api/graphql (without /v3)
            base = self._base_url.rstrip("/")
            if base.endswith("/v3"):
                base = base[:-3]  # Remove '/v3', keeping '/api'
            graphql_url = f"{base}/graphql"

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            response = requests.post(graphql_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                logger.warning(f"GraphQL errors: {result['errors']}")
                return None

            return result.get("data")
        except requests.RequestException as e:
            logger.warning(f"GraphQL request failed: {e}")
            return None

    def _get_thread_id_for_comment(self, repo_name: str, pr_number: int, comment_id: int) -> str | None:
        """Get the thread node ID for a review comment using GraphQL.

        Args:
            repo_name: Repository in "owner/name" format
            pr_number: Pull request number
            comment_id: The review comment ID

        Returns:
            Thread node ID or None if not found
        """
        owner, name = repo_name.split("/")

        # Query to find the thread containing this comment
        # Fetch more comments per thread to find comments that might not be first
        query = """
        query($owner: String!, $name: String!, $prNumber: Int!) {
            repository(owner: $owner, name: $name) {
                pullRequest(number: $prNumber) {
                    reviewThreads(first: 100) {
                        nodes {
                            id
                            isResolved
                            comments(first: 50) {
                                nodes {
                                    databaseId
                                }
                            }
                        }
                    }
                }
            }
        }
        """

        variables = {"owner": owner, "name": name, "prNumber": pr_number}
        data = self._graphql_request(query, variables)

        if not data:
            logger.warning(f"GraphQL query returned no data for PR #{pr_number}")
            return None

        try:
            threads = data["repository"]["pullRequest"]["reviewThreads"]["nodes"]
            logger.debug(f"Found {len(threads)} review threads in PR #{pr_number}")
            for thread in threads:
                # Skip already resolved threads
                if thread.get("isResolved"):
                    continue
                # Check if this thread contains our comment
                comments = thread.get("comments", {}).get("nodes", [])
                for comment in comments:
                    if comment.get("databaseId") == comment_id:
                        logger.debug(f"Found thread {thread['id']} for comment {comment_id}")
                        return thread["id"]
        except (KeyError, TypeError) as e:
            logger.warning(f"Failed to parse thread data: {e}")

        logger.debug(f"Could not find thread for comment {comment_id} in PR #{pr_number}")
        return None

    def _resolve_review_thread(self, thread_id: str) -> bool:
        """Resolve a review thread using GraphQL.

        Args:
            thread_id: The thread's node ID

        Returns:
            True if successfully resolved
        """
        mutation = """
        mutation($threadId: ID!) {
            resolveReviewThread(input: {threadId: $threadId}) {
                thread {
                    isResolved
                }
            }
        }
        """

        logger.debug(f"Attempting to resolve thread {thread_id}")
        data = self._graphql_request(mutation, {"threadId": thread_id})

        if data and data.get("resolveReviewThread", {}).get("thread", {}).get("isResolved"):
            logger.info(f"Successfully resolved thread {thread_id}")
            return True

        logger.warning(f"Failed to resolve thread {thread_id}, response: {data}")
        return False

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

    # Known AI reviewer bot usernames
    AI_REVIEWER_USERS = {
        "github-actions[bot]",
        "cursor[bot]",
    }

    def get_previous_review_comments(
        self,
        pr: PullRequest,
        raw_comments: list[PullRequestComment] | None = None,
    ) -> list[PreviousComment]:
        """Get all previous review comments from AI reviewers.

        Args:
            pr: Pull request object
            raw_comments: Optional pre-fetched list of raw comments to avoid redundant API calls

        Returns:
            List of previous comments from AI reviewers
        """
        comments: list[PreviousComment] = []
        allowed_users = self._get_allowed_users()

        # Use provided comments or fetch from API
        review_comments = raw_comments if raw_comments is not None else list(pr.get_review_comments())

        for comment in review_comments:
            # Skip comments from deleted users or integrations
            if comment.user is None or comment.user.login is None:
                continue

            user_login = comment.user.login

            # Check if from known AI reviewer OR has AI reviewer format
            is_ai_reviewer = user_login in allowed_users
            has_ai_format = self._is_ai_reviewer_comment(comment.body)

            if not (is_ai_reviewer or has_ai_format):
                continue

            # Parse the comment to extract title and severity
            parsed = self._parse_review_comment(comment)
            if parsed:
                comments.append(parsed)

        logger.info(f"Found {len(comments)} previous AI review comments")
        return comments

    def _is_ai_reviewer_comment(self, body: str) -> bool:
        """Check if a comment looks like it came from an AI reviewer.

        Args:
            body: Comment body

        Returns:
            True if it matches AI reviewer format
        """
        # Check for severity emojis that we use
        ai_markers = ["🔴", "🟡", "💡", "📝", "**Suggested fix:**", "AI Code Reviewer"]
        return any(marker in body for marker in ai_markers)

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

        if not delta.fixed_findings:
            logger.debug("No fixed findings to resolve")
            return 0

        # Fetch comments once to avoid redundant API calls
        raw_comments = list(pr.get_review_comments())

        # Get all existing replies to avoid duplicates
        existing_replies = self._get_resolved_comment_ids(pr, raw_comments=raw_comments)
        logger.info(
            f"Resolving {len(delta.fixed_findings)} fixed comments, "
            f"{len(existing_replies)} already resolved"
        )

        repo_name = pr.base.repo.full_name

        for fixed in delta.fixed_findings:
            # Skip if we've already marked this as resolved
            if fixed.id in existing_replies:
                logger.info(f"Comment {fixed.id} already has resolved reply, skipping")
                continue

            try:
                # Find the comment and reply to it
                comment = pr.get_review_comment(fixed.id)

                # Add reaction (may already exist, that's ok)
                with contextlib.suppress(Exception):
                    comment.create_reaction("hooray")  # 🎉 reaction

                # Post a reply indicating it's fixed
                pr.create_review_comment_reply(
                    comment_id=fixed.id,
                    body="✅ **Resolved** - This issue has been addressed in the latest changes.",
                )
                resolved_count += 1
                logger.debug(f"Marked comment {fixed.id} as resolved")

                # Also resolve the thread in GitHub UI using GraphQL
                self._resolve_thread_for_comment(repo_name, pr.number, fixed.id)

            except Exception as e:
                logger.warning(f"Could not resolve comment {fixed.id}: {e}")

        return resolved_count

    def _resolve_thread_for_comment(self, repo_name: str, pr_number: int, comment_id: int) -> bool:
        """Resolve the GitHub review thread containing a comment.

        Args:
            repo_name: Repository in "owner/name" format
            pr_number: Pull request number
            comment_id: The review comment ID

        Returns:
            True if thread was successfully resolved
        """
        thread_id = self._get_thread_id_for_comment(repo_name, pr_number, comment_id)
        if not thread_id:
            logger.debug(f"Could not find thread ID for comment {comment_id}")
            return False

        if self._resolve_review_thread(thread_id):
            logger.debug(f"Resolved thread for comment {comment_id}")
            return True

        logger.debug(f"Failed to resolve thread for comment {comment_id}")
        return False

    def _get_resolved_comment_ids(
        self,
        pr: PullRequest,
        raw_comments: list[PullRequestComment] | None = None,
    ) -> set[int]:
        """Get IDs of comments that already have a 'Resolved' reply from this reviewer.

        Args:
            pr: Pull request object
            raw_comments: Optional pre-fetched list of raw comments to avoid redundant API calls

        Returns:
            Set of comment IDs that have been marked resolved
        """
        resolved_ids: set[int] = set()
        allowed_users = self._get_allowed_users()

        # Use provided comments or fetch from API
        review_comments = raw_comments if raw_comments is not None else list(pr.get_review_comments())

        for comment in review_comments:
            # Check if this is a "Resolved" reply from our reviewer
            if "✅ **Resolved**" not in comment.body:
                continue

            # Skip comments from deleted users or integrations
            if comment.user is None or comment.user.login is None:
                continue

            # Only count resolved comments from allowed users
            if comment.user.login not in allowed_users:
                continue

            # Check if this comment has a valid in_reply_to_id
            # PyGithub may return NotSet, None, or 0 for non-reply comments
            reply_to_id = getattr(comment, "in_reply_to_id", None)
            if reply_to_id is not None and reply_to_id != 0:
                # Handle PyGithub's NotSet sentinel
                try:
                    reply_id = int(reply_to_id)
                    resolved_ids.add(reply_id)
                    logger.debug(f"Found existing resolved reply for comment {reply_id}")
                except (TypeError, ValueError):
                    # in_reply_to_id was NotSet or invalid
                    pass

        logger.debug(f"Found {len(resolved_ids)} already-resolved comment IDs")
        return resolved_ids
