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

# Sentinel for failed user login fetch (distinct from empty string)
_USER_FETCH_FAILED = "__FETCH_FAILED__"


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
        self._current_user_login: str | None = None
        self._allowed_users: set[str] | None = None

        if base_url:
            self._gh = Github(token, base_url=base_url)
        else:
            self._gh = Github(token)

    def _get_current_user_login(self) -> str | None:
        """Get the current authenticated user's login, with caching.

        Returns:
            The user login string, or None if fetch failed
        """
        if self._current_user_login is None:
            try:
                self._current_user_login = self._gh.get_user().login
            except Exception as e:
                logger.warning(f"Could not fetch current user: {e}")
                self._current_user_login = _USER_FETCH_FAILED

        if self._current_user_login == _USER_FETCH_FAILED:
            return None
        return self._current_user_login

    def _get_allowed_users(self) -> set[str]:
        """Get the set of allowed AI reviewer users, with caching.

        Returns:
            Set of allowed usernames (current user + known bot users)
        """
        if self._allowed_users is None:
            current_user = self._get_current_user_login()
            if current_user:
                self._allowed_users = self.AI_REVIEWER_USERS | {current_user}
            else:
                self._allowed_users = self.AI_REVIEWER_USERS.copy()
        return self._allowed_users

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
            current_user = self._get_current_user_login()
            if not current_user:
                logger.warning("Could not fetch current user, skipping pending review dismissal")
                return False

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
    }

    def get_previous_review_comments(self, pr: PullRequest) -> list[PreviousComment]:
        """Get all previous review comments from AI reviewers.

        Args:
            pr: Pull request object

        Returns:
            List of previous comments from AI reviewers
        """
        comments: list[PreviousComment] = []
        allowed_users = self._get_allowed_users()

        # Get review comments (inline comments on code)
        for comment in pr.get_review_comments():
            # Skip our own "Resolved" replies - they are not findings
            if "✅ **Resolved**" in comment.body:
                continue

            user_login = comment.user.login

            # Only process comments authored by our bot - never human or other bot comments.
            # Using format-based matching (has_ai_format) caused false positives where we
            # would reply "Resolved" to human comments that happened to use similar formatting.
            if user_login not in allowed_users:
                continue

            # Parse the comment to extract title and severity
            parsed = self._parse_review_comment(comment)
            if parsed:
                comments.append(parsed)

        logger.info(f"Found {len(comments)} previous AI review comments")
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

        # Determine which unmatched previous comments are likely fixed.
        # We mark as fixed when:
        # 1. The file is no longer in the diff (removed/renamed), OR
        # 2. The commented line was modified AND the AI didn't find the issue again
        #
        # This avoids false "Resolved" on unmodified code while still detecting
        # actual fixes when the relevant lines were changed.
        pr_files = list(pr.get_files())
        changed_files = {f.filename for f in pr_files}

        # Build a mapping of file -> modified lines
        file_modified_lines: dict[str, set[int]] = {}
        for f in pr_files:
            if f.patch:
                file_modified_lines[f.filename] = self._parse_modified_lines(f.patch)

        for comment in previous_comments:
            if comment.id not in matched_previous:
                file_path = comment.file_path

                if file_path not in changed_files:
                    # File was removed/renamed - definitely fixed
                    delta.fixed_findings.append(comment)
                elif file_path in file_modified_lines:
                    # File is still in diff - check if the commented line was modified
                    modified_lines = file_modified_lines[file_path]
                    if self._is_line_in_modified_range(comment.line, modified_lines):
                        # The line was modified AND the AI didn't find the issue again
                        # → likely fixed
                        delta.fixed_findings.append(comment)
                # else: file in diff but line wasn't modified → don't mark as fixed

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

    def _parse_modified_lines(self, patch: str) -> set[int]:
        """Parse a unified diff patch to extract modified line numbers.

        Args:
            patch: Unified diff patch string

        Returns:
            Set of line numbers that were added or modified in the new version
        """
        modified_lines: set[int] = set()
        if not patch:
            return modified_lines

        current_line = 0
        for line in patch.split("\n"):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            if line.startswith("@@"):
                # Extract new file line number
                match = re.search(r"\+(\d+)", line)
                if match:
                    current_line = int(match.group(1))
                continue

            if line.startswith("+") and not line.startswith("+++"):
                # Added line - this is a modification
                modified_lines.add(current_line)
                current_line += 1
            elif line.startswith("-") and not line.startswith("---"):
                # Deleted line - don't increment (it's not in the new file)
                # But mark the current position as "modified" since something changed here
                modified_lines.add(current_line)
            else:
                # Context line or other - increment line counter
                # Skip "\ No newline at end of file"
                if not line.startswith("\\"):
                    current_line += 1

        return modified_lines

    def _is_line_in_modified_range(
        self, line: int, modified_lines: set[int], tolerance: int = 3
    ) -> bool:
        """Check if a line number falls within or near modified lines.

        Args:
            line: Line number to check
            modified_lines: Set of modified line numbers
            tolerance: How many lines away still counts as "near"

        Returns:
            True if the line is within tolerance of any modified line
        """
        for mod_line in modified_lines:
            if abs(line - mod_line) <= tolerance:
                return True
        return False

    def _graphql_request(self, query: str, variables: dict | None = None) -> dict | None:
        """Make a GraphQL request to GitHub API.

        Args:
            query: GraphQL query string
            variables: Optional query variables

        Returns:
            Response data dict or None on error
        """
        # Determine GraphQL endpoint
        if self._base_url:
            # GitHub Enterprise: /api/v3 -> /api/graphql
            base = self._base_url.rstrip("/")
            if base.endswith("/api/v3"):
                # Replace /v3 with /graphql
                graphql_url = base[:-3] + "/graphql"
            else:
                graphql_url = f"{base}/graphql"
        else:
            graphql_url = "https://api.github.com/graphql"

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
        except Exception as e:
            logger.warning(f"GraphQL request failed: {e}")
            return None

    # Maximum pages to fetch when paginating GraphQL results (prevents runaway loops)
    _MAX_GRAPHQL_PAGES = 20

    def _fetch_thread_mapping(self, repo_name: str, pr_number: int) -> dict[int, str]:
        """Fetch all review threads and build a mapping of comment_id to thread_id.

        This batches the GraphQL calls to avoid N+1 queries when resolving multiple
        threads.

        Args:
            repo_name: Repository in "owner/name" format
            pr_number: Pull request number

        Returns:
            Dict mapping comment database IDs to their thread's GraphQL node ID
            (only includes unresolved threads)
        """
        owner, name = repo_name.split("/")
        comment_to_thread: dict[int, str] = {}

        query = """
        query($owner: String!, $name: String!, $pr_number: Int!, $cursor: String) {
          repository(owner: $owner, name: $name) {
            pullRequest(number: $pr_number) {
              reviewThreads(first: 100, after: $cursor) {
                pageInfo {
                  hasNextPage
                  endCursor
                }
                nodes {
                  id
                  isResolved
                  comments(first: 100) {
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

        cursor = None
        pages_fetched = 0

        while pages_fetched < self._MAX_GRAPHQL_PAGES:
            variables = {
                "owner": owner,
                "name": name,
                "pr_number": pr_number,
                "cursor": cursor,
            }

            data = self._graphql_request(query, variables)
            if not data:
                break

            pages_fetched += 1
            pr_data = (data.get("repository") or {}).get("pullRequest") or {}
            threads_data = pr_data.get("reviewThreads", {})
            threads = threads_data.get("nodes", [])

            for thread in threads:
                if thread.get("isResolved"):
                    continue  # Skip already resolved threads

                thread_id = thread.get("id")
                if not thread_id:
                    continue

                comments = (thread.get("comments") or {}).get("nodes") or []
                for comment in comments:
                    db_id = comment.get("databaseId")
                    if db_id:
                        comment_to_thread[db_id] = thread_id

            # Check for more pages
            page_info = threads_data.get("pageInfo", {})
            if page_info.get("hasNextPage"):
                cursor = page_info.get("endCursor")
            else:
                break

        if pages_fetched >= self._MAX_GRAPHQL_PAGES:
            logger.warning(
                f"Hit max page limit ({self._MAX_GRAPHQL_PAGES}) fetching threads for PR #{pr_number}"
            )

        return comment_to_thread

    def _resolve_review_thread(self, thread_id: str) -> bool:
        """Resolve a review thread using GraphQL.

        Args:
            thread_id: The GraphQL node ID of the thread

        Returns:
            True if successfully resolved
        """
        mutation = """
        mutation($thread_id: ID!) {
          resolveReviewThread(input: {threadId: $thread_id}) {
            thread {
              isResolved
            }
          }
        }
        """

        data = self._graphql_request(mutation, {"thread_id": thread_id})
        if data:
            thread = (data.get("resolveReviewThread") or {}).get("thread") or {}
            return thread.get("isResolved", False)
        return False

    def _resolve_thread_for_comment(
        self,
        comment_id: int,
        thread_mapping: dict[int, str],
    ) -> bool:
        """Resolve the review thread containing a specific comment.

        Args:
            comment_id: The comment ID whose thread to resolve
            thread_mapping: Pre-fetched mapping of comment_id to thread_id

        Returns:
            True if thread was resolved
        """
        thread_id = thread_mapping.get(comment_id)
        if not thread_id:
            logger.debug(f"Could not find thread for comment {comment_id}")
            return False

        if self._resolve_review_thread(thread_id):
            logger.info(f"Resolved thread for comment {comment_id}")
            return True

        return False

    def resolve_fixed_comments(self, pr: PullRequest, delta: ReviewDelta) -> int:
        """Mark fixed issues as resolved by replying to them.

        Args:
            pr: Pull request object
            delta: Review delta with fixed findings

        Returns:
            Number of comments marked as resolved
        """
        if not delta.fixed_findings:
            logger.debug("No fixed findings to resolve")
            return 0

        resolved_count = 0

        # Fetch comments once and pass to helper to avoid redundant API calls
        raw_comments = list(pr.get_review_comments())

        # Get all existing replies to avoid duplicates
        existing_replies = self._get_resolved_comment_ids(pr, raw_comments)
        logger.info(
            f"Found {len(existing_replies)} already-resolved comments, "
            f"processing {len(delta.fixed_findings)} fixed findings"
        )

        repo_name = pr.base.repo.full_name

        # Batch-fetch all thread mappings once (avoids N+1 GraphQL calls)
        thread_mapping = self._fetch_thread_mapping(repo_name, pr.number)

        for fixed in delta.fixed_findings:
            # Skip if we've already marked this as resolved (avoid duplicate "Resolved" on re-review)
            if fixed.id in existing_replies:
                logger.debug(f"Comment {fixed.id} already has resolved reply, skipping")
                continue
            if self._comment_already_has_resolved_reply(fixed.id, raw_comments):
                logger.debug(f"Comment {fixed.id} already has resolved reply (re-check), skipping")
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

                # Hand-in-hand: also resolve the thread in GitHub UI (collapse the conversation).
                # Without this, the reply would show but the thread would stay "open".
                if not self._resolve_thread_for_comment(fixed.id, thread_mapping):
                    logger.warning(
                        f"Posted 'Resolved' reply on comment {fixed.id} but could not "
                        "resolve the thread (GraphQL resolve failed or thread not found). "
                        "Thread may still appear open in the PR."
                    )

                resolved_count += 1
                logger.debug(f"Marked comment {fixed.id} as resolved")
            except Exception as e:
                logger.warning(f"Could not resolve comment {fixed.id}: {e}")

        return resolved_count

    def _get_resolved_comment_ids(
        self, pr: PullRequest, raw_comments: list | None = None
    ) -> set[int]:
        """Get IDs of comments that already have a 'Resolved' reply from us.

        Args:
            pr: Pull request object
            raw_comments: Optional pre-fetched comments to avoid redundant API calls

        Returns:
            Set of comment IDs that have been marked resolved
        """
        resolved_ids: set[int] = set()
        allowed_users = self._get_allowed_users()

        comments = raw_comments if raw_comments is not None else pr.get_review_comments()

        for comment in comments:
            # Check if this is a "Resolved" reply with a parent
            if "✅ **Resolved**" not in comment.body:
                continue

            # Safely get in_reply_to_id (may be NotSet, None, or 0)
            reply_to = getattr(comment, "in_reply_to_id", None)
            if reply_to is None or reply_to == 0:
                continue

            # Handle PyGithub's NotSet sentinel (isinstance never raises)
            if not isinstance(reply_to, int):
                continue

            # Only count resolved comments from allowed users
            if comment.user is None or comment.user.login is None:
                continue
            if comment.user.login not in allowed_users:
                continue

            resolved_ids.add(reply_to)

        return resolved_ids

    def _comment_already_has_resolved_reply(
        self,
        comment_id: int,
        raw_comments: list,
    ) -> bool:
        """Check if a comment already has a 'Resolved' reply from us in the given list.

        Used as a safeguard before posting to avoid duplicate Resolved replies
        (e.g. when the same thread is marked fixed on every re-review).

        Args:
            comment_id: The original comment (finding) ID
            raw_comments: Full list of PR review comments

        Returns:
            True if we already posted a Resolved reply to this comment
        """
        allowed_users = self._get_allowed_users()
        for c in raw_comments:
            reply_to = getattr(c, "in_reply_to_id", None)
            if reply_to is None or not isinstance(reply_to, int) or reply_to != comment_id:
                continue
            if "✅ **Resolved**" not in (c.body or ""):
                continue
            if c.user and c.user.login and c.user.login in allowed_users:
                return True
        return False
