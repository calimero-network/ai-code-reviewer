"""GitHub API client for PR operations."""

from __future__ import annotations

import contextlib
import enum
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

import requests
import yaml
from github import Github
from github.GithubException import GithubException
from github.PullRequest import PullRequest, ReviewComment
from github.PullRequestComment import PullRequestComment
from github.Repository import Repository

from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import ConsolidatedFinding, Severity, compute_fuzzy_hash
from ai_reviewer.models.review import ConsolidatedReview

logger = logging.getLogger(__name__)

# Sentinel for failed user login fetch (distinct from empty string)
_USER_FETCH_FAILED = "__FETCH_FAILED__"


def _raise_if_forbidden(exc: Exception) -> None:
    """Re-raise 403 Forbidden immediately — it won't resolve on retry.

    Handles both PyGithub GithubException (REST calls via PyGithub) and
    requests.HTTPError (direct requests.post calls, e.g. GraphQL endpoint).
    """
    if isinstance(exc, GithubException) and exc.status == 403:
        raise PermissionError("GitHub REST API 403 Forbidden — check token scopes") from exc
    if isinstance(exc, requests.exceptions.HTTPError):
        resp = getattr(exc, "response", None)
        if resp is not None and resp.status_code == 403:
            raise PermissionError("GitHub GraphQL 403 Forbidden") from exc


_RESOLVE_COMMENT_DELAY_S: float = float(os.environ.get("AI_REVIEWER_RESOLVE_DELAY", "0.2"))
_MAX_RESOLVE_COMMENTS: int = int(os.environ.get("AI_REVIEWER_MAX_RESOLVE", "100"))
_NO_LONGER_DETECTED_REPLY = (
    "✅ **No longer detected** - This issue was not re-detected after the latest changes."
)

_RESOLVED_REPLY_MARKERS = (
    _NO_LONGER_DETECTED_REPLY,
    "✅ **Resolved**",
    "✅ **No longer detected**",
)


def _is_resolved_reply(body: str) -> bool:
    return any(marker in body for marker in _RESOLVED_REPLY_MARKERS)


_SEVERITY_ORDER: list[Severity] = [
    Severity.CRITICAL,
    Severity.WARNING,
    Severity.SUGGESTION,
    Severity.NITPICK,
]


def _parse_severity(severity_str: str) -> Severity | None:
    """Convert a severity string (from a previous comment) to a Severity enum.

    Returns None for unrecognised values so callers can skip stabilization
    rather than guessing.
    """
    try:
        return Severity(severity_str.lower())
    except ValueError:
        return None


def compute_findings_hash(finding_hashes: list[str]) -> str:
    """Deterministic hash of a sorted list of finding hashes.

    Enables quick convergence checks: if the findings_hash from the previous
    review matches the current one, the issue set is identical.
    """
    key = ":".join(sorted(finding_hashes))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


_REVIEW_META_RE = re.compile(r"<!-- ai-reviewer-meta: ({.*?}) -->")


@dataclass
class ReviewMeta:
    """Metadata embedded in top-level review comments for cross-run tracking."""

    commit_sha: str
    review_count: int
    timestamp: str  # ISO 8601
    findings_hash: str  # hash of sorted finding hashes for quick equality check

    def to_html_comment(self) -> str:
        payload = json.dumps(
            {
                "commit_sha": self.commit_sha,
                "review_count": self.review_count,
                "timestamp": self.timestamp,
                "findings_hash": self.findings_hash,
            },
            separators=(",", ":"),
        )
        return f"<!-- ai-reviewer-meta: {payload} -->"

    @classmethod
    def parse(cls, body: str) -> ReviewMeta | None:
        """Extract ReviewMeta from a comment body containing the HTML comment tag.

        Returns None when the tag is missing or the JSON is malformed.
        """
        match = _REVIEW_META_RE.search(body)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            return None
        try:
            return cls(
                commit_sha=str(data["commit_sha"]),
                review_count=int(data["review_count"]),
                timestamp=str(data["timestamp"]),
                findings_hash=str(data["findings_hash"]),
            )
        except (KeyError, ValueError, TypeError):
            return None

    @classmethod
    def build(
        cls,
        commit_sha: str,
        review_count: int,
        finding_hashes: list[str],
    ) -> ReviewMeta:
        """Construct a ReviewMeta for the current review run."""
        return cls(
            commit_sha=commit_sha,
            review_count=review_count,
            timestamp=datetime.now(UTC).isoformat(),
            findings_hash=compute_findings_hash(finding_hashes),
        )


def stabilize_severity(
    current: Severity,
    previous: Severity,
    review_count: int,
) -> Severity:
    """Decide the effective severity for a matched finding.

    * Same severity → keep current.
    * Upgrade (more severe) → always allowed.
    * Downgrade (less severe) → blocked once ``review_count >= 2``.
    """
    if current is previous:
        return current

    cur_idx = _SEVERITY_ORDER.index(current)
    prev_idx = _SEVERITY_ORDER.index(previous)

    is_upgrade = cur_idx < prev_idx
    if is_upgrade:
        return current

    if review_count >= 2:
        return previous
    return current


def apply_comment_limits(
    findings: list[ConsolidatedFinding],
    max_total: int = 50,
    max_per_file: int = 10,
) -> list[ConsolidatedFinding]:
    """Select top findings respecting per-file and total caps.

    Findings are sorted by priority_score descending, then selected greedily:
    each finding is included only if its file hasn't already hit max_per_file
    and the overall count hasn't hit max_total.
    """
    sorted_findings = sorted(findings, key=lambda f: f.priority_score, reverse=True)
    per_file: dict[str, int] = defaultdict(int)
    result: list[ConsolidatedFinding] = []

    for finding in sorted_findings:
        if len(result) >= max_total:
            break
        if per_file[finding.file_path] >= max_per_file:
            continue
        result.append(finding)
        per_file[finding.file_path] += 1

    return result


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
    finding_hash: str | None = None  # Embedded hash for stable cross-run matching

    @property
    def finding_hash_fuzzy(self) -> str | None:
        """Fuzzy hash for cross-run matching (ignores line, category).

        Mirrors ConsolidatedFinding.finding_hash_fuzzy so that a previous
        comment can be matched to a current finding even when the line number
        or category has drifted between review runs.
        """
        return compute_fuzzy_hash(self.file_path, self.title)


@dataclass
class ReviewDelta:
    """Tracks changes between review runs."""

    new_findings: list[ConsolidatedFinding] = field(default_factory=list)
    fixed_findings: list[PreviousComment] = field(default_factory=list)
    open_findings: list[ConsolidatedFinding] = field(default_factory=list)
    previous_comments: list[PreviousComment] = field(default_factory=list)
    suppressed_findings: list[ConsolidatedFinding] = field(default_factory=list)

    @property
    def all_issues_resolved(self) -> bool:
        """Check if all previously found issues are now resolved."""
        return len(self.open_findings) == 0 and len(self.new_findings) == 0


def has_converged(delta: ReviewDelta) -> bool:
    """Return True when the issue set is unchanged since the last review.

    "Converged" means no new findings appeared and no previously-reported
    findings were fixed — the delta is stable.  Open findings may still exist;
    convergence is *not* the same as ``ReviewDelta.all_issues_resolved``.
    """
    return len(delta.new_findings) == 0 and len(delta.fixed_findings) == 0


def should_skip_review(review_count: int, delta: ReviewDelta) -> bool:
    """Decide whether to skip posting a review for this run.

    Rules:
    * review_count <= 1  → never skip (first review always posts).
    * review_count >= 2  → skip when ``has_converged(delta)`` is True.
    * review_count >= 3  → also skip when the only new findings are NITPICKs
      (the issue set is "effectively" unchanged).
    """
    if review_count <= 1:
        return False

    if has_converged(delta):
        return True

    if review_count >= 3 and delta.new_findings and not delta.fixed_findings:
        all_nitpicks = all(f.severity == Severity.NITPICK for f in delta.new_findings)
        if all_nitpicks:
            return True

    return False


def estimate_review_count(delta: ReviewDelta) -> int:
    """Approximate how many review rounds have occurred.

    Uses the number of previous comments as a proxy.  Zero previous comments
    means this is the first review (count=1).  Any previous comments imply at
    least a second review (count=2+).  The exact count is intentionally coarse;
    refine later without changing the gate interface.
    """
    if not delta.previous_comments:
        return 1
    return max(2, len(delta.previous_comments) // 3 + 1)


def lgtm_placeholder_review(repo: str, pr_number: int) -> ConsolidatedReview:
    """Build a minimal ConsolidatedReview for the LGTM fast path."""
    return ConsolidatedReview(
        id="lgtm-fast-path",
        created_at=datetime.now(UTC),
        repo=repo,
        pr_number=pr_number,
        findings=[],
        summary="All previously identified issues addressed",
        agent_count=0,
        review_quality_score=1.0,
        total_review_time_ms=0,
    )


class SkipReason(enum.Enum):
    """Reason for skipping a review before running agents."""

    ALREADY_REVIEWED = "already_reviewed"
    FINDINGS_UNCHANGED = "findings_unchanged"


def should_skip_before_agents(
    meta: ReviewMeta | None,
    current_sha: str,
    force_review: bool = False,
    diff_files: set[str] | None = None,
    previous_comments: list[PreviousComment] | None = None,
) -> SkipReason | None:
    """Decide whether to skip the review entirely before running agents.

    Returns a ``SkipReason`` when the review should be skipped, or ``None``
    when agents should proceed.  The LGTM fast path is **not** handled here
    (it requires a lightweight delta computation); see the webhook/CLI callers.

    Checks (in order):
    * ``force_review`` → never skip.
    * Same ``commit_sha`` → ``ALREADY_REVIEWED``.
    * ``findings_hash`` available and diff doesn't touch any file with previous
      findings → ``FINDINGS_UNCHANGED``.
    """
    if force_review:
        return None

    if meta is None:
        return None

    if meta.commit_sha == current_sha:
        return SkipReason.ALREADY_REVIEWED

    if (
        meta.findings_hash
        and diff_files is not None
        and previous_comments is not None
    ):
        previous_files = {c.file_path for c in previous_comments}
        if not diff_files & previous_files:
            return SkipReason.FINDINGS_UNCHANGED

    return None


class GitHubClient:
    """Client for GitHub API operations."""

    def __init__(
        self,
        token: str,
        base_url: str | None = None,
        extra_reviewer_users: list[str] | None = None,
    ) -> None:
        """Initialize the GitHub client.

        Args:
            token: GitHub personal access token or app token
            base_url: Optional base URL for GitHub Enterprise
            extra_reviewer_users: Additional bot/user logins to treat as AI reviewer accounts
        """
        self._token = token
        self._base_url = base_url
        self._current_user_login: str | None = None
        self._allowed_users: set[str] | None = None
        self._extra_reviewer_users: set[str] = set(extra_reviewer_users or [])
        self._previous_comments_cache: OrderedDict[int, list[PreviousComment]] = OrderedDict()
        self._previous_comments_cache_max = 50

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
                # Do NOT raise here — this method is used by callers that swallow
                # exceptions (e.g. _dismiss_pending_reviews). Cache the failure so
                # we don't retry a 403 or other permanent error on every call.
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
                self._allowed_users = (
                    self.AI_REVIEWER_USERS | {current_user} | self._extra_reviewer_users
                )
            else:
                self._allowed_users = self.AI_REVIEWER_USERS | self._extra_reviewer_users
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
                _raise_if_forbidden(e)
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

    _CONVENTION_FILES = [
        "AGENTS.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        ".cursor/rules/README.md",
    ]
    _CONVENTION_PER_FILE_LIMIT = 10000
    _CONVENTION_TOTAL_LIMIT = 30000

    def load_repo_config(self, repo_name: str, ref: str) -> dict | None:
        """Best-effort load of ``.ai-reviewer.yaml`` from the repo at *ref*.

        Returns the parsed dict on success, or ``None`` when the file is
        missing or contains invalid YAML.  403 Forbidden is re-raised via
        ``_raise_if_forbidden`` so callers surface permission problems.
        """
        repo = self._gh.get_repo(repo_name)
        try:
            content = repo.get_contents(".ai-reviewer.yaml", ref=ref)
            if isinstance(content, list):
                logger.warning(".ai-reviewer.yaml resolved to multiple entries; ignoring")
                return None
            parsed = yaml.safe_load(content.decoded_content)
            if isinstance(parsed, dict):
                return parsed
            logger.warning(".ai-reviewer.yaml did not parse to a dict; ignoring")
            return None
        except Exception as e:
            _raise_if_forbidden(e)
            logger.debug("Could not load .ai-reviewer.yaml: %s", e)
            return None

    def load_repo_conventions(self, repo_name: str, ref: str) -> str | None:
        """Best-effort load and concatenation of convention docs from the repo.

        Reads each file in ``_CONVENTION_FILES`` (in order), truncates each to
        ``_CONVENTION_PER_FILE_LIMIT`` chars, joins them with a header, and
        truncates the combined result to ``_CONVENTION_TOTAL_LIMIT`` chars.

        Returns ``None`` when no convention files are found.
        """
        repo = self._gh.get_repo(repo_name)
        parts: list[str] = []
        for path in self._CONVENTION_FILES:
            try:
                content = repo.get_contents(path, ref=ref)
                if isinstance(content, list):
                    logger.debug("Convention path %s resolved to multiple entries; skipping", path)
                    continue
                text = content.decoded_content.decode("utf-8", errors="replace")
                if len(text) > self._CONVENTION_PER_FILE_LIMIT:
                    text = text[: self._CONVENTION_PER_FILE_LIMIT] + "\n…(truncated)"
                parts.append(f"### {path}\n{text}")
            except Exception as e:
                _raise_if_forbidden(e)
                logger.debug("Convention file %s not found or unreadable: %s", path, e)

        if not parts:
            return None

        combined = "\n\n".join(parts)
        if len(combined) > self._CONVENTION_TOTAL_LIMIT:
            combined = combined[: self._CONVENTION_TOTAL_LIMIT] + "\n…(truncated)"
        return combined

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
            _raise_if_forbidden(e)
            logger.warning(f"Could not dismiss pending reviews: {e}")

        return False

    def post_review(
        self,
        pr: PullRequest,
        body: str,
        event: str = "COMMENT",
        inline_findings: list[ConsolidatedFinding] | None = None,
    ) -> int:
        """Post a review to a PR, optionally with inline comments in a single API call.

        When *inline_findings* is provided they must already be filtered via
        ``get_postable_inline_findings``.  The top-level body and all inline
        comments are submitted atomically via one ``create_review`` call.

        Returns the number of inline comments included in the review.
        """
        comments = self._build_review_comments(inline_findings)

        logger.info(
            "Posting review to PR #%d: %s (%d inline comments)",
            pr.number,
            event,
            len(comments),
        )

        try:
            if comments:
                pr.create_review(body=body, event=event, comments=comments)
            else:
                pr.create_review(body=body, event=event)
        except GithubException as e:
            if e.status == 422 and "pending review" in str(e.data).lower():
                logger.warning(
                    "Pending review detected on PR #%d, attempting dismiss-and-retry",
                    pr.number,
                )
                self._dismiss_pending_reviews(pr)
                try:
                    if comments:
                        pr.create_review(body=body, event=event, comments=comments)
                    else:
                        pr.create_review(body=body, event=event)
                except GithubException as retry_exc:
                    if retry_exc.status == 422 and "pending review" in str(retry_exc.data).lower():
                        if comments:
                            logger.warning(
                                "Retry failed — posting body as issue comment. "
                                "%d inline comment(s) could not be posted.",
                                len(comments),
                            )
                            self._post_as_comment_with_inline_warning(pr, body, len(comments))
                        else:
                            logger.warning("Retry failed — posting body as issue comment")
                            self._post_as_comment(pr, body)
                    else:
                        raise
            else:
                raise

        return len(comments)

    @staticmethod
    def _build_review_comments(
        inline_findings: list[ConsolidatedFinding] | None,
    ) -> list[ReviewComment]:
        """Build atomic review comments from pre-filtered findings.

        Callers are responsible for filtering via ``get_postable_inline_findings``
        before passing findings here.  Each entry matches PyGithub's review
        comment shape: ``path``, ``line``, and ``body``, suitable for
        ``PullRequest.create_review(..., comments=...)``.
        """
        if not inline_findings:
            return []

        comments: list[ReviewComment] = []
        for finding in inline_findings:
            severity_emoji = {
                "critical": "🔴",
                "warning": "🟡",
                "suggestion": "💡",
                "nitpick": "📝",
            }.get(finding.severity.value, "ℹ️")

            comment_body = f"{severity_emoji} **{finding.title}**\n\n{finding.description}"
            if finding.suggested_fix:
                comment_body += f"\n\n**Suggested fix:**\n```\n{finding.suggested_fix}\n```"
            comment_body += f"\n\n<!-- ai-reviewer-id: {finding.finding_hash} -->"

            comments.append(
                ReviewComment(
                    path=finding.file_path,
                    line=finding.line_start,
                    body=comment_body,
                )
            )

        return comments

    def get_postable_inline_findings(
        self,
        pr: PullRequest,
        inline_findings: list[ConsolidatedFinding] | None,
        max_total: int,
        max_per_file: int,
    ) -> list[ConsolidatedFinding]:
        """Return inline findings that can actually be posted on this PR diff."""
        if not inline_findings:
            return []

        file_modified_lines: dict[str, set[int]] = {}
        for pr_file in pr.get_files():
            if pr_file.patch:
                file_modified_lines[pr_file.filename] = self._parse_modified_lines(pr_file.patch)

        postable_findings: list[ConsolidatedFinding] = []
        for finding in apply_comment_limits(inline_findings, max_total, max_per_file):
            modified_lines = file_modified_lines.get(finding.file_path)
            if not modified_lines or finding.line_start not in modified_lines:
                logger.warning(
                    "Skipping inline comment on %s:%d because the line is not resolvable in the PR diff",
                    finding.file_path,
                    finding.line_start,
                )
                continue

            postable_findings.append(finding)

        return postable_findings

    def _post_as_comment(self, pr: PullRequest, body: str) -> None:
        """Post review as a regular issue comment (fallback).

        Args:
            pr: Pull request object
            body: Comment body
        """
        comment_body = (
            f"⚠️ *Posted as comment because you have a pending review on this PR.*\n\n---\n\n{body}"
        )
        pr.create_issue_comment(comment_body)
        logger.info(f"Posted review as issue comment on PR #{pr.number}")

    def _post_as_comment_with_inline_warning(
        self, pr: PullRequest, body: str, dropped_inline_count: int
    ) -> None:
        """Post review as issue comment with a warning about lost inline comments."""
        comment_body = (
            f"⚠️ *Posted as comment because a pending review could not be cleared.*\n"
            f"**{dropped_inline_count} inline comment(s) could not be posted** — "
            f"please check the review body below for details.\n\n---\n\n{body}"
        )
        pr.create_issue_comment(comment_body)
        logger.info(f"Posted review as issue comment on PR #{pr.number}")

    # Known AI reviewer bot usernames
    AI_REVIEWER_USERS = {
        "github-actions[bot]",
    }

    def get_review_metadata(self, pr: PullRequest) -> ReviewMeta | None:
        """Extract ReviewMeta from the most recent bot review on this PR.

        Only checks the *most recent* review (and issue comment) from allowed
        users.  If that review/comment lacks embedded metadata (legacy format),
        returns ``None`` rather than searching older entries — this prevents
        returning stale metadata from a much earlier review run.
        """
        allowed_users = self._get_allowed_users()
        try:
            reviews = list(pr.get_reviews())
        except Exception as e:
            _raise_if_forbidden(e)
            logger.warning("Could not fetch PR reviews for metadata: %s", e)
            return None

        for review in reversed(reviews):
            if review.user is None or review.user.login not in allowed_users:
                continue
            body = review.body or ""
            meta = ReviewMeta.parse(body)
            if meta is not None:
                logger.debug(
                    "Found review metadata: commit=%s count=%d",
                    meta.commit_sha[:8],
                    meta.review_count,
                )
                return meta
            logger.debug(
                "Most recent bot review (id=%s) has no metadata; not searching older reviews",
                review.id,
            )
            break

        # Fallback: check the most recent issue comment from allowed users
        try:
            for comment in pr.get_issue_comments().reversed:
                if comment.user is None or comment.user.login not in allowed_users:
                    continue
                meta = ReviewMeta.parse(comment.body or "")
                if meta is not None:
                    logger.debug(
                        "Found review metadata in issue comment: commit=%s count=%d",
                        meta.commit_sha[:8],
                        meta.review_count,
                    )
                    return meta
                logger.debug(
                    "Most recent bot issue comment (id=%s) has no metadata; not searching older comments",
                    comment.id,
                )
                break
        except Exception as e:
            _raise_if_forbidden(e)
            logger.warning("Could not fetch issue comments for metadata: %s", e)

        return None

    def check_lgtm_fast_path(
        self,
        pr: PullRequest,
        meta: ReviewMeta,
    ) -> ReviewDelta | None:
        """Check if the PR is an LGTM *candidate* based on diff heuristics.

        Computes a lightweight delta with an empty findings list to see whether
        every previously-reported comment has been fixed in the new diff.

        Returns the candidate delta when ``all_issues_resolved`` is True and
        ``meta.review_count >= 2``, otherwise ``None``.

        **Important:** A non-None return only means the PR *looks* clean from
        the diff.  Callers must run a lightweight 1-agent re-check to confirm
        before posting the LGTM review.
        """
        if meta.review_count < 2:
            return None

        delta = self.compute_review_delta(pr, current_findings=[])
        if not delta.previous_comments:
            return None
        if delta.all_issues_resolved:
            logger.info(
                "LGTM fast path: all %d previous issues resolved (review_count=%d)",
                len(delta.fixed_findings),
                meta.review_count,
            )
            return delta
        return None

    def get_previous_review_comments(self, pr: PullRequest) -> list[PreviousComment]:
        """Get all previous review comments from AI reviewers.

        Results are cached per PR number for the lifetime of this client
        instance to avoid redundant API calls (e.g. LGTM fast path check
        followed by the main review flow).

        Args:
            pr: Pull request object

        Returns:
            List of previous comments from AI reviewers
        """
        if pr.number in self._previous_comments_cache:
            self._previous_comments_cache.move_to_end(pr.number)
            return self._previous_comments_cache[pr.number]
        comments: list[PreviousComment] = []
        allowed_users = self._get_allowed_users()

        for comment in pr.get_review_comments():
            if _is_resolved_reply(comment.body):
                continue

            user_login = comment.user.login

            if user_login not in allowed_users:
                continue

            parsed = self._parse_review_comment(comment)
            if parsed:
                comments.append(parsed)

        logger.info(f"Found {len(comments)} previous AI review comments")
        self._previous_comments_cache[pr.number] = comments
        if len(self._previous_comments_cache) > self._previous_comments_cache_max:
            self._previous_comments_cache.popitem(last=False)
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

        # Extract embedded hash for stable cross-run matching
        hash_match = re.search(r"<!-- ai-reviewer-id: ([a-f0-9]{12}) -->", body)
        finding_hash = hash_match.group(1) if hash_match else None

        return PreviousComment(
            id=comment.id,
            file_path=comment.path,
            line=comment.line or comment.original_line or 0,
            title=title,
            severity=severity,
            body=body,
            is_resolved=False,  # GitHub API doesn't expose this directly
            finding_hash=finding_hash,
        )

    def compute_review_delta(
        self,
        pr: PullRequest,
        current_findings: list[ConsolidatedFinding],
        review_count: int | None = None,
    ) -> ReviewDelta:
        """Compare current findings with previous comments to compute delta.

        Args:
            pr: Pull request object
            current_findings: Current review findings
            review_count: Accurate review count from metadata. When ``None``,
                falls back to ``estimate_review_count()`` heuristic.

        Returns:
            ReviewDelta showing new, fixed, and open issues
        """
        previous_comments = self.get_previous_review_comments(pr)
        delta = ReviewDelta(previous_comments=previous_comments)

        # Build three lookups: strict hash, fuzzy hash, and title-based (legacy fallback)
        hash_lookup: dict[str, PreviousComment] = {}
        fuzzy_lookup: dict[str, PreviousComment] = {}
        title_lookup: dict[tuple[str, int, str], PreviousComment] = {}
        for comment in previous_comments:
            if comment.finding_hash:
                hash_lookup[comment.finding_hash] = comment
            fuzzy = comment.finding_hash_fuzzy
            if fuzzy:
                fuzzy_lookup[fuzzy] = comment
            key = (comment.file_path, comment.line, self._normalize_title(comment.title))
            title_lookup[key] = comment

        # Track which previous comments are still open
        matched_previous: set[int] = set()

        effective_review_count = (
            review_count if review_count is not None else estimate_review_count(delta)
        )

        # Collect candidate new findings; fix-zone filtering happens after
        # fixed_findings are determined below.
        candidate_new: list[ConsolidatedFinding] = []

        for finding in current_findings:
            # Three-tier matching: strict hash → fuzzy hash → title+line
            matched_comment = hash_lookup.get(finding.finding_hash)
            if matched_comment is None:
                fuzzy_hash = finding.finding_hash_fuzzy
                if fuzzy_hash is not None:
                    matched_comment = fuzzy_lookup.get(fuzzy_hash)
            if matched_comment is None:
                key = (finding.file_path, finding.line_start, self._normalize_title(finding.title))
                matched_comment = title_lookup.get(key)

            if matched_comment is not None:
                prev_sev = _parse_severity(matched_comment.severity)
                if prev_sev is not None:
                    finding.severity = stabilize_severity(
                        finding.severity, prev_sev, effective_review_count
                    )
                delta.open_findings.append(finding)
                matched_previous.add(matched_comment.id)
            else:
                candidate_new.append(finding)

        # Determine which unmatched previous comments are likely fixed.
        # We mark as fixed when:
        # 1. The file is no longer in the diff (removed/renamed), OR
        # 2. The file was deleted (status=removed) - no patch for binary/large files, OR
        # 3. The commented line was modified AND the AI didn't find the issue again
        #
        # This avoids false "no longer detected" replies on unmodified code while
        # still detecting actual fixes when the relevant lines were changed.
        pr_files = list(pr.get_files())
        changed_files = {f.filename for f in pr_files}
        removed_files = {f.filename for f in pr_files if getattr(f, "status", None) == "removed"}

        file_modified_lines: dict[str, set[int]] = {}
        for f in pr_files:
            if f.patch:
                file_modified_lines[f.filename] = self._parse_modified_lines(f.patch)

        for comment in previous_comments:
            if comment.id not in matched_previous:
                file_path = comment.file_path

                if file_path not in changed_files:
                    delta.fixed_findings.append(comment)
                elif file_path in removed_files:
                    delta.fixed_findings.append(comment)
                elif file_path in file_modified_lines:
                    modified_lines = file_modified_lines[file_path]
                    if self._is_line_in_modified_range(comment.line, modified_lines):
                        delta.fixed_findings.append(comment)

        # Deduplicate by comment ID to prevent exponential growth on re-reviews
        seen: set[int] = set()
        delta.fixed_findings = [
            f
            for f in delta.fixed_findings
            if f.id not in seen and not seen.add(f.id)  # type: ignore[func-returns-value]
        ]

        # Build fix zones: file → set of line numbers (with tolerance) where a
        # previous finding was fixed.  New low-severity findings landing in these
        # zones are suppressed to break the noise loop.
        fix_zones: dict[str, set[int]] = defaultdict(set)
        fix_zone_tolerance = 3
        for fixed_comment in delta.fixed_findings:
            line = fixed_comment.line
            for offset in range(-fix_zone_tolerance, fix_zone_tolerance + 1):
                fix_zones[fixed_comment.file_path].add(line + offset)

        _SUPPRESSED_SEVERITIES = {Severity.SUGGESTION, Severity.NITPICK}

        for finding in candidate_new:
            if (
                finding.file_path in fix_zones
                and finding.line_start in fix_zones[finding.file_path]
                and finding.severity in _SUPPRESSED_SEVERITIES
            ):
                delta.suppressed_findings.append(finding)
            else:
                delta.new_findings.append(finding)

        if delta.suppressed_findings:
            logger.info(
                "Suppressed %d low-severity finding(s) on fix-zone lines",
                len(delta.suppressed_findings),
            )

        logger.info(
            f"Review delta: {len(delta.new_findings)} new, "
            f"{len(delta.fixed_findings)} fixed, "
            f"{len(delta.open_findings)} open, "
            f"{len(delta.suppressed_findings)} suppressed"
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
        return any(abs(line - mod_line) <= tolerance for mod_line in modified_lines)

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
            graphql_url = base[:-3] + "/graphql" if base.endswith("/api/v3") else f"{base}/graphql"
        else:
            graphql_url = "https://api.github.com/graphql"

        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

        payload: dict[str, object] = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            response = requests.post(graphql_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                logger.warning("GraphQL request returned errors (use DEBUG for details)")
                logger.debug("GraphQL errors: %s", result["errors"])
                return None

            return result.get("data")
        except Exception as e:
            _raise_if_forbidden(e)
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

        findings_to_process = delta.fixed_findings[:_MAX_RESOLVE_COMMENTS]
        if len(delta.fixed_findings) > _MAX_RESOLVE_COMMENTS:
            logger.warning(
                "Capping resolved comment processing at %d (have %d)",
                _MAX_RESOLVE_COMMENTS,
                len(delta.fixed_findings),
            )

        for fixed in findings_to_process:
            # Skip if we've already marked this as no longer detected
            # (avoid duplicate replies on re-review)
            if fixed.id in existing_replies:
                logger.debug(f"Comment {fixed.id} already has resolved reply, skipping")
                continue

            try:
                # Find the comment and reply to it
                comment = pr.get_review_comment(fixed.id)

                # Add reaction (may already exist, that's ok)
                with contextlib.suppress(Exception):
                    comment.create_reaction("hooray")  # 🎉 reaction

                # Post a reply indicating the issue was not re-detected.
                pr.create_review_comment_reply(
                    comment_id=fixed.id,
                    body=_NO_LONGER_DETECTED_REPLY,
                )

                # Hand-in-hand: also resolve the thread in GitHub UI (collapse the conversation).
                # Without this, the reply would show but the thread would stay "open".
                if not self._resolve_thread_for_comment(fixed.id, thread_mapping):
                    logger.warning(
                        f"Posted 'no longer detected' reply on comment {fixed.id} but could not "
                        "resolve the thread (GraphQL resolve failed or thread not found). "
                        "Thread may still appear open in the PR."
                    )

                resolved_count += 1
                logger.debug(f"Marked comment {fixed.id} as resolved")
                time.sleep(_RESOLVE_COMMENT_DELAY_S)
            except Exception as e:
                _raise_if_forbidden(e)
                logger.warning(f"Could not resolve comment {fixed.id}: {e}")

        return resolved_count

    def _get_resolved_comment_ids(
        self, pr: PullRequest, raw_comments: list | None = None
    ) -> set[int]:
        """Get IDs of comments that already have a "no longer detected" reply from us.

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
            # Check if this is our resolved / "no longer detected" reply with a parent
            if not _is_resolved_reply(comment.body):
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
