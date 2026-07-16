"""GitHub webhook server for automatic PR reviews."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request

from ai_reviewer.github.task_queue import enqueue_review, is_queue_enabled

logger = logging.getLogger(__name__)

_handler_lock = threading.Lock()


def _log_task_error(task: asyncio.Task) -> None:
    """Done-callback that logs exceptions from fire-and-forget async tasks."""
    if not task.cancelled():
        exc = task.exception()
        if exc is not None:
            logger.error("Background task %r failed: %s", task.get_name(), exc)


def _get_env_int(key: str, default: int) -> int:
    """Parse env var as int; on ValueError log and return default."""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning("%s invalid, using default %s", key, default)
        return default


def _get_env_float(key: str, default: float) -> float:
    """Parse env var as float; on ValueError log and return default."""
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning("%s invalid, using default %s", key, default)
        return default


@dataclass
class PREvent:
    """Represents a PR webhook event."""

    repo: str
    pr_number: int
    action: str
    sender: str = ""
    installation_id: int | None = None
    head_sha: str = ""


# Cloud Run degrades and drops streaming connections when too many reviews run at
# once (concurrency, not size, triggered the "Review Incomplete" storms). Cap
# simultaneous reviews so a webhook burst queues instead of overloading the App.
# Lazily created so MAX_CONCURRENT_REVIEWS is read after the process env is set.
_review_semaphore: asyncio.Semaphore | None = None


def _get_review_semaphore() -> asyncio.Semaphore:
    global _review_semaphore
    if _review_semaphore is None:
        _review_semaphore = asyncio.Semaphore(_get_env_int("MAX_CONCURRENT_REVIEWS", 2))
    return _review_semaphore


# Review trigger - will be set by the application
_review_handler: Callable | None = None
# Push trigger for doc-update-on-merge - will be set by the application
_push_handler: Callable | None = None


def set_review_handler(handler: Callable) -> None:
    """Set the review handler function."""
    global _review_handler
    with _handler_lock:
        _review_handler = handler


def set_push_handler(handler: Callable) -> None:
    """Set the push handler used to trigger doc updates on merge."""
    global _push_handler
    with _handler_lock:
        _push_handler = handler


def _get_github_app_token(app_id: str, private_key: str, repo: str) -> str | None:
    """Generate an installation access token for a GitHub App.

    Args:
        app_id: GitHub App ID
        private_key: GitHub App private key (PEM format)
        repo: Repository in "owner/name" format

    Returns:
        Installation access token or None on failure
    """
    import time

    import jwt
    import requests

    try:
        # Generate JWT for GitHub App
        now = int(time.time())
        payload = {
            "iat": now - 60,  # Issued 60 seconds ago (clock drift)
            "exp": now + 600,  # Expires in 10 minutes
            "iss": app_id,
        }
        token = jwt.encode(payload, private_key, algorithm="RS256")

        # Get installation ID for the repo
        owner = repo.split("/")[0]
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

        # Try to get installation for the repo owner (org or user)
        response = requests.get(
            f"https://api.github.com/orgs/{owner}/installation",
            headers=headers,
            timeout=10,
        )
        if response.status_code == 404:
            # Try user installation
            response = requests.get(
                f"https://api.github.com/users/{owner}/installation",
                headers=headers,
                timeout=10,
            )

        if response.status_code != 200:
            # Truncate body to avoid logging sensitive auth details
            logger.error("Failed to get installation: status=%d", response.status_code)
            return None

        installation_id = response.json()["id"]

        # Generate installation access token
        response = requests.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens",
            headers=headers,
            timeout=10,
        )
        if response.status_code != 201:
            logger.error("Failed to get access token: status=%d", response.status_code)
            return None

        return response.json()["token"]

    except Exception as e:
        logger.exception(f"Failed to generate GitHub App token: {e}")
        return None


def _resolve_github_token(repo: str) -> str | None:
    """Resolve a GitHub token via App installation or PAT, preferring the App."""
    github_app_id = os.environ.get("GITHUB_APP_ID")
    github_app_private_key = os.environ.get("GITHUB_APP_PRIVATE_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")

    if github_app_id and github_app_private_key:
        logger.info(f"Using GitHub App authentication for {repo}")
        app_token = _get_github_app_token(github_app_id, github_app_private_key, repo)
        if app_token is None and github_token:
            logger.warning(
                "GitHub App token minting failed for %s; falling back to GITHUB_TOKEN", repo
            )
        # Keep the PAT as a fallback so a transient App-auth failure does not
        # silently disable reviews when a valid GITHUB_TOKEN is also configured.
        github_token = app_token or github_token

    return github_token


def _enqueue_review_job(repo: str, pr_number: int, head_sha: str, trigger: str) -> None:
    """Build the job payload and enqueue it on Cloud Tasks."""
    task_name = enqueue_review(
        {"repo": repo, "pr_number": pr_number, "head_sha": head_sha, "trigger": trigger}
    )
    logger.info("enqueued review job %s", task_name)


def _task_retry_count(request: Request) -> int:
    """Read the 0-based Cloud Tasks retry count from either header spelling."""
    for header in ("X-CloudTasks-TaskRetryCount", "X-Cloud-Tasks-Task-Retry-Count"):
        value = request.headers.get(header)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return 0
    return 0


def _already_reviewed(repo: str, pr_number: int, head_sha: str) -> bool:
    """True when the latest posted review metadata already matches head_sha."""
    from ai_reviewer.github.client import GitHubClient

    token = _resolve_github_token(repo)
    if not token:
        return False
    try:
        gh = GitHubClient(token)
        pr = gh.get_pull_request(repo, pr_number)
        meta = gh.get_review_metadata(pr)
    except Exception as e:
        logger.warning("Idempotency check failed for %s PR #%d: %s", repo, pr_number, e)
        return False
    return meta is not None and meta.commit_sha == head_sha


def _post_review_failed_notice(repo: str, pr_number: int) -> None:
    """Post a visible 'review could not complete' comment after the final attempt fails."""
    from ai_reviewer.github.client import GitHubClient

    token = _resolve_github_token(repo)
    if not token:
        logger.error("Cannot post failure notice for %s PR #%d - no GitHub token", repo, pr_number)
        return
    body = "\n".join(
        [
            "## 🤖 AI Code Reviewer",
            "",
            "### ⚠️ Review could not complete",
            "",
            "Review could not complete after multiple attempts (infrastructure error). "
            "No code has been reviewed. Re-trigger with `/ai-review` or by reopening the PR.",
            "",
        ]
    )
    try:
        gh = GitHubClient(token)
        pr = gh.get_pull_request(repo, pr_number)
        gh.post_review(pr, body, "COMMENT")
        logger.info("Posted 'review could not complete' notice to %s PR #%d", repo, pr_number)
    except Exception as e:
        logger.exception("Failed to post review-could-not-complete notice: %s", e)


async def run_review(repo: str, pr_number: int) -> None:
    """Run the full review flow for a PR, reading config from the environment.

    Raises on infrastructure failure so callers (the Cloud Tasks worker) can
    decide whether to retry. The inline webhook path wraps this and swallows
    exceptions for best-effort fire-and-forget behavior.
    """
    from ai_reviewer.cli import _run_doc_review
    from ai_reviewer.config import AnthropicApiConfig, Config, GitHubConfig, load_config
    from ai_reviewer.github.client import (
        GitHubClient,
        ReviewMeta,
        estimate_review_count,
        is_convergence_all_clear,
        lgtm_placeholder_review,
        should_skip_before_agents,
        should_skip_review,
    )
    from ai_reviewer.github.formatter import GitHubFormatter
    from ai_reviewer.review import review_pr

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    github_token = _resolve_github_token(repo)

    # Raise (not return) so a missing credential propagates to the Cloud Tasks
    # worker's retry/dead-letter path — a silent return would let /process-review
    # mark the job "ok" and drop it, recreating the exact total-silence outage
    # this PR fixes.
    if not anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    if not github_token:
        raise RuntimeError("GITHUB_TOKEN not set")

    anthropic_timeout = _get_env_int("ANTHROPIC_TIMEOUT", 600)
    num_agents = _get_env_int("NUM_AGENTS", 3)
    min_agreement = _get_env_float("MIN_VALIDATION_AGREEMENT", 2 / 3)

    anthropic_cfg = AnthropicApiConfig(
        api_key=anthropic_api_key,
        base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        timeout_seconds=anthropic_timeout,
    )

    enable_cross_review = os.environ.get("ENABLE_CROSS_REVIEW", "true").lower() != "false"

    try:
        webhook_config = load_config()
    except Exception as e:
        logger.warning("Failed to load config file, using defaults: %s", e)
        webhook_config = None

    gh = GitHubClient(github_token)
    pr = gh.get_pull_request(repo, pr_number)
    formatter = GitHubFormatter("AI Code Reviewer")

    force_review = any(label.name.lower() == "force-review" for label in pr.get_labels())

    meta = gh.get_review_metadata(pr)
    current_sha = pr.head.sha

    diff_files = {f.filename for f in pr.get_files()}
    previous_comments = gh.get_previous_review_comments(pr) if meta else []
    # Dismissal ledger only matters once a prior review exists to have been resolved.
    dismissed = gh.get_dismissed_findings(pr) if meta else []
    skip_reason = should_skip_before_agents(
        meta,
        current_sha,
        force_review,
        diff_files=diff_files,
        previous_comments=previous_comments,
    )
    if skip_reason is not None:
        logger.info(
            "Pre-agent skip for %s PR #%d: %s (sha=%s)",
            repo,
            pr_number,
            skip_reason.value,
            current_sha[:8],
        )
        return

    recheck_review = None
    if meta is not None and not force_review:
        lgtm_delta = gh.check_lgtm_fast_path(pr, meta)
        if lgtm_delta is not None:
            logger.info(
                "LGTM candidate for %s PR #%d — running 1-agent re-check",
                repo,
                pr_number,
            )
            try:
                # Bound by the same concurrency cap as the full review: the
                # recheck is also a streaming Anthropic call and must not
                # oversaturate connections under an LGTM-eligible burst.
                async with _get_review_semaphore():
                    recheck_review = await review_pr(
                        repo=repo,
                        pr_number=pr_number,
                        anthropic_cfg=anthropic_cfg,
                        github_token=github_token,
                        num_agents=1,
                        enable_cross_review=False,
                        min_validation_agreement=min_agreement,
                        config=webhook_config,
                    )
            except Exception as e:
                logger.warning(
                    "LGTM re-check failed for %s PR #%d; falling back to normal review: %s",
                    repo,
                    pr_number,
                    e,
                )

            if (
                recheck_review is not None
                and not recheck_review.findings
                and not recheck_review.all_agents_failed
            ):
                lgtm_review_count = meta.review_count + 1
                new_meta = ReviewMeta.build(
                    commit_sha=current_sha,
                    review_count=lgtm_review_count,
                    finding_hashes=[],
                )
                lgtm_review = lgtm_placeholder_review(repo, pr_number)
                body = formatter.format_review_with_delta_compact(
                    lgtm_review, lgtm_delta, meta=new_meta
                )
                gh.post_review(pr, body, "COMMENT")
                if lgtm_delta.fixed_findings:
                    resolved = gh.resolve_fixed_comments(pr, lgtm_delta)
                    logger.info(f"LGTM: resolved {resolved} comments")
                logger.info(
                    "LGTM for %s PR #%d — verified clean by re-check",
                    repo,
                    pr_number,
                )
                return

            if recheck_review is not None:
                logger.info(
                    "Re-check found %d issue(s) for %s PR #%d — falling back to normal flow",
                    len(recheck_review.findings),
                    repo,
                    pr_number,
                )

    semaphore = _get_review_semaphore()
    if semaphore.locked():
        logger.info(
            "Review for %s PR #%d queued behind %d in-flight review(s)",
            repo,
            pr_number,
            _get_env_int("MAX_CONCURRENT_REVIEWS", 2),
        )
    async with semaphore:
        review = await review_pr(
            repo=repo,
            pr_number=pr_number,
            anthropic_cfg=anthropic_cfg,
            github_token=github_token,
            num_agents=num_agents,
            enable_cross_review=enable_cross_review,
            min_validation_agreement=min_agreement,
            config=webhook_config,
            dismissed=dismissed,
        )

    # all_agents_failed is a terminal, non-retryable outcome: we ran, every agent
    # produced a failure result, and we post an honest visible notice right here.
    # Deliberately return (not raise) so the Cloud Tasks worker does not retry and
    # the inline handler still surfaces the notice — retry/dead-letter is reserved
    # for unexpected exceptions that posted nothing.
    if review.all_agents_failed:
        logger.error(f"All agents failed for {repo} PR #{pr_number}")
        gh.post_review(pr, formatter.format_all_agents_failed(review), "COMMENT")
        logger.info("Posted 'review could not complete' notice to %s PR #%d", repo, pr_number)
        return

    meta_review_count = (meta.review_count + 1) if meta is not None else None
    delta = gh.compute_review_delta(pr, review.findings, review_count=meta_review_count)

    review_count: int
    if meta_review_count is not None:
        review_count = meta_review_count
    else:
        review_count = estimate_review_count(delta)

    if delta.previous_comments and not force_review and should_skip_review(review_count, delta):
        logger.info(
            "Convergence detected for %s PR #%d — skipping post "
            "(review_count=%d, open=%d, new=%d, fixed=%d)",
            repo,
            pr_number,
            review_count,
            len(delta.open_findings),
            len(delta.new_findings),
            len(delta.fixed_findings),
        )
        return

    finding_hashes = [f.finding_hash for f in review.findings]
    new_meta = ReviewMeta.build(
        commit_sha=current_sha,
        review_count=review_count,
        finding_hashes=finding_hashes,
    )

    max_total = _get_env_int("MAX_TOTAL_FINDINGS", 50)
    max_per_file = _get_env_int("MAX_FINDINGS_PER_FILE", 10)

    if is_convergence_all_clear(review, delta):
        # Explicit convergence verdict; never APPROVE unless policy opts in.
        body = formatter.format_all_clear(review, delta, meta=new_meta)
        auto_approve = bool(
            webhook_config
            and getattr(webhook_config, "review_policy", None)
            and webhook_config.review_policy.auto_approve_if_no_findings
        )
        action = "APPROVE" if (auto_approve and not review.failed_agents) else "COMMENT"
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
                    review,
                    delta,
                    meta=new_meta,
                    inline_new_findings=postable_inline_findings,
                )
                if use_compact_body
                else formatter.format_review_with_delta(review, delta, meta=new_meta)
            )
            action = formatter.get_review_action_with_delta(review, delta, allow_approve=False)
        else:
            body = (
                formatter.format_review_compact(
                    review,
                    meta=new_meta,
                    inline_findings=postable_inline_findings,
                )
                if use_compact_body
                else formatter.format_review(review, meta=new_meta)
            )
            action = formatter.get_review_action(review, allow_approve=False)

    posted = gh.post_review(
        pr,
        body,
        action,
        inline_findings=postable_inline_findings or None,
    )
    logger.info(
        "Posted review to %s PR #%d (%s, %d inline comments)",
        repo,
        pr_number,
        action,
        posted,
    )

    if delta.fixed_findings:
        resolved = gh.resolve_fixed_comments(pr, delta)
        logger.info(f"Resolved {resolved} comments")

    _run_doc_review(
        gh=gh,
        pr=pr,
        repo=repo,
        config=webhook_config or Config(anthropic=None, github=GitHubConfig(token=""), agents=[]),
        doc_check=None,
        dry_run=False,
    )


def _setup_default_review_handler() -> Callable:
    """Return the inline review handler used when the task queue is disabled.

    Wraps run_review and swallows exceptions for best-effort fire-and-forget
    behavior (Cloud Run standalone / local mode).
    """

    async def default_review_handler(repo: str, pr_number: int) -> None:
        try:
            await run_review(repo, pr_number)
        except Exception as e:
            logger.exception(f"Error reviewing {repo} PR #{pr_number}: {e}")

    return default_review_handler


def _setup_default_push_handler() -> Callable:
    """Build and return the default push handler that runs update-docs on merges to main/master."""
    from ai_reviewer.cli import _update_docs_async

    async def default_push_handler(repo: str, ref: str, head_commit_message: str) -> None:
        branch = ref.removeprefix("refs/heads/")
        if branch not in ("main", "master"):
            logger.debug("Push to %s — not main/master, skipping doc update", branch)
            return

        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            logger.error("GITHUB_TOKEN not set — cannot run update-docs")
            return

        # Extract merged PR number from commit message ("Merge pull request #123")
        import re

        match = re.search(r"pull request #(\d+)", head_commit_message)
        if not match:
            logger.info("Push to %s — no merged PR number in commit message, skipping", branch)
            return

        pr_number = int(match.group(1))
        logger.info("Running update-docs for %s PR #%d (push to %s)", repo, pr_number, branch)
        try:
            await _update_docs_async(
                repo=repo,
                pr_number=pr_number,
                dry_run=False,
                base=branch,
                config_path=None,
            )
        except Exception as e:
            logger.exception("update-docs failed for %s PR #%d: %s", repo, pr_number, e)

    return default_push_handler


async def handle_push_event(payload: dict) -> None:
    """Handle a push event by running update-docs when a PR merges to main/master."""
    ref = payload.get("ref", "")
    repo = payload.get("repository", {}).get("full_name", "")
    head_commit = payload.get("head_commit") or {}
    head_commit_message = head_commit.get("message", "") if isinstance(head_commit, dict) else ""

    if _push_handler:
        await _push_handler(repo=repo, ref=ref, head_commit_message=head_commit_message)
    else:
        logger.warning("No push handler configured")


async def handle_pr_event(event: PREvent) -> None:
    """Handle a PR event by triggering a review if appropriate.

    Args:
        event: PR event data
    """
    # Only review on these actions
    trigger_actions = {"opened", "synchronize", "reopened"}

    if event.action not in trigger_actions:
        logger.debug(f"Ignoring PR action: {event.action}")
        return

    logger.info(f"Triggering review for {event.repo} PR #{event.pr_number}")

    if is_queue_enabled():
        _enqueue_review_job(event.repo, event.pr_number, event.head_sha, "pull_request")
        return

    if _review_handler:
        await _review_handler(repo=event.repo, pr_number=event.pr_number)
    else:
        logger.warning("No review handler configured")


async def _handle_issue_comment_event(payload: dict) -> None:
    """Handle issue_comment webhook events — trigger review on /ai-review command."""
    if payload.get("action") != "created":
        return

    comment_body = payload.get("comment", {}).get("body", "").strip()
    if not comment_body.startswith("/ai-review"):
        return

    # Only valid on PR comments (not plain issues)
    if "pull_request" not in payload.get("issue", {}):
        logger.debug("Ignoring /ai-review on non-PR issue")
        return

    repo = payload.get("repository", {}).get("full_name", "")
    pr_number = payload["issue"]["number"]

    logger.info(f"Triggering review via /ai-review comment on {repo} PR #{pr_number}")

    if is_queue_enabled():
        # Comment payloads carry no head sha; /process-review resolves and
        # dedups against the PR's current head instead.
        _enqueue_review_job(repo, pr_number, "", "comment")
        return

    if _review_handler:
        await _review_handler(repo=repo, pr_number=pr_number)
    else:
        logger.warning("No review handler configured for /ai-review trigger")


def create_webhook_app(webhook_secret: str | None = None) -> FastAPI:
    """Create the FastAPI webhook application.

    Args:
        webhook_secret: Optional GitHub webhook secret for signature verification.
                       If not provided, reads from GITHUB_WEBHOOK_SECRET env var.

    Returns:
        FastAPI application
    """
    import os

    # Allow webhook secret from env var (for Cloud Run / container deployments)
    if webhook_secret is None:
        webhook_secret = os.environ.get("GITHUB_WEBHOOK_SECRET")

    # Set up review and push handlers from environment if not already set.
    # Lock prevents a race if create_webhook_app is called concurrently (e.g. in tests).
    with _handler_lock:
        global _review_handler, _push_handler
        if _review_handler is None:
            _review_handler = _setup_default_review_handler()
        if _push_handler is None:
            _push_handler = _setup_default_push_handler()

    app = FastAPI(
        title="AI Code Reviewer Webhook",
        description="Webhook server for AI-powered code reviews",
        version="0.1.0",
    )

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "ai-code-reviewer"}

    @app.post("/webhook")
    async def github_webhook(request: Request):
        """Handle GitHub webhook events."""
        # Read body once (required for signature verification and parsing)
        body = await request.body()

        if webhook_secret:
            signature = request.headers.get("X-Hub-Signature-256", "")
            if not verify_signature(body, signature, webhook_secret):
                raise HTTPException(status_code=401, detail="Invalid signature")

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

        event_type = request.headers.get("X-GitHub-Event", "")

        if event_type == "pull_request":
            try:
                pr_event = PREvent(
                    repo=payload["repository"]["full_name"],
                    pr_number=payload["pull_request"]["number"],
                    action=payload["action"],
                    sender=payload.get("sender", {}).get("login", ""),
                    installation_id=payload.get("installation", {}).get("id"),
                    head_sha=payload["pull_request"].get("head", {}).get("sha", ""),
                )
            except KeyError as e:
                raise HTTPException(
                    status_code=400, detail=f"Malformed pull_request payload: missing {e}"
                ) from e

            # Process async to respond quickly; log exceptions so they aren't silently lost
            asyncio.create_task(handle_pr_event(pr_event)).add_done_callback(_log_task_error)

        elif event_type == "issue_comment":
            asyncio.create_task(_handle_issue_comment_event(payload)).add_done_callback(
                _log_task_error
            )

        elif event_type == "push":
            asyncio.create_task(handle_push_event(payload)).add_done_callback(_log_task_error)

        elif event_type == "ping":
            logger.info("Received ping from GitHub")
            return {"status": "pong"}

        else:
            logger.debug(f"Ignoring event type: {event_type}")

        return {"status": "ok"}

    @app.post("/process-review")
    async def process_review(request: Request):
        """Cloud Tasks worker: run one durable review job.

        Auth via X-Task-Auth (fail closed if TASK_AUTH_TOKEN unset). Dedups on
        head_sha, retries transient failures via HTTP 500, and terminates the
        task on the final attempt after posting a visible failure notice.
        """
        auth_token = os.environ.get("TASK_AUTH_TOKEN")
        provided = request.headers.get("X-Task-Auth", "")
        if not auth_token or not hmac.compare_digest(provided, auth_token):
            raise HTTPException(status_code=401, detail="Invalid task auth")

        body = await request.body()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

        repo = payload.get("repo", "")
        pr_number = payload.get("pr_number")
        head_sha = payload.get("head_sha", "")
        if not repo or pr_number is None:
            raise HTTPException(status_code=400, detail="Missing repo or pr_number")

        if head_sha and _already_reviewed(repo, pr_number, head_sha):
            logger.info(
                "Review already posted for %s PR #%d at %s - duplicate task",
                repo,
                pr_number,
                head_sha[:8],
            )
            return {"status": "duplicate"}

        retry_count = _task_retry_count(request)
        max_attempts = _get_env_int("TASK_MAX_ATTEMPTS", 4)
        try:
            await run_review(repo, pr_number)
        except Exception as e:
            if retry_count < max_attempts - 1:
                logger.error(
                    "Review job failed for %s PR #%d (attempt %d/%d): %s",
                    repo,
                    pr_number,
                    retry_count + 1,
                    max_attempts,
                    e,
                )
                raise HTTPException(status_code=500, detail="review failed, will retry") from e
            # Final attempt: a raised exception means run_review posted nothing,
            # so it is safe to post the visible failure notice without double-posting.
            logger.error(
                "review-job-dead repo=%s pr=%s sha=%s error=%s", repo, pr_number, head_sha, e
            )
            _post_review_failed_notice(repo, pr_number)
            return {"status": "dead"}

        return {"status": "ok"}

    @app.get("/")
    async def root():
        """Root endpoint with service info."""
        return {
            "service": "AI Code Reviewer",
            "version": "0.1.0",
            "endpoints": {
                "health": "/health",
                "webhook": "/webhook",
            },
        }

    return app


def verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature.

    Args:
        payload: Raw request body
        signature: X-Hub-Signature-256 header value
        secret: Webhook secret

    Returns:
        True if signature is valid
    """
    if not signature.startswith("sha256="):
        return False

    expected = (
        "sha256="
        + hmac.new(
            secret.encode(),
            payload,
            hashlib.sha256,
        ).hexdigest()
    )

    return hmac.compare_digest(expected, signature)
