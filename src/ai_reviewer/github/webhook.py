"""GitHub webhook server for automatic PR reviews."""

import asyncio
import hashlib
import hmac
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, Request

logger = logging.getLogger(__name__)


@dataclass
class PREvent:
    """Represents a PR webhook event."""

    repo: str
    pr_number: int
    action: str
    sender: str = ""
    installation_id: int | None = None


# Review trigger - will be set by the application
_review_handler: Callable | None = None


def set_review_handler(handler: Callable) -> None:
    """Set the review handler function."""
    global _review_handler
    _review_handler = handler


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
            logger.error(f"Failed to get installation: {response.status_code} {response.text}")
            return None

        installation_id = response.json()["id"]

        # Generate installation access token
        response = requests.post(
            f"https://api.github.com/app/installations/{installation_id}/access_tokens",
            headers=headers,
            timeout=10,
        )
        if response.status_code != 201:
            logger.error(f"Failed to get access token: {response.status_code} {response.text}")
            return None

        return response.json()["token"]

    except Exception as e:
        logger.exception(f"Failed to generate GitHub App token: {e}")
        return None


def _setup_default_review_handler() -> None:
    """Set up the default review handler using environment config.

    This is used when running as a standalone server (e.g., Cloud Run)
    without the CLI's explicit handler setup.
    """
    import os

    from ai_reviewer.review import review_pr_with_cursor_agent
    from ai_reviewer.agents.cursor_client import CursorConfig
    from ai_reviewer.github.client import GitHubClient
    from ai_reviewer.github.formatter import GitHubFormatter

    async def default_review_handler(repo: str, pr_number: int) -> None:
        """Default review handler that reads config from environment."""
        cursor_api_key = os.environ.get("CURSOR_API_KEY")

        # Support both GitHub App and PAT authentication
        github_app_id = os.environ.get("GITHUB_APP_ID")
        github_app_private_key = os.environ.get("GITHUB_APP_PRIVATE_KEY")
        github_token = os.environ.get("GITHUB_TOKEN")

        # If GitHub App is configured, generate installation token
        if github_app_id and github_app_private_key:
            logger.info(f"Using GitHub App authentication for {repo}")
            github_token = _get_github_app_token(github_app_id, github_app_private_key, repo)
            if not github_token:
                logger.error("Failed to get GitHub App installation token")
                return

        if not cursor_api_key:
            logger.error("CURSOR_API_KEY not set")
            return
        if not github_token:
            logger.error("GITHUB_TOKEN not set")
            return

        cursor_config = CursorConfig(
            api_key=cursor_api_key,
            base_url=os.environ.get("CURSOR_BASE_URL", "https://api.cursor.com/v0"),
            timeout=int(os.environ.get("CURSOR_TIMEOUT", "300")),
        )

        try:
            review = await review_pr_with_cursor_agent(
                repo=repo,
                pr_number=pr_number,
                cursor_config=cursor_config,
                github_token=github_token,
                num_agents=int(os.environ.get("NUM_AGENTS", "3")),
            )

            if review.all_agents_failed:
                logger.error(f"All agents failed for {repo} PR #{pr_number}")
                return

            # Post review to GitHub
            gh = GitHubClient(github_token)
            pr = gh.get_pull_request(repo, pr_number)
            formatter = GitHubFormatter("AI Code Reviewer")

            delta = gh.compute_review_delta(pr, review.findings)
            new_findings = delta.new_findings if delta.previous_comments else review.findings
            use_compact_body = len(new_findings) > 0

            if delta.previous_comments:
                body = (
                    formatter.format_review_with_delta_compact(review, delta)
                    if use_compact_body
                    else formatter.format_review_with_delta(review, delta)
                )
                action = formatter.get_review_action_with_delta(review, delta, allow_approve=False)
            else:
                body = (
                    formatter.format_review_compact(review)
                    if use_compact_body
                    else formatter.format_review(review)
                )
                action = formatter.get_review_action(review, allow_approve=False)

            gh.post_review(pr, review, body, action)
            logger.info(f"Posted review to {repo} PR #{pr_number} ({action})")

            # Resolve fixed comments
            if delta.fixed_findings:
                resolved = gh.resolve_fixed_comments(pr, delta)
                logger.info(f"Resolved {resolved} comments")

            if new_findings:
                from ai_reviewer.models.review import ConsolidatedReview as CR

                new_only_review = CR(
                    id=review.id,
                    created_at=review.created_at,
                    repo=review.repo,
                    pr_number=review.pr_number,
                    findings=new_findings,
                    summary=review.summary,
                    agent_count=review.agent_count,
                    review_quality_score=review.review_quality_score,
                    total_review_time_ms=review.total_review_time_ms,
                )
                posted = gh.post_inline_comments(pr, new_only_review)
                logger.info(f"Posted {posted} inline comments")

        except Exception as e:
            logger.exception(f"Error reviewing {repo} PR #{pr_number}: {e}")

    set_review_handler(default_review_handler)


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

    if _review_handler:
        await _review_handler(repo=event.repo, pr_number=event.pr_number)
    else:
        logger.warning("No review handler configured")


async def review_pr(repo: str, pr_number: int) -> None:
    """Placeholder for review function - will be implemented with full app."""
    logger.info(f"Would review {repo} PR #{pr_number}")


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

    # Set up review handler from environment if not already set
    if _review_handler is None:
        _setup_default_review_handler()

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
            pr_event = PREvent(
                repo=payload["repository"]["full_name"],
                pr_number=payload["pull_request"]["number"],
                action=payload["action"],
                sender=payload.get("sender", {}).get("login", ""),
                installation_id=payload.get("installation", {}).get("id"),
            )

            # Process async to respond quickly
            asyncio.create_task(handle_pr_event(pr_event))

        elif event_type == "ping":
            logger.info("Received ping from GitHub")
            return {"status": "pong"}

        else:
            logger.debug(f"Ignoring event type: {event_type}")

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
