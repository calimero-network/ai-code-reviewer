"""GitHub webhook server for automatic PR reviews."""

import asyncio
import hashlib
import hmac
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
        webhook_secret: Optional GitHub webhook secret for signature verification

    Returns:
        FastAPI application
    """
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
        # Verify signature if secret is configured
        if webhook_secret:
            signature = request.headers.get("X-Hub-Signature-256", "")
            body = await request.body()

            if not verify_signature(body, signature, webhook_secret):
                raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse event
        event_type = request.headers.get("X-GitHub-Event", "")
        payload = await request.json()

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
