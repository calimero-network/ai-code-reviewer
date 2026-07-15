"""Cloud Tasks job queue for durable, retryable PR reviews.

This is the only module that imports ``google.cloud``. The import is lazy so the
package runs without the GCP SDK installed - unset queue env vars fall back to
inline reviews in the webhook.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def is_queue_enabled() -> bool:
    """True when both the queue path and this service's target URL are configured."""
    return bool(os.environ.get("TASK_QUEUE_PATH") and os.environ.get("TASK_TARGET_URL"))


def enqueue_review(payload: dict) -> str:
    """Create a Cloud Tasks HTTP task that POSTs ``payload`` to ``/process-review``.

    Returns the created task name. Raises RuntimeError if the queue env vars are
    unset or the google-cloud-tasks package is not installed.
    """
    queue_path = os.environ.get("TASK_QUEUE_PATH")
    target_url = os.environ.get("TASK_TARGET_URL")
    if not queue_path or not target_url:
        raise RuntimeError("TASK_QUEUE_PATH and TASK_TARGET_URL must be set to enqueue reviews")

    try:
        from google.cloud import tasks_v2
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-tasks is not installed; install the 'gcp' extra "
            "(pip install 'ai-code-reviewer[gcp]') to enable the task queue"
        ) from e

    headers = {"Content-Type": "application/json"}
    auth_token = os.environ.get("TASK_AUTH_TOKEN")
    if auth_token:
        headers["X-Task-Auth"] = auth_token

    task = {
        "http_request": {
            "http_method": tasks_v2.HttpMethod.POST,
            "url": f"{target_url.rstrip('/')}/process-review",
            "headers": headers,
            "body": json.dumps(payload).encode(),
        },
        # A review runs inside the /process-review request and can take 15+ min.
        # The default 600s dispatch deadline made Cloud Tasks abandon every
        # dispatch mid-review (504 -> retry from scratch -> dead-letter). 1800s
        # is the Cloud Tasks maximum; the Cloud Run service timeout must be
        # raised to match (gcloud run services update --timeout=1800).
        "dispatch_deadline": {"seconds": 1800},
    }

    client = tasks_v2.CloudTasksClient()
    response = client.create_task(parent=queue_path, task=task)
    return response.name
