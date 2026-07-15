"""Tests for the Cloud Tasks review job queue and /process-review worker."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_reviewer.github import task_queue, webhook


@pytest.fixture
def client():
    return TestClient(webhook.create_webhook_app())


class TestEnqueuePath:
    """Webhook enqueues instead of running inline when the queue is enabled."""

    @pytest.mark.asyncio
    async def test_queue_disabled_runs_inline(self, monkeypatch):
        monkeypatch.delenv("TASK_QUEUE_PATH", raising=False)
        monkeypatch.delenv("TASK_TARGET_URL", raising=False)

        handler = AsyncMock()
        event = webhook.PREvent(repo="o/r", pr_number=7, action="opened", head_sha="deadbeef")
        with patch.object(webhook, "_review_handler", handler):
            await webhook.handle_pr_event(event)
        handler.assert_called_once_with(repo="o/r", pr_number=7)

    @pytest.mark.asyncio
    async def test_queue_enabled_enqueues_and_skips_inline(self, monkeypatch):
        monkeypatch.setenv("TASK_QUEUE_PATH", "projects/p/locations/l/queues/q")
        monkeypatch.setenv("TASK_TARGET_URL", "https://svc.example")

        handler = AsyncMock()
        enqueue = MagicMock(return_value="task-123")
        event = webhook.PREvent(repo="o/r", pr_number=7, action="synchronize", head_sha="abc123")
        with (
            patch.object(webhook, "_review_handler", handler),
            patch.object(webhook, "enqueue_review", enqueue),
        ):
            await webhook.handle_pr_event(event)

        handler.assert_not_called()
        enqueue.assert_called_once_with(
            {"repo": "o/r", "pr_number": 7, "head_sha": "abc123", "trigger": "pull_request"}
        )

    @pytest.mark.asyncio
    async def test_comment_trigger_enqueues_without_sha(self, monkeypatch):
        monkeypatch.setenv("TASK_QUEUE_PATH", "projects/p/locations/l/queues/q")
        monkeypatch.setenv("TASK_TARGET_URL", "https://svc.example")

        handler = AsyncMock()
        enqueue = MagicMock(return_value="task-9")
        payload = {
            "action": "created",
            "comment": {"body": "/ai-review"},
            "issue": {"number": 5, "pull_request": {"url": "https://..."}},
            "repository": {"full_name": "o/r"},
        }
        with (
            patch.object(webhook, "_review_handler", handler),
            patch.object(webhook, "enqueue_review", enqueue),
        ):
            await webhook._handle_issue_comment_event(payload)

        handler.assert_not_called()
        enqueue.assert_called_once_with(
            {"repo": "o/r", "pr_number": 5, "head_sha": "", "trigger": "comment"}
        )


class TestProcessReviewAuth:
    """X-Task-Auth gating on /process-review."""

    def _body(self):
        return {"repo": "o/r", "pr_number": 1, "head_sha": "", "trigger": "pull_request"}

    def test_401_when_token_unset(self, client, monkeypatch):
        monkeypatch.delenv("TASK_AUTH_TOKEN", raising=False)
        resp = client.post("/process-review", json=self._body())
        assert resp.status_code == 401

    def test_401_wrong_token(self, client, monkeypatch):
        monkeypatch.setenv("TASK_AUTH_TOKEN", "secret")
        resp = client.post("/process-review", json=self._body(), headers={"X-Task-Auth": "nope"})
        assert resp.status_code == 401

    def test_401_missing_header(self, client, monkeypatch):
        monkeypatch.setenv("TASK_AUTH_TOKEN", "secret")
        resp = client.post("/process-review", json=self._body())
        assert resp.status_code == 401


class TestProcessReviewFlow:
    """Review execution, retry, dead-letter, and idempotency on /process-review."""

    def _post(self, client, headers=None):
        body = {"repo": "o/r", "pr_number": 1, "head_sha": "sha1", "trigger": "pull_request"}
        h = {"X-Task-Auth": "secret"}
        if headers:
            h.update(headers)
        return client.post("/process-review", json=body, headers=h)

    def test_happy_path(self, client, monkeypatch):
        monkeypatch.setenv("TASK_AUTH_TOKEN", "secret")
        run = AsyncMock()
        with (
            patch.object(webhook, "run_review", run),
            patch.object(webhook, "_already_reviewed", return_value=False),
        ):
            resp = self._post(client)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        run.assert_called_once_with("o/r", 1)

    def test_retryable_failure_returns_500(self, client, monkeypatch):
        monkeypatch.setenv("TASK_AUTH_TOKEN", "secret")
        run = AsyncMock(side_effect=RuntimeError("boom"))
        with (
            patch.object(webhook, "run_review", run),
            patch.object(webhook, "_already_reviewed", return_value=False),
        ):
            resp = self._post(client, headers={"X-CloudTasks-TaskRetryCount": "0"})
        assert resp.status_code == 500

    def test_final_attempt_terminates_and_posts_notice(self, client, monkeypatch, caplog):
        monkeypatch.setenv("TASK_AUTH_TOKEN", "secret")
        monkeypatch.setenv("TASK_MAX_ATTEMPTS", "4")
        run = AsyncMock(side_effect=RuntimeError("boom"))
        notice = MagicMock()
        with (
            patch.object(webhook, "run_review", run),
            patch.object(webhook, "_already_reviewed", return_value=False),
            patch.object(webhook, "_post_review_failed_notice", notice),
            caplog.at_level("ERROR"),
        ):
            resp = self._post(client, headers={"X-CloudTasks-TaskRetryCount": "3"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "dead"
        notice.assert_called_once_with("o/r", 1)
        assert any("review-job-dead" in r.message for r in caplog.records)

    def test_duplicate_skips_review(self, client, monkeypatch):
        monkeypatch.setenv("TASK_AUTH_TOKEN", "secret")
        run = AsyncMock()
        with (
            patch.object(webhook, "run_review", run),
            patch.object(webhook, "_already_reviewed", return_value=True),
        ):
            resp = self._post(client)
        assert resp.status_code == 200
        assert resp.json()["status"] == "duplicate"
        run.assert_not_called()


class TestEnqueueReview:
    def test_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("TASK_QUEUE_PATH", raising=False)
        monkeypatch.delenv("TASK_TARGET_URL", raising=False)
        with pytest.raises(RuntimeError, match="TASK_QUEUE_PATH"):
            task_queue.enqueue_review({"repo": "o/r"})

    def test_missing_library_raises_clear_error(self, monkeypatch):
        monkeypatch.setenv("TASK_QUEUE_PATH", "projects/p/locations/l/queues/q")
        monkeypatch.setenv("TASK_TARGET_URL", "https://svc.example")
        # Force the lazy `from google.cloud import tasks_v2` to fail.
        monkeypatch.setitem(sys.modules, "google.cloud.tasks_v2", None)
        with pytest.raises(RuntimeError, match="google-cloud-tasks is not installed"):
            task_queue.enqueue_review({"repo": "o/r"})
