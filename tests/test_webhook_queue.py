"""Tests for the Cloud Tasks review job queue and /process-review worker."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ai_reviewer.github import task_queue, webhook


@pytest.fixture
def client():
    return TestClient(webhook.create_webhook_app())


def _comment_payload(association="OWNER"):
    return {
        "action": "created",
        "comment": {
            "body": "/ai-review",
            "id": 42,
            "author_association": association,
            "user": {"login": "someone"},
        },
        "issue": {"number": 5, "pull_request": {"url": "https://..."}},
        "repository": {"full_name": "o/r"},
    }


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
        with (
            patch.object(webhook, "_review_handler", handler),
            patch.object(webhook, "enqueue_review", enqueue),
            patch.object(webhook, "_ack_comment"),
        ):
            await webhook._handle_issue_comment_event(_comment_payload())

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
        run.assert_called_once_with("o/r", 1, retry_transient=True)

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


def _fake_review(*, failed_agents, agent_count, transient, findings=None):
    from datetime import datetime

    from ai_reviewer.models.review import ConsolidatedReview

    return ConsolidatedReview(
        id="r1",
        created_at=datetime.now(),
        repo="o/r",
        pr_number=1,
        findings=findings or [],
        summary="s",
        agent_count=agent_count,
        review_quality_score=0.5,
        total_review_time_ms=1,
        failed_agents=failed_agents,
        transient_failure=transient,
    )


class TestTransientPartialRetry:
    """A partial caused purely by transient infra must retry the whole review
    instead of posting "Review Incomplete" - and post nothing while it retries."""

    @pytest.fixture
    def gh(self, monkeypatch):
        from ai_reviewer.github.client import ReviewDelta

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
        monkeypatch.delenv("GITHUB_APP_ID", raising=False)
        gh = MagicMock()
        gh.get_review_metadata.return_value = None
        gh.compute_review_delta.return_value = ReviewDelta()
        gh.get_pull_request.return_value.get_labels.return_value = []
        gh.get_pull_request.return_value.get_files.return_value = []
        gh.get_pull_request.return_value.head.sha = "sha1"
        return gh

    @pytest.fixture
    def run(self, gh):
        """Run webhook.run_review with review_pr stubbed to a given result."""

        async def _run(review, *, retry_transient):
            with (
                patch("ai_reviewer.github.client.GitHubClient", return_value=gh),
                patch("ai_reviewer.github.client.should_skip_before_agents", return_value=None),
                patch("ai_reviewer.review.review_pr", AsyncMock(return_value=review)),
                patch("ai_reviewer.cli._run_doc_review"),
            ):
                await webhook.run_review("o/r", 1, retry_transient=retry_transient)

        return _run

    @pytest.mark.asyncio
    async def test_transient_partial_raises_and_posts_nothing(self, gh, run):
        review = _fake_review(failed_agents=["logic"], agent_count=3, transient=True)
        with pytest.raises(webhook.TransientInfraFailure):
            await run(review, retry_transient=True)
        gh.post_review.assert_not_called()

    @pytest.mark.asyncio
    async def test_final_attempt_posts_honest_notice(self, gh, run):
        """retry_transient=False (the final attempt) keeps today's behavior: the
        honest incomplete notice is posted, never silence."""
        review = _fake_review(failed_agents=["a", "b"], agent_count=2, transient=True)
        await run(review, retry_transient=False)
        gh.post_review.assert_called_once()
        assert "could not complete" in gh.post_review.call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_non_transient_partial_posts_immediately(self, gh, run):
        """A real bug is not worth retrying - post the partial even on attempt 1."""
        review = _fake_review(failed_agents=["a", "b"], agent_count=2, transient=False)
        await run(review, retry_transient=True)
        gh.post_review.assert_called_once()

    @pytest.mark.asyncio
    async def test_clean_review_posts_and_never_raises(self, gh, run):
        review = _fake_review(failed_agents=[], agent_count=3, transient=False)
        await run(review, retry_transient=True)
        gh.post_review.assert_called_once()


class TestProcessReviewTransient:
    """Wiring of the transient retry into the Cloud Tasks worker."""

    def _post(self, client, retry_count):
        return client.post(
            "/process-review",
            json={"repo": "o/r", "pr_number": 1, "head_sha": "sha1", "trigger": "pull_request"},
            headers={"X-Task-Auth": "secret", "X-CloudTasks-TaskRetryCount": str(retry_count)},
        )

    def test_transient_non_final_returns_500_and_posts_nothing(self, client, monkeypatch, caplog):
        monkeypatch.setenv("TASK_AUTH_TOKEN", "secret")
        monkeypatch.setenv("TASK_MAX_ATTEMPTS", "4")
        run = AsyncMock(side_effect=webhook.TransientInfraFailure("529"))
        notice = MagicMock()
        with (
            patch.object(webhook, "run_review", run),
            patch.object(webhook, "_already_reviewed", return_value=False),
            patch.object(webhook, "_post_review_failed_notice", notice),
            caplog.at_level("WARNING"),
        ):
            resp = self._post(client, retry_count=1)

        assert resp.status_code == 500
        notice.assert_not_called()
        assert run.await_args.kwargs["retry_transient"] is True
        assert any("Cloud Tasks retries the whole review" in r.message for r in caplog.records)
        assert any("attempt 2/4" in r.getMessage() for r in caplog.records)

    def test_final_attempt_does_not_defer(self, client, monkeypatch):
        """On the last attempt run_review posts the honest partial itself, so the
        worker must not ask it to defer."""
        monkeypatch.setenv("TASK_AUTH_TOKEN", "secret")
        monkeypatch.setenv("TASK_MAX_ATTEMPTS", "4")
        run = AsyncMock()
        with (
            patch.object(webhook, "run_review", run),
            patch.object(webhook, "_already_reviewed", return_value=False),
        ):
            resp = self._post(client, retry_count=3)

        assert resp.status_code == 200
        assert run.await_args.kwargs["retry_transient"] is False


class TestCommentTriggerGate:
    """/ai-review burns a full review, so only write-access users may fire it."""

    async def _handle(self, payload, handler):
        with (
            patch.object(webhook, "_review_handler", handler),
            patch.object(webhook, "is_queue_enabled", return_value=False),
            patch.object(webhook, "_ack_comment") as ack,
        ):
            await webhook._handle_issue_comment_event(payload)
        return ack

    @pytest.mark.parametrize("association", ["OWNER", "MEMBER", "COLLABORATOR"])
    @pytest.mark.asyncio
    async def test_write_access_triggers(self, association):
        handler = AsyncMock()
        await self._handle(_comment_payload(association), handler)
        handler.assert_called_once_with(repo="o/r", pr_number=5)

    @pytest.mark.parametrize("association", ["CONTRIBUTOR", "NONE", "FIRST_TIME_CONTRIBUTOR", ""])
    @pytest.mark.asyncio
    async def test_read_only_association_ignored(self, association):
        handler = AsyncMock()
        ack = await self._handle(_comment_payload(association), handler)
        handler.assert_not_called()
        ack.assert_not_called()

    @pytest.mark.asyncio
    async def test_accepted_trigger_is_acknowledged(self):
        ack = await self._handle(_comment_payload(), AsyncMock())
        ack.assert_called_once_with("o/r", 5, 42)

    @pytest.mark.asyncio
    async def test_reaction_failure_does_not_break_trigger(self, monkeypatch):
        """The 👀 is a courtesy - a failing reaction API must not lose the review."""
        gh = MagicMock()
        gh.add_issue_comment_reaction.side_effect = RuntimeError("403")
        monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
        monkeypatch.delenv("GITHUB_APP_ID", raising=False)
        handler = AsyncMock()
        with (
            patch.object(webhook, "_review_handler", handler),
            patch.object(webhook, "is_queue_enabled", return_value=False),
            patch("ai_reviewer.github.client.GitHubClient", return_value=gh),
        ):
            await webhook._handle_issue_comment_event(_comment_payload())

        gh.add_issue_comment_reaction.assert_called_once_with("o/r", 5, 42, "eyes")
        handler.assert_called_once_with(repo="o/r", pr_number=5)
