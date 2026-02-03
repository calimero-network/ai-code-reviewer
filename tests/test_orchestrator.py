"""Tests for the agent orchestrator."""

import asyncio
from datetime import datetime
from unittest.mock import MagicMock

import pytest


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator."""

    @pytest.mark.asyncio
    async def test_parallel_execution(self, sample_vulnerable_diff, mock_review_context):
        """Test that agents are executed in parallel."""
        from ai_reviewer.models.review import AgentReview
        from ai_reviewer.orchestrator.orchestrator import AgentOrchestrator

        # Create mock agents
        mock_agent_1 = MagicMock()
        mock_agent_1.agent_id = "agent-1"
        mock_agent_1.focus_areas = ["security"]

        mock_agent_2 = MagicMock()
        mock_agent_2.agent_id = "agent-2"
        mock_agent_2.focus_areas = ["performance"]

        # Track execution times to verify parallelism
        execution_times = []

        async def mock_review_1(*_args, **_kwargs):
            start = datetime.now()
            await asyncio.sleep(0.1)  # Simulate work
            execution_times.append(("agent-1", start, datetime.now()))
            return AgentReview(
                agent_id="agent-1",
                agent_type="claude",
                focus_areas=["security"],
                findings=[],
                summary="No issues",
                review_time_ms=100,
            )

        async def mock_review_2(*_args, **_kwargs):
            start = datetime.now()
            await asyncio.sleep(0.1)  # Simulate work
            execution_times.append(("agent-2", start, datetime.now()))
            return AgentReview(
                agent_id="agent-2",
                agent_type="gpt4",
                focus_areas=["performance"],
                findings=[],
                summary="No issues",
                review_time_ms=100,
            )

        mock_agent_1.review = mock_review_1
        mock_agent_2.review = mock_review_2

        orchestrator = AgentOrchestrator(
            agents=[mock_agent_1, mock_agent_2],
            timeout_seconds=10,
            min_agents_required=2,
        )

        start_time = datetime.now()
        results = await orchestrator.review(
            diff=sample_vulnerable_diff,
            file_contents={},
            context=mock_review_context,
        )
        total_time = (datetime.now() - start_time).total_seconds()

        # Both agents should complete
        assert len(results) == 2

        # Total time should be ~0.1s (parallel), not ~0.2s (sequential)
        # Allow some margin for overhead
        assert total_time < 0.15

    @pytest.mark.asyncio
    async def test_handles_agent_timeout(self, sample_vulnerable_diff, mock_review_context):
        """Test that orchestrator handles agent timeouts gracefully."""
        from ai_reviewer.models.review import AgentReview
        from ai_reviewer.orchestrator.orchestrator import AgentOrchestrator

        # Create mock agents - one fast, one slow
        fast_agent = MagicMock()
        fast_agent.agent_id = "fast-agent"
        fast_agent.focus_areas = ["security"]

        slow_agent = MagicMock()
        slow_agent.agent_id = "slow-agent"
        slow_agent.focus_areas = ["performance"]

        async def fast_review(*_args, **_kwargs):
            await asyncio.sleep(0.01)
            return AgentReview(
                agent_id="fast-agent",
                agent_type="claude",
                focus_areas=["security"],
                findings=[],
                summary="Fast review",
                review_time_ms=10,
            )

        async def slow_review(*_args, **_kwargs):
            await asyncio.sleep(10)  # Will timeout
            return AgentReview(
                agent_id="slow-agent",
                agent_type="gpt4",
                focus_areas=["performance"],
                findings=[],
                summary="Slow review",
                review_time_ms=10000,
            )

        fast_agent.review = fast_review
        slow_agent.review = slow_review

        orchestrator = AgentOrchestrator(
            agents=[fast_agent, slow_agent],
            timeout_seconds=0.5,  # Short timeout
            min_agents_required=1,  # Only need 1 to succeed
        )

        results = await orchestrator.review(
            diff=sample_vulnerable_diff,
            file_contents={},
            context=mock_review_context,
        )

        # Should get result from fast agent, slow agent times out
        assert len(results) >= 1
        assert any(r.agent_id == "fast-agent" for r in results)

    @pytest.mark.asyncio
    async def test_fails_if_insufficient_agents(self, sample_vulnerable_diff, mock_review_context):
        """Test that orchestrator fails if too few agents succeed."""
        from ai_reviewer.orchestrator.orchestrator import (
            AgentOrchestrator,
            InsufficientAgentsError,
        )

        # Create mock agent that always fails
        failing_agent = MagicMock()
        failing_agent.agent_id = "failing-agent"
        failing_agent.focus_areas = ["security"]

        async def failing_review(*_args, **_kwargs):
            raise Exception("Agent failed")

        failing_agent.review = failing_review

        orchestrator = AgentOrchestrator(
            agents=[failing_agent],
            timeout_seconds=10,
            min_agents_required=1,
        )

        with pytest.raises(InsufficientAgentsError):
            await orchestrator.review(
                diff=sample_vulnerable_diff,
                file_contents={},
                context=mock_review_context,
            )
