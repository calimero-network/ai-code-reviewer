from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_reviewer.agents.anthropic_client import AnthropicReviewResult, UsageStats
from ai_reviewer.agents.base import ReviewAgent
from ai_reviewer.models.context import ReviewContext


class DummyAgent(ReviewAgent):
    MODEL = "claude-opus-4-6"
    AGENT_TYPE = "dummy"
    FOCUS_AREAS = ["security"]
    SYSTEM_PROMPT = "You are a dummy reviewer."
    THINKING_ENABLED = True
    THINKING_BUDGET = 4096


@pytest.mark.asyncio
async def test_review_agent_uses_anthropic_client():
    client = MagicMock()
    client.run_review = AsyncMock(return_value=AnthropicReviewResult(
        parsed={
            "findings": [{
                "file_path": "a.py",
                "line_start": 1,
                "severity": "warning",
                "category": "security",
                "title": "t",
                "description": "d",
                "confidence": 0.9,
            }],
            "summary": "sum",
        },
        raw_text="",
        usage=UsageStats(input_tokens=100, output_tokens=20),
    ))

    agent = DummyAgent(
        client=client,
        agent_id="dummy-1",
        system_blocks=[{"type": "text", "text": "sys"}],
        user_blocks=[{"type": "text", "text": "u"}],
        tool_registry=None,
        max_tokens=4096,
        temperature=0.2,
    )
    ctx = ReviewContext(
        repo_name="o/r",
        pr_number=1,
        pr_title="t",
        pr_description="d",
        base_branch="main",
        head_branch="feat",
        author="u",
        changed_files_count=1,
        additions=1,
        deletions=0,
    )
    review = await agent.review(diff="d", file_contents={}, context=ctx)

    assert review.agent_id == "dummy-1"
    assert review.agent_type == "dummy"
    assert len(review.findings) == 1
    assert review.summary == "sum"
    kwargs = client.run_review.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-6"
    assert kwargs["thinking_budget"] == 4096
