from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_reviewer.agents.anthropic_client import AnthropicReviewResult, UsageStats
from ai_reviewer.agents.base import ReviewAgent
from ai_reviewer.models.context import ReviewContext


class DummyAgent(ReviewAgent):
    MODEL = "claude-sonnet-4-6"
    AGENT_TYPE = "dummy"
    FOCUS_AREAS = ["security"]
    SYSTEM_PROMPT = "You are a dummy reviewer."
    THINKING_ENABLED = False


class OtherAgent(ReviewAgent):
    MODEL = "claude-sonnet-4-6"
    AGENT_TYPE = "other"
    FOCUS_AREAS = ["performance"]
    SYSTEM_PROMPT = "You are a totally different reviewer."
    THINKING_ENABLED = False


def _make_client(caching: bool = True):
    client = MagicMock()
    client.config.enable_prompt_caching = caching
    client.run_review = AsyncMock(
        return_value=AnthropicReviewResult(parsed={"findings": [], "summary": "s"}, raw_text="")
    )
    return client


def _ctx():
    return ReviewContext(
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


@pytest.mark.asyncio
async def test_shared_prefix_identical_across_agents_and_role_appended_last():
    """The cacheable prefix [system][shared user blocks] must be byte-identical
    across agents so caching's strict prefix match holds; the per-agent role sits
    as the LAST user block, after a cache breakpoint on the last shared block. The
    shared user_blocks list (handed to every agent at once) must not be mutated."""
    system_blocks = [{"type": "text", "text": "shared system"}]
    user_blocks = [
        {"type": "text", "text": "shared diff"},
        {"type": "text", "text": "shared files"},
    ]
    original = [dict(b) for b in user_blocks]

    a = DummyAgent(
        client=_make_client(),
        agent_id="a",
        system_blocks=system_blocks,
        user_blocks=user_blocks,
        tool_registry=None,
    )
    b = OtherAgent(
        client=_make_client(),
        agent_id="b",
        system_blocks=system_blocks,
        user_blocks=user_blocks,
        tool_registry=None,
    )
    await a.review(diff="d", file_contents={}, context=_ctx())
    await b.review(diff="d", file_contents={}, context=_ctx())

    a_kwargs = a.client.run_review.call_args.kwargs
    b_kwargs = b.client.run_review.call_args.kwargs

    # System blocks pass through unchanged and identical.
    assert a_kwargs["system_blocks"] == system_blocks
    assert a_kwargs["system_blocks"] == b_kwargs["system_blocks"]

    a_user = a_kwargs["user_blocks"]
    b_user = b_kwargs["user_blocks"]
    # All blocks except the last (the shared prefix) are byte-identical.
    assert a_user[:-1] == b_user[:-1]
    # The last block is the per-agent role and differs between agents.
    assert a_user[-1]["text"] == "## Your reviewer role\nYou are a dummy reviewer."
    assert b_user[-1]["text"] == "## Your reviewer role\nYou are a totally different reviewer."
    # Cache breakpoint on the last SHARED block (second-to-last overall).
    assert a_user[-2]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in a_user[-1]
    # The shared list itself is never mutated.
    assert user_blocks == original


@pytest.mark.asyncio
async def test_caching_disabled_adds_no_breakpoint_but_still_appends_role():
    user_blocks = [{"type": "text", "text": "shared"}]
    agent = DummyAgent(
        client=_make_client(caching=False),
        agent_id="a",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=user_blocks,
        tool_registry=None,
    )
    await agent.review(diff="d", file_contents={}, context=_ctx())
    sent = agent.client.run_review.call_args.kwargs["user_blocks"]
    assert all("cache_control" not in b for b in sent)
    assert sent[-1]["text"].endswith("You are a dummy reviewer.")


@pytest.mark.asyncio
async def test_review_agent_uses_anthropic_client():
    client = MagicMock()
    client.run_review = AsyncMock(
        return_value=AnthropicReviewResult(
            parsed={
                "findings": [
                    {
                        "file_path": "a.py",
                        "line_start": 1,
                        "severity": "warning",
                        "category": "security",
                        "title": "t",
                        "description": "d",
                        "confidence": 0.9,
                    }
                ],
                "summary": "sum",
            },
            raw_text="",
            usage=UsageStats(input_tokens=100, output_tokens=20),
        )
    )

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
    assert kwargs["model"] == "claude-sonnet-4-6"
    assert kwargs["enable_thinking"] is False


@pytest.mark.asyncio
async def test_config_model_overrides_class_default():
    """The model passed in (from AgentConfig) takes precedence over class MODEL."""
    client = MagicMock()
    client.run_review = AsyncMock(
        return_value=AnthropicReviewResult(parsed={"findings": [], "summary": "s"}, raw_text="")
    )
    agent = DummyAgent(
        client=client,
        agent_id="dummy-1",
        system_blocks=[{"type": "text", "text": "sys"}],
        user_blocks=[{"type": "text", "text": "u"}],
        tool_registry=None,
        model="claude-opus-4-8",  # differs from DummyAgent.MODEL
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
    await agent.review(diff="d", file_contents={}, context=ctx)
    assert client.run_review.call_args.kwargs["model"] == "claude-opus-4-8"
