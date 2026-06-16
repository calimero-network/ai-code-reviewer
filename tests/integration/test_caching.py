"""Opt-in integration test: prove prompt caching produces a real cache hit.

Unit tests can only assert the cache_control breakpoint is *placed*. This hits
the live API and checks the authoritative signal — the usage counters — to prove
caching actually works: a repeated large system prefix must read from cache.

Requires ANTHROPIC_API_KEY. Enable with: pytest -m integration
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_prompt_caching_second_call_reads_from_cache() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    from ai_reviewer.agents.anthropic_client import AnthropicClient
    from ai_reviewer.config import AnthropicApiConfig
    from ai_reviewer.context.builder import FINDINGS_SCHEMA

    cfg = AnthropicApiConfig(api_key=os.environ["ANTHROPIC_API_KEY"], enable_prompt_caching=True)
    # System prefix must exceed the model's minimum cacheable length
    # (~1024 tokens for Sonnet); pad well past it so a cache entry is created.
    big_system = "You are a code reviewer.\n" + ("Cacheable context line. " * 600)
    system_blocks = [{"type": "text", "text": big_system}]
    user_blocks = [{"type": "text", "text": "Review this diff:\n+print('hello')"}]

    async with AnthropicClient(cfg) as client:
        first = await client.run_review(
            model="claude-sonnet-4-6",
            system_blocks=system_blocks,
            user_blocks=user_blocks,
            output_schema=FINDINGS_SCHEMA,
            tool_registry=None,
            max_tokens=256,
        )
        second = await client.run_review(
            model="claude-sonnet-4-6",
            system_blocks=system_blocks,
            user_blocks=user_blocks,
            output_schema=FINDINGS_SCHEMA,
            tool_registry=None,
            max_tokens=256,
        )

    # Definitive proof: the identical second request reads the cached prefix.
    assert second.usage.cache_read_input_tokens > 0, (
        f"expected a cache read on the second call; second usage={second.usage}"
    )
    # The prefix is cacheable: the first call either created the entry or read a
    # warm one left by a prior run within the 5-minute TTL.
    assert first.usage.cache_creation_input_tokens > 0 or first.usage.cache_read_input_tokens > 0, (
        f"expected the first call to create or read a cache entry; first usage={first.usage}"
    )
