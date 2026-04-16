from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_reviewer.agents.anthropic_client import AnthropicClient, AnthropicReviewResult
from ai_reviewer.config import AnthropicApiConfig


def _text_block(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _fake_response(text: str, stop_reason: str = "end_turn"):
    msg = MagicMock()
    msg.stop_reason = stop_reason
    msg.content = [_text_block(text)]
    msg.usage.input_tokens = 100
    msg.usage.output_tokens = 50
    msg.usage.cache_read_input_tokens = 0
    msg.usage.cache_creation_input_tokens = 0
    return msg


def _tool_use_block(tool_id: str, name: str, input_: dict):
    b = MagicMock()
    b.type = "tool_use"
    b.id = tool_id
    b.name = name
    b.input = input_
    return b


def _tool_use_response(tool_id: str, name: str, input_: dict):
    msg = MagicMock()
    msg.stop_reason = "tool_use"
    msg.content = [_tool_use_block(tool_id, name, input_)]
    msg.usage.input_tokens = 10
    msg.usage.output_tokens = 5
    msg.usage.cache_read_input_tokens = 0
    msg.usage.cache_creation_input_tokens = 0
    return msg


@pytest.mark.asyncio
async def test_run_review_happy_path_parses_json():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    result = await client.run_review(
        model="claude-opus-4-6",
        system_blocks=[{"type": "text", "text": "You are a reviewer."}],
        user_blocks=[{"type": "text", "text": "diff..."}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )

    assert isinstance(result, AnthropicReviewResult)
    assert result.parsed == {"findings": [], "summary": "ok"}
    assert result.usage.input_tokens == 100


@pytest.mark.asyncio
async def test_run_review_passes_output_schema_as_json_schema():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    schema = {"type": "object", "properties": {"findings": {"type": "array"}}}
    await client.run_review(
        model="claude-opus-4-6",
        system_blocks=[{"type": "text", "text": "sys"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema=schema,
        tool_registry=None,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )

    kwargs = client._sdk.messages.create.call_args.kwargs
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["format"]["schema"] == schema


@pytest.mark.asyncio
async def test_run_review_with_thinking_budget_sets_thinking_config():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    await client.run_review(
        model="claude-opus-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=8192,
        max_tokens=16384,
        temperature=1.0,
    )
    kwargs = client._sdk.messages.create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 8192}


@pytest.mark.asyncio
async def test_run_review_without_thinking_omits_config():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )
    kwargs = client._sdk.messages.create.call_args.kwargs
    assert "thinking" not in kwargs


@pytest.mark.asyncio
async def test_thinking_forces_temperature_one():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )
    await client.run_review(
        model="claude-opus-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=8192,
        max_tokens=16384,
        temperature=0.2,
    )
    assert client._sdk.messages.create.call_args.kwargs["temperature"] == 1.0


@pytest.mark.asyncio
async def test_tool_use_loop_dispatches_and_feeds_result_back():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("t1", "read_file", {"path": "x.py"}),
            _fake_response('{"findings": [], "summary": "done"}'),
        ]
    )

    registry = MagicMock()
    registry.tool_specs.return_value = [{"name": "read_file", "input_schema": {}}]
    registry.execute = AsyncMock(return_value="file-contents")

    result = await client.run_review(
        model="claude-opus-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=registry,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )

    assert result.parsed == {"findings": [], "summary": "done"}
    registry.execute.assert_awaited_once_with("read_file", {"path": "x.py"})
    assert client._sdk.messages.create.await_count == 2

    second_kwargs = client._sdk.messages.create.await_args_list[1].kwargs
    last_msg = second_kwargs["messages"][-1]
    assert last_msg["role"] == "user"
    assert last_msg["content"][0]["type"] == "tool_result"
    assert last_msg["content"][0]["tool_use_id"] == "t1"
    assert last_msg["content"][0]["content"] == "file-contents"


@pytest.mark.asyncio
async def test_caching_marks_last_system_block_when_enabled():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    await client.run_review(
        model="claude-opus-4-6",
        system_blocks=[
            {"type": "text", "text": "role"},
            {"type": "text", "text": "conventions"},
        ],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )
    sent = client._sdk.messages.create.call_args.kwargs["system"]
    assert sent[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent[0]


@pytest.mark.asyncio
async def test_caching_disabled_leaves_system_unchanged():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    await client.run_review(
        model="claude-opus-4-6",
        system_blocks=[{"type": "text", "text": "role"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        thinking_budget=None,
        max_tokens=4096,
        temperature=0.3,
    )
    sent = client._sdk.messages.create.call_args.kwargs["system"]
    assert "cache_control" not in sent[0]
