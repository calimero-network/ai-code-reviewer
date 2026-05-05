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
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "You are a reviewer."}],
        user_blocks=[{"type": "text", "text": "diff..."}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
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
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "sys"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema=schema,
        tool_registry=None,
        enable_thinking=False,
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
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=True,
        max_tokens=16384,
        temperature=1.0,
    )
    kwargs = client._sdk.messages.create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}


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
        enable_thinking=False,
        max_tokens=4096,
        temperature=0.3,
    )
    kwargs = client._sdk.messages.create.call_args.kwargs
    assert "thinking" not in kwargs


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
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=registry,
        enable_thinking=False,
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
        model="claude-sonnet-4-6",
        system_blocks=[
            {"type": "text", "text": "role"},
            {"type": "text", "text": "conventions"},
        ],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
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
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "role"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
        max_tokens=4096,
        temperature=0.3,
    )
    sent = client._sdk.messages.create.call_args.kwargs["system"]
    assert "cache_control" not in sent[0]


@pytest.mark.asyncio
async def test_run_completion_returns_plain_text():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        return_value=_fake_response("# Updated README\n\nNew content here.")
    )

    result = await client.run_completion(
        model="claude-sonnet-4-6",
        system="You are a technical writer.",
        user="Update these docs.",
        max_tokens=2048,
    )

    assert result == "# Updated README\n\nNew content here."
    call_kwargs = client._sdk.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-sonnet-4-6"
    assert call_kwargs["max_tokens"] == 2048
    # run_completion must not pass output_config or tools
    assert "output_config" not in call_kwargs
    assert "tools" not in call_kwargs


@pytest.mark.asyncio
async def test_caching_marks_last_tool_result_when_enabled():
    """cache_control is placed on the last tool_result block so the conversation
    prefix is cached for the next round — reducing re-billed input tokens by ~90%."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("t1", "read_file", {"path": "a.py"}),
            _tool_use_response("t2", "read_file", {"path": "b.py"}),
            _fake_response('{"findings": [], "summary": "done"}'),
        ]
    )

    registry = MagicMock()
    registry.tool_specs.return_value = [{"name": "read_file", "input_schema": {}}]
    registry.execute = AsyncMock(return_value="contents")

    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=registry,
        enable_thinking=False,
        max_tokens=4096,
        temperature=0.3,
    )

    # Round 2: the tool_result user turn appended after round 1 must carry cache_control
    round2_kwargs = client._sdk.messages.create.await_args_list[1].kwargs
    round2_last_user_msg = round2_kwargs["messages"][-1]
    assert round2_last_user_msg["role"] == "user"
    last_block = round2_last_user_msg["content"][-1]
    assert last_block["type"] == "tool_result"
    assert last_block.get("cache_control") == {"type": "ephemeral"}, (
        "Last tool_result block must carry cache_control so the conversation "
        "prefix is cached before the next messages.create call"
    )

    # Round 3: same invariant — the tool_result from round 2 is also marked
    round3_kwargs = client._sdk.messages.create.await_args_list[2].kwargs
    round3_last_user_msg = round3_kwargs["messages"][-1]
    last_block_r3 = round3_last_user_msg["content"][-1]
    assert last_block_r3.get("cache_control") == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_caching_disabled_leaves_tool_result_unmarked():
    """When caching is off, no cache_control is added to tool_result blocks."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(
        side_effect=[
            _tool_use_response("t1", "read_file", {"path": "a.py"}),
            _fake_response('{"findings": [], "summary": "done"}'),
        ]
    )

    registry = MagicMock()
    registry.tool_specs.return_value = [{"name": "read_file", "input_schema": {}}]
    registry.execute = AsyncMock(return_value="contents")

    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=registry,
        enable_thinking=False,
        max_tokens=4096,
        temperature=0.3,
    )

    round2_kwargs = client._sdk.messages.create.await_args_list[1].kwargs
    last_user_msg = round2_kwargs["messages"][-1]
    for block in last_user_msg["content"]:
        assert "cache_control" not in block, (
            "cache_control must not appear on tool_result when caching is disabled"
        )


@pytest.mark.asyncio
async def test_run_completion_uses_system_and_user():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(return_value=_fake_response("result"))

    await client.run_completion(
        model="claude-sonnet-4-6",
        system="sys prompt",
        user="user prompt",
    )

    call_kwargs = client._sdk.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "sys prompt"
    assert call_kwargs["messages"] == [{"role": "user", "content": "user prompt"}]
