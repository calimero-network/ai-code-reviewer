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
async def test_run_review_with_thinking_enabled_sets_adaptive_config():
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


def _count_cache_control(kwargs: dict) -> int:
    """Count cache_control breakpoints across the system + messages of a single
    messages.create payload — exactly what Anthropic caps at 4 per request."""
    n = 0
    system = kwargs.get("system")
    if isinstance(system, list):
        n += sum(1 for blk in system if isinstance(blk, dict) and "cache_control" in blk)
    for msg in kwargs.get("messages", []):
        content = msg.get("content")
        if isinstance(content, list):
            n += sum(1 for blk in content if isinstance(blk, dict) and "cache_control" in blk)
    return n


@pytest.mark.asyncio
async def test_cache_control_breakpoints_never_exceed_four_across_tool_rounds():
    """Regression for #67: cache_control breakpoints must not accumulate past the
    Anthropic 4-per-request cap as the tool-use loop runs 5+ rounds.

    Previously one breakpoint was appended per tool round and never pruned, so
    system(1) + N accumulated tool-result breakpoints hit 5 on the 5th round and
    the request was rejected with a 400 (silently dropping the whole review)."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()

    # Snapshot the breakpoint count at call time — run_review reuses and mutates
    # the same messages list across rounds, so inspecting await_args_list after
    # the fact would only ever show the final state, not what each request sent.
    counts: list[int] = []
    counter = {"n": 0}

    def fake_create(**kwargs):
        counts.append(_count_cache_control(kwargs))
        counter["n"] += 1
        # Always request another tool round to drive the loop to its cap.
        return _tool_use_response(f"t{counter['n']}", "read_file", {"path": "a.py"})

    client._sdk.messages.create = AsyncMock(side_effect=fake_create)

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
        max_tool_rounds=8,
    )

    assert len(counts) >= 6, f"expected the loop to run past 4 rounds, ran {len(counts)}"
    assert max(counts) <= 4, f"cache_control breakpoints exceeded the 4-per-request cap: {counts}"


@pytest.mark.asyncio
async def test_breakpoint_cap_holds_when_loop_terminates_normally():
    """Companion to the cap regression: the loop runs 5+ tool rounds and then
    ends with a real end_turn response (not the max_tool_rounds sentinel). The
    breakpoint cap must hold on that path too, and the final review must parse."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()

    counts: list[int] = []
    counter = {"n": 0}

    def fake_create(**kwargs):
        counts.append(_count_cache_control(kwargs))
        counter["n"] += 1
        # Five tool rounds, then a normal completion on the sixth request.
        if counter["n"] <= 5:
            return _tool_use_response(f"t{counter['n']}", "read_file", {"path": "a.py"})
        return _fake_response('{"findings": [], "summary": "done"}')

    client._sdk.messages.create = AsyncMock(side_effect=fake_create)

    registry = MagicMock()
    registry.tool_specs.return_value = [{"name": "read_file", "input_schema": {}}]
    registry.execute = AsyncMock(return_value="contents")

    result = await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=registry,
        enable_thinking=False,
        max_tokens=4096,
        temperature=0.3,
        max_tool_rounds=8,
    )

    assert len(counts) == 6, f"expected 5 tool rounds + 1 final request, got {len(counts)}"
    assert max(counts) <= 4, f"cache_control breakpoints exceeded the 4-per-request cap: {counts}"
    # The loop ended normally, so the real model output is parsed — not the cap sentinel.
    assert result.parsed == {"findings": [], "summary": "done"}


@pytest.mark.asyncio
async def test_caller_user_block_cache_control_survives_tool_rounds():
    """The strip pass prunes only the breakpoints the client adds to appended
    tool_result turns — it must never strip a cache_control the caller placed on
    the initial user turn. Guards the messages[1:] boundary (PR #68 review)."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()

    # Snapshot at call time whether the original user turn still carries its
    # caller-supplied breakpoint on every request.
    user_cc_present: list[bool] = []
    counter = {"n": 0}

    def fake_create(**kwargs):
        first_user_content = kwargs["messages"][0]["content"]
        user_cc_present.append(
            any(isinstance(b, dict) and "cache_control" in b for b in first_user_content)
        )
        counter["n"] += 1
        if counter["n"] < 3:
            return _tool_use_response(f"t{counter['n']}", "read_file", {"path": "a.py"})
        return _fake_response('{"findings": [], "summary": "done"}')

    client._sdk.messages.create = AsyncMock(side_effect=fake_create)

    registry = MagicMock()
    registry.tool_specs.return_value = [{"name": "read_file", "input_schema": {}}]
    registry.execute = AsyncMock(return_value="contents")

    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u", "cache_control": {"type": "ephemeral"}}],
        output_schema={"type": "object"},
        tool_registry=registry,
        enable_thinking=False,
        max_tokens=4096,
        temperature=0.3,
    )

    assert all(user_cc_present), (
        f"caller's user-block cache_control was stripped: per-request presence={user_cc_present}"
    )


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


@pytest.mark.asyncio
async def test_complete_simple_returns_text_without_tools_or_schema():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(return_value=_fake_response("assessment json"))

    out = await client.complete_simple(
        model="claude-sonnet-4-6",
        system=[{"type": "text", "text": "You are a validator."}],
        user="findings...",
        max_tokens=4096,
        temperature=0.2,
    )

    assert out == "assessment json"
    kw = client._sdk.messages.create.call_args.kwargs
    assert "tools" not in kw and "output_config" not in kw
    assert kw["messages"] == [{"role": "user", "content": "findings..."}]
    assert kw["temperature"] == 0.2


@pytest.mark.asyncio
async def test_complete_simple_caches_last_system_block_when_enabled():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._sdk.messages.create = AsyncMock(return_value=_fake_response("ok"))

    await client.complete_simple(
        model="m",
        system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        user="u",
    )
    sent = client._sdk.messages.create.call_args.kwargs["system"]
    assert sent[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent[0]


@pytest.mark.asyncio
async def test_run_review_logs_cache_usage(caplog):
    """run_review surfaces cache counters so caching is observable on real runs."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    resp = _fake_response('{"findings": [], "summary": "ok"}')
    resp.usage.cache_read_input_tokens = 70
    resp.usage.cache_creation_input_tokens = 120
    client._sdk.messages.create = AsyncMock(return_value=resp)

    with caplog.at_level("INFO", logger="ai_reviewer.agents.anthropic_client"):
        result = await client.run_review(
            model="claude-sonnet-4-6",
            system_blocks=[{"type": "text", "text": "s"}],
            user_blocks=[{"type": "text", "text": "u"}],
            output_schema={"type": "object"},
            tool_registry=None,
        )

    assert result.usage.cache_read_input_tokens == 70
    assert "cache_read=70" in caplog.text
    assert "cache_creation=120" in caplog.text
