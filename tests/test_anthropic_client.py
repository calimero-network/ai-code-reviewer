from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from ai_reviewer.agents import anthropic_client as ac
from ai_reviewer.agents.anthropic_client import AnthropicClient, AnthropicReviewResult
from ai_reviewer.config import AnthropicApiConfig

# The SDK is banned outside anthropic_client (invariant I1); reach its exception
# type through the module under test instead of importing it directly.
BadRequestError = ac.anthropic.BadRequestError


def _bad_request(message: str) -> BadRequestError:
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    return BadRequestError(message, response=resp, body=None)


def _mock_stream_boundary(client: AnthropicClient, side_effect):
    """Mock the raw _sdk.messages.stream async-context-manager boundary.

    Each entry in side_effect is either a Message to return from
    get_final_message() or an exception to raise on entering the context.
    """
    calls = {"n": 0}

    def make_cm(**_kwargs):
        idx = calls["n"]
        calls["n"] += 1
        item = side_effect[idx]
        cm = MagicMock()
        if isinstance(item, BaseException):
            cm.__aenter__ = AsyncMock(side_effect=item)
        else:
            stream_obj = MagicMock()
            stream_obj.get_final_message = AsyncMock(return_value=item)
            cm.__aenter__ = AsyncMock(return_value=stream_obj)
        cm.__aexit__ = AsyncMock(return_value=None)
        return cm

    client._sdk = MagicMock()
    client._sdk.messages.stream = MagicMock(side_effect=make_cm)
    return client._sdk.messages.stream, calls


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


def _thinking_block(text: str = "reasoning", signature: str = "sig"):
    b = MagicMock()
    b.type = "thinking"
    b.thinking = text
    b.signature = signature
    return b


def _thinking_tool_use_response(tool_id: str, name: str, input_: dict):
    """A thinking-on assistant turn: a signed thinking block plus a tool_use."""
    msg = MagicMock()
    msg.stop_reason = "tool_use"
    msg.content = [_thinking_block(), _tool_use_block(tool_id, name, input_)]
    msg.usage.input_tokens = 10
    msg.usage.output_tokens = 5
    msg.usage.cache_read_input_tokens = 0
    msg.usage.cache_creation_input_tokens = 0
    return msg


def _has_thinking(content) -> bool:
    return any(isinstance(b, dict) and b.get("type") == "thinking" for b in content)


@pytest.mark.asyncio
async def test_create_message_uses_streaming_not_blocking_create():
    """The API call must go through messages.stream() + get_final_message(), not
    the non-streaming messages.create(): long non-streaming calls on large prompts
    drop the connection (APIConnectionError) and corrupt the response body."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)

    final = _fake_response('{"findings": [], "summary": "ok"}')
    stream_obj = MagicMock()
    stream_obj.get_final_message = AsyncMock(return_value=final)
    stream_cm = MagicMock()
    stream_cm.__aenter__ = AsyncMock(return_value=stream_obj)
    stream_cm.__aexit__ = AsyncMock(return_value=None)

    client._sdk = MagicMock()
    client._sdk.messages.stream = MagicMock(return_value=stream_cm)
    client._sdk.messages.create = AsyncMock()  # must NOT be used

    result = await client._create_message(model="claude-sonnet-5", max_tokens=8192)

    client._sdk.messages.stream.assert_called_once()
    assert client._sdk.messages.stream.call_args.kwargs["model"] == "claude-sonnet-5"
    stream_obj.get_final_message.assert_awaited_once()
    client._sdk.messages.create.assert_not_called()
    assert result is final


@pytest.mark.asyncio
async def test_run_review_happy_path_parses_json():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
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
    client._create_message = AsyncMock(
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

    kwargs = client._create_message.call_args.kwargs
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["output_config"]["format"]["schema"] == schema


@pytest.mark.asyncio
async def test_run_review_with_thinking_enabled_sets_adaptive_config():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
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
    kwargs = client._create_message.call_args.kwargs
    assert kwargs["thinking"] == {"type": "adaptive"}


@pytest.mark.asyncio
async def test_run_review_thinking_sets_medium_effort():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    await client.run_review(
        model="claude-sonnet-5",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=True,
        max_tokens=32000,
        temperature=1.0,
    )
    kwargs = client._create_message.call_args.kwargs
    assert kwargs["output_config"]["effort"] == "medium"


@pytest.mark.asyncio
async def test_run_review_without_thinking_omits_effort():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
        return_value=_fake_response('{"findings": [], "summary": "ok"}')
    )

    await client.run_review(
        model="claude-sonnet-5",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
        max_tokens=8192,
        temperature=0.3,
    )
    kwargs = client._create_message.call_args.kwargs
    assert "effort" not in kwargs["output_config"]


@pytest.mark.asyncio
async def test_run_review_without_thinking_sends_disabled():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
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
    kwargs = client._create_message.call_args.kwargs
    assert kwargs["thinking"] == {"type": "disabled"}


def _client_with_mocked_sdk(stop_reason: str = "end_turn", text: str | None = None):
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    body = '{"findings": [], "summary": "ok"}' if text is None else text
    mock_create = AsyncMock(return_value=_fake_response(body, stop_reason=stop_reason))
    client._create_message = mock_create
    return client, mock_create


@pytest.mark.asyncio
async def test_run_review_thinking_off_sends_explicit_disabled():
    """Omitted `thinking` means adaptive-ON for Sonnet 5 — must send explicit disabled."""
    client, mock_create = _client_with_mocked_sdk()
    await client.run_review(
        model="claude-sonnet-5",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
    )
    kwargs = mock_create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "disabled"}
    assert "temperature" not in kwargs  # sonnet-5 rejects it


@pytest.mark.asyncio
async def test_run_review_sonnet46_thinking_off_keeps_temperature():
    client, mock_create = _client_with_mocked_sdk()
    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
        temperature=0.3,
    )
    kwargs = mock_create.call_args.kwargs
    assert kwargs["thinking"] == {"type": "disabled"}
    assert kwargs["temperature"] == 0.3


@pytest.mark.asyncio
async def test_run_review_fable_omits_thinking_entirely():
    """Fable rejects explicit {"type": "disabled"} — the field must be absent."""
    client, mock_create = _client_with_mocked_sdk()
    await client.run_review(
        model="claude-fable-5",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
    )
    kwargs = mock_create.call_args.kwargs
    assert "thinking" not in kwargs
    assert "temperature" not in kwargs


@pytest.mark.asyncio
async def test_tool_use_loop_dispatches_and_feeds_result_back():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
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
    assert client._create_message.await_count == 2

    second_kwargs = client._create_message.await_args_list[1].kwargs
    last_msg = second_kwargs["messages"][-1]
    assert last_msg["role"] == "user"
    assert last_msg["content"][0]["type"] == "tool_result"
    assert last_msg["content"][0]["tool_use_id"] == "t1"
    assert last_msg["content"][0]["content"] == "file-contents"


@pytest.mark.asyncio
async def test_tool_budget_exhausted_drops_tools_and_finishes():
    """A registry that hits its tool-call budget must finish with real findings,
    not thrash on failing tool calls until the round cap marks it incomplete."""
    from ai_reviewer.tools.repo_tools import ToolBudgetExhausted

    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
        side_effect=[
            _tool_use_response("t1", "read_file", {"path": "x.py"}),
            _fake_response('{"findings": [{"title": "bug"}], "summary": "real review"}'),
        ]
    )

    registry = MagicMock()
    registry.tool_specs.return_value = [{"name": "read_file", "input_schema": {}}]
    registry.execute = AsyncMock(side_effect=ToolBudgetExhausted("exceeded max_tool_calls=8"))

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

    # Round 2 must omit tools entirely so the model produces its final JSON.
    second_kwargs = client._create_message.await_args_list[1].kwargs
    assert "tools" not in second_kwargs

    # The exhaustion message was fed back for the round-1 tool_use.
    first_tool_result = second_kwargs["messages"][-1]["content"][0]
    assert first_tool_result["tool_use_id"] == "t1"
    assert "tool budget exhausted" in first_tool_result["content"]

    # Final review parses real findings with no incomplete marker.
    from ai_reviewer.agents.anthropic_client import INCOMPLETE_SUMMARY_MARKERS

    assert result.parsed["findings"] == [{"title": "bug"}]
    assert not any(m in result.parsed["summary"] for m in INCOMPLETE_SUMMARY_MARKERS)


@pytest.mark.asyncio
async def test_round_cap_forces_final_findings_emission():
    """An agent that keeps calling tools until the round cap must still emit
    real findings: the final round drops tools so the model produces its JSON,
    instead of discarding the whole review with the TOOL_LOOP_CAP sentinel."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()

    counter = {"n": 0}

    def fake_create(**kwargs):
        counter["n"] += 1
        # Model keeps requesting tools whenever they're offered; when the final
        # round withholds them, it emits its findings.
        if "tools" in kwargs:
            return _tool_use_response(f"t{counter['n']}", "read_file", {"path": "a.py"})
        return _fake_response('{"findings": [{"title": "real bug"}], "summary": "reviewed"}')

    client._create_message = AsyncMock(side_effect=fake_create)

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
        max_tool_rounds=3,
    )

    from ai_reviewer.agents.anthropic_client import INCOMPLETE_SUMMARY_MARKERS

    # The last request omitted tools, forcing the emission.
    assert "tools" not in client._create_message.await_args_list[-1].kwargs
    # Real findings survive instead of the cap sentinel.
    assert result.parsed["findings"] == [{"title": "real bug"}]
    assert not any(m in result.parsed["summary"] for m in INCOMPLETE_SUMMARY_MARKERS)


@pytest.mark.asyncio
async def test_circuit_breaker_trips_on_cache_read_context():
    """The breaker must count cache-read tokens. A tool_use response with tiny
    input_tokens but huge cache_read_input_tokens (the shape once prompt caching
    engages) has a real per-request context above the limit, so the next round
    must abort with the circuit-breaker marker — the cumulative input_tokens sum
    used previously would have stayed near zero and never tripped."""
    from ai_reviewer.agents.anthropic_client import CIRCUIT_BREAKER_MARKER

    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)  # limit = 80_000 * 2 = 160_000
    client._sdk = MagicMock()

    resp = _tool_use_response("t1", "read_file", {"path": "a.py"})
    resp.usage.input_tokens = 1000
    resp.usage.cache_read_input_tokens = 200_000  # true context 201k > 160k
    resp.usage.cache_creation_input_tokens = 0
    client._create_message = AsyncMock(return_value=resp)

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
        max_tool_rounds=20,
    )

    assert result.parsed["summary"] == CIRCUIT_BREAKER_MARKER
    assert result.parsed["findings"] == []
    # Round 0 sent, round 1 aborted before sending — exactly 1 call, no runaway.
    assert client._create_message.await_count == 1


@pytest.mark.asyncio
async def test_circuit_breaker_does_not_trip_on_small_per_request_context():
    """With small per-request usage on every round the breaker never trips even
    across many rounds — the cumulative sum would balloon, the per-request size
    stays flat. Normal completion path is unaffected."""
    from ai_reviewer.agents.anthropic_client import INCOMPLETE_SUMMARY_MARKERS

    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()

    counter = {"n": 0}

    def fake_create(**_kwargs):
        counter["n"] += 1
        if counter["n"] <= 10:
            r = _tool_use_response(f"t{counter['n']}", "read_file", {"path": "a.py"})
            r.usage.input_tokens = 5000
            r.usage.cache_read_input_tokens = 5000  # ~10k per request, well under 160k
            return r
        return _fake_response('{"findings": [{"title": "bug"}], "summary": "done"}')

    client._create_message = AsyncMock(side_effect=fake_create)

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
        max_tool_rounds=20,
    )

    assert result.parsed["findings"] == [{"title": "bug"}]
    assert not any(m in result.parsed["summary"] for m in INCOMPLETE_SUMMARY_MARKERS)


@pytest.mark.asyncio
async def test_thinking_stripped_from_all_but_last_assistant_turn():
    """With thinking enabled the signed block is required only on the last
    assistant turn; earlier turns' thinking is dead weight re-sent every round.
    Only the most recent assistant turn keeps its thinking in the final request;
    the first request (no prior assistant) is untouched."""
    import copy

    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()

    snapshots: list[list] = []
    counter = {"n": 0}

    def fake_create(**kwargs):
        snapshots.append(copy.deepcopy(kwargs["messages"]))
        counter["n"] += 1
        if counter["n"] <= 3:
            return _thinking_tool_use_response(f"t{counter['n']}", "read_file", {"path": "a.py"})
        return _fake_response('{"findings": [], "summary": "done"}')

    client._create_message = AsyncMock(side_effect=fake_create)

    registry = MagicMock()
    registry.tool_specs.return_value = [{"name": "read_file", "input_schema": {}}]
    registry.execute = AsyncMock(return_value="contents")

    await client.run_review(
        model="claude-sonnet-5",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=registry,
        enable_thinking=True,
        max_tokens=8192,
        temperature=1.0,
    )

    # First request: only the caller's user turn, no assistant yet — untouched.
    assert len(snapshots[0]) == 1
    assert snapshots[0][0]["role"] == "user"

    # Final request carries 3 appended assistant turns; only the most recent one
    # keeps its thinking block, the earlier two were stripped.
    final = snapshots[-1]
    assistant_turns = [m for m in final if m["role"] == "assistant"]
    assert len(assistant_turns) == 3
    assert not _has_thinking(assistant_turns[0]["content"])
    assert not _has_thinking(assistant_turns[1]["content"])
    assert _has_thinking(assistant_turns[2]["content"])
    # The kept thinking block still carries its signature (required by the API).
    kept = next(b for b in assistant_turns[2]["content"] if b.get("type") == "thinking")
    assert kept["signature"] == "sig"
    # Stripping never removes the tool_use blocks themselves.
    assert all(any(b.get("type") == "tool_use" for b in t["content"]) for t in assistant_turns)


@pytest.mark.asyncio
async def test_soft_finalize_forces_emission_before_hard_breaker():
    """When per-request context crosses 75% of the breaker mid-loop, the agent
    must stop offering tools and finalize with real findings — instead of
    growing another 25% into the hard breaker and being discarded."""
    import copy

    from ai_reviewer.agents.anthropic_client import (
        _CONTEXT_BUDGET_MSG,
        INCOMPLETE_SUMMARY_MARKERS,
    )

    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)  # circuit_limit = 160_000, soft = 120_000
    client._sdk = MagicMock()

    snapshots: list[list] = []
    tools_present: list[bool] = []
    counter = {"n": 0}

    def fake_create(**kwargs):
        snapshots.append(copy.deepcopy(kwargs["messages"]))
        tools_present.append("tools" in kwargs)
        counter["n"] += 1
        if "tools" not in kwargs:
            return _fake_response('{"findings": [{"title": "real bug"}], "summary": "reviewed"}')
        r = _tool_use_response(f"t{counter['n']}", "read_file", {"path": "a.py"})
        if counter["n"] == 2:
            # Round 1 crosses the 75% soft threshold, still under the hard breaker.
            r.usage.input_tokens = 1000
            r.usage.cache_read_input_tokens = 130_000  # 131k > 120k, < 160k
        return r

    client._create_message = AsyncMock(side_effect=fake_create)

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
        max_tool_rounds=20,
    )

    # Rounds 0 and 1 offered tools; round 2 (post soft-finalize) must omit them.
    assert tools_present[0] and tools_present[1]
    assert not tools_present[2]

    # Round 1's pending tool_use (t2) was answered with the context-budget message.
    last_user = snapshots[2][-1]
    assert last_user["role"] == "user"
    assert last_user["content"][0]["tool_use_id"] == "t2"
    assert last_user["content"][0]["content"] == _CONTEXT_BUDGET_MSG

    # Review completed with real findings and no incomplete / circuit marker.
    assert result.parsed["findings"] == [{"title": "real bug"}]
    assert not any(m in result.parsed["summary"] for m in INCOMPLETE_SUMMARY_MARKERS)


@pytest.mark.asyncio
async def test_hard_breaker_wins_over_soft_finalize_on_huge_first_jump():
    """A single jump straight past the hard limit (not just 75%) must still abort
    with CIRCUIT_BREAKER_MARKER. Soft-finalize sets its flag the same round, but
    the hard breaker at the top of the next round fires before the salvage can
    complete — the last-resort abort for pathological context blow-ups."""
    from ai_reviewer.agents.anthropic_client import CIRCUIT_BREAKER_MARKER

    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)  # circuit_limit = 160_000
    client._sdk = MagicMock()

    resp = _tool_use_response("t1", "read_file", {"path": "a.py"})
    resp.usage.input_tokens = 1000
    resp.usage.cache_read_input_tokens = 200_000  # 201k > 160k hard, past 1.0x
    client._create_message = AsyncMock(return_value=resp)

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
        max_tool_rounds=20,
    )

    assert result.parsed["summary"] == CIRCUIT_BREAKER_MARKER
    assert result.parsed["findings"] == []
    # Round 0 sent, round 1 aborted before sending — no runaway.
    assert client._create_message.await_count == 1


@pytest.mark.asyncio
async def test_caching_marks_last_system_block_when_enabled():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
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
    sent = client._create_message.call_args.kwargs["system"]
    assert sent[-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in sent[0]


@pytest.mark.asyncio
async def test_caching_disabled_leaves_system_unchanged():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
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
    sent = client._create_message.call_args.kwargs["system"]
    assert "cache_control" not in sent[0]


@pytest.mark.asyncio
async def test_run_completion_returns_plain_text():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
        return_value=_fake_response("# Updated README\n\nNew content here.")
    )

    result = await client.run_completion(
        model="claude-sonnet-4-6",
        system="You are a technical writer.",
        user="Update these docs.",
        max_tokens=2048,
    )

    assert result == "# Updated README\n\nNew content here."
    call_kwargs = client._create_message.call_args.kwargs
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
    client._create_message = AsyncMock(
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
    round2_kwargs = client._create_message.await_args_list[1].kwargs
    round2_last_user_msg = round2_kwargs["messages"][-1]
    assert round2_last_user_msg["role"] == "user"
    last_block = round2_last_user_msg["content"][-1]
    assert last_block["type"] == "tool_result"
    assert last_block.get("cache_control") == {"type": "ephemeral"}, (
        "Last tool_result block must carry cache_control so the conversation "
        "prefix is cached before the next messages.create call"
    )

    # Round 3: same invariant — the tool_result from round 2 is also marked
    round3_kwargs = client._create_message.await_args_list[2].kwargs
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

    client._create_message = AsyncMock(side_effect=fake_create)

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

    client._create_message = AsyncMock(side_effect=fake_create)

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

    client._create_message = AsyncMock(side_effect=fake_create)

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
async def test_caller_messages0_breakpoint_survives_and_total_stays_under_cap():
    """Cross-agent cache sharing shape: the caller marks the last shared block on
    the initial user turn (messages[0]). Across tool rounds that breakpoint must
    survive the prune pass, and total breakpoints per request - system(1) +
    messages[0](1) + moving tool_result(1) = 3 - must stay <= 4."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()

    counts: list[int] = []
    msg0_cc: list[bool] = []
    counter = {"n": 0}

    def fake_create(**kwargs):
        counts.append(_count_cache_control(kwargs))
        first_user_content = kwargs["messages"][0]["content"]
        msg0_cc.append(
            any(isinstance(b, dict) and "cache_control" in b for b in first_user_content)
        )
        counter["n"] += 1
        if counter["n"] < 4:
            return _tool_use_response(f"t{counter['n']}", "read_file", {"path": "a.py"})
        return _fake_response('{"findings": [], "summary": "done"}')

    client._create_message = AsyncMock(side_effect=fake_create)

    registry = MagicMock()
    registry.tool_specs.return_value = [{"name": "read_file", "input_schema": {}}]
    registry.execute = AsyncMock(return_value="contents")

    # Mirrors what base.py builds: shared prefix block carries the breakpoint,
    # per-agent role block is appended last (unmarked).
    await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[
            {"type": "text", "text": "shared", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "## Your reviewer role\nrole"},
        ],
        output_schema={"type": "object"},
        tool_registry=registry,
        enable_thinking=False,
        max_tool_rounds=8,
    )

    assert len(counts) >= 3, f"expected multiple tool rounds, ran {len(counts)}"
    assert all(msg0_cc), f"caller messages[0] breakpoint was stripped: {msg0_cc}"
    assert max(counts) <= 4, f"breakpoints exceeded the 4-per-request cap: {counts}"
    # Steady-state (round >= 1) carries all three breakpoints.
    assert max(counts) == 3, f"expected system+messages0+tool_result=3, got {counts}"


@pytest.mark.asyncio
async def test_caching_disabled_leaves_tool_result_unmarked():
    """When caching is off, no cache_control is added to tool_result blocks."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(
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

    round2_kwargs = client._create_message.await_args_list[1].kwargs
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
    client._create_message = AsyncMock(return_value=_fake_response("result"))

    await client.run_completion(
        model="claude-sonnet-4-6",
        system="sys prompt",
        user="user prompt",
    )

    call_kwargs = client._create_message.call_args.kwargs
    assert call_kwargs["system"] == "sys prompt"
    assert call_kwargs["messages"] == [{"role": "user", "content": "user prompt"}]


@pytest.mark.asyncio
async def test_complete_simple_returns_text_without_tools_or_schema():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(return_value=_fake_response("assessment json"))

    out = await client.complete_simple(
        model="claude-sonnet-4-6",
        system=[{"type": "text", "text": "You are a validator."}],
        user="findings...",
        max_tokens=4096,
        temperature=0.2,
    )

    assert out == "assessment json"
    kw = client._create_message.call_args.kwargs
    assert "tools" not in kw and "output_config" not in kw
    assert kw["messages"] == [{"role": "user", "content": "findings..."}]
    assert kw["temperature"] == 0.2


@pytest.mark.asyncio
async def test_complete_simple_caches_last_system_block_when_enabled():
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=True)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    client._create_message = AsyncMock(return_value=_fake_response("ok"))

    await client.complete_simple(
        model="m",
        system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        user="u",
    )
    sent = client._create_message.call_args.kwargs["system"]
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
    client._create_message = AsyncMock(return_value=resp)

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


def test_incomplete_markers_cover_all_giveup_paths():
    from ai_reviewer.agents import anthropic_client as ac

    assert ac.TOOL_LOOP_CAP_MARKER in ac.INCOMPLETE_SUMMARY_MARKERS
    assert ac.PARSE_ERROR_MARKER in ac.INCOMPLETE_SUMMARY_MARKERS
    assert ac.TRUNCATED_MARKER in ac.INCOMPLETE_SUMMARY_MARKERS
    assert any(m.startswith("[circuit breaker") for m in ac.INCOMPLETE_SUMMARY_MARKERS)


@pytest.mark.asyncio
async def test_run_review_truncated_response_is_marked_incomplete():
    """stop_reason=max_tokens must not read as a clean zero-finding review."""
    client, mock_create = _client_with_mocked_sdk(stop_reason="max_tokens", text="")
    result = await client.run_review(
        model="claude-sonnet-5",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=None,
        enable_thinking=False,
    )
    from ai_reviewer.agents.anthropic_client import TRUNCATED_MARKER

    assert TRUNCATED_MARKER in result.parsed["summary"]


@pytest.fixture
def _no_retry_sleep(monkeypatch):
    monkeypatch.setattr(
        "ai_reviewer.agents.anthropic_client.asyncio.sleep", AsyncMock(return_value=None)
    )


@pytest.mark.asyncio
async def test_create_message_retries_grammar_timeout_then_succeeds(_no_retry_sleep):
    """A 'Grammar compilation timed out' 400 is retried; the retry's result is returned."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    final = _fake_response('{"findings": [], "summary": "ok"}')
    stream, calls = _mock_stream_boundary(
        client,
        [_bad_request("Error code: 400 - Grammar compilation timed out."), final],
    )

    result = await client._create_message(model="claude-sonnet-5", max_tokens=8192)

    assert result is final
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_create_message_does_not_retry_other_bad_request():
    """A 400 without the grammar-timeout marker propagates on the first failure."""
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    stream, calls = _mock_stream_boundary(
        client,
        [_bad_request("Error code: 400 - invalid model"), _fake_response("{}")],
    )

    with pytest.raises(BadRequestError):
        await client._create_message(model="claude-sonnet-5", max_tokens=8192)

    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_create_message_grammar_timeout_retries_are_exhausted(_no_retry_sleep):
    """After _GRAMMAR_TIMEOUT_MAX_RETRIES consecutive failures the error is raised."""
    from ai_reviewer.agents.anthropic_client import _GRAMMAR_TIMEOUT_MAX_RETRIES

    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    total_attempts = _GRAMMAR_TIMEOUT_MAX_RETRIES + 1
    stream, calls = _mock_stream_boundary(
        client,
        [_bad_request("Grammar compilation timed out") for _ in range(total_attempts)],
    )

    with pytest.raises(BadRequestError):
        await client._create_message(model="claude-sonnet-5", max_tokens=8192)

    assert calls["n"] == total_attempts


def test_run_review_default_tool_rounds_matches_tool_call_budget():
    import inspect

    from ai_reviewer.agents.anthropic_client import AnthropicClient

    sig = inspect.signature(AnthropicClient.run_review)
    assert sig.parameters["max_tool_rounds"].default == 20


def _api_connection_error() -> "ac.anthropic.APIConnectionError":
    return ac.anthropic.APIConnectionError(
        message="Server disconnected without sending a response",
        request=httpx.Request("POST", "http://x"),
    )


@pytest.mark.asyncio
async def test_create_message_bounds_connection_error_retries(monkeypatch):
    """A persistently dropped connection is retried at most twice (1 + 2) then
    re-raised — seconds-scale failure, not a multi-minute retry storm."""
    monkeypatch.setattr(ac.asyncio, "sleep", AsyncMock())
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    _, calls = _mock_stream_boundary(client, [_api_connection_error() for _ in range(3)])

    with pytest.raises(ac.anthropic.APIConnectionError):
        await client._create_message(model="claude-sonnet-5", max_tokens=8192)

    assert calls["n"] == 3  # initial attempt + 2 bounded retries


@pytest.mark.asyncio
async def test_create_message_connection_error_recovers_on_retry(monkeypatch):
    """A transient connection drop that clears on the second attempt returns the
    response instead of failing the review."""
    monkeypatch.setattr(ac.asyncio, "sleep", AsyncMock())
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    final = _fake_response('{"findings": [], "summary": "ok"}')
    _, calls = _mock_stream_boundary(client, [_api_connection_error(), final])

    result = await client._create_message(model="claude-sonnet-5", max_tokens=8192)

    assert result is final
    assert calls["n"] == 2


def _overloaded_error() -> "ac.anthropic.APIStatusError":
    resp = httpx.Response(529, request=httpx.Request("POST", "http://x"))
    return ac.anthropic.APIStatusError("Error code: 529 - Overloaded", response=resp, body=None)


@pytest.mark.asyncio
async def test_create_message_retries_overloaded_then_succeeds(monkeypatch):
    """A 529 Overloaded is retried with backoff; the recovered response is returned."""
    monkeypatch.setattr(ac.asyncio, "sleep", AsyncMock())
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    final = _fake_response('{"findings": [], "summary": "ok"}')
    _, calls = _mock_stream_boundary(client, [_overloaded_error(), final])

    result = await client._create_message(model="claude-sonnet-5", max_tokens=8192)

    assert result is final
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_create_message_bounds_overloaded_retries(monkeypatch):
    """A sustained 529 is retried a bounded number of times, then re-raised."""
    from ai_reviewer.agents.anthropic_client import _OVERLOADED_MAX_RETRIES

    monkeypatch.setattr(ac.asyncio, "sleep", AsyncMock())
    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    total = _OVERLOADED_MAX_RETRIES + 1
    _, calls = _mock_stream_boundary(client, [_overloaded_error() for _ in range(total)])

    with pytest.raises(ac.anthropic.APIStatusError):
        await client._create_message(model="claude-sonnet-5", max_tokens=8192)

    assert calls["n"] == total


@pytest.mark.asyncio
async def test_run_review_returns_deadline_marker_when_budget_exceeded():
    """An agent that keeps requesting tools past its wall-clock budget must stop
    with DEADLINE_MARKER (a recognized incomplete marker), not loop for minutes."""
    from ai_reviewer.agents.anthropic_client import DEADLINE_MARKER, INCOMPLETE_SUMMARY_MARKERS

    cfg = AnthropicApiConfig(api_key="sk-test", enable_prompt_caching=False)
    client = AnthropicClient(cfg)
    client._sdk = MagicMock()
    # Always ask for another tool, so only the deadline can end the loop.
    client._create_message = AsyncMock(
        return_value=_tool_use_response("t1", "read_file", {"path": "a.py"})
    )
    registry = MagicMock()
    registry.tool_specs.return_value = [{"name": "read_file", "input_schema": {}}]
    registry.execute = AsyncMock(return_value="contents")

    result = await client.run_review(
        model="claude-sonnet-4-6",
        system_blocks=[{"type": "text", "text": "s"}],
        user_blocks=[{"type": "text", "text": "u"}],
        output_schema={"type": "object"},
        tool_registry=registry,
        max_tool_rounds=1000,
        max_review_seconds=0,
    )

    assert result.parsed["summary"] == DEADLINE_MARKER
    assert DEADLINE_MARKER in INCOMPLETE_SUMMARY_MARKERS
    # Deadline tripped before an unbounded number of rounds ran.
    assert client._create_message.await_count < 1000


_FENCE = "`" * 3


def test_parse_json_ignores_stray_suggestion_fence_in_prose():
    # Regression: a ```suggestion``` fence mentioned in prose must not hijack
    # extraction and discard the real findings JSON that follows (PR #118).
    resp = (
        "For style issues, wrap the change in a " + _FENCE + "suggestion" + _FENCE + " block.\n\n"
        '{"findings": [{"file_path": "a.py", "line_start": 1, '
        '"severity": "warning", "description": "x"}], "summary": "ok"}\n'
    )
    out = ac._parse_json(resp)
    assert out["summary"] == "ok"
    assert len(out["findings"]) == 1


def test_parse_json_handles_json_fence_and_plain_fence():
    wrapped_json = _FENCE + 'json\n{"findings": [], "summary": "a"}\n' + _FENCE
    assert ac._parse_json(wrapped_json)["summary"] == "a"
    wrapped_plain = _FENCE + '\n{"findings": [], "summary": "b"}\n' + _FENCE
    assert ac._parse_json(wrapped_plain)["summary"] == "b"


def test_parse_json_unparseable_demotes_to_marker():
    out = ac._parse_json("no json here at all")
    assert out["summary"] == ac.PARSE_ERROR_MARKER
    assert out["findings"] == []
