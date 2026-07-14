"""Anthropic Messages API client with tool-use loop, thinking, caching."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic  # noqa: TID251 — the single allowed SDK importer (architecture invariant I1)

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.tools.repo_tools import ToolBudgetExhausted

logger = logging.getLogger(__name__)

# Fed back to the model when the tool-call budget is exhausted mid-turn. Tells it
# to stop requesting tools and emit its final JSON review from evidence gathered.
_TOOL_BUDGET_EXHAUSTED_MSG = (
    "[tool budget exhausted — no more tool calls are available. Produce your "
    "final JSON review now from the evidence you have already gathered. Do not "
    "request more tools.]"
)

# Models that reject temperature/top_p/top_k outright (400 invalid_request_error).
# ponytail: hardcoded set, add the next rejecting model here when it ships.
_NO_SAMPLING_PARAMS_MODELS = {
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
}


# Models that reject an explicit thinking={"type": "disabled"} (thinking is always on).
_ALWAYS_THINKING_MODELS = {"claude-fable-5", "claude-mythos-5"}


# Adaptive thinking shares max_tokens with the response; the default effort "high"
# can exhaust the budget and truncate the findings JSON, so cap thinking effort here.
_THINKING_EFFORT = "medium"


# Server-side grammar (JSON-schema constrained decoding) compilation can time out on
# large context + schema requests, returning a 400. The SDK's max_retries only covers
# 408/429/5xx, so it never retries this; Anthropic's guidance treats it as often
# transient, so we retry it specifically here.
_GRAMMAR_TIMEOUT_MARKER = "Grammar compilation timed out"
_GRAMMAR_TIMEOUT_MAX_RETRIES = 2


# Summary markers for reviews that did NOT complete. aggregate_findings() treats
# any summary containing one of these as a failed agent — a give-up must never
# be indistinguishable from a genuinely clean review.
TOOL_LOOP_CAP_MARKER = "[tool loop cap]"
PARSE_ERROR_MARKER = "[parse error]"
CIRCUIT_BREAKER_MARKER = "[circuit breaker: context limit exceeded]"
TRUNCATED_MARKER = "[truncated at max_tokens]"
INCOMPLETE_SUMMARY_MARKERS: tuple[str, ...] = (
    TOOL_LOOP_CAP_MARKER,
    PARSE_ERROR_MARKER,
    "[circuit breaker",
    TRUNCATED_MARKER,
)


def _accepts_temperature(model: str) -> bool:
    return model not in _NO_SAMPLING_PARAMS_MODELS


def _sampling_params(
    model: str, enable_thinking: bool, temperature: float | None
) -> dict[str, Any]:
    """Thinking + temperature request params that are safe for the target model.

    Sonnet 5 (and Opus 4.7+) treat an *omitted* `thinking` field as adaptive-ON,
    so thinking-off agents must send an explicit {"type": "disabled"} or thinking
    silently eats the max_tokens budget. Fable/Mythos reject explicit "disabled"
    (thinking is always on there) — omit the field for those instead. Temperature
    is only sent to models that still accept it.
    """
    params: dict[str, Any] = {}
    if enable_thinking:
        params["thinking"] = {"type": "adaptive"}
        if _accepts_temperature(model):
            params["temperature"] = 1.0  # API requires temp=1 alongside thinking
    else:
        if model not in _ALWAYS_THINKING_MODELS:
            params["thinking"] = {"type": "disabled"}
        if temperature is not None and _accepts_temperature(model):
            params["temperature"] = temperature
    return params


class ToolRegistryProtocol(Protocol):
    """Structural interface the tool-use loop needs from a tool registry."""

    def tool_specs(self) -> list[dict[str, Any]]: ...

    async def execute(self, name: str, tool_input: dict[str, Any]) -> str: ...


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class AnthropicReviewResult:
    parsed: dict[str, Any]
    raw_text: str
    usage: UsageStats = field(default_factory=UsageStats)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class AnthropicClient:
    """Thin wrapper over the official anthropic SDK for review agents."""

    def __init__(self, config: AnthropicApiConfig) -> None:
        self.config = config
        self._sdk = anthropic.AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    async def _create_message(self, **kwargs: Any) -> Any:
        """Send one Messages request via streaming and return the final Message.

        Streaming keeps the connection active for the whole generation. Long
        non-streaming calls on large prompts leave the connection idle long
        enough for the network to drop it (httpx.ReadError -> APIConnectionError)
        and can deliver a corrupted/partial body that fails JSON parsing. The
        stream helper accumulates the response cleanly and returns the same
        Message object messages.create() would, so callers are unchanged.

        A "Grammar compilation timed out" 400 is retried here (see
        _GRAMMAR_TIMEOUT_MARKER); any other error propagates immediately.
        """
        for attempt in range(1, _GRAMMAR_TIMEOUT_MAX_RETRIES + 2):
            try:
                async with self._sdk.messages.stream(**kwargs) as stream:
                    return await stream.get_final_message()
            except anthropic.BadRequestError as exc:
                retriable = _GRAMMAR_TIMEOUT_MARKER in str(exc)
                if not retriable or attempt > _GRAMMAR_TIMEOUT_MAX_RETRIES:
                    raise
                logger.warning(
                    "Grammar compilation timed out, retrying (%d/%d)",
                    attempt,
                    _GRAMMAR_TIMEOUT_MAX_RETRIES,
                )
                await asyncio.sleep(attempt)  # 1s, 2s backoff

    async def run_completion(
        self,
        model: str,
        system: str,
        user: str,
        max_tokens: int = 4096,
    ) -> str:
        """Plain text completion — no tool use, no JSON schema.

        Used for prose generation tasks (e.g. doc drafting) where structured
        output is not needed.
        """
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
        }
        kwargs.update(_sampling_params(model, enable_thinking=False, temperature=None))
        response = await self._create_message(**kwargs)
        return _extract_text(response)

    async def complete_simple(
        self,
        model: str,
        system: str | list[dict[str, Any]],
        user: str,
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> str:
        """Single completion with no tools and no JSON schema.

        Used for lightweight calls such as cross-review that must still go through
        the client (architecture invariant I1) rather than the raw SDK, and to get
        usage logging. Logs token usage on every call.

        Caching: when a block list is given and caching is enabled, a cache_control
        breakpoint is placed on the last system block. This only yields a real cache
        hit when that system prefix exceeds the model's minimum cacheable length
        (~1024 tokens for Sonnet/Opus). For a small system prompt + large *user*
        message (the cross-review shape) it is a no-op — the breakpoint is set but
        nothing is cached. Put the large reusable content in a system block to
        benefit.
        """
        system_to_send = system
        if self.config.enable_prompt_caching and isinstance(system, list) and system:
            system_to_send = [dict(b) for b in system]
            system_to_send[-1]["cache_control"] = {"type": "ephemeral"}

        # Pass via a dict[str, Any] kwargs bag (as run_review does) so mypy does
        # not reject the structural TextBlockParam dicts.
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system_to_send,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens,
        }
        kwargs.update(_sampling_params(model, enable_thinking=False, temperature=temperature))
        response = await self._create_message(**kwargs)
        usage = UsageStats()
        _accumulate_usage(usage, response)
        logger.info(
            "complete_simple usage: input=%d output=%d cache_read=%d",
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
        )
        return _extract_text(response)

    async def close(self) -> None:
        await self._sdk.close()

    async def __aenter__(self) -> AnthropicClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def run_review(
        self,
        model: str,
        system_blocks: list[dict[str, Any]],
        user_blocks: list[dict[str, Any]],
        output_schema: dict[str, Any],
        tool_registry: ToolRegistryProtocol | None,
        enable_thinking: bool = False,
        max_tokens: int = 8192,
        temperature: float = 0.3,
        # 20 matches AgentConfig.max_tool_calls — the registry's per-review call
        # budget is the binding cap; an 8-round loop cap below it made the last
        # 12 calls unreachable and silently truncated agentic reviews on Sonnet 5.
        max_tool_rounds: int = 20,
    ) -> AnthropicReviewResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_blocks}]
        usage = UsageStats()
        tool_calls: list[dict[str, Any]] = []

        tools = tool_registry.tool_specs() if tool_registry else None
        # Once the registry's tool-call budget is exhausted, drop `tools` from
        # every subsequent request so the model must produce its final JSON
        # instead of thrashing on tool calls that can only fail (which used to
        # burn the round cap and mark the review incomplete).
        tool_budget_exhausted = False

        system_to_send = system_blocks
        if self.config.enable_prompt_caching and system_blocks:
            system_to_send = [dict(b) for b in system_blocks]
            system_to_send[-1]["cache_control"] = {"type": "ephemeral"}

        circuit_limit = self.config.max_combined_context_tokens * 2

        for round_idx in range(max_tool_rounds + 1):
            # On the final allowed round, stop offering tools so the model is
            # forced to emit its findings JSON from what it already gathered.
            # Without this, an agent that keeps calling tools until the cap exits
            # the loop with empty findings — discarding a full review and flipping
            # the whole PR to "Review Incomplete". Same salvage as tool-budget
            # exhaustion, applied to the round cap.
            last_round = round_idx == max_tool_rounds
            # Circuit breaker: abort before sending if accumulated input from prior rounds
            # already exceeds twice the context limit — paying for this call would just
            # make it worse without producing useful output.
            if usage.input_tokens > circuit_limit:
                logger.warning(
                    "Circuit breaker: accumulated input_tokens=%d exceeds 2× context limit=%d — "
                    "aborting tool loop before next request to contain cost",
                    usage.input_tokens,
                    self.config.max_combined_context_tokens,
                )
                return AnthropicReviewResult(
                    parsed={"findings": [], "summary": CIRCUIT_BREAKER_MARKER},
                    raw_text="",
                    usage=usage,
                    tool_calls=tool_calls,
                )

            kwargs: dict[str, Any] = {
                "model": model,
                "system": system_to_send,
                "messages": messages,
                "max_tokens": max_tokens,
                "output_config": {
                    "format": {"type": "json_schema", "schema": output_schema},
                },
            }
            if enable_thinking:
                kwargs["output_config"]["effort"] = _THINKING_EFFORT
            kwargs.update(_sampling_params(model, enable_thinking, temperature))
            if tools and not tool_budget_exhausted and not last_round:
                kwargs["tools"] = tools

            response = await self._create_message(**kwargs)
            _accumulate_usage(usage, response)

            stop = getattr(response, "stop_reason", None)
            if stop != "tool_use" or not tool_registry:
                total_tokens = usage.input_tokens + usage.output_tokens
                if total_tokens > 100_000:
                    logger.warning(
                        "High token usage: input=%d output=%d total=%d — "
                        "consider reducing max_tool_rounds or context size",
                        usage.input_tokens,
                        usage.output_tokens,
                        total_tokens,
                    )
                # Surface usage — including the cache counters — so prompt caching
                # can be validated from logs on a real review: cache_read > 0 on a
                # later round/agent proves a cache hit.
                logger.info(
                    "Review usage: input=%d output=%d cache_read=%d cache_creation=%d",
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.cache_read_input_tokens,
                    usage.cache_creation_input_tokens,
                )
                raw_text = _extract_text(response)
                parsed = _parse_json(raw_text)
                if stop == "max_tokens":
                    logger.warning(
                        "Response truncated at max_tokens=%d — marking review incomplete",
                        max_tokens,
                    )
                    parsed["summary"] = f"{TRUNCATED_MARKER} {parsed.get('summary', '')}".strip()
                return AnthropicReviewResult(
                    parsed=parsed,
                    raw_text=raw_text,
                    usage=usage,
                    tool_calls=tool_calls,
                )

            assistant_blocks = list(getattr(response, "content", []) or [])
            messages.append({"role": "assistant", "content": _serialize_blocks(assistant_blocks)})
            tool_result_blocks: list[dict[str, Any]] = []
            for block in assistant_blocks:
                if getattr(block, "type", None) != "tool_use":
                    continue
                tool_calls.append({"name": block.name, "input": block.input})
                if tool_budget_exhausted:
                    # Budget already blew earlier in this same turn — answer the
                    # remaining tool_use blocks without attempting execution.
                    tool_output = _TOOL_BUDGET_EXHAUSTED_MSG
                else:
                    try:
                        tool_output = await tool_registry.execute(block.name, block.input)
                    except ToolBudgetExhausted:
                        tool_budget_exhausted = True
                        tool_output = _TOOL_BUDGET_EXHAUSTED_MSG
                    except Exception as e:  # noqa: BLE001
                        tool_output = f"[tool error: {e}]"
                        logger.warning("Tool %s failed: %s", block.name, e)
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_output,
                    }
                )

            if self.config.enable_prompt_caching and tool_result_blocks:
                # Single MOVING breakpoint: caching is a prefix match, so
                # cache_control on the latest tool_result caches everything before
                # it — earlier breakpoints are redundant and must be pruned, else
                # system(1) + one-per-round exceeds Anthropic's 4-per-request cap
                # by the 5th round (a 400 that silently drops the whole review).
                # Skip messages[0], the caller's user turn: we only ever mark
                # appended tool_result turns (index >= 1), so the in-place pop
                # touches only internally-built dicts and never clobbers a
                # caller-supplied breakpoint.
                for msg in messages[1:]:
                    content = msg.get("content")
                    if isinstance(content, list):
                        for blk in content:
                            if isinstance(blk, dict):
                                blk.pop("cache_control", None)
                tool_result_blocks[-1] = dict(tool_result_blocks[-1])
                tool_result_blocks[-1]["cache_control"] = {"type": "ephemeral"}

            messages.append({"role": "user", "content": tool_result_blocks})

        logger.warning("Tool-use loop exceeded max_tool_rounds=%d", max_tool_rounds)
        return AnthropicReviewResult(
            parsed={"findings": [], "summary": TOOL_LOOP_CAP_MARKER},
            raw_text="",
            usage=usage,
            tool_calls=tool_calls,
        )


def _accumulate_usage(u: UsageStats, response: Any) -> None:
    ru = getattr(response, "usage", None)
    if not ru:
        return
    u.input_tokens += getattr(ru, "input_tokens", 0) or 0
    u.output_tokens += getattr(ru, "output_tokens", 0) or 0
    u.cache_read_input_tokens += getattr(ru, "cache_read_input_tokens", 0) or 0
    u.cache_creation_input_tokens += getattr(ru, "cache_creation_input_tokens", 0) or 0


def _extract_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts)


def _serialize_blocks(blocks: list[Any]) -> list[dict[str, Any]]:
    """Convert SDK block objects back to the dict form the API expects."""
    out: list[dict[str, Any]] = []
    for b in blocks:
        t = getattr(b, "type", None)
        if t == "text":
            out.append({"type": "text", "text": getattr(b, "text", "")})
        elif t == "tool_use":
            out.append(
                {
                    "type": "tool_use",
                    "id": b.id,
                    "name": b.name,
                    "input": b.input,
                }
            )
        elif t == "thinking":
            out.append(
                {
                    "type": "thinking",
                    "thinking": getattr(b, "thinking", ""),
                    "signature": getattr(b, "signature", ""),
                }
            )
    return out


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if "```json" in text:
        m = re.search(r"```json\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    elif "```" in text:
        m = re.search(r"```\s*([\s\S]*?)```", text)
        if m:
            text = m.group(1).strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON: %r", text[:200])
        return {"findings": [], "summary": PARSE_ERROR_MARKER}
