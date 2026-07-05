"""Anthropic Messages API client with tool-use loop, thinking, caching."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic  # noqa: TID251 — the single allowed SDK importer (architecture invariant I1)

from ai_reviewer.config import AnthropicApiConfig

logger = logging.getLogger(__name__)

# Models that reject temperature/top_p/top_k outright (400 invalid_request_error).
# ponytail: hardcoded set, add the next rejecting model here when it ships.
_NO_SAMPLING_PARAMS_MODELS = {
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
}


def _accepts_temperature(model: str) -> bool:
    return model not in _NO_SAMPLING_PARAMS_MODELS


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
        response = await self._sdk.messages.create(
            model=model,
            system=system,
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens,
        )
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
        if _accepts_temperature(model):
            kwargs["temperature"] = temperature
        response = await self._sdk.messages.create(**kwargs)
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
        max_tool_rounds: int = 8,
    ) -> AnthropicReviewResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_blocks}]
        usage = UsageStats()
        tool_calls: list[dict[str, Any]] = []

        tools = tool_registry.tool_specs() if tool_registry else None

        system_to_send = system_blocks
        if self.config.enable_prompt_caching and system_blocks:
            system_to_send = [dict(b) for b in system_blocks]
            system_to_send[-1]["cache_control"] = {"type": "ephemeral"}

        circuit_limit = self.config.max_combined_context_tokens * 2

        for _ in range(max_tool_rounds + 1):
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
                    parsed={"findings": [], "summary": "[circuit breaker: context limit exceeded]"},
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
            if _accepts_temperature(model):
                kwargs["temperature"] = 1.0 if enable_thinking else temperature
            if tools:
                kwargs["tools"] = tools
            if enable_thinking:
                kwargs["thinking"] = {"type": "adaptive"}

            response = await self._sdk.messages.create(**kwargs)
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
                return AnthropicReviewResult(
                    parsed=_parse_json(raw_text),
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
                try:
                    tool_output = await tool_registry.execute(block.name, block.input)
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
            parsed={"findings": [], "summary": "[tool loop cap]"},
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
        return {"findings": [], "summary": "[parse error]"}
