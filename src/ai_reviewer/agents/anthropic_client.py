"""Anthropic Messages API client with tool-use loop, thinking, caching."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic

from ai_reviewer.config import AnthropicApiConfig

logger = logging.getLogger(__name__)


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
        tool_registry: Any,
        thinking_budget: int | None,
        max_tokens: int,
        temperature: float,
        max_tool_rounds: int = 30,
    ) -> AnthropicReviewResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_blocks}]
        usage = UsageStats()
        tool_calls: list[dict[str, Any]] = []

        tools = tool_registry.tool_specs() if tool_registry else None

        system_to_send = system_blocks
        if self.config.enable_prompt_caching and system_blocks:
            system_to_send = [dict(b) for b in system_blocks]
            system_to_send[-1]["cache_control"] = {"type": "ephemeral"}

        for _ in range(max_tool_rounds + 1):
            kwargs: dict[str, Any] = {
                "model": model,
                "system": system_to_send,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "output_config": {
                    "format": {"type": "json_schema", "schema": output_schema},
                },
            }
            if tools:
                kwargs["tools"] = tools
            if thinking_budget:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
                if temperature != 1.0:
                    logger.info("Thinking enabled; overriding temperature -> 1.0")
                    kwargs["temperature"] = 1.0

            response = await self._sdk.messages.create(**kwargs)
            _accumulate_usage(usage, response)

            stop = getattr(response, "stop_reason", None)
            if stop != "tool_use" or not tool_registry:
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
