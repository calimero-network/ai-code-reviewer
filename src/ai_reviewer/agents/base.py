"""Base class for review agents (Anthropic Messages API backed)."""

from __future__ import annotations

import logging
import time
from typing import Any

from ai_reviewer.agents.anthropic_client import AnthropicClient
from ai_reviewer.context.builder import FINDINGS_SCHEMA
from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import Category, ReviewFinding, Severity
from ai_reviewer.models.review import AgentReview

logger = logging.getLogger(__name__)


class ReviewAgent:
    """Base class for all review agents."""

    MODEL: str = "claude-opus-4-6"
    AGENT_TYPE: str = "base"
    FOCUS_AREAS: list[str] = []
    SYSTEM_PROMPT: str = "You are a code reviewer."
    THINKING_ENABLED: bool = False

    def __init__(
        self,
        client: AnthropicClient,
        agent_id: str,
        system_blocks: list[dict[str, Any]],
        user_blocks: list[dict[str, Any]],
        tool_registry: Any,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        thinking_enabled: bool | None = None,
    ) -> None:
        self.client = client
        self._agent_id = agent_id
        self._system_blocks = system_blocks
        self._user_blocks = user_blocks
        self._tool_registry = tool_registry
        self._max_tokens = max_tokens
        self._temperature = temperature
        # Config override takes precedence over class-level default
        self._thinking_enabled = thinking_enabled if thinking_enabled is not None else self.THINKING_ENABLED

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def focus_areas(self) -> list[str]:
        return self.FOCUS_AREAS

    async def review(
        self,
        diff: str,
        file_contents: dict[str, str],
        context: ReviewContext | dict[str, Any],
    ) -> AgentReview:
        start_time = time.monotonic()

        system_blocks = self._prepend_role(self._system_blocks)

        try:
            result = await self.client.run_review(
                model=self.MODEL,
                system_blocks=system_blocks,
                user_blocks=self._user_blocks,
                output_schema=FINDINGS_SCHEMA,
                tool_registry=self._tool_registry,
                enable_thinking=self._thinking_enabled,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        except Exception as e:
            logger.error("Agent %s failed: %s", self.agent_id, e)
            raise

        findings = _parse_findings(result.parsed)
        summary = result.parsed.get("summary", "Review completed")
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return AgentReview(
            agent_id=self.agent_id,
            agent_type=self.AGENT_TYPE,
            focus_areas=self.focus_areas,
            findings=findings,
            summary=summary,
            review_time_ms=elapsed_ms,
        )

    def _prepend_role(self, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Inject this agent's SYSTEM_PROMPT as the first block."""
        role_block = {"type": "text", "text": self.SYSTEM_PROMPT}
        return [role_block, *blocks]


def _parse_findings(parsed: dict[str, Any]) -> list[ReviewFinding]:
    findings: list[ReviewFinding] = []
    for raw in parsed.get("findings", []) or []:
        try:
            findings.append(
                ReviewFinding(
                    file_path=raw["file_path"],
                    line_start=int(raw["line_start"]),
                    line_end=int(raw["line_end"]) if raw.get("line_end") else None,
                    severity=Severity(str(raw["severity"]).lower()),
                    category=Category(str(raw["category"]).lower()),
                    title=raw["title"],
                    description=raw["description"],
                    suggested_fix=raw.get("suggested_fix"),
                    confidence=float(raw.get("confidence", 0.8)),
                )
            )
        except (KeyError, ValueError) as e:
            logger.warning("Failed to parse finding: %s, raw=%r", e, raw)
    return findings
