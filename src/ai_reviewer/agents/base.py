"""Base class for review agents."""

import logging
import time
from typing import Any

from ai_reviewer.agents.cursor_client import CursorClient
from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import Category, ReviewFinding, Severity
from ai_reviewer.models.review import AgentReview

logger = logging.getLogger(__name__)


class ReviewAgent:
    """Base class for all review agents."""

    # Subclasses should override these
    MODEL: str = "claude-3-opus-20240229"
    AGENT_TYPE: str = "base"
    FOCUS_AREAS: list[str] = []
    SYSTEM_PROMPT: str = "You are a code reviewer."

    def __init__(self, client: CursorClient, agent_id: str | None = None) -> None:
        """Initialize the agent.

        Args:
            client: Cursor API client for LLM access
            agent_id: Optional custom agent ID (defaults to class-based ID)
        """
        self.client = client
        self._agent_id = agent_id or f"{self.AGENT_TYPE}-{id(self)}"

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent instance."""
        return self._agent_id

    @property
    def focus_areas(self) -> list[str]:
        """Categories this agent specializes in."""
        return self.FOCUS_AREAS

    async def review(
        self,
        diff: str,
        file_contents: dict[str, str],
        context: ReviewContext | dict[str, Any],
    ) -> AgentReview:
        """Perform code review and return findings.

        Args:
            diff: The git diff to review
            file_contents: Full contents of changed files
            context: Additional context (PR description, repo info, etc.)

        Returns:
            AgentReview with findings and summary
        """
        start_time = time.monotonic()

        # Build the review prompt
        user_prompt = self._build_review_prompt(diff, file_contents, context)

        # Get review from LLM
        try:
            response = await self.client.complete_json(
                model=self.MODEL,
                system_prompt=self._get_system_prompt(),
                user_prompt=user_prompt,
                temperature=0.3,
            )
            findings = self._parse_findings(response)
            summary = response.get("summary", "Review completed")
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed: {e}")
            raise

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return AgentReview(
            agent_id=self.agent_id,
            agent_type=self.AGENT_TYPE,
            focus_areas=self.focus_areas,
            findings=findings,
            summary=summary,
            review_time_ms=elapsed_ms,
        )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return f"""{self.SYSTEM_PROMPT}

You MUST respond with valid JSON in this exact format:
{{
    "findings": [
        {{
            "file_path": "path/to/file.py",
            "line_start": 10,
            "line_end": 15,
            "severity": "critical|warning|suggestion|nitpick",
            "category": "security|performance|logic|style|architecture|testing|documentation",
            "title": "Short descriptive title",
            "description": "Detailed description of the issue",
            "suggested_fix": "Optional code or description of fix",
            "confidence": 0.95
        }}
    ],
    "summary": "Brief summary of the review"
}}

Rules:
- Only report issues you can clearly identify in the code
- Be specific about file paths and line numbers
- Confidence should reflect how certain you are (0.0 to 1.0)
- Do not make up issues - if the code is fine, return empty findings
- Focus on your specialty areas: {", ".join(self.focus_areas)}
"""

    def _build_review_prompt(
        self,
        diff: str,
        file_contents: dict[str, str],
        context: ReviewContext | dict[str, Any],
    ) -> str:
        """Build the user prompt for the review request."""
        # Handle both ReviewContext objects and dicts
        if isinstance(context, ReviewContext):
            context_str = context.to_prompt_context()
        else:
            context_str = f"Repository: {context.get('repo_name', 'unknown')}\nPR: #{context.get('pr_number', 0)}"

        files_str = ""
        if file_contents:
            files_str = "\n\n## Full File Contents\n"
            for path, content in file_contents.items():
                files_str += f"\n### {path}\n```\n{content}\n```\n"

        return f"""{context_str}

## Code Changes (Diff)
```diff
{diff}
```
{files_str}

Please review the code changes above and identify any issues within your focus areas: {", ".join(self.focus_areas)}.
"""

    def _parse_findings(self, response: dict[str, Any]) -> list[ReviewFinding]:
        """Parse findings from LLM response."""
        findings = []
        raw_findings = response.get("findings", [])

        for raw in raw_findings:
            try:
                finding = ReviewFinding(
                    file_path=raw["file_path"],
                    line_start=int(raw["line_start"]),
                    line_end=int(raw["line_end"]) if raw.get("line_end") else None,
                    severity=Severity(raw["severity"].lower()),
                    category=Category(raw["category"].lower()),
                    title=raw["title"],
                    description=raw["description"],
                    suggested_fix=raw.get("suggested_fix"),
                    confidence=float(raw.get("confidence", 0.8)),
                )
                findings.append(finding)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse finding: {e}, raw: {raw}")
                continue

        return findings
