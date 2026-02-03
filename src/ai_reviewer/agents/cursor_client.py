"""Cursor Background Agent API client for code review."""

import asyncio
import base64
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CURSOR_API_BASE = "https://api.cursor.com/v0"
TERMINAL_STATUSES = {"COMPLETED", "FINISHED", "FAILED", "EXPIRED"}


@dataclass
class CursorConfig:
    """Configuration for Cursor API client."""

    api_key: str
    base_url: str = CURSOR_API_BASE
    timeout: int = 120
    poll_interval_seconds: int = 5
    max_wait_seconds: int = 15 * 60  # 15 minutes


@dataclass
class AgentResult:
    """Result from a Cursor agent execution."""

    id: str
    status: str
    content: str = ""
    branch_name: str = ""
    pr_url: str | None = None


class CursorClient:
    """Client for Cursor Background Agent API."""

    def __init__(self, config: CursorConfig) -> None:
        """Initialize the Cursor client.

        Args:
            config: Configuration for the client
        """
        self.config = config
        # Cursor uses Basic auth with key:
        auth_string = f"{config.api_key}:"
        encoded = base64.b64encode(auth_string.encode()).decode()

        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Basic {encoded}",
                "Content-Type": "application/json",
            },
            timeout=config.timeout,
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "CursorClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def create_agent(
        self,
        repo_url: str,
        ref: str,
        prompt: str,
        branch_name: str | None = None,
        auto_create_pr: bool = False,
    ) -> dict[str, Any]:
        """Create a Cursor background agent.

        Args:
            repo_url: GitHub repository URL (e.g., "https://github.com/owner/repo")
            ref: Branch or commit ref to analyze
            prompt: The prompt/instructions for the agent
            branch_name: Optional branch name for agent to create
            auto_create_pr: Whether to auto-create a PR

        Returns:
            Agent creation response with id and status
        """
        if branch_name is None:
            import time

            branch_name = f"ai-code-reviewer/review-{int(time.time())}"

        body = {
            "prompt": {"text": prompt},
            "source": {"repository": repo_url, "ref": ref},
            "target": {
                "branchName": branch_name,
                "autoCreatePr": auto_create_pr,
                "skipReviewerRequest": True,
            },
        }

        logger.debug(f"Creating agent for {repo_url} ref={ref}")
        response = await self._client.post("/agents", json=body)
        response.raise_for_status()

        data = response.json()
        agent = data.get("agent", data)

        if not agent.get("id"):
            raise RuntimeError("Cursor API did not return an agent id")

        logger.info(f"Created agent {agent['id']} with status {agent.get('status')}")
        return agent

    async def get_agent(self, agent_id: str) -> dict[str, Any]:
        """Get agent status.

        Args:
            agent_id: The agent ID

        Returns:
            Agent status response
        """
        response = await self._client.get(f"/agents/{agent_id}")
        response.raise_for_status()
        data = response.json()
        return data.get("agent", data)

    async def get_agent_conversation(self, agent_id: str) -> list[dict[str, Any]]:
        """Get the conversation/messages from an agent.

        Args:
            agent_id: The agent ID

        Returns:
            List of messages from the conversation
        """
        response = await self._client.get(f"/agents/{agent_id}/conversation")
        response.raise_for_status()
        data = response.json()
        return data.get("messages", data) or []

    async def wait_for_agent(
        self,
        agent_id: str,
        on_status: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Wait for an agent to complete.

        Args:
            agent_id: The agent ID to wait for
            on_status: Optional callback for status updates

        Returns:
            Final agent state
        """
        start = asyncio.get_event_loop().time()
        max_wait = self.config.max_wait_seconds

        while (asyncio.get_event_loop().time() - start) < max_wait:
            agent = await self.get_agent(agent_id)
            status = agent.get("status", "UNKNOWN")

            if on_status:
                on_status(status)

            logger.debug(f"Agent {agent_id} status: {status}")

            if status in TERMINAL_STATUSES:
                return agent

            await asyncio.sleep(self.config.poll_interval_seconds)

        raise TimeoutError(f"Timeout waiting for agent {agent_id}")

    async def run_review_agent(
        self,
        repo_url: str,
        ref: str,
        prompt: str,
        on_status: Callable[..., Any] | None = None,
    ) -> AgentResult:
        """Run a complete review agent flow.

        Creates an agent, waits for completion, and extracts the response.

        Args:
            repo_url: GitHub repository URL
            ref: Branch or commit ref
            prompt: Review prompt
            on_status: Optional status callback

        Returns:
            AgentResult with the review content
        """
        # Create agent
        agent = await self.create_agent(repo_url, ref, prompt)
        agent_id = agent["id"]
        branch_name = agent.get("target", {}).get("branchName", "")

        if on_status:
            on_status("CREATING")

        # Wait for completion
        final_agent = await self.wait_for_agent(agent_id, on_status)
        status = final_agent.get("status", "UNKNOWN")

        if status not in {"COMPLETED", "FINISHED"}:
            raise RuntimeError(f"Agent finished with status: {status}")

        # Get conversation and extract content
        messages = await self.get_agent_conversation(agent_id)
        content = self._extract_assistant_content(messages)

        return AgentResult(
            id=agent_id,
            status=status,
            content=content,
            branch_name=branch_name,
            pr_url=final_agent.get("target", {}).get("prUrl"),
        )

    def _extract_assistant_content(self, messages: list[dict]) -> str:
        """Extract the assistant's response from conversation messages."""
        if not messages:
            return ""

        # Cursor API may not include role field - look for assistant messages
        # or just get the last message that looks like a response
        for m in reversed(messages):
            role = m.get("role") or m.get("author")
            content = m.get("content", "") or m.get("text", "")

            # If it's explicitly an assistant message, use it
            if role == "assistant":
                return content

            # If no role but content contains JSON-like response, use it
            if not role and content and '{"findings"' in content:
                return content

        # Fallback: return last message content
        if messages:
            last = messages[-1]
            return last.get("content", "") or last.get("text", "")

        return ""

    # Legacy method for compatibility with existing agent code
    # Parameters kept for API compatibility even if unused in implementation
    async def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        response_format: str | None = None,
    ) -> str:
        """Legacy completion method - runs a quick agent for the prompt.

        Note: This uses the Background Agent API, so it may take longer
        than a direct chat completion API would.
        """
        # Combine prompts
        full_prompt = f"""## System Instructions
{system_prompt}

## User Request
{user_prompt}

Respond with the requested format."""

        # For code review, we don't need a repo - just analyze the provided diff
        # We'll use a minimal agent that just responds to the prompt
        result = await self._run_prompt_only_agent(full_prompt)
        return result

    async def _run_prompt_only_agent(self, prompt: str) -> str:
        """Run an agent with just a prompt (no repo context).

        This is a workaround since Cursor's API is designed for repo analysis.
        We create a minimal agent and extract its response.
        """
        # The Cursor API requires a repository, so we need to use a different approach
        # For now, we'll simulate the response structure

        # Actually, we need to think about this differently.
        # The Cursor Background Agent API is for repo-level tasks.
        # For simple chat completions, we might need to use direct API calls to Anthropic/OpenAI

        raise NotImplementedError(
            "Cursor Background Agent API requires a repository context. "
            "For direct chat completions, configure Anthropic or OpenAI API keys directly."
        )

    async def complete_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Legacy JSON completion method."""
        content = await self.complete(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format="json",
        )
        return self._parse_json_response(content)

    def _parse_json_response(self, content: str) -> dict[str, Any]:
        """Parse JSON from response, handling markdown code blocks."""
        content = content.strip()

        # Handle markdown code blocks
        if "```json" in content:
            match = re.search(r"```json\s*([\s\S]*?)```", content)
            if match:
                content = match.group(1).strip()
        elif "```" in content:
            match = re.search(r"```\s*([\s\S]*?)```", content)
            if match:
                content = match.group(1).strip()

        # Try to find JSON object
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            content = json_match.group(0)

        return json.loads(content)
