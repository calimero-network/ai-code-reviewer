"""Unified Cursor API client for accessing multiple LLM models."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class CursorConfig:
    """Configuration for Cursor API client."""

    api_key: str
    base_url: str = "https://api.cursor.com/v1"
    timeout: int = 120
    max_retries: int = 3


@dataclass
class CompletionResponse:
    """Response from a completion request."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"


class CursorClient:
    """Unified client for accessing multiple LLM models via Cursor API."""

    def __init__(self, config: CursorConfig) -> None:
        """Initialize the Cursor client.

        Args:
            config: Configuration for the client
        """
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
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

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def complete(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        response_format: Optional[str] = None,
    ) -> str:
        """Send a completion request to the specified model.

        Args:
            model: Model identifier (e.g., "claude-3-opus-20240229", "gpt-4-turbo")
            system_prompt: System prompt for the model
            user_prompt: User prompt/question
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 - 1.0)
            response_format: Optional response format ("json" for JSON mode)

        Returns:
            The model's response content as a string
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if response_format == "json":
            payload["response_format"] = {"type": "json_object"}

        logger.debug(f"Sending request to model {model}")

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        logger.debug(f"Received response from {model}: {len(content)} chars")

        return content

    async def complete_json(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Send a completion request expecting JSON response.

        Args:
            model: Model identifier
            system_prompt: System prompt (should instruct JSON output)
            user_prompt: User prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Parsed JSON response as a dictionary
        """
        content = await self.complete(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format="json",
        )

        # Try to parse JSON, handling potential markdown code blocks
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        return json.loads(content)
