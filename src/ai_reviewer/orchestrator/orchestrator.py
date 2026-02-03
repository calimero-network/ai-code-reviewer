"""Agent orchestrator for parallel review execution."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ai_reviewer.agents.base import ReviewAgent
from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.review import AgentReview

logger = logging.getLogger(__name__)


class InsufficientAgentsError(Exception):
    """Raised when too few agents succeed to produce a valid review."""

    pass


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""

    timeout_seconds: int = 120
    min_agents_required: int = 2
    max_parallel_agents: int = 5
    retry_on_failure: bool = True
    max_retries: int = 2


class AgentOrchestrator:
    """Coordinates multiple agents to review code in parallel."""

    def __init__(
        self,
        agents: list[ReviewAgent],
        timeout_seconds: int = 120,
        min_agents_required: int = 2,
        config: OrchestratorConfig | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            agents: List of review agents to coordinate
            timeout_seconds: Maximum time to wait for each agent
            min_agents_required: Minimum successful agents for valid review
            config: Optional full configuration (overrides other params)
        """
        self.agents = agents
        self.config = config or OrchestratorConfig(
            timeout_seconds=timeout_seconds,
            min_agents_required=min_agents_required,
        )

    async def review(
        self,
        diff: str,
        file_contents: dict[str, str],
        context: ReviewContext | dict[str, Any],
    ) -> list[AgentReview]:
        """Execute all agents in parallel and collect results.

        Args:
            diff: Git diff to review
            file_contents: Full contents of changed files
            context: Review context information

        Returns:
            List of successful agent reviews

        Raises:
            InsufficientAgentsError: If too few agents succeed
        """
        logger.info(f"Starting review with {len(self.agents)} agents")

        # Create tasks for all agents
        tasks = [
            asyncio.create_task(
                self._run_agent_with_timeout(agent, diff, file_contents, context),
                name=f"agent-{agent.agent_id}",
            )
            for agent in self.agents
        ]

        # Wait for all tasks, collecting results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful_reviews: list[AgentReview] = []
        failed_agents: list[str] = []

        for agent, result in zip(self.agents, results):
            if isinstance(result, AgentReview):
                successful_reviews.append(result)
                logger.info(
                    f"Agent {agent.agent_id} completed: {len(result.findings)} findings"
                )
            elif isinstance(result, asyncio.TimeoutError):
                failed_agents.append(f"{agent.agent_id} (timeout)")
                logger.warning(f"Agent {agent.agent_id} timed out")
            elif isinstance(result, Exception):
                failed_agents.append(f"{agent.agent_id} ({type(result).__name__})")
                logger.error(f"Agent {agent.agent_id} failed: {result}")
            else:
                failed_agents.append(f"{agent.agent_id} (unknown)")
                logger.error(f"Agent {agent.agent_id} returned unexpected: {result}")

        # Check if we have enough successful reviews
        if len(successful_reviews) < self.config.min_agents_required:
            raise InsufficientAgentsError(
                f"Only {len(successful_reviews)} agents succeeded, "
                f"minimum {self.config.min_agents_required} required. "
                f"Failed: {', '.join(failed_agents)}"
            )

        logger.info(
            f"Review complete: {len(successful_reviews)} successful, "
            f"{len(failed_agents)} failed"
        )

        return successful_reviews

    async def _run_agent_with_timeout(
        self,
        agent: ReviewAgent,
        diff: str,
        file_contents: dict[str, str],
        context: ReviewContext | dict[str, Any],
    ) -> AgentReview:
        """Run a single agent with timeout.

        Args:
            agent: Agent to run
            diff: Git diff to review
            file_contents: File contents for context
            context: Review context

        Returns:
            AgentReview from the agent

        Raises:
            asyncio.TimeoutError: If agent exceeds timeout
        """
        return await asyncio.wait_for(
            agent.review(diff, file_contents, context),
            timeout=self.config.timeout_seconds,
        )

    async def review_with_retry(
        self,
        diff: str,
        file_contents: dict[str, str],
        context: ReviewContext | dict[str, Any],
    ) -> list[AgentReview]:
        """Execute review with retry logic for failed agents.

        Args:
            diff: Git diff to review
            file_contents: Full contents of changed files
            context: Review context information

        Returns:
            List of successful agent reviews
        """
        all_reviews: list[AgentReview] = []
        remaining_agents = list(self.agents)
        attempts = 0

        while remaining_agents and attempts < self.config.max_retries + 1:
            if attempts > 0:
                logger.info(f"Retry attempt {attempts} with {len(remaining_agents)} agents")

            tasks = [
                asyncio.create_task(
                    self._run_agent_with_timeout(agent, diff, file_contents, context)
                )
                for agent in remaining_agents
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            still_failing = []
            for agent, result in zip(remaining_agents, results):
                if isinstance(result, AgentReview):
                    all_reviews.append(result)
                else:
                    still_failing.append(agent)

            remaining_agents = still_failing if self.config.retry_on_failure else []
            attempts += 1

        if len(all_reviews) < self.config.min_agents_required:
            raise InsufficientAgentsError(
                f"Only {len(all_reviews)} agents succeeded after retries"
            )

        return all_reviews
