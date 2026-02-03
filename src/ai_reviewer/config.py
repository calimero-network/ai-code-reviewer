"""Configuration loading and validation for AI Code Reviewer."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AgentConfig:
    """Configuration for a single agent."""

    name: str
    model: str
    focus_areas: list[str]
    max_tokens: int = 4096
    temperature: float = 0.3
    custom_prompt_append: str | None = None
    include_codebase_context: bool = False


@dataclass
class CursorApiConfig:
    """Cursor API configuration."""

    api_key: str
    base_url: str = "https://api.cursor.com/v0"
    timeout_seconds: int = 120


@dataclass
class GitHubConfig:
    """GitHub integration configuration."""

    token: str
    webhook_secret: str | None = None
    app_id: str | None = None
    private_key_path: str | None = None


@dataclass
class OrchestratorSettings:
    """Orchestrator configuration."""

    timeout_seconds: int = 120
    min_agents_required: int = 2
    max_parallel_agents: int = 5
    retry_on_failure: bool = True
    max_retries: int = 2


@dataclass
class AggregatorSettings:
    """Aggregator configuration."""

    similarity_threshold: float = 0.85
    min_consensus_for_critical: float = 0.5
    use_embeddings: bool = False


@dataclass
class OutputSettings:
    """Output configuration."""

    include_agent_breakdown: bool = True
    include_confidence_scores: bool = True
    max_findings_per_file: int = 10
    max_total_findings: int = 50


@dataclass
class ReviewPolicy:
    """Review policy configuration."""

    auto_approve_if_no_findings: bool = False
    block_on_critical: bool = True
    require_human_review_for: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=list)


@dataclass
class ServerSettings:
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    health_check_path: str = "/health"
    metrics_enabled: bool = True


@dataclass
class Config:
    """Complete application configuration."""

    cursor: CursorApiConfig
    github: GitHubConfig
    agents: list[AgentConfig]
    orchestrator: OrchestratorSettings = field(default_factory=OrchestratorSettings)
    aggregator: AggregatorSettings = field(default_factory=AggregatorSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    review_policy: ReviewPolicy = field(default_factory=ReviewPolicy)
    server: ServerSettings = field(default_factory=ServerSettings)


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file and environment.

    Args:
        config_path: Path to config file (default: config.yaml)

    Returns:
        Loaded configuration
    """
    # Find config file
    if config_path is None:
        config_path = Path("config.yaml")
        if not config_path.exists():
            config_path = Path("config.example.yaml")

    # Load from file if exists
    raw_config: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            raw_config = yaml.safe_load(f) or {}

    # Expand environment variables
    raw_config = _expand_env_vars(raw_config)

    # Parse configuration
    return _parse_config(raw_config)


def _expand_env_vars(obj: Any) -> Any:
    """Recursively expand environment variables in config."""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.environ.get(env_var, "")
        return obj
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(v) for v in obj]
    return obj


def _parse_config(raw: dict[str, Any]) -> Config:
    """Parse raw config dict into Config object."""
    # Cursor config
    cursor_raw = raw.get("cursor", {})
    cursor = CursorApiConfig(
        api_key=cursor_raw.get("api_key") or os.environ.get("CURSOR_API_KEY", ""),
        base_url=cursor_raw.get("base_url", "https://api.cursor.com/v0"),
        timeout_seconds=cursor_raw.get("timeout_seconds", 120),
    )

    # GitHub config
    github_raw = raw.get("github", {})
    github = GitHubConfig(
        token=github_raw.get("token") or os.environ.get("GITHUB_TOKEN", ""),
        webhook_secret=github_raw.get("webhook_secret"),
        app_id=github_raw.get("app_id"),
        private_key_path=github_raw.get("private_key_path"),
    )

    # Agents config
    agents = []
    for agent_raw in raw.get("agents", []):
        agents.append(
            AgentConfig(
                name=agent_raw["name"],
                model=agent_raw["model"],
                focus_areas=agent_raw.get("focus_areas", []),
                max_tokens=agent_raw.get("max_tokens", 4096),
                temperature=agent_raw.get("temperature", 0.3),
                custom_prompt_append=agent_raw.get("custom_prompt_append"),
                include_codebase_context=agent_raw.get("include_codebase_context", False),
            )
        )

    # Default agents if none configured
    if not agents:
        agents = [
            AgentConfig(
                name="security-reviewer",
                model="claude-3-opus-20240229",
                focus_areas=["security", "authentication"],
            ),
            AgentConfig(
                name="performance-reviewer",
                model="gpt-4-turbo-preview",
                focus_areas=["performance", "complexity"],
            ),
            AgentConfig(
                name="patterns-reviewer",
                model="claude-3-opus-20240229",
                focus_areas=["consistency", "patterns"],
            ),
        ]

    # Orchestrator settings
    orch_raw = raw.get("orchestrator", {})
    orchestrator = OrchestratorSettings(
        timeout_seconds=orch_raw.get("timeout_seconds", 120),
        min_agents_required=orch_raw.get("min_agents_required", 2),
        max_parallel_agents=orch_raw.get("max_parallel_agents", 5),
        retry_on_failure=orch_raw.get("retry_on_failure", True),
        max_retries=orch_raw.get("max_retries", 2),
    )

    # Aggregator settings
    agg_raw = raw.get("aggregator", {})
    aggregator = AggregatorSettings(
        similarity_threshold=agg_raw.get("similarity_threshold", 0.85),
        min_consensus_for_critical=agg_raw.get("min_consensus_for_critical", 0.5),
        use_embeddings=agg_raw.get("use_embeddings", False),
    )

    # Output settings
    out_raw = raw.get("output", {})
    output = OutputSettings(
        include_agent_breakdown=out_raw.get("include_agent_breakdown", True),
        include_confidence_scores=out_raw.get("include_confidence_scores", True),
        max_findings_per_file=out_raw.get("max_findings_per_file", 10),
        max_total_findings=out_raw.get("max_total_findings", 50),
    )

    # Review policy
    policy_raw = raw.get("review_policy", {})
    review_policy = ReviewPolicy(
        auto_approve_if_no_findings=policy_raw.get("auto_approve_if_no_findings", False),
        block_on_critical=policy_raw.get("block_on_critical", True),
        require_human_review_for=policy_raw.get("require_human_review_for", []),
        ignore_patterns=policy_raw.get("ignore_patterns", []),
    )

    # Server settings
    server_raw = raw.get("server", {})
    server = ServerSettings(
        host=server_raw.get("host", "0.0.0.0"),
        port=server_raw.get("port", 8080),
        health_check_path=server_raw.get("health_check_path", "/health"),
        metrics_enabled=server_raw.get("metrics_enabled", True),
    )

    return Config(
        cursor=cursor,
        github=github,
        agents=agents,
        orchestrator=orchestrator,
        aggregator=aggregator,
        output=output,
        review_policy=review_policy,
        server=server,
    )


def validate_config(config: Config) -> list[str]:
    """Validate configuration and return list of errors.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not config.cursor.api_key:
        errors.append("Missing Cursor API key (set CURSOR_API_KEY or cursor.api_key)")

    if not config.github.token:
        errors.append("Missing GitHub token (set GITHUB_TOKEN or github.token)")

    if not config.agents:
        errors.append("No agents configured")

    if config.orchestrator.min_agents_required > len(config.agents):
        errors.append(
            f"min_agents_required ({config.orchestrator.min_agents_required}) "
            f"exceeds available agents ({len(config.agents)})"
        )

    return errors
