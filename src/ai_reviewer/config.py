"""Configuration loading and validation for AI Code Reviewer."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Single source of truth for the current Anthropic model generation.
# Bump these two when a new model ships; every default below inherits from them.
DEFAULT_SONNET_MODEL = "claude-sonnet-5"
DEFAULT_HAIKU_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class AgentConfig:
    """Configuration for a single agent."""

    name: str
    model: str
    focus_areas: list[str]
    max_tokens: int = 8192
    temperature: float = 0.3
    custom_prompt_append: str | None = None
    include_codebase_context: bool = False
    thinking_enabled: bool = False
    allow_tool_use: bool = True
    max_tool_calls: int = 20


@dataclass
class AnthropicApiConfig:
    """Anthropic Messages API configuration."""

    api_key: str
    base_url: str = "https://api.anthropic.com"
    timeout_seconds: int = 300
    # Long non-streaming review calls occasionally have their connection dropped
    # by the network (httpx.ReadError -> APIConnectionError). These are transient
    # and retryable; 1 retry was too few (one of N parallel agents reliably
    # failed and the whole review posted "incomplete"). 3 rescues the vast majority.
    max_retries: int = 3
    # Wall-clock budget for one agent's whole run_review call (all tool rounds and
    # in-call retries). Backstops the SDK retry x per-attempt-timeout product that
    # let a single dead streaming connection burn ~20 minutes.
    agent_review_deadline_seconds: int = 900
    default_model: str = DEFAULT_SONNET_MODEL
    enable_prompt_caching: bool = True
    max_combined_context_tokens: int = 80_000
    per_file_max_bytes: int = 512 * 1024
    per_review_github_request_budget: int = 200
    max_tool_result_bytes: int = 16 * 1024
    # Pull-based context: changed files longer than this are sent as hunk excerpts
    # (agents pull the rest via read_file) instead of full contents.
    full_file_max_lines: int = 300
    hunk_context_lines: int = 60
    # Aggregate cap on the concatenated project-conventions block.
    conventions_max_chars: int = 16_000


@dataclass
class GitHubConfig:
    """GitHub integration configuration."""

    token: str
    webhook_secret: str | None = None
    app_id: str | None = None
    private_key_path: str | None = None
    extra_reviewer_users: list[str] = field(default_factory=list)


@dataclass
class OrchestratorSettings:
    """Orchestrator configuration."""

    timeout_seconds: int = 120
    min_agents_required: int = 2
    max_parallel_agents: int = 5
    retry_on_failure: bool = True
    max_retries: int = 1


@dataclass
class AggregatorSettings:
    """Aggregator configuration."""

    similarity_threshold: float = 0.85
    min_consensus_for_critical: float = 0.5
    use_embeddings: bool = False
    min_confidence_critical: float = 0.3
    min_confidence_warning: float = 0.4
    min_confidence_suggestion: float = 0.5
    min_confidence_nitpick: float = 0.6


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
    secret_scan_exclude: list[str] = field(default_factory=list)


@dataclass
class ServerSettings:
    """Server configuration."""

    host: str = "0.0.0.0"
    port: int = 8080
    health_check_path: str = "/health"
    metrics_enabled: bool = True


@dataclass
class DocReviewSettings:
    """Documentation review configuration."""

    enabled: bool = True
    architecture_paths: list[str] = field(
        default_factory=lambda: ["architecture/", "docs/", "doc/", "docs-static/"]
    )
    convention_files: list[str] = field(
        default_factory=lambda: [
            "AGENTS.md",
            "CLAUDE.md",
            "CONTRIBUTING.md",
            ".cursor/rules/README.md",
        ]
    )
    comment_marker: str = "<!-- AI-CODE-REVIEWER-DOC-BOT -->"


@dataclass
class DocGenerationSettings:
    """AI-powered documentation draft generation configuration.

    When enabled, the doc reviewer calls Claude Sonnet to draft updated sections
    for stale documentation files and posts them as expandable suggestions inside
    the existing doc-bot PR comment.  Disabled by default — opt in per-repo via
    ``doc_generation.enabled: true`` in ``.ai-reviewer.yaml``.
    """

    enabled: bool = False
    model: str = DEFAULT_HAIKU_MODEL
    max_files: int = 15
    # Directories containing static HTML docs (GitHub Pages sites).
    # When set, update-docs scans these dirs for HTML pages to update on merge.
    static_docs_dirs: list[str] = field(
        default_factory=lambda: ["architecture/", "docs/", "docs-static/"]
    )
    # Labels to apply to auto-generated doc update PRs.
    pr_labels: list[str] = field(default_factory=lambda: ["automated-docs", "documentation"])
    # Open doc update PRs as drafts so they don't appear ready to merge.
    pr_draft: bool = True
    # Stage models for the Understand → Route → Apply → Verify pipeline.
    understanding_model: str = DEFAULT_SONNET_MODEL
    apply_model: str = DEFAULT_HAIKU_MODEL
    verify_model: str = DEFAULT_HAIKU_MODEL
    # Full-PR read budget for stage 1; above this, map-reduce per file.
    max_understanding_diff_chars: int = 250_000
    # Capability gates.
    allow_new_pages: bool = True
    allow_new_sections: bool = True
    # Below this verifier confidence, flag instead of ship: "low" | "medium" | "high".
    verify_confidence_threshold: str = "medium"


@dataclass
class Config:
    """Complete application configuration."""

    anthropic: AnthropicApiConfig | None
    github: GitHubConfig
    agents: list[AgentConfig]
    orchestrator: OrchestratorSettings = field(default_factory=OrchestratorSettings)
    aggregator: AggregatorSettings = field(default_factory=AggregatorSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    review_policy: ReviewPolicy = field(default_factory=ReviewPolicy)
    server: ServerSettings = field(default_factory=ServerSettings)
    doc_review: DocReviewSettings = field(default_factory=DocReviewSettings)
    doc_generation: DocGenerationSettings = field(default_factory=DocGenerationSettings)


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
    # Anthropic config — always constructed; falls back to env var when YAML block absent
    anthropic_raw = raw.get("anthropic", {})
    anthropic = AnthropicApiConfig(
        api_key=anthropic_raw.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", ""),
        base_url=anthropic_raw.get("base_url", "https://api.anthropic.com"),
        timeout_seconds=anthropic_raw.get("timeout_seconds", 300),
        max_retries=anthropic_raw.get("max_retries", 3),
        agent_review_deadline_seconds=anthropic_raw.get("agent_review_deadline_seconds", 900),
        default_model=anthropic_raw.get("default_model", DEFAULT_SONNET_MODEL),
        enable_prompt_caching=anthropic_raw.get("enable_prompt_caching", True),
        max_combined_context_tokens=anthropic_raw.get("max_combined_context_tokens", 80_000),
        per_file_max_bytes=anthropic_raw.get("per_file_max_bytes", 512 * 1024),
        per_review_github_request_budget=anthropic_raw.get("per_review_github_request_budget", 200),
        max_tool_result_bytes=anthropic_raw.get("max_tool_result_bytes", 16 * 1024),
        full_file_max_lines=anthropic_raw.get("full_file_max_lines", 300),
        hunk_context_lines=anthropic_raw.get("hunk_context_lines", 60),
        conventions_max_chars=anthropic_raw.get("conventions_max_chars", 16_000),
    )

    # GitHub config
    github_raw = raw.get("github", {})
    github = GitHubConfig(
        token=github_raw.get("token") or os.environ.get("GITHUB_TOKEN", ""),
        webhook_secret=github_raw.get("webhook_secret"),
        app_id=github_raw.get("app_id"),
        private_key_path=github_raw.get("private_key_path"),
        extra_reviewer_users=github_raw.get("extra_reviewer_users", []),
    )

    # Agents config
    agents = []
    for i, agent_raw in enumerate(raw.get("agents", [])):
        name = agent_raw.get("name")
        model = agent_raw.get("model")
        if not name:
            raise ValueError(f"Agent #{i} is missing required field 'name'")
        if not model:
            raise ValueError(f"Agent '{name}' is missing required field 'model'")
        agents.append(
            AgentConfig(
                name=name,
                model=model,
                focus_areas=agent_raw.get("focus_areas", []),
                max_tokens=agent_raw.get("max_tokens", 8192),
                temperature=agent_raw.get("temperature", 0.3),
                custom_prompt_append=agent_raw.get("custom_prompt_append"),
                include_codebase_context=agent_raw.get("include_codebase_context", False),
                thinking_enabled=agent_raw.get("thinking_enabled", False),
                allow_tool_use=agent_raw.get("allow_tool_use", True),
                max_tool_calls=agent_raw.get("max_tool_calls", 20),
            )
        )

    # Default agents if none configured
    if not agents:
        agents = [
            AgentConfig(
                name="security-reviewer",
                model=DEFAULT_SONNET_MODEL,
                focus_areas=["security", "authentication"],
            ),
            AgentConfig(
                name="performance-reviewer",
                model=DEFAULT_SONNET_MODEL,
                focus_areas=["performance", "complexity"],
            ),
            AgentConfig(
                name="patterns-reviewer",
                model=DEFAULT_SONNET_MODEL,
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
        max_retries=orch_raw.get("max_retries", 1),
    )

    # Aggregator settings
    agg_raw = raw.get("aggregator", {})
    aggregator = AggregatorSettings(
        similarity_threshold=agg_raw.get("similarity_threshold", 0.85),
        min_consensus_for_critical=agg_raw.get("min_consensus_for_critical", 0.5),
        use_embeddings=agg_raw.get("use_embeddings", False),
        min_confidence_critical=agg_raw.get("min_confidence_critical", 0.3),
        min_confidence_warning=agg_raw.get("min_confidence_warning", 0.4),
        min_confidence_suggestion=agg_raw.get("min_confidence_suggestion", 0.5),
        min_confidence_nitpick=agg_raw.get("min_confidence_nitpick", 0.6),
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
        secret_scan_exclude=policy_raw.get("secret_scan_exclude", []),
    )

    # Server settings
    server_raw = raw.get("server", {})
    server = ServerSettings(
        host=server_raw.get("host", "0.0.0.0"),
        port=server_raw.get("port", 8080),
        health_check_path=server_raw.get("health_check_path", "/health"),
        metrics_enabled=server_raw.get("metrics_enabled", True),
    )

    # Doc review settings
    doc_raw = raw.get("doc_review", {})
    doc_review = DocReviewSettings(
        enabled=doc_raw.get("enabled", True),
        architecture_paths=doc_raw.get(
            "architecture_paths", ["architecture/", "docs/", "doc/", "docs-static/"]
        ),
        convention_files=doc_raw.get(
            "convention_files",
            ["AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", ".cursor/rules/README.md"],
        ),
        comment_marker=doc_raw.get("comment_marker", "<!-- AI-CODE-REVIEWER-DOC-BOT -->"),
    )

    # Doc generation settings
    docgen_raw = raw.get("doc_generation", {})
    _dg_model = docgen_raw.get("model", DEFAULT_HAIKU_MODEL)
    doc_generation = DocGenerationSettings(
        enabled=docgen_raw.get("enabled", False),
        model=_dg_model,
        max_files=docgen_raw.get("max_files", 15),
        static_docs_dirs=docgen_raw.get(
            "static_docs_dirs", ["architecture/", "docs/", "docs-static/"]
        ),
        pr_labels=docgen_raw.get("pr_labels", ["automated-docs", "documentation"]),
        pr_draft=docgen_raw.get("pr_draft", True),
        understanding_model=docgen_raw.get("understanding_model", DEFAULT_SONNET_MODEL),
        apply_model=docgen_raw.get("apply_model", _dg_model),
        verify_model=docgen_raw.get("verify_model", _dg_model),
        max_understanding_diff_chars=int(docgen_raw.get("max_understanding_diff_chars", 250_000)),
        allow_new_pages=bool(docgen_raw.get("allow_new_pages", True)),
        allow_new_sections=bool(docgen_raw.get("allow_new_sections", True)),
        verify_confidence_threshold=docgen_raw.get("verify_confidence_threshold", "medium"),
    )

    return Config(
        anthropic=anthropic,
        github=github,
        agents=agents,
        orchestrator=orchestrator,
        aggregator=aggregator,
        output=output,
        review_policy=review_policy,
        server=server,
        doc_review=doc_review,
        doc_generation=doc_generation,
    )


def validate_config(config: Config) -> list[str]:
    """Validate configuration and return list of errors.

    Args:
        config: Configuration to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if not config.anthropic or not config.anthropic.api_key:
        errors.append("Missing Anthropic API key (set ANTHROPIC_API_KEY or anthropic.api_key)")

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
