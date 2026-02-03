"""Command-line interface for AI Code Reviewer."""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from ai_reviewer import __version__
from ai_reviewer.agents.cursor_client import CursorClient, CursorConfig
from ai_reviewer.agents.performance import LogicAgent, PerformanceAgent
from ai_reviewer.agents.patterns import PatternsAgent, StyleAgent
from ai_reviewer.agents.security import AuthenticationAgent, SecurityAgent
from ai_reviewer.config import Config, load_config, validate_config
from ai_reviewer.github.client import GitHubClient
from ai_reviewer.github.formatter import GitHubFormatter, format_review_as_json
from ai_reviewer.github.webhook import create_webhook_app, set_review_handler
from ai_reviewer.orchestrator.aggregator import ReviewAggregator
from ai_reviewer.orchestrator.orchestrator import AgentOrchestrator

console = Console()

# Mapping of focus areas to agent classes
AGENT_CLASSES = {
    "security": SecurityAgent,
    "authentication": AuthenticationAgent,
    "performance": PerformanceAgent,
    "logic": LogicAgent,
    "patterns": PatternsAgent,
    "consistency": PatternsAgent,
    "style": StyleAgent,
}


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def cli(verbose: bool) -> None:
    """AI Code Reviewer - Multi-agent code review system."""
    setup_logging(verbose)


@cli.command("review-pr")
@click.argument("repo")
@click.argument("pr_number", type=int)
@click.option("--output", type=click.Choice(["github", "json", "markdown"]), default="github")
@click.option("--dry-run", is_flag=True, help="Don't post to GitHub")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def review_pr(
    repo: str,
    pr_number: int,
    output: str,
    dry_run: bool,
    config_path: Optional[str],
) -> None:
    """Review a GitHub pull request."""
    asyncio.run(review_pr_async(
        repo=repo,
        pr_number=pr_number,
        output=output,
        dry_run=dry_run,
        config_path=Path(config_path) if config_path else None,
    ))


async def review_pr_async(
    repo: str,
    pr_number: int,
    output: str = "github",
    dry_run: bool = False,
    config_path: Optional[Path] = None,
) -> None:
    """Async implementation of PR review."""
    config = load_config(config_path)
    errors = validate_config(config)
    if errors:
        for error in errors:
            console.print(f"[red]Config error:[/red] {error}")
        sys.exit(1)

    console.print(f"🔍 Reviewing PR #{pr_number} in [bold]{repo}[/bold]...")

    # Initialize GitHub client
    gh = GitHubClient(config.github.token)
    pr = gh.get_pull_request(repo, pr_number)
    repo_obj = gh.get_repo(repo)

    # Get PR data
    diff = gh.get_pr_diff(pr)
    files = gh.get_changed_files(pr)
    context = gh.build_review_context(pr, repo_obj)

    console.print(f"📄 {len(files)} files changed (+{context.additions}/-{context.deletions})")

    # Create agents
    cursor_client = CursorClient(CursorConfig(
        api_key=config.cursor.api_key,
        base_url=config.cursor.base_url,
        timeout=config.cursor.timeout_seconds,
    ))

    agents = create_agents(config, cursor_client)
    console.print(f"🤖 Launching {len(agents)} agents...")

    # Run orchestrator
    orchestrator = AgentOrchestrator(
        agents=agents,
        timeout_seconds=config.orchestrator.timeout_seconds,
        min_agents_required=config.orchestrator.min_agents_required,
    )

    try:
        reviews = await orchestrator.review(diff, files, context)
    finally:
        await cursor_client.close()

    # Aggregate results
    aggregator = ReviewAggregator()
    consolidated = aggregator.aggregate(reviews, repo, pr_number)

    console.print(f"✅ Review complete: {consolidated.summary}")

    # Output
    if output == "json":
        print(json.dumps(format_review_as_json(consolidated), indent=2))
    elif output == "markdown":
        formatter = GitHubFormatter()
        print(formatter.format_review(consolidated))
    else:  # github
        if dry_run:
            console.print("\n[yellow]Dry run - not posting to GitHub[/yellow]")
            formatter = GitHubFormatter()
            print(formatter.format_review(consolidated))
        else:
            formatter = GitHubFormatter()
            body = formatter.format_review(consolidated)
            action = formatter.get_review_action(consolidated)
            gh.post_review(pr, consolidated, body, action)
            console.print(f"📝 Posted review to GitHub ({action})")


@cli.command("review")
@click.option("--diff", "diff_file", type=click.Path(exists=True), help="Path to diff file")
@click.option("--output", type=click.Choice(["json", "markdown"]), default="markdown")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def review(diff_file: Optional[str], output: str, config_path: Optional[str]) -> None:
    """Review a diff from file or stdin."""
    # Read diff
    if diff_file:
        diff = Path(diff_file).read_text()
    else:
        if sys.stdin.isatty():
            console.print("[red]Error:[/red] Provide --diff or pipe diff to stdin")
            sys.exit(1)
        diff = sys.stdin.read()

    asyncio.run(review_diff_async(
        diff=diff,
        output=output,
        config_path=Path(config_path) if config_path else None,
    ))


async def review_diff_async(
    diff: str,
    output: str = "markdown",
    config_path: Optional[Path] = None,
) -> None:
    """Async implementation of diff review."""
    config = load_config(config_path)
    errors = validate_config(config)
    if errors:
        for error in errors:
            console.print(f"[red]Config error:[/red] {error}")
        sys.exit(1)

    console.print("🔍 Reviewing diff...")

    # Create agents
    cursor_client = CursorClient(CursorConfig(
        api_key=config.cursor.api_key,
        base_url=config.cursor.base_url,
        timeout=config.cursor.timeout_seconds,
    ))

    agents = create_agents(config, cursor_client)
    console.print(f"🤖 Launching {len(agents)} agents...")

    # Run orchestrator
    orchestrator = AgentOrchestrator(
        agents=agents,
        timeout_seconds=config.orchestrator.timeout_seconds,
        min_agents_required=config.orchestrator.min_agents_required,
    )

    # Create minimal context
    context = {
        "repo_name": "local",
        "pr_number": 0,
        "pr_title": "Local review",
        "pr_description": "",
        "base_branch": "main",
        "head_branch": "current",
        "author": "local",
        "changed_files_count": 0,
        "additions": 0,
        "deletions": 0,
    }

    try:
        reviews = await orchestrator.review(diff, {}, context)
    finally:
        await cursor_client.close()

    # Aggregate results
    aggregator = ReviewAggregator()
    consolidated = aggregator.aggregate(reviews)

    console.print(f"✅ Review complete: {consolidated.summary}")

    # Output
    if output == "json":
        print(json.dumps(format_review_as_json(consolidated), indent=2))
    else:
        formatter = GitHubFormatter()
        print(formatter.format_review(consolidated))


@cli.group("config")
def config_group() -> None:
    """Configuration commands."""
    pass


@config_group.command("validate")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def config_validate(config_path: Optional[str]) -> None:
    """Validate configuration file."""
    try:
        config = load_config(Path(config_path) if config_path else None)
        errors = validate_config(config)

        if errors:
            console.print("[red]Configuration is invalid:[/red]")
            for error in errors:
                console.print(f"  • {error}")
            sys.exit(1)
        else:
            console.print("[green]✓ Configuration is valid[/green]")
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        sys.exit(1)


@config_group.command("show")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def config_show(config_path: Optional[str]) -> None:
    """Show current configuration."""
    config = load_config(Path(config_path) if config_path else None)

    console.print("\n[bold]Current Configuration[/bold]\n")

    # Agents table
    table = Table(title="Agents")
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("Focus Areas")

    for agent in config.agents:
        table.add_row(agent.name, agent.model, ", ".join(agent.focus_areas))

    console.print(table)

    # Other settings
    console.print(f"\n[bold]Orchestrator:[/bold] timeout={config.orchestrator.timeout_seconds}s, min_agents={config.orchestrator.min_agents_required}")
    console.print(f"[bold]Server:[/bold] {config.server.host}:{config.server.port}")


@cli.group("agents")
def agents_group() -> None:
    """Agent management commands."""
    pass


@agents_group.command("list")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def agents_list(config_path: Optional[str]) -> None:
    """List configured agents."""
    config = load_config(Path(config_path) if config_path else None)

    table = Table(title="Configured Agents")
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("Focus Areas")
    table.add_column("Max Tokens")

    for agent in config.agents:
        table.add_row(
            agent.name,
            agent.model,
            ", ".join(agent.focus_areas),
            str(agent.max_tokens),
        )

    console.print(table)


@cli.command("serve")
@click.option("--port", default=8080, help="Port to listen on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def serve(port: int, host: str, config_path: Optional[str]) -> None:
    """Start the webhook server."""
    config = load_config(Path(config_path) if config_path else None)
    errors = validate_config(config)
    if errors:
        for error in errors:
            console.print(f"[red]Config error:[/red] {error}")
        sys.exit(1)

    # Set up review handler
    async def review_handler(repo: str, pr_number: int) -> None:
        await review_pr_async(repo=repo, pr_number=pr_number, output="github")

    set_review_handler(review_handler)

    # Create and run app
    app = create_webhook_app(config.github.webhook_secret)

    console.print(f"🚀 Starting webhook server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


def create_agents(config: Config, cursor_client: CursorClient) -> list:
    """Create agent instances from configuration."""
    agents = []

    for agent_config in config.agents:
        # Find appropriate agent class based on focus areas
        agent_class = None
        for focus in agent_config.focus_areas:
            if focus in AGENT_CLASSES:
                agent_class = AGENT_CLASSES[focus]
                break

        if agent_class is None:
            agent_class = SecurityAgent  # Default

        agent = agent_class(cursor_client, agent_id=agent_config.name)
        # Override model if specified
        agent.MODEL = agent_config.model
        agents.append(agent)

    return agents


if __name__ == "__main__":
    cli()
