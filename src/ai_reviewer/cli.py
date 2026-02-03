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
from ai_reviewer.agents.cursor_client import CursorConfig
from ai_reviewer.config import load_config, validate_config
from ai_reviewer.github.formatter import GitHubFormatter, format_review_as_json
from ai_reviewer.github.client import GitHubClient
from ai_reviewer.github.webhook import create_webhook_app, set_review_handler
from ai_reviewer.review import review_pr_with_cursor_agent

console = Console()


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
@click.option("--agents", type=int, default=1, help="Number of agents (1-3): 1=comprehensive, 2+=specialized")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def review_pr(
    repo: str,
    pr_number: int,
    output: str,
    dry_run: bool,
    agents: int,
    config_path: Optional[str],
) -> None:
    """Review a GitHub pull request using Cursor AI agent(s).
    
    With --agents=1 (default): Single comprehensive review
    With --agents=2: Security + Performance agents
    With --agents=3: Security + Performance + Quality agents
    """
    asyncio.run(review_pr_async(
        repo=repo,
        pr_number=pr_number,
        output=output,
        dry_run=dry_run,
        num_agents=agents,
        config_path=Path(config_path) if config_path else None,
    ))


async def review_pr_async(
    repo: str,
    pr_number: int,
    output: str = "github",
    dry_run: bool = False,
    num_agents: int = 1,
    config_path: Optional[Path] = None,
) -> None:
    """Async implementation of PR review using Cursor Background Agent(s)."""
    config = load_config(config_path)
    errors = validate_config(config)
    if errors:
        for error in errors:
            console.print(f"[red]Config error:[/red] {error}")
        sys.exit(1)

    console.print(f"🔍 Reviewing PR #{pr_number} in [bold]{repo}[/bold]...")
    
    if num_agents == 1:
        console.print("[yellow]Using 1 comprehensive agent (2-5 min)[/yellow]")
    else:
        agent_types = ["security", "performance", "quality"][:num_agents]
        console.print(f"[yellow]Using {num_agents} specialized agents: {', '.join(agent_types)} (3-8 min)[/yellow]")

    # Status callback
    last_status = [None]
    def on_status(status: str) -> None:
        if status != last_status[0]:
            console.print(f"  → Agent status: [cyan]{status}[/cyan]")
            last_status[0] = status

    # Run review with Cursor agent
    cursor_config = CursorConfig(
        api_key=config.cursor.api_key,
        base_url=config.cursor.base_url,
        timeout=config.cursor.timeout_seconds,
    )

    try:
        review = await review_pr_with_cursor_agent(
            repo=repo,
            pr_number=pr_number,
            cursor_config=cursor_config,
            github_token=config.github.token,
            on_status=on_status,
            num_agents=num_agents,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    console.print(f"✅ Review complete: {review.summary}")
    console.print(f"   Time: {review.total_review_time_ms / 1000:.1f}s | Findings: {len(review.findings)}")

    # Output
    if output == "json":
        print(json.dumps(format_review_as_json(review), indent=2))
    elif output == "markdown":
        formatter = GitHubFormatter()
        print(formatter.format_review(review))
    else:  # github
        if dry_run:
            console.print("\n[yellow]Dry run - not posting to GitHub[/yellow]")
            formatter = GitHubFormatter()
            print(formatter.format_review(review))
        else:
            gh = GitHubClient(config.github.token)
            pr = gh.get_pull_request(repo, pr_number)
            formatter = GitHubFormatter()
            body = formatter.format_review(review)
            action = formatter.get_review_action(review)
            gh.post_review(pr, review, body, action)
            console.print(f"📝 Posted review to GitHub")
            
            # Post inline comments for each finding
            if review.findings:
                console.print(f"💬 Posting inline comments for {min(len(review.findings), 10)} findings...")
                posted = gh.post_inline_comments(pr, review)
                console.print(f"✅ Posted {posted} inline comments")


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
    table = Table(title="Configured Agents")
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("Focus Areas")

    for agent in config.agents:
        table.add_row(agent.name, agent.model, ", ".join(agent.focus_areas))

    console.print(table)

    # Other settings
    console.print(f"\n[bold]Cursor API:[/bold] {config.cursor.base_url}")
    console.print(f"[bold]Timeout:[/bold] {config.cursor.timeout_seconds}s")


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


if __name__ == "__main__":
    cli()
