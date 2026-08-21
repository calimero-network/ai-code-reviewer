"""Command-line interface for AI Code Reviewer."""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import click
import uvicorn
from github.PullRequest import PullRequest
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from ai_reviewer import __version__
from ai_reviewer.config import (
    AnthropicApiConfig,
    Config,
    DocReviewSettings,
    load_config,
    validate_config,
)
from ai_reviewer.context.local_source import PRMeta
from ai_reviewer.context.pr_checkout import (
    PreparedPR,
    create_pr_worktree,
    parse_pr_target,
    remove_pr_worktree,
    resolve_clone,
)
from ai_reviewer.docs.analyzer import DocAnalyzer, format_doc_comment
from ai_reviewer.docs.updater import run_doc_update
from ai_reviewer.github.client import (
    GitHubClient,
    ReviewMeta,
    lgtm_placeholder_review,
    should_skip_before_agents,
)
from ai_reviewer.github.formatter import (
    GitHubFormatter,
    format_local_report,
    format_review_as_json,
)
from ai_reviewer.github.publish import publish_review
from ai_reviewer.github.webhook import create_webhook_app, set_review_handler
from ai_reviewer.models.review import ConsolidatedReview
from ai_reviewer.review import build_agent_prompts, consolidate_agent_findings, review_local
from ai_reviewer.review import review_pr as run_review

console = Console()
# The reviewer subagents parse stdout brief lines; preparation progress must not land there.
err_console = Console(stderr=True)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def github_token(config: Config) -> str:
    """The token to post with, falling back to the gh CLI's own credential.

    A developer machine usually has `gh auth login` and no GITHUB_TOKEN.
    """
    if config.github.token:
        return config.github.token
    try:
        proc = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=False)
        token = proc.stdout.strip()
    except FileNotFoundError:
        token = ""
    if not token:
        raise click.ClickException("no GitHub token: set GITHUB_TOKEN, or run `gh auth login`")
    return token


def _github_client(config: Config) -> GitHubClient:
    """Without extra_reviewer_users the repo's bot identity is not an AI reviewer
    here, so everything it already posted is invisible and gets posted again."""
    return GitHubClient(
        github_token(config), extra_reviewer_users=config.github.extra_reviewer_users
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
@click.option(
    "--agents",
    type=int,
    default=3,
    help="Number of agents (1-3): 1=comprehensive, 2+=specialized (default: 3)",
)
@click.option(
    "--no-approve", is_flag=True, help="Don't use APPROVE action (auto-enabled in GitHub Actions)"
)
@click.option(
    "--reviewer-name", default="AI Code Reviewer", help="Custom name to display in review header"
)
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
@click.option(
    "--no-cross-review",
    "no_cross_review",
    is_flag=True,
    help="Disable second round where agents validate and rank findings (default: cross-review on when --agents>=2)",
)
@click.option(
    "--min-agreement",
    "min_agreement",
    type=click.FloatRange(0.0, 1.0),
    default=2 / 3,
    help="Fraction of assessing agents that must mark a finding valid (default: 2/3; with 2 agents, both must agree)",
)
@click.option(
    "--force-review",
    is_flag=True,
    help="Bypass convergence detection and always post a review",
)
@click.option(
    "--doc-check/--no-doc-check",
    default=None,
    help="Enable/disable documentation review (default: follow config doc_review.enabled)",
)
def review_pr(
    repo: str,
    pr_number: int,
    output: str,
    dry_run: bool,
    agents: int,
    no_approve: bool,
    reviewer_name: str,
    config_path: str | None,
    no_cross_review: bool,
    min_agreement: float,
    force_review: bool,
    doc_check: bool | None,
) -> None:
    """Review a GitHub pull request using Anthropic Claude agent(s).

    With --agents=1: Single comprehensive review
    With --agents=2: Security + Performance agents (cross-review on by default)
    With --agents=3 (default): Security + Performance + Quality agents (cross-review on by default)
    Use --no-cross-review to skip the second round (validate & rank findings).
    Use --min-agreement to tune how many agents must validate a finding to keep it (0-1).
    """
    asyncio.run(
        review_pr_async(
            repo=repo,
            pr_number=pr_number,
            output=output,
            dry_run=dry_run,
            num_agents=agents,
            no_approve=no_approve,
            reviewer_name=reviewer_name,
            config_path=Path(config_path) if config_path else None,
            enable_cross_review=not no_cross_review,
            min_validation_agreement=min_agreement,
            force_review=force_review,
            doc_check=doc_check,
        )
    )


async def review_pr_async(
    repo: str,
    pr_number: int,
    output: str = "github",
    dry_run: bool = False,
    num_agents: int = 3,
    no_approve: bool = False,
    reviewer_name: str = "AI Code Reviewer",
    config_path: Path | None = None,
    enable_cross_review: bool = True,
    min_validation_agreement: float = 2 / 3,
    force_review: bool = False,
    doc_check: bool | None = None,
) -> None:
    """Async implementation of PR review using Anthropic Claude agents."""
    # Auto-detect GitHub Actions environment - never allow APPROVE there
    is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"
    allow_approve = not no_approve and not is_github_actions

    if is_github_actions and not no_approve:
        console.print("[dim]ℹ️  Running in GitHub Actions - APPROVE disabled automatically[/dim]")
    config = load_config(config_path)
    errors = validate_config(config)
    if errors:
        for error in errors:
            console.print(f"[red]Config error:[/red] {error}")
        sys.exit(1)

    console.print(f"🔍 Reviewing PR #{pr_number} in [bold]{repo}[/bold]...")
    console.print(
        f"[dim]Requested {num_agents} agent(s); actual count may be reduced for small PRs[/dim]"
    )

    if not config.anthropic or not config.anthropic.api_key:
        console.print(
            "[red]error:[/red] anthropic.api_key not configured "
            "(set ANTHROPIC_API_KEY or anthropic.api_key in config.yaml)"
        )
        sys.exit(2)
    anthropic_cfg = config.anthropic

    # Pre-agent checks (github output only — json/markdown always run agents)
    gh: GitHubClient | None = None
    pr: PullRequest | None = None
    meta: ReviewMeta | None = None
    recheck_review: ConsolidatedReview | None = None
    dismissed: list = []

    if output == "github":
        gh = GitHubClient(config.github.token)
        pr = gh.get_pull_request(repo, pr_number)
        current_sha = pr.head.sha

        meta = gh.get_review_metadata(pr)
        # Dismissal ledger only matters once a prior review exists to have been resolved.
        if meta is not None:
            dismissed = gh.get_dismissed_findings(pr)
        diff_files = {f.filename for f in pr.get_files()}
        previous_comments = gh.get_previous_review_comments(pr) if meta else []
        skip_reason = should_skip_before_agents(
            meta,
            current_sha,
            force_review,
            diff_files=diff_files,
            previous_comments=previous_comments,
        )
        if skip_reason is not None:
            console.print(
                f"[dim]⏭️  Skipping review: {skip_reason.value} "
                f"(use --force-review to override)[/dim]"
            )
            return

        if meta is not None and not force_review:
            lgtm_delta = gh.check_lgtm_fast_path(pr, meta)
            if lgtm_delta is not None:
                console.print(
                    "[dim]🔍 LGTM candidate detected — running lightweight 1-agent re-check…[/dim]"
                )
                try:
                    recheck_review = await run_review(
                        repo=repo,
                        pr_number=pr_number,
                        anthropic_cfg=anthropic_cfg,
                        github_token=config.github.token,
                        num_agents=1,
                        enable_cross_review=False,
                        min_validation_agreement=min_validation_agreement,
                        config=config,
                    )
                except Exception as e:
                    console.print(f"[red]Error during LGTM re-check:[/red] {e}")
                    console.print(
                        "[yellow]Falling back to normal review flow after re-check failure[/yellow]"
                    )

                if (
                    recheck_review is not None
                    and not recheck_review.findings
                    and not recheck_review.all_agents_failed
                ):
                    formatter = GitHubFormatter(reviewer_name)
                    lgtm_review_count = meta.review_count + 1
                    new_meta = ReviewMeta.build(
                        commit_sha=current_sha,
                        review_count=lgtm_review_count,
                        finding_hashes=[],
                    )
                    lgtm_review = lgtm_placeholder_review(repo, pr_number)
                    if dry_run:
                        console.print(
                            "\n[yellow]Dry run - LGTM (verified by 1-agent re-check)[/yellow]"
                        )
                        print(
                            formatter.format_review_with_delta(
                                lgtm_review, lgtm_delta, meta=new_meta
                            )
                        )
                    else:
                        body = formatter.format_review_with_delta_compact(
                            lgtm_review, lgtm_delta, meta=new_meta
                        )
                        gh.post_review(pr, body, "COMMENT")
                        if lgtm_delta.fixed_findings:
                            resolved = gh.resolve_fixed_comments(pr, lgtm_delta)
                            console.print(f"✅ LGTM: resolved {resolved} comments")
                        console.print(
                            "[green]🎉 LGTM — all issues resolved (verified by re-check)[/green]"
                        )
                    return

                if recheck_review is not None:
                    console.print(
                        f"[yellow]Re-check found {len(recheck_review.findings)} issue(s) "
                        f"— proceeding with normal review flow[/yellow]"
                    )

    # Status callback
    last_status: list[str | None] = [None]

    def on_status(status: str) -> None:
        if status != last_status[0]:
            console.print(f"  → Agent status: [cyan]{status}[/cyan]")
            last_status[0] = status

    try:
        review = await run_review(
            repo=repo,
            pr_number=pr_number,
            anthropic_cfg=anthropic_cfg,
            github_token=config.github.token,
            on_status=on_status,
            num_agents=num_agents,
            enable_cross_review=enable_cross_review,
            min_validation_agreement=min_validation_agreement,
            config=config,
            dismissed=dismissed,
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Check if all agents failed
    if review.all_agents_failed:
        console.print(f"[red]❌ All {review.agent_count} agents failed![/red]")
        console.print(f"   Time: {review.total_review_time_ms / 1000:.1f}s")
        # Never stay silent on GitHub: post an honest "could not complete" comment so
        # the author can tell the reviewer ran and failed, then re-trigger.
        if output == "github" and gh is not None and pr is not None:
            formatter = GitHubFormatter(reviewer_name)
            if dry_run:
                console.print(
                    "\n[yellow]Dry run - would post 'review could not complete' notice[/yellow]"
                )
                print(formatter.format_all_agents_failed(review))
            else:
                gh.post_review(pr, formatter.format_all_agents_failed(review), "COMMENT")
                console.print(
                    "[yellow]Posted 'review could not complete' notice to GitHub.[/yellow]"
                )
        else:
            console.print("\n[yellow]Not posting to GitHub - all agents failed.[/yellow]")
        console.print("\n[bold]Possible causes:[/bold]")
        console.print("  • Invalid or expired Anthropic API key")
        console.print("  • Rate limit exceeded")
        console.print("  • Network connectivity issues")
        console.print("\nCheck your ANTHROPIC_API_KEY and try again.")
        sys.exit(1)

    effective_agents = review.agent_count
    cross_review_ran = effective_agents > 1 and enable_cross_review
    if effective_agents == 1:
        msg = (
            "[yellow]Used 1 comprehensive agent (PR too small for multi-agent)[/yellow]"
            if num_agents > 1
            else "[yellow]Used 1 comprehensive agent[/yellow]"
        )
        console.print(msg)
    else:
        agent_types = ["security", "performance", "quality"][:effective_agents]
        console.print(
            f"[yellow]Used {effective_agents} specialized agents: {', '.join(agent_types)}[/yellow]"
        )
    if effective_agents > 1:
        if cross_review_ran:
            console.print("[dim]Cross-review ran: findings validated and ranked[/dim]")
        else:
            console.print("[dim]Cross-review disabled[/dim]")

    console.print(f"✅ Review complete: {review.summary}")
    console.print(
        f"   Time: {review.total_review_time_ms / 1000:.1f}s | Findings: {len(review.findings)}"
    )

    # Warn about partial failures
    if review.failed_agents:
        console.print(
            f"[yellow]⚠️  {len(review.failed_agents)}/{review.agent_count} agents failed: {', '.join(review.failed_agents)}[/yellow]"
        )

    # Output
    if output == "json":
        print(json.dumps(format_review_as_json(review), indent=2))
    elif output == "markdown":
        formatter = GitHubFormatter(reviewer_name)
        print(formatter.format_review(review))
    else:  # github
        if gh is None or pr is None:
            raise click.ClickException(
                "--output github requires a valid GitHub token and accessible PR"
            )
        result = publish_review(
            gh=gh,
            pr=pr,
            review=review,
            config=config,
            meta=meta,
            reviewer_name=reviewer_name,
            force_review=force_review,
            dry_run=dry_run,
            allow_approve=allow_approve,
            emit=console.print,
        )
        if dry_run and result.body:
            print(result.body)
        if result.skipped:
            return

        # Doc review (runs for github output after the main review)
        _run_doc_review(
            gh=gh,
            pr=pr,
            repo=repo,
            config=config,
            doc_check=doc_check,
            dry_run=dry_run,
        )


def _run_doc_review(
    *,
    gh: GitHubClient | None,
    pr: PullRequest | None,
    repo: str,
    config: Config,
    doc_check: bool | None,
    dry_run: bool,
) -> None:
    """Run documentation review and post/update the doc-bot comment.

    Factored out of ``review_pr_async`` to keep the main flow readable.
    """
    if gh is None or pr is None:
        return

    doc_settings = getattr(config, "doc_review", None)
    if not isinstance(doc_settings, DocReviewSettings):
        doc_settings = DocReviewSettings()

    # Resolve effective enabled flag: CLI flag > config
    enabled = doc_settings.enabled if doc_check is None else doc_check
    if not enabled:
        return

    console.print("📄 Running documentation review...")

    static_docs_dirs = config.doc_generation.static_docs_dirs

    # Probe the repo for convention files, architecture directories, and static doc dirs.
    # static_docs_dirs must be included here so check_static_html_docs can find them in
    # existing_repo_paths — entries absent from the probe are silently skipped.
    probe_paths = list(
        dict.fromkeys(
            doc_settings.convention_files + doc_settings.architecture_paths + static_docs_dirs
        )
    )
    try:
        existing_repo_paths = gh.probe_repo_paths(repo, pr.head.sha, probe_paths)
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not probe repo paths for doc review: {e}[/yellow]")
        return

    # Build changed paths with status from PR files
    pr_files = list(pr.get_files())
    changed_paths = [f.filename for f in pr_files]
    changed_paths_with_status = {f.filename: getattr(f, "status", "modified") for f in pr_files}

    # Load doc_config from repo's .ai-reviewer.yaml if present
    repo_config = gh.load_repo_config(repo, pr.head.sha)
    doc_config = repo_config.get("documentation") if repo_config else None

    if doc_config is not None and not doc_config.get("enabled", True):
        console.print("[dim]ℹ️  Documentation review disabled in repo .ai-reviewer.yaml[/dim]")
        return

    analyzer = DocAnalyzer(
        changed_paths=changed_paths,
        changed_paths_with_status=changed_paths_with_status,
        existing_repo_paths=existing_repo_paths,
        doc_config=doc_config,
        architecture_dirs=doc_settings.architecture_paths,
        convention_files=doc_settings.convention_files,
        static_docs_dirs=static_docs_dirs,
    )
    suggestions = analyzer.run()

    marker = doc_settings.comment_marker
    body = format_doc_comment(suggestions, marker)

    if dry_run:
        if suggestions:
            console.print(f"[yellow]Dry run - {len(suggestions)} doc suggestion(s):[/yellow]")
            print(body)
        else:
            console.print("[dim]Dry run - documentation looks current[/dim]")
        return

    if suggestions:
        console.print(f"📄 Posting {len(suggestions)} doc suggestion(s)...")
        gh.post_or_update_doc_comment(pr, body, marker)
    else:
        existing_comment_id = gh.find_doc_bot_comment(pr, marker)
        if existing_comment_id is not None:
            console.print("[dim]📄 Updating doc-bot comment: all documentation looks current[/dim]")
            gh.post_or_update_doc_comment(pr, body, marker)
        else:
            console.print("[dim]📄 Documentation looks current — no comment needed[/dim]")


@cli.command("review")
@click.option("--staged", is_flag=True, help="Review the index instead of the working tree")
@click.option("--base", default=None, help="Review base...HEAD instead of uncommitted changes")
@click.option("--output", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--agents", type=click.IntRange(1, 5), default=3, help="Number of agents")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
@click.option("--no-cross-review", "no_cross_review", is_flag=True)
def review_local_command(
    staged: bool,
    base: str | None,
    output: str,
    agents: int,
    config_path: str | None,
    no_cross_review: bool,
) -> None:
    """Review local changes with no pull request.

    Reviews uncommitted work by default, the index with --staged, or a branch
    range with --base. Use --output json to drive a fix loop.
    """
    config = load_config(Path(config_path) if config_path else None)
    # Local review builds its own inputs; a config without an anthropic section is
    # usable rather than fatal.
    anthropic_cfg = config.anthropic or AnthropicApiConfig(api_key="")

    try:
        review = asyncio.run(
            review_local(
                root=os.getcwd(),
                anthropic_cfg=anthropic_cfg,
                staged=staged,
                base=base,
                num_agents=agents,
                enable_cross_review=not no_cross_review,
                config=config,
            )
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if output == "json":
        print(json.dumps(format_review_as_json(review), indent=2))
    else:
        print(GitHubFormatter("AI Code Reviewer").format_review(review))


@cli.command("prompts")
@click.option("--out", "out_dir", required=True, type=click.Path(), help="Directory to write to")
@click.option("--staged", is_flag=True, help="Prompt for the index instead of the working tree")
@click.option("--base", default=None, help="Prompt for base...HEAD")
@click.option(
    "--pr", "pr_target", default=None, help="Prompt for a pull request (URL or owner/repo#N)"
)
@click.option(
    "--repo-path",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Clone of the pull request's repository to take the worktree from",
)
@click.option(
    "--agents", type=click.IntRange(1, 5), default=3, help="How many reviewer profiles to emit"
)
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def prompts_command(
    out_dir: str,
    staged: bool,
    base: str | None,
    pr_target: str | None,
    repo_path: str | None,
    agents: int,
    config_path: str | None,
) -> None:
    """Write one self-contained review prompt per reviewer profile.

    For orchestrating reviewer subagents from a coding session: this makes no LLM
    calls, it only assembles the same prompts the API path would send. With --pr it
    also prepares a worktree, which `publish` removes.
    """
    if pr_target and (staged or base):
        raise click.ClickException("--pr cannot be combined with --staged or --base")

    config = load_config(Path(config_path) if config_path else None)
    target = Path(out_dir)

    if pr_target:
        target.mkdir(parents=True, exist_ok=True)
        _prompts_for_pr(pr_target, repo_path, target, agents, config)
        return

    try:
        built = asyncio.run(
            build_agent_prompts(
                root=os.getcwd(),
                staged=staged,
                base=base,
                num_agents=agents,
                anthropic_cfg=config.anthropic,
                config=config,
            )
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    target.mkdir(parents=True, exist_ok=True)
    _write_briefs(built, target)


def _write_briefs(built: dict, target: Path) -> None:
    """Brief paths go to stdout; anything else must not, callers parse these lines."""
    for name, spec in built.items():
        (target / f"{name}.md").write_text(spec["prompt"])
        print(f"{name}\t{spec['model']}\t{target / f'{name}.md'}")


def _prompts_for_pr(
    pr_target: str, repo_path: str | None, target: Path, agents: int, config: Config
) -> None:
    """Prepare a pull request's worktree and write its briefs.

    The worktree outlives this process: the reviewer subagents read it, and
    ``publish`` removes it.
    """
    # Resolving the pull request and taking the worktree from it fail as readily as
    # anything after, so the whole preparation reports rather than tracebacks.
    prepared: PreparedPR | None = None
    try:
        slug, number = parse_pr_target(pr_target)
        gh = _github_client(config)
        pull = gh.get_pull_request(slug, number)

        err_console.print(f"[bold]Preparing {slug}#{number}[/bold]")
        err_console.print(f"  {pull.title}")

        clone = resolve_clone(slug, repo_path)
        prepared = create_pr_worktree(clone, slug, number, pull.base.ref, target / "wt")
        err_console.print(f"  worktree from {clone}  (detached at {prepared.head_sha[:7]})")
        err_console.print(f"  base {pull.base.ref} @ {prepared.base_sha[:7]}")
        built = asyncio.run(
            build_agent_prompts(
                root=prepared.root,
                base=prepared.base_sha,
                num_agents=agents,
                anthropic_cfg=config.anthropic,
                config=config,
                pr_meta=PRMeta(
                    repo=slug, number=number, title=pull.title or "", body=pull.body or ""
                ),
            )
        )
        prepared.write(target / "target.json")
        _write_briefs(built, target)
    except Exception as e:  # noqa: BLE001
        if prepared is not None:
            remove_pr_worktree(prepared)
        err_console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command("consolidate")
@click.argument("findings_files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--output", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option("--all", "show_all", is_flag=True, help="Include suggestions and nitpicks")
@click.option("--staged", is_flag=True, help="Findings came from the index")
@click.option("--base", default=None, help="Findings came from base...HEAD")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def consolidate_command(
    findings_files: tuple[str, ...],
    output: str,
    show_all: bool,
    staged: bool,
    base: str | None,
    config_path: str | None,
) -> None:
    """Consolidate per-agent findings JSON into one reviewed result.

    Each file holds one agent's findings and is named for that agent, so the
    filename drives consensus scoring. Clustering, confidence floors, cross-file
    dedup and fix validation all run here.
    """
    from ai_reviewer.context.local_source import (
        build_local_context,
        changed_files,
        local_diff,
        read_repo_file,
        scope_label,
    )

    config = load_config(Path(config_path) if config_path else None)
    root = Path(os.getcwd())

    def read_local(path: str) -> str | None:
        # file_path comes from agent-produced JSON, so it is untrusted; the shared
        # reader confines it to the repository.
        return read_repo_file(str(root), path)

    try:
        diff = local_diff(str(root), staged=staged, base=base)
        reviewed = build_local_context(str(root), diff, changed_files(str(root), staged, base))
        review = consolidate_agent_findings(
            list(findings_files),
            repo=root.name,
            config=config,
            read_file=read_local,
            total_lines=reviewed.additions + reviewed.deletions,
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if output == "json":
        print(json.dumps(format_review_as_json(review), indent=2))
    else:
        print(format_local_report(review, scope=scope_label(staged, base), show_all=show_all))


@cli.command("publish")
@click.argument("session_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--dry-run", is_flag=True, help="Print the review instead of posting it")
@click.option("--all", "show_all", is_flag=True, help="Include suggestions and nitpicks locally")
@click.option("--force-review", is_flag=True, help="Post even when findings are unchanged")
@click.option("--reviewer-name", default="AI Code Reviewer", help="Name shown in the review header")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def publish_command(
    session_dir: str,
    dry_run: bool,
    show_all: bool,
    force_review: bool,
    reviewer_name: str,
    config_path: str | None,
) -> None:
    """Consolidate a prepared pull request review and post it to GitHub.

    Takes the directory `prompts --pr` wrote: what was reviewed is read from
    target.json rather than re-described, so the two phases cannot disagree.
    """
    from ai_reviewer.context.local_source import (
        build_local_context,
        changed_files,
        local_diff,
        read_repo_file,
    )

    session = Path(session_dir)
    target_file = session / "target.json"
    if not target_file.is_file():
        raise click.ClickException(f"{target_file} not found - run `prompts --pr` first")
    target = PreparedPR.read(target_file)

    try:
        findings_files = sorted(str(p) for p in (session / "out").glob("*.json"))
        if not findings_files:
            raise click.ClickException(f"no agent findings in {session / 'out'}")

        config = load_config(Path(config_path) if config_path else None)

        diff = local_diff(target.root, base=target.base_sha)
        reviewed = build_local_context(
            target.root, diff, changed_files(target.root, base=target.base_sha)
        )
        review = consolidate_agent_findings(
            findings_files,
            repo=target.repo,
            config=config,
            read_file=lambda path: read_repo_file(target.root, path),
            total_lines=reviewed.additions + reviewed.deletions,
        )
        print(
            format_local_report(review, scope=f"{target.repo}#{target.number}", show_all=show_all)
        )

        gh = _github_client(config)
        pull = gh.get_pull_request(target.repo, target.number)
        result = publish_review(
            gh=gh,
            pr=pull,
            review=review,
            config=config,
            meta=gh.get_review_metadata(pull),
            reviewer_name=reviewer_name,
            force_review=force_review,
            dry_run=dry_run,
            # This posts under a person's GitHub identity, so it never approves for them.
            allow_approve=False,
            emit=console.print,
        )
        if result.posted:
            # The session quotes this link verbatim, so it goes out unwrapped: the
            # rich console would fold a long owner/repo across lines.
            print(f"🔗 {pull.html_url}")
        if dry_run and result.body:
            print(result.body)
    except click.ClickException:
        raise
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    finally:
        # A dry run is a rehearsal: keep the worktree so the real post needs no refetch.
        # Every other exit from here on - success, ClickException, or any other error - removes it.
        if not dry_run:
            remove_pr_worktree(target)


@cli.command("update-docs")
@click.argument("repo")
@click.argument("pr_number", type=int)
@click.option("--dry-run", is_flag=True, help="Print what would change without opening a PR")
@click.option(
    "--base", default=None, help="Base branch to target for the doc PR (default: auto-detect)"
)
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def update_docs_cmd(
    repo: str,
    pr_number: int,
    dry_run: bool,
    base: str | None,
    config_path: str | None,
) -> None:
    """Generate and commit AI-drafted doc updates for a merged PR.

    Detects stale documentation files via source_to_docs_mapping in
    .ai-reviewer.yaml, generates full updated file content using Claude Sonnet,
    then commits the changes to a new branch and opens a PR for human review.

    Use --dry-run to preview the generated content locally without opening a PR.
    """
    try:
        asyncio.run(
            _update_docs_async(
                repo=repo,
                pr_number=pr_number,
                dry_run=dry_run,
                base=base,
                config_path=Path(config_path) if config_path else None,
            )
        )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


async def _update_docs_async(
    repo: str,
    pr_number: int,
    dry_run: bool,
    base: str | None,
    config_path: Path | None,
) -> None:
    config = load_config(config_path)
    errors = validate_config(config)
    if errors:
        for error in errors:
            console.print(f"[red]Config error:[/red] {error}")
        raise RuntimeError(f"Invalid config: {'; '.join(errors)}")

    if not config.anthropic or not config.anthropic.api_key:
        console.print("[red]error:[/red] ANTHROPIC_API_KEY not set")
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    gh = GitHubClient(config.github.token)
    console.print(f"📄 Fetching PR #{pr_number} from [bold]{repo}[/bold]...")

    result = await run_doc_update(
        repo=repo,
        pr_number=pr_number,
        gh=gh,
        anthropic_cfg=config.anthropic,
        doc_generation=config.doc_generation,
        base=base,
        dry_run=dry_run,
    )

    if result.skipped:
        for d in result.failed:
            console.print(f"[yellow]⚠️  Skipped {d.target_path}: {d.error}[/yellow]")
        if result.flagged:
            for d in result.flagged:
                console.print(f"[yellow]⚑  Flagged {d.target_path}: {d.flagged_reason}[/yellow]")
        console.print(f"[dim]ℹ️  {result.skip_reason}[/dim]")
        return

    for d in result.failed:
        console.print(f"[yellow]⚠️  Skipped {d.target_path}: {d.error}[/yellow]")

    if result.flagged:
        for d in result.flagged:
            console.print(f"[yellow]⚑  Flagged {d.target_path}: {d.flagged_reason}[/yellow]")

    if not result.successful:
        if result.flagged:
            console.print(
                "[yellow]⚑ All candidate updates were flagged for human review — none shipped.[/yellow]"
            )
        else:
            console.print("[green]✅ No doc updates needed after scanning all candidates.[/green]")
        return

    if dry_run:
        console.print(
            f"\n[yellow]Dry run — would update {len(result.successful)} file(s):[/yellow]\n"
        )
        for draft in result.successful:
            console.print(f"[bold]━━ {draft.target_path} ━━[/bold]")
            preview_lines = draft.updated_content.splitlines()[:60]
            console.print("\n".join(preview_lines))
            if len(draft.updated_content.splitlines()) > 60:
                console.print(
                    f"[dim]… ({len(draft.updated_content.splitlines()) - 60} more lines)[/dim]"
                )
            console.print()
        return

    if result.pr_url:
        console.print(f"[green]✅ Doc update PR opened: {result.pr_url}[/green]")
    else:
        raise RuntimeError("PR creation returned no URL")


@cli.group("config")
def config_group() -> None:
    """Configuration commands."""
    pass


@config_group.command("validate")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def config_validate(config_path: str | None) -> None:
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
def config_show(config_path: str | None) -> None:
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
    if config.anthropic:
        console.print(f"\n[bold]Anthropic API:[/bold] {config.anthropic.base_url}")
        console.print(f"[bold]Model:[/bold] {config.anthropic.default_model}")
        console.print(f"[bold]Timeout:[/bold] {config.anthropic.timeout_seconds}s")


@cli.command("serve")
@click.option("--port", default=8080, help="Port to listen on")
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--config", "config_path", type=click.Path(exists=True), help="Config file path")
def serve(port: int, host: str, config_path: str | None) -> None:
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
