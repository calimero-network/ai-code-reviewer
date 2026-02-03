"""Main review flow using Cursor Background Agent API with multi-agent support."""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from uuid import uuid4

from ai_reviewer.agents.cursor_client import CursorClient, CursorConfig
from ai_reviewer.github.client import GitHubClient
from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity
from ai_reviewer.models.review import AgentReview, ConsolidatedReview

logger = logging.getLogger(__name__)


# Different agent configurations for multi-perspective review
AGENT_CONFIGS = [
    {
        "name": "security-agent",
        "focus": "security",
        "prompt_addition": """
**YOUR FOCUS: SECURITY**
You are a security expert. Focus ONLY on:
- Injection vulnerabilities (SQL, command, XSS)
- Authentication/authorization flaws
- Cryptographic issues
- Data exposure and validation
- Trust boundary violations

Ignore performance, style, and other non-security issues.
""",
    },
    {
        "name": "performance-agent",
        "focus": "performance",
        "prompt_addition": """
**YOUR FOCUS: PERFORMANCE & CORRECTNESS**
You are a performance engineer. Focus ONLY on:
- Algorithm complexity (O(n²) where O(n) possible)
- Memory leaks and resource management
- N+1 queries, unnecessary allocations
- Race conditions and concurrency bugs
- Logic errors and edge cases

Ignore security and style issues unless they cause bugs.
""",
    },
    {
        "name": "quality-agent",
        "focus": "quality",
        "prompt_addition": """
**YOUR FOCUS: CODE QUALITY & PATTERNS**
You are a code quality reviewer. Focus ONLY on:
- API design and consistency
- Error handling patterns
- Code organization and maintainability
- Missing tests for critical paths
- Documentation accuracy

Ignore security vulnerabilities and performance unless severe.
""",
    },
]


def get_base_prompt(context: ReviewContext, diff: str, file_contents: dict[str, str]) -> str:
    """Build the base review prompt."""
    
    files_context = ""
    if file_contents:
        files_context = "\n\n## Full File Contents (for context)\n"
        for path, content in list(file_contents.items())[:5]:
            files_context += f"\n### {path}\n```\n{content[:5000]}\n```\n"

    return f"""You are performing a **code review** of a pull request.

## Pull Request Information
- **Repository**: {context.repo_name}
- **PR #{context.pr_number}**: {context.pr_title}
- **Author**: {context.author}
- **Branch**: {context.head_branch} → {context.base_branch}
- **Changes**: +{context.additions} / -{context.deletions} in {context.changed_files_count} files
- **Languages**: {', '.join(context.repo_languages) if context.repo_languages else 'Unknown'}

## PR Description
{context.pr_description or 'No description provided.'}

## Code Changes (Diff)
```diff
{diff[:50000]}
```
{files_context}
"""


def get_output_format() -> str:
    """Get the JSON output format instructions."""
    return """
## Output Format

You MUST respond with a single valid JSON object (no markdown fences around it):

{"findings": [
  {
    "file_path": "path/to/file.rs",
    "line_start": 42,
    "line_end": 45,
    "severity": "critical|warning|suggestion|nitpick",
    "category": "security|performance|logic|style|architecture|testing|documentation",
    "title": "Short descriptive title",
    "description": "Detailed description of the issue and why it matters",
    "suggested_fix": "How to fix it (optional)",
    "confidence": 0.95
  }
],
"summary": "Brief overall summary of the review"
}

**Rules**:
- Only report issues you can clearly identify in the diff
- Be specific about file paths and line numbers
- Use "critical" only for security bugs or data corruption risks
- If the code looks good for your focus area, return empty findings array
- Maximum 5 findings per agent

Analyze the PR and output your JSON review.
"""


def parse_review_response(content: str) -> tuple[list[dict], str]:
    """Parse the agent's response into findings and summary."""
    content = content.strip()
    
    # Try to extract JSON from response
    if "```json" in content:
        match = re.search(r"```json\s*([\s\S]*?)```", content)
        if match:
            content = match.group(1).strip()
    elif "```" in content:
        match = re.search(r"```\s*([\s\S]*?)```", content)
        if match:
            content = match.group(1).strip()

    # Try to find JSON object
    json_match = re.search(r'\{[\s\S]*"findings"[\s\S]*\}', content)
    if json_match:
        content = json_match.group(0)
    
    try:
        data = json.loads(content)
        findings = data.get("findings", [])
        summary = data.get("summary", "Review completed")
        return findings, summary
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse review JSON: {e}")
        return [], "Failed to parse review response"


async def run_single_agent(
    client: CursorClient,
    repo_url: str,
    ref: str,
    prompt: str,
    agent_name: str,
    on_status: Optional[callable] = None,
) -> tuple[str, list[dict], str]:
    """Run a single agent and return its findings."""
    try:
        result = await client.run_review_agent(
            repo_url=repo_url,
            ref=ref,
            prompt=prompt,
            on_status=on_status,
        )
        findings, summary = parse_review_response(result.content)
        return agent_name, findings, summary
    except Exception as e:
        logger.error(f"Agent {agent_name} failed: {e}")
        return agent_name, [], f"Agent failed: {e}"


def aggregate_findings(
    all_findings: list[tuple[str, list[dict], str]],
    repo: str,
    pr_number: int,
) -> ConsolidatedReview:
    """Aggregate findings from multiple agents."""

    consolidated = []
    summaries = []
    failed_agents = []

    # Track which findings are similar (for consensus scoring)
    finding_clusters: dict[str, list[tuple[str, dict]]] = {}

    for agent_name, findings, summary in all_findings:
        summaries.append(f"**{agent_name}**: {summary}")

        # Track failed agents (summary contains error message)
        if "Agent failed:" in summary or "401 Unauthorized" in summary:
            failed_agents.append(agent_name)
        
        for raw in findings:
            # Create a key for clustering similar findings
            key = f"{raw.get('file_path', '')}:{raw.get('line_start', 0)}:{raw.get('category', '')}"
            
            if key not in finding_clusters:
                finding_clusters[key] = []
            finding_clusters[key].append((agent_name, raw))
    
    # Process clusters
    for key, cluster in finding_clusters.items():
        # Use the first finding as base, but track all agreeing agents
        agent_name, raw = cluster[0]
        agreeing_agents = [a for a, _ in cluster]
        
        try:
            severity = Severity(raw.get("severity", "suggestion").lower())
        except ValueError:
            severity = Severity.SUGGESTION
            
        try:
            category = Category(raw.get("category", "logic").lower())
        except ValueError:
            category = Category.LOGIC

        # Consensus score based on how many agents found this
        total_agents = len(all_findings)
        consensus_score = len(cluster) / total_agents if total_agents > 0 else 1.0

        finding = ConsolidatedFinding(
            id=f"finding-{len(consolidated)+1}",
            file_path=raw.get("file_path", "unknown"),
            line_start=int(raw.get("line_start", 1)),
            line_end=raw.get("line_end"),
            severity=severity,
            category=category,
            title=raw.get("title", "Issue found"),
            description=raw.get("description", ""),
            suggested_fix=raw.get("suggested_fix"),
            consensus_score=consensus_score,
            agreeing_agents=agreeing_agents,
            confidence=float(raw.get("confidence", 0.8)),
        )
        consolidated.append(finding)
    
    # Sort by priority
    consolidated.sort(key=lambda f: f.priority_score, reverse=True)
    
    # Build combined summary
    combined_summary = "\n".join(summaries) if summaries else "Review completed"
    
    return ConsolidatedReview(
        id=f"review-{uuid4().hex[:8]}",
        created_at=datetime.now(),
        repo=repo,
        pr_number=pr_number,
        findings=consolidated,
        summary=combined_summary,
        agent_count=len(all_findings),
        review_quality_score=0.85 if consolidated else 0.95,
        total_review_time_ms=0,
        failed_agents=failed_agents,
    )


async def review_pr_with_cursor_agent(
    repo: str,
    pr_number: int,
    cursor_config: CursorConfig,
    github_token: str,
    on_status: Optional[callable] = None,
    num_agents: int = 1,
) -> ConsolidatedReview:
    """Review a PR using Cursor Background Agent(s).
    
    Args:
        repo: Repository in "owner/name" format
        pr_number: Pull request number
        cursor_config: Cursor API configuration
        github_token: GitHub token for PR access
        on_status: Optional callback for status updates
        num_agents: Number of agents to run (1-3)
        
    Returns:
        ConsolidatedReview with findings
    """
    import time
    start_time = time.time()
    
    # Get PR information
    gh = GitHubClient(github_token)
    pr = gh.get_pull_request(repo, pr_number)
    repo_obj = gh.get_repo(repo)
    
    diff = gh.get_pr_diff(pr)
    files = gh.get_changed_files(pr)
    context = gh.build_review_context(pr, repo_obj)
    
    logger.info(f"Reviewing PR #{pr_number}: {context.pr_title}")
    logger.info(f"Files changed: {context.changed_files_count} (+{context.additions}/-{context.deletions})")
    
    # Build base prompt
    base_prompt = get_base_prompt(context, diff, files)
    output_format = get_output_format()
    repo_url = f"https://github.com/{repo}"
    
    # Select agents to run
    agents_to_run = AGENT_CONFIGS[:min(num_agents, len(AGENT_CONFIGS))]
    
    if num_agents == 1:
        # Single comprehensive agent
        prompt = base_prompt + """
**Analyze from ALL perspectives**: security, performance, logic, and code quality.
""" + output_format
        
        async with CursorClient(cursor_config) as client:
            if on_status:
                on_status("CREATING")
            
            result = await client.run_review_agent(
                repo_url=repo_url,
                ref=pr.base.ref,
                prompt=prompt,
                on_status=on_status,
            )
        
        raw_findings, summary = parse_review_response(result.content)
        all_findings = [("cursor-agent", raw_findings, summary)]
    else:
        # Multi-agent review
        logger.info(f"Running {len(agents_to_run)} specialized agents in parallel...")
        
        async with CursorClient(cursor_config) as client:
            tasks = []
            for agent_config in agents_to_run:
                prompt = base_prompt + agent_config["prompt_addition"] + output_format
                
                def make_status_callback(name: str):
                    def callback(status: str):
                        if on_status:
                            on_status(f"{name}: {status}")
                    return callback
                
                task = run_single_agent(
                    client=client,
                    repo_url=repo_url,
                    ref=pr.base.ref,
                    prompt=prompt,
                    agent_name=agent_config["name"],
                    on_status=make_status_callback(agent_config["name"]),
                )
                tasks.append(task)
            
            # Run all agents in parallel
            all_findings = await asyncio.gather(*tasks)
    
    # Aggregate findings
    review = aggregate_findings(list(all_findings), repo, pr_number)
    review.total_review_time_ms = int((time.time() - start_time) * 1000)
    
    logger.info(f"Review complete: {len(review.findings)} findings from {review.agent_count} agent(s) in {review.total_review_time_ms}ms")
    
    return review
