"""Main review flow using Cursor Background Agent API."""

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
from ai_reviewer.models.review import ConsolidatedReview

logger = logging.getLogger(__name__)


def get_review_prompt(context: ReviewContext, diff: str, file_contents: dict[str, str]) -> str:
    """Build the review prompt for the Cursor agent."""
    
    files_context = ""
    if file_contents:
        files_context = "\n\n## Full File Contents (for context)\n"
        for path, content in list(file_contents.items())[:5]:  # Limit files
            files_context += f"\n### {path}\n```\n{content[:5000]}\n```\n"

    return f"""You are performing a **comprehensive code review** of a pull request.

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

---

## Review Instructions

Analyze the code changes from **multiple perspectives**:

### 1. Security
- Injection vulnerabilities (SQL, command, XSS)
- Authentication/authorization issues
- Data exposure or validation problems
- Cryptographic misuse

### 2. Performance
- Algorithm complexity issues
- Memory leaks or resource management
- N+1 queries or unnecessary work
- Missing timeouts or limits

### 3. Logic & Correctness
- Off-by-one errors, wrong conditions
- Edge cases not handled
- Error handling issues
- State management problems

### 4. Code Quality
- Consistency with codebase patterns
- API design issues
- Missing documentation
- Code duplication

---

## Output Format

You MUST respond with a single valid JSON object (no markdown fences around it):

{{"findings": [
  {{
    "file_path": "path/to/file.rs",
    "line_start": 42,
    "line_end": 45,
    "severity": "critical|warning|suggestion|nitpick",
    "category": "security|performance|logic|style|architecture|testing|documentation",
    "title": "Short descriptive title",
    "description": "Detailed description of the issue and why it matters",
    "suggested_fix": "How to fix it (optional)",
    "confidence": 0.95
  }}
],
"summary": "Brief overall summary of the review"
}}

**Rules**:
- Only report issues you can clearly identify in the diff
- Be specific about file paths and line numbers
- Use "critical" only for security bugs or data corruption risks
- Use "warning" for bugs and important issues
- Use "suggestion" for improvements
- Use "nitpick" for style/minor issues
- If the code looks good, return empty findings array

Analyze the PR and output your JSON review.
"""


def parse_review_response(content: str) -> tuple[list[dict], str]:
    """Parse the agent's response into findings and summary."""
    content = content.strip()
    
    # Try to extract JSON from response
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


def findings_to_consolidated(
    raw_findings: list[dict],
    summary: str,
    repo: str,
    pr_number: int,
) -> ConsolidatedReview:
    """Convert raw findings to a ConsolidatedReview."""
    
    consolidated = []
    for i, raw in enumerate(raw_findings):
        try:
            severity = Severity(raw.get("severity", "suggestion").lower())
        except ValueError:
            severity = Severity.SUGGESTION
            
        try:
            category = Category(raw.get("category", "logic").lower())
        except ValueError:
            category = Category.LOGIC

        finding = ConsolidatedFinding(
            id=f"finding-{i+1}",
            file_path=raw.get("file_path", "unknown"),
            line_start=int(raw.get("line_start", 1)),
            line_end=raw.get("line_end"),
            severity=severity,
            category=category,
            title=raw.get("title", "Issue found"),
            description=raw.get("description", ""),
            suggested_fix=raw.get("suggested_fix"),
            consensus_score=1.0,  # Single agent
            agreeing_agents=["cursor-agent"],
            confidence=float(raw.get("confidence", 0.8)),
        )
        consolidated.append(finding)
    
    # Sort by priority
    consolidated.sort(key=lambda f: f.priority_score, reverse=True)
    
    return ConsolidatedReview(
        id=f"review-{uuid4().hex[:8]}",
        created_at=datetime.now(),
        repo=repo,
        pr_number=pr_number,
        findings=consolidated,
        summary=summary,
        agent_count=1,
        review_quality_score=0.85 if consolidated else 0.95,
        total_review_time_ms=0,  # Will be updated by caller
    )


async def review_pr_with_cursor_agent(
    repo: str,
    pr_number: int,
    cursor_config: CursorConfig,
    github_token: str,
    on_status: Optional[callable] = None,
) -> ConsolidatedReview:
    """Review a PR using Cursor Background Agent.
    
    This creates a Cursor agent that analyzes the PR and returns findings.
    The agent runs async and may take several minutes.
    
    Args:
        repo: Repository in "owner/name" format
        pr_number: Pull request number
        cursor_config: Cursor API configuration
        github_token: GitHub token for PR access
        on_status: Optional callback for status updates
        
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
    
    # Build prompt
    prompt = get_review_prompt(context, diff, files)
    
    # Create and run Cursor agent
    async with CursorClient(cursor_config) as client:
        repo_url = f"https://github.com/{repo}"
        
        if on_status:
            on_status("CREATING")
        
        result = await client.run_review_agent(
            repo_url=repo_url,
            ref=pr.base.ref,  # Analyze against base branch
            prompt=prompt,
            on_status=on_status,
        )
    
    # Parse response
    raw_findings, summary = parse_review_response(result.content)
    
    # Build consolidated review
    review = findings_to_consolidated(raw_findings, summary, repo, pr_number)
    review.total_review_time_ms = int((time.time() - start_time) * 1000)
    
    logger.info(f"Review complete: {len(review.findings)} findings in {review.total_review_time_ms}ms")
    
    return review
