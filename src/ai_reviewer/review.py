"""Main review flow using Cursor Background Agent API with multi-agent support.

Review standard (embedded in prompts):
- Favor approving when the CL improves overall code health; no perfectionism.
- Use severity nitpick and prefix "Nit: " for optional/style points.
- Comment on the code not the author; be courteous; explain why when asking for a change.

What to look for (order of impact): Design → Functionality → Complexity → Tests
→ Naming, comments (why not what), style, consistency, documentation.

Design principles considered (when relevant): SOLID, DRY, KISS, YAGNI,
Composition over Inheritance, Law of Demeter, Convention over Configuration.
Only flag violations that meaningfully hurt maintainability or clarity.

Severity semantics: see ai_reviewer.models.findings.Severity.
"""

import asyncio
import json
import logging
import re
from collections.abc import Callable
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

from ai_reviewer.agents.cursor_client import CursorClient, CursorConfig
from ai_reviewer.github.client import GitHubClient
from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import Category, ConsolidatedFinding, Severity
from ai_reviewer.models.review import ConsolidatedReview

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
**YOUR FOCUS: CODE QUALITY & DESIGN PRINCIPLES**
You are a code quality reviewer. Focus on the design principles listed above (SOLID, DRY, KISS, YAGNI, Composition over Inheritance, Law of Demeter) and on: API design, error handling patterns, maintainability, tests for critical paths, documentation. Ignore security and performance unless they affect maintainability or correctness.
""",
    },
]


def _detect_pr_type(changed_paths: list[str]) -> str:
    """Detect PR type from changed file paths for context-aware review instructions."""
    if not changed_paths:
        return "code"
    if all(
        p.endswith(".md") or p.endswith(".mdx") for p in changed_paths
    ):
        return "docs"
    if all(
        p.startswith(".github/") or p.endswith(".yml") or p.endswith(".yaml")
        for p in changed_paths
    ):
        return "ci"
    return "code"


def get_base_prompt(
    context: ReviewContext,
    diff: str,
    file_contents: dict[str, str],
    changed_paths: list[str] | None = None,
) -> str:
    """Build the base review prompt."""
    pr_type = _detect_pr_type(changed_paths or list(file_contents.keys()))

    pr_type_instruction = ""
    if pr_type == "docs":
        pr_type_instruction = "\n**This PR is docs-only (markdown).** Only report factual errors, broken links, or security-sensitive content. Do not suggest code style, tests, or nitpicks.\n"
    elif pr_type == "ci":
        pr_type_instruction = "\n**This PR is CI/workflow-only.** Focus on workflow correctness (paths, steps, secrets). Do not report code style or nitpicks.\n"

    files_context = ""
    if file_contents:
        files_context = "\n\n## Full File Contents (for context)\n"
        for path, content in list(file_contents.items())[:5]:
            files_context += f"\n### {path}\n```\n{content[:5000]}\n```\n"

    review_standard = """
**Review standard:** Favor approving when the CL improves overall code health, even if it isn't perfect. There is no "perfect" code—only better code. Do not block on minor polish. For optional or style-only points, use severity "nitpick" and prefix the title with "Nit: " so the author knows it's optional. Comment on the code, not the author; be courteous and explain *why* when asking for a change.
"""
    what_to_look_for = """
**What to look for (in order of impact):** Design (does the change make sense and integrate well?) → Functionality (edge cases, concurrency, correct behavior) → Complexity (no over-engineering) → Tests (present and meaningful) → Naming, comments (explain why, not what), style, consistency with existing code, documentation if behavior/build/test changes.
"""
    design_principles = """
**Design principles to consider (when relevant):** SOLID (single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion); DRY (no duplicate logic—extract and reuse); KISS (keep it simple; avoid over-engineering); YAGNI (don't add code for hypothetical future needs); Composition over Inheritance (prefer composing over deep hierarchies); Law of Demeter (talk to immediate collaborators only, avoid long chains); Convention over Configuration where it fits. Only flag violations that meaningfully hurt maintainability or clarity—use "Nit:" for minor style preferences.
"""
    return f"""You are performing a **code review** of a pull request.
{review_standard}
{what_to_look_for}
{design_principles}
{pr_type_instruction}
## Pull Request Information
- **Repository**: {context.repo_name}
- **PR #{context.pr_number}**: {context.pr_title}
- **Author**: {context.author}
- **Branch**: {context.head_branch} → {context.base_branch}
- **Changes**: +{context.additions} / -{context.deletions} in {context.changed_files_count} files
- **Languages**: {", ".join(context.repo_languages) if context.repo_languages else "Unknown"}

## PR Description
{context.pr_description or "No description provided."}

## Code Changes (Diff)
```diff
{diff[:50000]}
```
{files_context}
"""


def get_output_format(pr_type: str = "code") -> str:
    """Get the JSON output format instructions."""
    concise_rules = [
        "- Be concise: one short sentence per finding description. Do not repeat the same point.",
        "- Only report issues on changed lines; do not suggest pre-existing improvements.",
        "- **Severity semantics:** critical = must fix (security bugs or data corruption risks only); warning = should fix (other serious correctness or maintainability issues); suggestion = consider; nitpick = optional polish—always prefix title with \"Nit: \" for nitpicks.",
        "- If the code looks good for your focus area, return empty findings array.",
        "- Maximum 5 findings per agent.",
        "- If something is done well (e.g. clear naming, good tests), mention it briefly in the summary.",
    ]
    if pr_type == "docs":
        concise_rules.append(
            "- Do not report style, nitpicks, or \"add tests\". Only factual errors or security."
        )
    elif pr_type == "ci":
        concise_rules.append(
            "- Do not report code style or nitpicks. Focus on workflow correctness only."
        )

    return f"""
## Output Format

You MUST respond with a single valid JSON object (no markdown fences around it):

{{"findings": [
  {{
    "file_path": "path/to/file.rs",
    "line_start": 42,
    "line_end": 45,
    "severity": "critical|warning|suggestion|nitpick",
    "category": "security|performance|logic|style|architecture|testing|documentation",
    "title": "Short descriptive title (use \\"Nit: \\" prefix for nitpick severity)",
    "description": "One concise sentence; explain why it matters when helpful",
    "suggested_fix": "How to fix it (optional)",
    "confidence": 0.95
  }}
],
"summary": "One brief sentence. Include one positive if something is done well."
}}

**Rules**:
{chr(10).join(concise_rules)}

Analyze the PR and output your JSON review.
"""


# --- Cross-review round (agents validate and rank each other's findings) ---

# Max findings to send in cross-review (avoid huge prompts)
_CROSS_REVIEW_MAX_FINDINGS = 20
# Max diff chars in cross-review prompt
_CROSS_REVIEW_DIFF_MAX_CHARS = 15000


def get_cross_review_prompt(
    context: ReviewContext,
    review: ConsolidatedReview,
    diff: str,
) -> str:
    """Build prompt for cross-review round: validate findings and rank by importance."""
    findings_blob = []
    for i, f in enumerate(review.findings[: _CROSS_REVIEW_MAX_FINDINGS], 1):
        line_ref = f"{f.line_start}" + (f"-{f.line_end}" if f.line_end else "")
        findings_blob.append(
            f"{i}. [id={f.id}] {f.file_path}:{line_ref} [{f.severity.value}] {f.title}\n   {f.description}"
        )
    findings_text = "\n".join(findings_blob)

    return f"""You are in a **cross-review round**. Multiple agents already produced the findings below for this PR. Your job is to validate them and rank by importance.

## PR
- **Repo**: {context.repo_name} | **PR #{context.pr_number}**: {context.pr_title}
- **Changes**: +{context.additions}/-{context.deletions} in {context.changed_files_count} files

## Code diff (excerpt)
```diff
{diff[:_CROSS_REVIEW_DIFF_MAX_CHARS]}
```

## Findings to validate and rank
{findings_text}

For each finding, decide:
1. **Valid** (true/false): Does it make sense? Does it follow review best practices (concrete, actionable, not nitpicky)? Would you keep it in the final report?
2. **Rank** (integer): Importance for the author, 1 = most important. Ties allowed.

Output the list of assessments in the exact JSON format below (use the finding `id` from the list, e.g. finding-1, finding-2).
"""


def get_cross_review_output_format() -> str:
    """JSON schema for cross-review round response."""
    return '''
## Output format (valid JSON only, no markdown fences)
{"assessments": [{"id": "finding-1", "valid": true, "rank": 1}, {"id": "finding-2", "valid": false, "rank": 5}, ...], "summary": "One sentence on overall quality of the findings."}

- "id" must match the finding id from the list (e.g. finding-1, finding-2).
- "valid": true if the finding should stay in the report, false if it should be dropped or is not actionable.
- "rank": integer, 1 = most important. Lower rank = higher priority.
- Include every finding id from the list in assessments.
'''


def parse_cross_review_response(content: str) -> tuple[list[dict[str, Any]], str]:
    """Parse cross-review JSON into list of {id, valid, rank} and summary."""
    content = content.strip()
    if "```json" in content:
        match = re.search(r"```json\s*([\s\S]*?)```", content)
        if match:
            content = match.group(1).strip()
    elif "```" in content:
        match = re.search(r"```\s*([\s\S]*?)```", content)
        if match:
            content = match.group(1).strip()
    json_match = re.search(r'\{[\s\S]*"assessments"[\s\S]*\}', content)
    if json_match:
        content = json_match.group(0)
    try:
        data = json.loads(content)
        assessments = data.get("assessments", [])
        summary = data.get("summary", "")
        return assessments, summary
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse cross-review JSON: {e}")
        return [], ""


def apply_cross_review(
    review: ConsolidatedReview,
    all_assessments: list[tuple[str, list[dict[str, Any]]]],
    min_validation_agreement: float = 0.5,
) -> ConsolidatedReview:
    """Filter and re-rank findings using cross-review assessments.

    - Drops findings where the fraction of agents that said valid is < min_validation_agreement.
    - Re-orders by average rank (1 = first), then by severity.
    """
    if not all_assessments:
        return review

    finding_ids = [f.id for f in review.findings]
    id_to_finding = {f.id: f for f in review.findings}

    # Per finding: list of (valid, rank) from each agent
    id_to_votes: dict[str, list[tuple[bool, int]]] = {fid: [] for fid in finding_ids}

    for _agent_name, assessments in all_assessments:
        for a in assessments:
            fid = a.get("id") or a.get("finding_id")
            if not fid or fid not in id_to_finding:
                continue
            valid = a.get("valid", True)
            rank = a.get("rank", 99)
            if isinstance(rank, (int, float)):
                rank = max(1, int(rank))
            else:
                rank = 99
            id_to_votes[fid].append((valid, rank))

    n_agents = len(all_assessments)
    kept: list[tuple[ConsolidatedFinding, float, float]] = []  # (finding, valid_ratio, avg_rank)

    for fid in finding_ids:
        votes = id_to_votes.get(fid, [])
        if not votes:
            kept.append((id_to_finding[fid], 1.0, 99.0))
            continue
        valid_count = sum(1 for v, _ in votes if v)
        valid_ratio = valid_count / n_agents if n_agents else 1.0
        if valid_ratio < min_validation_agreement:
            continue  # Drop finding
        avg_rank = sum(r for _, r in votes) / len(votes) if votes else 99.0
        kept.append((id_to_finding[fid], valid_ratio, avg_rank))

    # Sort by avg_rank ascending, then by severity (critical first)
    severity_order = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.SUGGESTION: 2, Severity.NITPICK: 3}
    kept.sort(key=lambda x: (x[2], severity_order.get(x[0].severity, 4)))

    new_findings = [x[0] for x in kept]
    n_dropped = len(review.findings) - len(new_findings)
    original_ids = [f.id for f in review.findings]
    new_ids = [f.id for f in new_findings]
    order_changed = original_ids != new_ids

    summary = review.summary
    if n_dropped > 0 or order_changed:
        parts = []
        if n_dropped > 0:
            parts.append(f"{n_dropped} finding(s) dropped by cross-review")
        if order_changed:
            parts.append("re-ranked by agent consensus")
        summary = review.summary + "\n\nCross-review: " + "; ".join(parts) + "."

    return ConsolidatedReview(
        id=review.id,
        created_at=review.created_at,
        repo=review.repo,
        pr_number=review.pr_number,
        findings=new_findings,
        summary=summary,
        agent_count=review.agent_count,
        review_quality_score=review.review_quality_score,
        total_review_time_ms=review.total_review_time_ms,
        failed_agents=review.failed_agents,
    )


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
    on_status: Callable[..., Any] | None = None,
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


# Similarity threshold for clustering findings (match aggregator default)
_SIMILARITY_THRESHOLD = 0.85


def _normalize_path(path: str) -> str:
    """Normalize file path for comparison."""
    if not path:
        return ""
    return path.strip().replace("\\", "/").lstrip("./")


def _raw_lines_overlap(raw1: dict, raw2: dict, tolerance: int = 5) -> bool:
    """Check if two raw findings have overlapping or close line ranges."""
    start1 = int(raw1.get("line_start", 1))
    end1 = raw1.get("line_end")
    end1 = int(end1) if end1 else start1
    start2 = int(raw2.get("line_start", 1))
    end2 = raw2.get("line_end")
    end2 = int(end2) if end2 else start2
    return not (end1 + tolerance < start2 or end2 + tolerance < start1)


def _raw_text_similarity(text1: str, text2: str) -> float:
    """Compute text similarity using SequenceMatcher (0.0–1.0)."""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def _raw_findings_similar(raw1: dict, raw2: dict, threshold: float = _SIMILARITY_THRESHOLD) -> bool:
    """Check if two raw findings describe the same issue (for consensus clustering)."""
    path1 = _normalize_path(raw1.get("file_path", ""))
    path2 = _normalize_path(raw2.get("file_path", ""))
    if path1 != path2:
        return False

    cat1 = (raw1.get("category") or "logic").lower().strip()
    cat2 = (raw2.get("category") or "logic").lower().strip()
    if cat1 != cat2:
        return False

    if not _raw_lines_overlap(raw1, raw2):
        return False

    title1 = (raw1.get("title") or "").strip()
    title2 = (raw2.get("title") or "").strip()
    desc1 = (raw1.get("description") or "").strip()
    desc2 = (raw2.get("description") or "").strip()
    title_sim = _raw_text_similarity(title1, title2)
    desc_sim = _raw_text_similarity(desc1, desc2)
    combined = (title_sim * 0.6) + (desc_sim * 0.4)
    return combined >= threshold


def _cluster_raw_findings(
    tagged: list[tuple[str, dict]], threshold: float = _SIMILARITY_THRESHOLD
) -> list[list[tuple[str, dict]]]:
    """Cluster similar raw findings so consensus = agents that found the same issue."""
    if not tagged:
        return []

    clusters: list[list[tuple[str, dict]]] = []
    used = set()

    for i, (agent_i, raw_i) in enumerate(tagged):
        if i in used:
            continue
        cluster = [(agent_i, raw_i)]
        used.add(i)
        for j, (agent_j, raw_j) in enumerate(tagged):
            if j in used:
                continue
            if _raw_findings_similar(raw_i, raw_j, threshold):
                cluster.append((agent_j, raw_j))
                used.add(j)
        clusters.append(cluster)

    return clusters


def aggregate_findings(
    all_findings: list[tuple[str, list[dict], str]],
    repo: str,
    pr_number: int,
) -> ConsolidatedReview:
    """Aggregate findings from multiple agents."""

    consolidated = []
    summaries = []
    failed_agents = []

    # Flatten to (agent_name, raw) and collect summaries
    tagged: list[tuple[str, dict]] = []
    for agent_name, findings, summary in all_findings:
        summaries.append(f"**{agent_name}**: {summary}")

        if "Agent failed:" in summary or "401 Unauthorized" in summary:
            failed_agents.append(agent_name)

        for raw in findings:
            tagged.append((agent_name, raw))

    # Cluster by similarity (same file, overlapping lines, same category, similar title/description)
    finding_clusters = _cluster_raw_findings(tagged)

    # Process clusters
    for cluster in finding_clusters:
        # Use the first finding as base, but track all agreeing agents
        agent_name, raw = cluster[0]
        # Deduplicate agents - a single agent may have multiple similar findings clustered
        agreeing_agents = list(dict.fromkeys(a for a, _ in cluster))

        try:
            severity = Severity(raw.get("severity", "suggestion").lower())
        except ValueError:
            severity = Severity.SUGGESTION

        try:
            category = Category(raw.get("category", "logic").lower())
        except ValueError:
            category = Category.LOGIC

        # Consensus score based on unique agents that found this issue
        total_agents = len(all_findings)
        consensus_score = len(agreeing_agents) / total_agents if total_agents > 0 else 1.0

        finding = ConsolidatedFinding(
            id=f"finding-{len(consolidated) + 1}",
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

    # Compute quality score from consensus and agent count
    total_agents = len(all_findings)
    if not consolidated:
        # No findings: higher score for more agents (max 0.95)
        quality_score = min(0.95, 0.7 + total_agents * 0.1)
    else:
        # With findings: score based on average consensus weighted by agent coverage
        avg_consensus = sum(f.consensus_score for f in consolidated) / len(consolidated)
        agent_factor = min(1.0, total_agents / 3)  # Full credit at 3+ agents
        quality_score = round(avg_consensus * agent_factor, 2)

    return ConsolidatedReview(
        id=f"review-{uuid4().hex[:8]}",
        created_at=datetime.now(),
        repo=repo,
        pr_number=pr_number,
        findings=consolidated,
        summary=combined_summary,
        agent_count=len(all_findings),
        review_quality_score=quality_score,
        total_review_time_ms=0,
        failed_agents=failed_agents,
    )


async def _run_single_cross_review_agent(
    client: CursorClient,
    repo_url: str,
    ref: str,
    cross_prompt: str,
    agent_name: str,
    on_status: Callable[..., Any] | None,
) -> tuple[str, list[dict[str, Any]] | None]:
    """Run one agent's cross-review; returns (name, assessments) or (name, None) on failure."""
    try:
        if on_status:
            on_status(f"Cross-review: {agent_name}")
        result = await client.run_review_agent(
            repo_url=repo_url,
            ref=ref,
            prompt=cross_prompt,
            on_status=on_status,
        )
        assessments, _ = parse_cross_review_response(result.content)
        return (agent_name, assessments)
    except Exception as e:
        logger.warning(f"Cross-review agent {agent_name} failed: {e}")
        return (agent_name, None)


async def run_cross_review_round(
    client: CursorClient,
    repo_url: str,
    ref: str,
    review: ConsolidatedReview,
    context: ReviewContext,
    diff: str,
    agents_to_run: list[dict],
    on_status: Callable[..., Any] | None = None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Run cross-review: each agent validates and ranks the consolidated findings (in parallel)."""
    if not review.findings:
        return []

    cross_prompt = get_cross_review_prompt(context, review, diff) + get_cross_review_output_format()
    tasks = [
        _run_single_cross_review_agent(
            client=client,
            repo_url=repo_url,
            ref=ref,
            cross_prompt=cross_prompt,
            agent_name=agent_config["name"],
            on_status=on_status,
        )
        for agent_config in agents_to_run
    ]
    gathered = await asyncio.gather(*tasks)
    return [(name, assessments) for name, assessments in gathered if assessments is not None]


async def review_pr_with_cursor_agent(
    repo: str,
    pr_number: int,
    cursor_config: CursorConfig,
    github_token: str,
    on_status: Callable[..., Any] | None = None,
    num_agents: int = 3,
    enable_cross_review: bool = True,
    min_validation_agreement: float = 0.5,
) -> ConsolidatedReview:
    """Review a PR using Cursor Background Agent(s).

    Args:
        repo: Repository in "owner/name" format
        pr_number: Pull request number
        cursor_config: Cursor API configuration
        github_token: GitHub token for PR access
        on_status: Optional callback for status updates
        num_agents: Number of agents to run (1-3)
        enable_cross_review: If True (default) and num_agents > 1, run a second round where
            agents validate and rank findings; drop low-agreement and re-order by rank.
        min_validation_agreement: Fraction of agents that must mark a finding valid to keep it (0-1).

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
    logger.info(
        f"Files changed: {context.changed_files_count} (+{context.additions}/-{context.deletions})"
    )

    changed_paths = list(files.keys())
    pr_type = _detect_pr_type(changed_paths)
    if pr_type != "code":
        logger.info(f"PR type: {pr_type} – using context-aware review rules")

    base_prompt = get_base_prompt(context, diff, files, changed_paths=changed_paths)
    output_format = get_output_format(pr_type)
    repo_url = f"https://github.com/{repo}"

    # Select agents to run
    agents_to_run = AGENT_CONFIGS[: min(num_agents, len(AGENT_CONFIGS))]

    if num_agents == 1:
        # Single comprehensive agent
        prompt = (
            base_prompt
            + """
**Analyze from ALL perspectives**: security, performance, logic, and code quality.
"""
            + output_format
        )

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

    # Optional: cross-review round (agents validate and rank findings)
    if (
        enable_cross_review
        and num_agents > 1
        and review.findings
        and not review.all_agents_failed
    ):
        logger.info("Running cross-review round (validate and rank findings)...")
        async with CursorClient(cursor_config) as client:
            cross_results = await run_cross_review_round(
                client=client,
                repo_url=repo_url,
                ref=pr.base.ref,
                review=review,
                context=context,
                diff=diff,
                agents_to_run=agents_to_run,
                on_status=on_status,
            )
        if cross_results:
            review = apply_cross_review(
                review, cross_results, min_validation_agreement
            )
            logger.info(
                f"Cross-review done: {len(review.findings)} findings after validation"
            )

    review.total_review_time_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"Review complete: {len(review.findings)} findings from {review.agent_count} agent(s) in {review.total_review_time_ms}ms"
    )

    return review
