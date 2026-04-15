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
import fnmatch
import json
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

from ai_reviewer.agents.anthropic_client import AnthropicClient
from ai_reviewer.agents.base import ReviewAgent
from ai_reviewer.agents.patterns import PatternsAgent, StyleAgent
from ai_reviewer.agents.performance import LogicAgent, PerformanceAgent
from ai_reviewer.agents.security import AuthenticationAgent, SecurityAgent
from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.context.builder import build_system_blocks, build_user_blocks
from ai_reviewer.context.fetch import build_repo_map, fetch_conventions
from ai_reviewer.context.neighbors import select_neighbors
from ai_reviewer.github.client import GitHubClient
from ai_reviewer.models.context import ReviewContext
from ai_reviewer.models.findings import Category, ConsolidatedFinding, ReviewFinding, Severity
from ai_reviewer.models.review import AgentReview, ConsolidatedReview, ScoreBreakdown
from ai_reviewer.security.scanner import scan_for_secrets
from ai_reviewer.session import ReviewSession
from ai_reviewer.tools.repo_tools import ToolRegistry

logger = logging.getLogger(__name__)


# Different agent configurations for multi-perspective review
AGENT_CONFIGS = [
    {
        "name": "security-agent",
        "focus": "security",
        "prompt_addition": """
**YOUR FOCUS: SECURITY**
You are a security expert with deep knowledge of OWASP Top 10 vulnerabilities.
Focus ONLY on security issues. Ignore performance, style, and other non-security concerns.

Review for these categories:

1. **Injection Vulnerabilities**
   - SQL injection (string interpolation in queries)
   - Command injection (os.system, subprocess with user input)
   - XSS (unescaped user input in HTML/JS)
   - LDAP/XPath injection

2. **Authentication & Authorization**
   - Hardcoded credentials or secrets
   - Weak password handling (MD5/SHA1 instead of bcrypt/argon2)
   - Missing authentication or authorization checks
   - Broken access control and privilege escalation

3. **Cryptographic Issues**
   - Weak algorithms (MD5, SHA1 for security purposes)
   - Hardcoded keys or IVs
   - Insecure random number generation
   - Missing encryption for sensitive data

4. **Data Exposure**
   - Sensitive data in logs or error messages
   - Insecure data transmission
   - Missing input validation

5. **Security Misconfigurations**
   - Debug mode in production
   - Permissive CORS policies
   - Missing security headers
   - Insecure defaults

Provide specific line numbers and concrete evidence for each finding.
Do not speculate about issues that might exist elsewhere in the codebase.
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
    if all(p.endswith(".md") or p.endswith(".mdx") for p in changed_paths):
        return "docs"
    if all(
        p.startswith(".github/") or p.endswith(".yml") or p.endswith(".yaml") for p in changed_paths
    ):
        return "ci"
    return "code"


def classify_pr(
    changed_paths: list[str],
    additions: int = 0,
    deletions: int = 0,
) -> tuple[str, str]:
    """Classify a PR by type (docs/ci/code) and size (trivial/small/medium/large).

    Reuses ``_detect_pr_type`` for the type half and buckets
    ``additions + deletions`` for size:
      < 50  → trivial
      < 200 → small
      < 1000 → medium
      ≥ 1000 → large
    """
    pr_type = _detect_pr_type(changed_paths)
    total = additions + deletions
    if total < 50:
        pr_size = "trivial"
    elif total < 200:
        pr_size = "small"
    elif total < 1000:
        pr_size = "medium"
    else:
        pr_size = "large"
    return pr_type, pr_size


_DIFF_FILE_HEADER_RE = re.compile(r"^diff --git a/.+ b/(.+)$")


def _compile_ignore_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    """Pre-compile fnmatch patterns into regex for efficient repeated matching."""
    return [re.compile(fnmatch.translate(p)) for p in patterns]


def filter_by_ignore_patterns(files: dict[str, str], patterns: list[str]) -> dict[str, str]:
    """Remove entries whose path matches any of the fnmatch *patterns*."""
    if not patterns:
        return files
    compiled = _compile_ignore_patterns(patterns)
    return {
        path: content for path, content in files.items() if not any(c.match(path) for c in compiled)
    }


def filter_diff_by_ignore_patterns(diff: str, patterns: list[str]) -> str:
    """Drop entire file sections from a unified diff when the path matches *patterns*.

    Splits on ``diff --git`` boundaries and reassembles only non-matching sections.
    """
    if not patterns:
        return diff

    compiled = _compile_ignore_patterns(patterns)
    sections: list[str] = []
    current_section: list[str] = []
    current_file: str | None = None

    for line in diff.splitlines(keepends=True):
        header_match = _DIFF_FILE_HEADER_RE.match(line.rstrip("\n"))
        if header_match:
            if (
                current_section
                and current_file is not None
                and not any(c.match(current_file) for c in compiled)
            ):
                sections.append("".join(current_section))
            current_section = [line]
            current_file = header_match.group(1)
        else:
            current_section.append(line)

    if (
        current_section
        and current_file is not None
        and not any(c.match(current_file) for c in compiled)
    ):
        sections.append("".join(current_section))

    return "".join(sections)


_LANGUAGE_RULES: dict[str, str] = {
    "python": (
        "For Python:\n"
        "- Mutable default arguments (e.g. `def f(x=[])`) — flag every occurrence.\n"
        "- Bare `except:` or `except Exception:` without re-raise — require specific exception types.\n"
        "- Missing type hints on public function signatures.\n"
        '- f-string injection in `logging.info(f"...")` — use `logging.info("...", arg)` instead.\n'
        "- Missing context managers (`with`) for file handles, DB connections, locks.\n"
        "- `subprocess` calls with `shell=True` — flag as security risk.\n"
        "- Shadowing built-in names (`id`, `type`, `list`, `dict`, `input`, `hash`).\n"
        "- `import *` usage — require explicit imports.\n"
        "- Missing `__all__` in library/package modules that define a public API.\n"
        "- `os.path` usage where `pathlib.Path` is preferred in modern Python."
    ),
    "rust": (
        "For Rust:\n"
        "- `.unwrap()` / `.expect()` in non-test code — require proper error propagation with `?` or `match`.\n"
        "- `unsafe` blocks without a `// SAFETY:` comment justifying invariants.\n"
        "- Unnecessary `.clone()` — flag when borrowing or references would suffice.\n"
        "- Unbounded allocations: `Vec::new()` in loops without pre-allocated capacity, or "
        "`collect()` on unbounded iterators without size hints.\n"
        "- Missing lifetime annotations where the compiler cannot elide them.\n"
        "- `panic!()` / `todo!()` / `unimplemented!()` in library code — should return `Result`.\n"
        "- Mutex poisoning: using `.lock().unwrap()` without handling `PoisonError`.\n"
        "- Large types on the stack — suggest `Box<T>` for types > ~1KB.\n"
        "- Missing `#[must_use]` on functions returning `Result` or important values.\n"
        "- `String` vs `&str` in function parameters — prefer `&str` / `impl AsRef<str>` for inputs."
    ),
    "javascript": (
        "For JavaScript:\n"
        "- Prototype pollution via unguarded `Object.assign` or bracket notation from user input.\n"
        "- `==` vs `===` — require strict equality everywhere.\n"
        "- Unhandled Promise rejections: missing `.catch()` or `try/catch` around `await`.\n"
        "- `eval()`, `new Function()`, `innerHTML` — flag as security risks.\n"
        "- Missing input sanitization on user-facing data (XSS vectors).\n"
        "- `var` usage — require `const`/`let`.\n"
        "- Callback hell — suggest async/await refactoring when nesting > 2 levels.\n"
        "- Missing `AbortController` for fetch calls that should be cancellable.\n"
        "- `JSON.parse()` without try/catch.\n"
        "- Regex denial-of-service (ReDoS) — flag catastrophic backtracking patterns."
    ),
    "typescript": (
        "For TypeScript:\n"
        "- `any` type escapes — flag every `any` that isn't explicitly justified.\n"
        "- Type assertions (`as T`) that bypass type safety — prefer type guards.\n"
        "- `@ts-ignore` / `@ts-expect-error` without justification comment.\n"
        "- Missing error boundaries in React components (when JSX is present).\n"
        "- `==` vs `===` — require strict equality everywhere.\n"
        "- Unhandled Promise rejections: missing `.catch()` or `try/catch` around `await`.\n"
        "- `eval()`, `new Function()`, `innerHTML` — flag as security risks.\n"
        "- Missing input sanitization on user-facing data (XSS vectors).\n"
        "- `JSON.parse()` without try/catch.\n"
        "- Regex denial-of-service (ReDoS) — flag catastrophic backtracking patterns."
    ),
    "go": (
        "For Go:\n"
        "- Unchecked errors: every returned `error` must be checked or explicitly ignored with `_`.\n"
        "- SQL string concatenation — use parameterized queries to prevent injection.\n"
        "- Goroutine leaks: ensure goroutines have a termination path (context, done channel, timeout).\n"
        "- Missing `defer` for resource cleanup (files, connections, locks).\n"
        "- Nil pointer dereference: check interface/pointer values before use after type assertions.\n"
        "- `sync.Mutex` without matching `Unlock` (prefer `defer mu.Unlock()` immediately after `Lock`).\n"
        "- Unbuffered channel sends in goroutines without a receiver — can block forever.\n"
        "- `init()` functions with side effects — prefer explicit initialization.\n"
        "- Exported functions missing doc comments.\n"
        "- `context.Background()` in request handlers — propagate the request context instead."
    ),
}


def get_language_rules(languages: list[str]) -> str:
    """Return language-specific review rules for the given languages."""
    rules = []
    for lang in languages:
        rule = _LANGUAGE_RULES.get(lang.lower(), "")
        if rule:
            rules.append(rule)
    return "\n\n".join(rules)


def get_base_prompt(
    context: ReviewContext,
    diff: str,
    file_contents: dict[str, str],
    changed_paths: list[str] | None = None,
) -> str:
    """Build the base review prompt."""
    paths = changed_paths or list(file_contents.keys())
    pr_type, pr_size = classify_pr(paths, context.additions, context.deletions)

    pr_type_instruction = ""
    if pr_type == "docs":
        pr_type_instruction = "\n**This PR is docs-only (markdown).** Only report factual errors, broken links, or security-sensitive content. Do not suggest code style, tests, or nitpicks.\n"
    elif pr_type == "ci":
        pr_type_instruction = "\n**This PR is CI/workflow-only.** Focus on workflow correctness (paths, steps, secrets). Do not report code style or nitpicks.\n"

    pr_size_instruction = ""
    if pr_size in ("trivial", "small"):
        pr_size_instruction = "\n**Small change — prioritize precision.** Only report findings you are confident about. Do not pad the review with low-value suggestions.\n"
    elif pr_size == "large":
        pr_size_instruction = "\n**Large change — prioritize high-severity issues.** Focus on architectural concerns, correctness, and security over minor style or nitpicks.\n"

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
    conventions_section = ""
    if context.conventions:
        conventions_section = f"\n## Repository Conventions\n{context.conventions}\n"

    _MAX_CUSTOM_RULES = 20
    _MAX_RULE_LENGTH = 500

    custom_rules_section = ""
    if context.repo_config and context.repo_config.get("custom_rules"):
        raw_rules = context.repo_config["custom_rules"]
        validated = [
            str(r)[:_MAX_RULE_LENGTH]
            for r in raw_rules[:_MAX_CUSTOM_RULES]
            if isinstance(r, (str, int, float))
        ]
        if validated:
            rules_list = "\n".join(f"- {r}" for r in validated)
            custom_rules_section = f"\n## Repository-Specific Rules\n{rules_list}\n"

    lang_rules_section = ""
    lang_rules = get_language_rules(context.repo_languages)
    if lang_rules:
        lang_rules_section = f"\n## Language-specific guidance\n{lang_rules}\n"

    return f"""You are performing a **code review** of a pull request.
{review_standard}
{what_to_look_for}
{design_principles}
{pr_type_instruction}{pr_size_instruction}
## Pull Request Information
- **Repository**: {context.repo_name}
- **PR #{context.pr_number}**: {context.pr_title}
- **Author**: {context.author}
- **Branch**: {context.head_branch} → {context.base_branch}
- **Changes**: +{context.additions} / -{context.deletions} in {context.changed_files_count} files
- **Languages**: {", ".join(context.repo_languages) if context.repo_languages else "Unknown"}

## PR Description
{context.pr_description or "No description provided."}
{conventions_section}{custom_rules_section}{lang_rules_section}
## Code Changes (Diff)
```diff
{diff[:50000]}
```
{files_context}
"""


def get_output_format(pr_type: str = "code", total_lines: int = 0) -> str:
    """Get the JSON output format instructions."""
    max_findings = max(3, min(10, total_lines // 100 + 3))
    concise_rules = [
        "- Be concise: one short sentence per finding description. Do not repeat the same point.",
        "- Only report issues on changed lines; do not suggest pre-existing improvements.",
        '- **Severity semantics:** critical = must fix (security bugs or data corruption risks only); warning = should fix (other serious correctness or maintainability issues); suggestion = consider; nitpick = optional polish—always prefix title with "Nit: " for nitpicks.',
        "- If the code looks good for your focus area, return empty findings array.",
        f"- Maximum {max_findings} findings per agent.",
        "- If something is done well (e.g. clear naming, good tests), mention it briefly in the summary.",
    ]
    if pr_type == "docs":
        concise_rules.append(
            '- Do not report style, nitpicks, or "add tests". Only factual errors or security.'
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

## Example of a GOOD finding (specific, actionable):
{{"file_path": "auth.py", "line_start": 45, "severity": "critical",
 "category": "security", "title": "SQL injection via string interpolation",
 "description": "User input interpolated directly into SQL query without parameterization.",
 "suggested_fix": "Use parameterized query: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
 "confidence": 0.95}}

## Example of a BAD finding (vague -- DO NOT produce these):
{{"file_path": "utils.py", "line_start": 1, "severity": "suggestion",
 "category": "testing", "title": "Consider adding more tests",
 "description": "The code could benefit from additional test coverage.",
 "confidence": 0.5}}

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
    if len(review.findings) > _CROSS_REVIEW_MAX_FINDINGS:
        logger.info(
            "Cross-review limited to first %s of %s findings",
            _CROSS_REVIEW_MAX_FINDINGS,
            len(review.findings),
        )
    findings_blob = []
    for i, f in enumerate(review.findings[:_CROSS_REVIEW_MAX_FINDINGS], 1):
        line_ref = f"{f.line_start}" + (f"-{f.line_end}" if f.line_end else "")
        findings_blob.append(
            f"{i}. [id={f.id}] {f.file_path}:{line_ref} [{f.severity.value}] {f.title}\n   {f.description}"
        )
    findings_text = "\n".join(findings_blob)

    # Truncate at newline boundary only when we actually truncated (avoid dropping last line when diff fits)
    diff_excerpt = diff[:_CROSS_REVIEW_DIFF_MAX_CHARS]
    if len(diff) > _CROSS_REVIEW_DIFF_MAX_CHARS and "\n" in diff_excerpt:
        diff_excerpt = diff_excerpt.rsplit("\n", 1)[0]

    return f"""You are in a **cross-review round**. Multiple agents already produced the findings below for this PR. Your job is to validate them and rank by importance.

## PR
- **Repo**: {context.repo_name} | **PR #{context.pr_number}**: {context.pr_title}
- **Changes**: +{context.additions}/-{context.deletions} in {context.changed_files_count} files

## Code diff (excerpt)
```diff
{diff_excerpt}
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
    return """
## Output format (valid JSON only, no markdown fences)
{"assessments": [{"id": "finding-1", "valid": true, "rank": 1}, {"id": "finding-2", "valid": false, "rank": 5}, ...], "summary": "One sentence on overall quality of the findings."}

- "id" must match the finding id from the list (e.g. finding-1, finding-2).
- "valid": true if the finding should stay in the report, false if it should be dropped or is not actionable.
- "rank": integer, 1 = most important. Lower rank = higher priority.
- Include every finding id from the list in assessments.
"""


def _extract_json_block(content: str, json_key: str) -> str:
    """Extract JSON block from LLM response: strip, unwrap markdown fences, find object containing json_key (non-greedy)."""
    content = content.strip()
    if "```json" in content:
        match = re.search(r"```json\s*([\s\S]*?)```", content)
        if match:
            content = match.group(1).strip()
    elif "```" in content:
        match = re.search(r"```\s*([\s\S]*?)```", content)
        if match:
            content = match.group(1).strip()
    # Greedy match: one JSON object containing json_key (trailing content may be included; json.loads will fail and callers return []).
    json_match = re.search(r"\{[\s\S]*" + re.escape(json_key) + r"[\s\S]*\}", content)
    return json_match.group(0) if json_match else content


def parse_cross_review_response(content: str) -> tuple[list[dict[str, Any]], str]:
    """Parse cross-review JSON into list of {id, valid, rank} and summary."""
    content = _extract_json_block(content, "assessments")
    try:
        data = json.loads(content)
        return data.get("assessments", []), data.get("summary", "")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse cross-review JSON: {e}")
        return [], ""


def apply_cross_review(
    review: ConsolidatedReview,
    all_assessments: list[tuple[str, list[dict[str, Any]]]],
    min_validation_agreement: float = 2 / 3,
) -> ConsolidatedReview:
    """Filter and re-rank findings using cross-review assessments.

    - Drops findings where the fraction of agents that said valid is < min_validation_agreement.
    - Re-orders by average rank (1 = first), then by severity.
    - Findings with no votes (e.g. omitted by all agents) are kept but assigned rank 99
      and appear at the end; assessments may use "id" or "finding_id" for the finding key.
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
            raw_valid = a.get("valid", True)
            if isinstance(raw_valid, bool):
                valid = raw_valid
            elif isinstance(raw_valid, str):
                valid = raw_valid.lower() in ("true", "1", "yes")
            else:
                valid = bool(raw_valid)
            rank = a.get("rank", 99)
            rank = max(1, int(rank)) if isinstance(rank, (int, float)) else 99
            id_to_votes[fid].append((valid, rank))

    kept: list[tuple[ConsolidatedFinding, float, float]] = []  # (finding, valid_ratio, avg_rank)

    for fid in finding_ids:
        finding = id_to_finding[fid]
        if finding.severity == Severity.CRITICAL and finding.category == Category.SECURITY:
            kept.append((finding, 1.0, 0))
            continue
        votes = id_to_votes.get(fid, [])
        if not votes:
            kept.append((finding, 1.0, 99.0))
            continue
        valid_count = sum(1 for v, _ in votes if v)
        # Use len(votes) not n_agents: only agents that assessed this finding count (omit = no vote)
        valid_ratio = valid_count / len(votes) if votes else 1.0
        if valid_ratio < min_validation_agreement:
            continue  # Drop finding
        avg_rank = sum(r for _, r in votes) / len(votes) if votes else 99.0
        kept.append((finding, valid_ratio, avg_rank))

    # Sort by avg_rank ascending, then by severity (critical first)
    severity_order = {
        Severity.CRITICAL: 0,
        Severity.WARNING: 1,
        Severity.SUGGESTION: 2,
        Severity.NITPICK: 3,
    }
    kept.sort(key=lambda x: (x[2], severity_order.get(x[0].severity, 4)))

    new_findings = [x[0] for x in kept]
    n_dropped = len(review.findings) - len(new_findings)
    new_ids = [f.id for f in new_findings]
    # Compare only relative order of retained findings (avoid false positive when findings dropped)
    remaining_original_order = [f.id for f in review.findings if f.id in set(new_ids)]
    order_changed = remaining_original_order != new_ids

    quality_score, score_breakdown = compute_quality_score(
        new_findings, review.agent_count, total_lines=0
    )

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
        review_quality_score=quality_score,
        total_review_time_ms=review.total_review_time_ms,
        failed_agents=review.failed_agents,
        score_breakdown=score_breakdown,
    )


def parse_review_response(content: str) -> tuple[list[dict], str]:
    """Parse the agent's response into findings and summary."""
    content = _extract_json_block(content, "findings")
    try:
        data = json.loads(content)
        return data.get("findings", []), data.get("summary", "Review completed")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse review JSON: {e}")
        return [], "Failed to parse review response"


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


CONFIDENCE_THRESHOLDS: dict[Severity, float] = {
    Severity.CRITICAL: 0.5,
    Severity.WARNING: 0.6,
    Severity.SUGGESTION: 0.7,
    Severity.NITPICK: 0.8,
}


_CROSS_FILE_ALSO_FOUND_CAP = 5


def dedup_cross_file(
    findings: list[ConsolidatedFinding],
) -> list[ConsolidatedFinding]:
    """Collapse repeated cross-file findings that share the same (category, title).

    Groups of 1–2 are left unchanged.  Groups of 3+ are collapsed to a single
    representative (the one with the highest ``priority_score``).  The
    representative's description gets an appended "Also found in: …" note
    listing the other file paths (capped to ``_CROSS_FILE_ALSO_FOUND_CAP``).
    """
    groups: dict[tuple[str, str], list[ConsolidatedFinding]] = defaultdict(list)
    for f in findings:
        key = (f.category.value.lower().strip(), f.title.lower().strip())
        groups[key].append(f)

    result: list[ConsolidatedFinding] = []
    for group in groups.values():
        if len(group) < 3:
            result.extend(group)
            continue

        group.sort(key=lambda f: f.priority_score, reverse=True)
        representative = deepcopy(group[0])

        other_paths = [f.file_path for f in group[1:]]
        if len(other_paths) > _CROSS_FILE_ALSO_FOUND_CAP:
            shown = other_paths[:_CROSS_FILE_ALSO_FOUND_CAP]
            note = ", ".join(shown) + f", and {len(other_paths) - _CROSS_FILE_ALSO_FOUND_CAP} more"
        else:
            note = ", ".join(other_paths)
        representative.description = representative.description.rstrip()
        representative.description += f"\n\nAlso found in: {note}"

        result.append(representative)

    return result


def compute_quality_score(
    findings: list[ConsolidatedFinding],
    agent_count: int,
    total_lines: int,
) -> tuple[float, ScoreBreakdown]:
    """Composite quality score factoring severity, density, consensus, and agents.

    Returns a score between 0.0 and 0.95 and a component breakdown.

    When ``total_lines`` is 0, density normalization is skipped (density penalty is 0).
    """
    if not findings:
        raw_score = 0.85
        agent_bonus = max(0.0, min(0.10, (agent_count - 1) * 0.05))
        agent_factor = (raw_score + agent_bonus) / raw_score
        combined = round(min(0.95, raw_score * agent_factor), 2)
        breakdown = ScoreBreakdown(
            severity_penalty=0.0,
            density_penalty=0.0,
            consensus_factor=1.0,
            agent_factor=round(agent_factor, 4),
            raw_score=raw_score,
        )
        return combined, breakdown

    severity_weights = {
        Severity.CRITICAL: 0.20,
        Severity.WARNING: 0.06,
        Severity.SUGGESTION: 0.02,
        Severity.NITPICK: 0.005,
    }
    severity_penalty = sum(severity_weights.get(f.severity, 0.02) * f.confidence for f in findings)

    if total_lines > 0:
        density = len(findings) / max(total_lines / 100, 1)
        density_penalty = min(0.15, density * 0.03)
    else:
        density_penalty = 0.0

    avg_consensus = sum(f.consensus_score for f in findings) / len(findings)
    consensus_factor = 0.8 + (avg_consensus * 0.2)
    agent_factor = min(1.0, agent_count / 3)

    raw_score = max(0.0, 1.0 - severity_penalty - density_penalty)
    combined = raw_score * consensus_factor * agent_factor
    breakdown = ScoreBreakdown(
        severity_penalty=severity_penalty,
        density_penalty=density_penalty,
        consensus_factor=consensus_factor,
        agent_factor=agent_factor,
        raw_score=raw_score,
    )
    return round(min(0.95, combined), 2), breakdown


def aggregate_findings(
    all_findings: list[tuple[str, list[dict], str]],
    repo: str,
    pr_number: int,
    confidence_thresholds: dict[Severity, float] | None = None,
    total_lines: int = 0,
) -> ConsolidatedReview:
    """Aggregate findings from multiple agents."""

    consolidated: list[ConsolidatedFinding] = []
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

    # Filter out low-confidence findings per severity
    thresholds = (
        confidence_thresholds if confidence_thresholds is not None else CONFIDENCE_THRESHOLDS
    )
    pre_filter_count = len(consolidated)
    consolidated = [f for f in consolidated if f.confidence >= thresholds.get(f.severity, 0.0)]
    filtered_count = pre_filter_count - len(consolidated)
    if filtered_count > 0:
        logger.info("Confidence filter dropped %d finding(s)", filtered_count)

    pre_dedup_count = len(consolidated)
    consolidated = dedup_cross_file(consolidated)
    dedup_count = pre_dedup_count - len(consolidated)
    if dedup_count > 0:
        logger.info("Cross-file dedup collapsed %d finding(s)", dedup_count)

    # Build combined summary
    combined_summary = "\n".join(summaries) if summaries else "Review completed"

    total_agents = len(all_findings)
    quality_score, score_breakdown = compute_quality_score(consolidated, total_agents, total_lines)

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
        score_breakdown=score_breakdown,
    )


_AGENT_CLASSES: dict[str, type[ReviewAgent]] = {
    "security-reviewer": SecurityAgent,
    "authentication-reviewer": AuthenticationAgent,
    "performance-reviewer": PerformanceAgent,
    "patterns-reviewer": PatternsAgent,
    "logic-reviewer": LogicAgent,
    "style-reviewer": StyleAgent,
}

DEFAULT_AGENT_ORDER = [
    "security-reviewer",
    "performance-reviewer",
    "patterns-reviewer",
    "logic-reviewer",
    "style-reviewer",
]

CONVENTION_PATHS = [
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    ".ai/rules/architecture.md",
    ".ai/rules/conventions.md",
    ".ai/rules/agents.md",
    ".cursor/rules/README.md",
]


def _review_finding_to_dict(f: ReviewFinding) -> dict[str, Any]:
    """Convert a parsed ReviewFinding back to the dict form aggregate_findings expects."""
    return {
        "file_path": f.file_path,
        "line_start": f.line_start,
        "line_end": f.line_end,
        "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
        "category": f.category.value if hasattr(f.category, "value") else str(f.category),
        "title": f.title,
        "description": f.description,
        "suggested_fix": f.suggested_fix,
        "confidence": f.confidence,
    }


async def _prepare_shared_context(
    session: ReviewSession,
    gh: GitHubClient,
    pr: Any,
    diff: str,
    changed_file_contents: dict[str, str],
    anthropic_cfg: AnthropicApiConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch conventions + repo map + neighbors and build system/user blocks."""
    import base64 as _b64

    conventions = fetch_conventions(session, gh, CONVENTION_PATHS)
    repo_map = build_repo_map(session, gh)

    tree = session.cached_tree() or []

    def _read(path: str) -> str:
        cached = session.cached_file(path)
        return cached if cached is not None else ""

    neighbor_paths = select_neighbors(
        changed_files=changed_file_contents,
        repo_paths=tree,
        read_file=_read,
        max_siblings=5,
        max_total=15,
    )

    neighbors: dict[str, str] = {}
    repo_obj = gh.client.get_repo(session.repo)
    for path in neighbor_paths:
        if session.is_github_budget_exhausted():
            break
        cached = session.cached_file(path)
        if cached is not None:
            neighbors[path] = cached
            continue
        try:
            session.consume_github_request()
            contents = repo_obj.get_contents(path, ref=session.head_sha)
            text = _b64.b64decode(getattr(contents, "content", "")).decode(
                "utf-8", errors="replace"
            )
            if len(text) > anthropic_cfg.per_file_max_bytes:
                text = text[: anthropic_cfg.per_file_max_bytes] + "\n[truncated]"
            session.store_file(path, text)
            neighbors[path] = text
        except Exception as e:  # noqa: BLE001
            logger.debug("Neighbor fetch failed %s: %s", path, e)

    system_blocks = build_system_blocks(
        agent_role=(
            "You are a specialized code reviewer. Each agent has its own "
            "focus area in the next system block."
        ),
        convention_texts=conventions,
        repo_map=repo_map,
    )
    user_blocks = build_user_blocks(
        pr_title=getattr(pr, "title", "") or "",
        pr_body=getattr(pr, "body", "") or "",
        diff=diff,
        changed_files=changed_file_contents,
        neighbor_files=neighbors,
        max_total_chars=anthropic_cfg.max_combined_context_tokens * 4,
    )
    return system_blocks, user_blocks


async def _run_agent_safe(
    agent: ReviewAgent,
    context: ReviewContext,
    on_status: Callable[..., Any] | None,
) -> AgentReview | Exception:
    """Run one agent; return its AgentReview or the exception for downstream handling."""
    name = agent.agent_id
    if on_status:
        on_status(f"{name}: RUNNING")
    try:
        review = await agent.review(diff="", file_contents={}, context=context)
        if on_status:
            on_status(f"{name}: DONE")
        return review
    except Exception as e:  # noqa: BLE001
        logger.exception("Agent %s failed", name)
        if on_status:
            on_status(f"{name}: FAILED")
        return e


async def run_cross_review_round(
    client: AnthropicClient,
    session: ReviewSession,
    gh: GitHubClient,
    pr: Any,
    review: ConsolidatedReview,
    context: ReviewContext,
    diff: str,
    agents_to_run: list[dict],
    anthropic_cfg: AnthropicApiConfig,
    on_status: Callable[..., Any] | None = None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Cross-review round: each agent re-evaluates the consolidated findings."""
    if not review.findings:
        return []

    prior_findings = [
        {
            "file_path": f.file_path,
            "line_start": f.line_start,
            "line_end": f.line_end,
            "severity": f.severity.value,
            "category": f.category.value,
            "title": f.title,
            "description": f.description,
        }
        for f in review.findings
    ]

    system_blocks, base_user_blocks = await _prepare_shared_context(
        session=session,
        gh=gh,
        pr=pr,
        diff=diff,
        changed_file_contents={},
        anthropic_cfg=anthropic_cfg,
    )
    cross_instructions = {
        "type": "text",
        "text": (
            "## Cross-review task\n\n"
            "The findings below were produced by earlier agents. Re-evaluate each: "
            "is it a true positive? What is your confidence? Return findings list in "
            "the same schema. For each finding you agree with, copy its file_path / "
            "line_start / title / category and set your own confidence. Drop findings "
            "you consider false positives.\n\n"
            f"```json\n{json.dumps(prior_findings, indent=2)}\n```"
        ),
    }
    cross_user_blocks = [*base_user_blocks, cross_instructions]

    tasks: list[Any] = []
    for i, cfg in enumerate(agents_to_run):
        name = cfg.get("name") if isinstance(cfg, dict) else cfg
        cls = _AGENT_CLASSES.get(name)
        if not cls:
            continue
        agent = cls(
            client=client,
            agent_id=f"{name}-cross-{i}",
            system_blocks=system_blocks,
            user_blocks=cross_user_blocks,
            tool_registry=None,
            max_tokens=8192,
            temperature=0.2,
        )
        tasks.append(_run_agent_safe(agent, context, on_status))

    results: list[tuple[str, list[dict[str, Any]]]] = []
    for cfg, res in zip(agents_to_run, await asyncio.gather(*tasks), strict=False):
        name = cfg.get("name") if isinstance(cfg, dict) else cfg
        if isinstance(res, Exception):
            continue
        as_dicts = [_review_finding_to_dict(f) for f in res.findings]
        if as_dicts:
            results.append((name, as_dicts))
    return results


def _effective_agent_count(
    additions: int, deletions: int, changed_files: int, requested: int
) -> int:
    """Scale agent count with PR size to save cost and latency on small PRs."""
    total = additions + deletions
    if total < 150 and changed_files <= 3:
        return min(1, requested)
    elif total < 500:
        return min(2, requested)
    return requested


async def review_pr_with_cursor_agent(
    repo: str,
    pr_number: int,
    anthropic_cfg: AnthropicApiConfig,
    github_token: str,
    on_status: Callable[..., Any] | None = None,
    num_agents: int = 3,
    enable_cross_review: bool = True,
    min_validation_agreement: float = 2 / 3,
    config: Any | None = None,
) -> ConsolidatedReview:
    """Review a PR using Anthropic Messages API agents.

    Function name is retained for backward compatibility with existing
    callers and test patches; the implementation now uses the official
    Anthropic SDK (see docs/superpowers/specs/2026-04-15-anthropic-
    messages-migration-design.md).

    Args:
        repo: Repository in "owner/name" format
        pr_number: Pull request number
        anthropic_cfg: Anthropic API configuration
        github_token: GitHub token for PR access
        on_status: Optional callback for status updates
        num_agents: Number of agents to run (1-5)
        enable_cross_review: If True and num_agents > 1, run a second round where
            agents validate and rank findings; drop low-agreement and re-order by rank.
        min_validation_agreement: Fraction of assessing agents that must mark a finding valid.
        config: Optional Config object; used for aggregator confidence thresholds and
            review_policy.secret_scan_exclude.

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

    context.repo_config = gh.load_repo_config(repo, ref=pr.head.sha)
    context.conventions = gh.load_repo_conventions(repo, ref=pr.head.sha)

    secret_scan_exclude = config.review_policy.secret_scan_exclude if config else []
    secret_findings = scan_for_secrets(diff, exclude_patterns=secret_scan_exclude)
    if secret_findings:
        logger.warning(
            "Secret scanner detected %d potential secret(s) — these bypass aggregation/cross-review",
            len(secret_findings),
        )

    raw_ignore = (context.repo_config or {}).get("ignore", [])
    ignore_patterns = (
        raw_ignore
        if isinstance(raw_ignore, list)
        else [raw_ignore]
        if isinstance(raw_ignore, str)
        else []
    )
    if ignore_patterns:
        pre_file_count = len(files)
        files = filter_by_ignore_patterns(files, ignore_patterns)
        diff = filter_diff_by_ignore_patterns(diff, ignore_patterns)
        logger.info(
            "Ignore patterns filtered %d file(s) from prompt inputs",
            pre_file_count - len(files),
        )

    logger.info(f"Reviewing PR #{pr_number}: {context.pr_title}")
    logger.info(
        f"Files changed: {context.changed_files_count} (+{context.additions}/-{context.deletions})"
    )

    effective = _effective_agent_count(
        context.additions, context.deletions, context.changed_files_count, num_agents
    )
    if effective != num_agents:
        logger.info(f"Effective agent count: {effective} (requested {num_agents})")
    num_agents = effective
    if num_agents <= 2:
        enable_cross_review = False

    changed_paths = list(files.keys())
    pr_type, _pr_size = classify_pr(changed_paths, context.additions, context.deletions)
    if pr_type != "code":
        logger.info(f"PR type: {pr_type} – using context-aware review rules")

    # Select agents to run (resolve from config.agents, fall back to defaults)
    configured_names = [a.name for a in (config.agents if config and config.agents else [])]
    effective_order = configured_names or DEFAULT_AGENT_ORDER
    agent_order = effective_order[: min(num_agents, len(effective_order))]
    agents_to_run = [{"name": n} for n in agent_order]

    session = ReviewSession(
        repo=repo,
        head_sha=pr.head.sha,
        github_budget=anthropic_cfg.per_review_github_request_budget,
    )

    async with AnthropicClient(anthropic_cfg) as client:
        system_blocks, user_blocks = await _prepare_shared_context(
            session=session,
            gh=gh,
            pr=pr,
            diff=diff,
            changed_file_contents=files,
            anthropic_cfg=anthropic_cfg,
        )

        if on_status:
            on_status("CREATING")

        tasks: list[Any] = []
        instantiated: list[tuple[str, ReviewAgent]] = []
        for i, agent_name in enumerate(agent_order):
            cls = _AGENT_CLASSES.get(agent_name)
            if not cls:
                logger.warning("Unknown agent %s; skipping", agent_name)
                continue
            agent_cfg = next(
                (a for a in (config.agents if config else []) if a.name == agent_name), None
            )
            allow_tools = agent_cfg.allow_tool_use if agent_cfg else True
            max_tool_calls = agent_cfg.max_tool_calls if agent_cfg else 20
            registry = (
                ToolRegistry(
                    session=session,
                    github_client=gh,
                    agent_id=f"{agent_name}-{i}",
                    max_calls=max_tool_calls,
                    per_file_max_bytes=anthropic_cfg.per_file_max_bytes,
                )
                if allow_tools
                else None
            )
            agent = cls(
                client=client,
                agent_id=f"{agent_name}-{i}",
                system_blocks=system_blocks,
                user_blocks=user_blocks,
                tool_registry=registry,
                max_tokens=agent_cfg.max_tokens if agent_cfg else 8192,
                temperature=agent_cfg.temperature if agent_cfg else 0.3,
            )
            instantiated.append((agent_name, agent))
            tasks.append(_run_agent_safe(agent, context, on_status))

        agent_results = await asyncio.gather(*tasks)

    all_findings: list[tuple[str, list[dict[str, Any]], str]] = []
    for (agent_name, _agent), result in zip(instantiated, agent_results, strict=False):
        if isinstance(result, Exception):
            all_findings.append((agent_name, [], f"Agent failed: {result}"))
            continue
        dicts = [_review_finding_to_dict(f) for f in result.findings]
        all_findings.append((agent_name, dicts, result.summary))

    # Aggregate findings
    confidence_thresholds = None
    if config:
        confidence_thresholds = {
            Severity.CRITICAL: config.aggregator.min_confidence_critical,
            Severity.WARNING: config.aggregator.min_confidence_warning,
            Severity.SUGGESTION: config.aggregator.min_confidence_suggestion,
            Severity.NITPICK: config.aggregator.min_confidence_nitpick,
        }
    total_lines = context.additions + context.deletions
    review = aggregate_findings(
        list(all_findings),
        repo,
        pr_number,
        confidence_thresholds=confidence_thresholds,
        total_lines=total_lines,
    )

    # Optional: cross-review round (agents validate and rank findings).
    # Note: cross-review doubles API calls; disable with --no-cross-review for cost-sensitive use.
    if enable_cross_review and num_agents > 1 and review.findings and not review.all_agents_failed:
        # Only run cross-review with agents that succeeded in round 1
        agents_for_cross = [c for c in agents_to_run if c["name"] not in review.failed_agents]
        if not agents_for_cross:
            logger.info("Skipping cross-review: no round-1 agents succeeded")
        else:
            logger.info("Running cross-review round (validate and rank findings)...")
            async with AnthropicClient(anthropic_cfg) as cross_client:
                cross_results = await run_cross_review_round(
                    client=cross_client,
                    session=session,
                    gh=gh,
                    pr=pr,
                    review=review,
                    context=context,
                    diff=diff,
                    agents_to_run=agents_for_cross,
                    anthropic_cfg=anthropic_cfg,
                    on_status=on_status,
                )
            if cross_results:
                review = apply_cross_review(review, cross_results, min_validation_agreement)
                logger.info(f"Cross-review done: {len(review.findings)} findings after validation")

    if secret_findings:
        review.findings = secret_findings + review.findings

    review.total_review_time_ms = int((time.time() - start_time) * 1000)

    logger.info(
        f"Review complete: {len(review.findings)} findings from {review.agent_count} agent(s) in {review.total_review_time_ms}ms"
    )

    return review
