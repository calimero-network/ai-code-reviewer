# AI Code Reviewer: Technical Improvement Document

## Context

**This document analyzes the existing AI code reviewer codebase and proposes concrete, implementable improvements across 10 areas. The reviewer currently uses a multi-agent architecture (security, performance, quality) orchestrated via Cursor Background Agent API, with consensus-based aggregation, cross-review validation, and delta tracking for incremental reviews. While functional, the system has significant gaps in prompt quality, scoring reliability, large-PR handling, and convergence behavior. Each section below identifies the current state, proposes specific changes with pseudocode, names the exact files/functions to modify, and flags tradeoffs.**

---

## 1. Review Prompt & Context Quality

### Current State (post Phase 1)

- Prompt built in `get_base_prompt()` (`review.py`): static review standard, design principles, PR metadata, truncated diff (50k chars), file contents
- ~~No few-shot examples~~ -> Good/bad finding examples added to `get_output_format()` (`review.py:220-231`)
- `ReviewContext.custom_instructions` field (`context.py:22`) exists but is never populated
- `.ai-reviewer.yaml` exists in repo root with `custom_rules`, `agents[].custom_prompt_append`, `ignore` patterns -- **none of it is loaded or used**
- No repo-specific context (architecture docs, conventions, CLAUDE.md)
- Same prompt regardless of PR size (50-line fix vs 5k-line refactor)

### Proposed Changes

**1a. Load `.ai-reviewer.yaml` from the target repository**

Add `load_repo_config()` to `github/client.py`:

```python
def load_repo_config(self, repo_name: str, ref: str) -> dict | None:
    try:
        content = self.get_repo(repo_name).get_contents(".ai-reviewer.yaml", ref=ref)
        return yaml.safe_load(content.decoded_content.decode("utf-8"))
    except Exception:
        return None
```

Integration point: In `review_pr_with_cursor_agent()` (`review.py:695`), after `context = gh.build_review_context(pr, repo_obj)`, call `repo_config = gh.load_repo_config(repo, pr.base.ref)`. Extract `custom_rules[]` and inject as a "Repository-Specific Rules" prompt section. Extract `ignore[]` and filter those paths from the diff before sending to agents.

**1b. Load architectural context files (CLAUDE.md, CONTRIBUTING.md)**

Best-effort fetch of convention files, capped at 3k chars total:

```python
CONTEXT_FILES = ["CLAUDE.md", ".cursor/rules/README.md", "CONTRIBUTING.md"]

def load_repo_conventions(self, repo_name, ref):
    context = []
    for path in CONTEXT_FILES:
        try:
            content = self.get_repo(repo_name).get_contents(path, ref=ref)
            context.append(f"### {path}\n{content.decoded_content.decode()[:1500]}")
        except Exception:
            continue
    return "\n".join(context)[:3000] if context else None
```

Inject as `## Repository Conventions` section before the diff in the base prompt.

**1c. PR size classification + adaptive prompt instructions**

Extend `_detect_pr_type()` (`review.py:80-93`) with size awareness:

```python
def _classify_pr(changed_paths, additions, deletions):
    pr_type = _detect_pr_type(changed_paths)
    total_lines = additions + deletions
    if total_lines < 50:       size = "trivial"
    elif total_lines < 200:    size = "small"
    elif total_lines < 1000:   size = "medium"
    else:                      size = "large"
    return pr_type, size
```

Inject size-specific instructions:

- **trivial/small**: "This is a small change. Be extra precise -- only flag genuine issues. Do not pad with low-value suggestions."
- **large**: "This is a large change. Focus on architectural concerns and high-severity issues first. Ignore minor style."

**1d. Few-shot examples to reduce generic feedback**

Add 2 exemplar findings to `get_output_format()` (`review.py:150-194`):

```
## Example of a GOOD finding (specific, actionable):
{"file_path": "auth.py", "line_start": 45, "severity": "critical",
 "title": "SQL injection via string interpolation",
 "description": "User input interpolated directly into SQL without parameterization."}

## Example of a BAD finding (vague -- DO NOT produce these):
{"title": "Consider adding more tests",
 "description": "The code could benefit from additional test coverage."}
```

**1e. Language-specific rules**

`context.repo_languages` is already populated (`review.py:137`). Maintain a small dict of 3-5 rules per common language. For Python: "Watch for: mutable default arguments, bare except clauses, f-string injection in logging." For TypeScript: "Watch for: `any` type escapes, missing error boundaries, unhandled Promise rejections."

### Files to Modify

- `src/ai_reviewer/review.py` -- `get_base_prompt()`, `_detect_pr_type()` -> `_classify_pr()`, `get_output_format()`
- `src/ai_reviewer/github/client.py` -- new `load_repo_config()`, `load_repo_conventions()`
- `src/ai_reviewer/models/context.py` -- add `repo_config: dict | None` and `conventions: str | None` fields

---

## 2. Review Accuracy & Quality

### Current State (post Phase 1)

- ~~Max 5 findings per agent hard-coded~~ -> Now dynamic: `max(3, min(10, total_lines // 100 + 3))` (`review.py:176-184`)
- ~~No confidence filtering~~ -> Per-severity thresholds in `AggregatorSettings` (`config.py:62-65`), applied in `aggregate_findings()` (`review.py:551-633`)
- ~~No adaptive agent count~~ -> `_effective_agent_count()` scales 1-3 agents by PR size (`review.py:723-732`), cross-review auto-skipped for <=2 agents
- Similarity threshold 0.85 still hard-coded (`review.py`), uses character-level `SequenceMatcher`
- `sentence-transformers` is in `pyproject.toml` but unused
- `AggregatorSettings.use_embeddings` exists in `config.py` but is never checked

### Completed (Phase 1)

- **2a. Confidence-based filtering** -- Done. Thresholds: critical=0.5, warning=0.6, suggestion=0.7, nitpick=0.8. Config-driven via `AggregatorSettings`.
- **2b. Adaptive agent count** -- Done. `_effective_agent_count()` at `review.py:723-732`. Cross-review skipped for <=2 agents.
- **2c. Adaptive max findings** -- Done. Dynamic formula at `review.py:176-184`.

### Remaining (Phase 2+)

**2d. Embedding-based semantic similarity (opt-in)**

When `config.aggregator.use_embeddings == True`, replace `_raw_text_similarity()` (`review.py:445`):

```python
from sentence_transformers import SentenceTransformer

_model = None
def _embedding_similarity(text1, text2):
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = _model.encode([text1, text2])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
```

Fall back to SequenceMatcher when library unavailable.

### Files to Modify

- `src/ai_reviewer/review.py` -- `aggregate_findings()`, `get_output_format()`, `review_pr_with_cursor_agent()`
- `src/ai_reviewer/config.py` -- add confidence thresholds to `AggregatorSettings`
- `src/ai_reviewer/orchestrator/aggregator.py` -- add embedding path to `_are_similar()`

---

## 3. Reliable Code Quality Scoring

### Current State

- Computed in `aggregate_findings()` (`review.py:574-582`) and `ReviewAggregator._compute_quality_score()` (`aggregator.py:264-281`) -- **duplicated logic**
- Formula: `avg_consensus * agent_factor` where `agent_factor = min(1.0, agent_count/3)`
- Clean review: `min(0.95, 0.7 + agent_count * 0.1)` regardless of PR complexity
- No severity weighting: 1 nitpick and 1 critical score the same
- Fluctuates based on agent count and whether findings happen to cluster

### Proposed Changes

**3a. Composite scoring formula**

```python
def compute_quality_score(findings, agent_count, total_lines):
    if not findings:
        base = 0.85
        agent_bonus = min(0.10, (agent_count - 1) * 0.05)
        return min(0.95, base + agent_bonus)

    severity_penalty = {
        Severity.CRITICAL: 0.15, Severity.WARNING: 0.06,
        Severity.SUGGESTION: 0.02, Severity.NITPICK: 0.005,
    }
    total_penalty = sum(severity_penalty[f.severity] * f.confidence for f in findings)

    # Density: many findings per 100 LOC is worse
    density = len(findings) / max(total_lines / 100, 1)
    density_penalty = min(0.15, density * 0.03)

    avg_consensus = sum(f.consensus_score for f in findings) / len(findings)
    consensus_factor = 0.8 + (avg_consensus * 0.2)
    agent_factor = min(1.0, agent_count / 3)

    raw_score = max(0.0, 1.0 - total_penalty - density_penalty)
    return round(raw_score * consensus_factor * agent_factor, 2)
```

**3b. Stability via damping on re-reviews**

Extract previous score from the last posted review comment header ("Quality score: XX%", formatted by `GitHubFormatter._format_header()` at `formatter.py:150-158`):

```python
DAMPING_FACTOR = 0.3  # 30% old, 70% new
if previous_score is not None:
    if previous_score - new_score > 0.2:
        score = new_score  # Skip damping on real regression
    else:
        score = DAMPING_FACTOR * previous_score + (1 - DAMPING_FACTOR) * new_score
```

**3c. Transparent score breakdown**

Add to `ConsolidatedReview` in `models/review.py`:

```python
@dataclass
class ScoreBreakdown:
    severity_penalty: float
    density_penalty: float
    consensus_factor: float
    agent_factor: float
    raw_score: float
    damped_score: float | None
```

Display in the review footer as a collapsed `<details>` section via `formatter.py`.

**3d. Consolidate dual computation**

Remove the duplicate in `aggregator.py:264-281`. Have `aggregate_findings()` in `review.py` be the single source of truth.

### Files to Modify

- `src/ai_reviewer/review.py` -- `aggregate_findings()` scoring
- `src/ai_reviewer/orchestrator/aggregator.py` -- remove `_compute_quality_score()`, delegate to review.py
- `src/ai_reviewer/models/review.py` -- add `ScoreBreakdown`
- `src/ai_reviewer/github/formatter.py` -- display breakdown
- `src/ai_reviewer/github/client.py` -- parse previous score from comment header

---

## 4. Incremental Review Intelligence

### Current State

- Delta tracking in `compute_review_delta()` (`client.py:441-534`) compares current findings vs previous inline comments
- Hash matching: `finding_hash` = SHA256 of `file_path:line_start:normalized_title` (findings.py:82-94)
- Fixed detection: file removed OR deleted OR line modified +/-3 lines (`client.py:586-602`)
- **Always reviews entire PR diff**, even on incremental pushes
- No tracking of which findings relate to which commits
- Force-pushes/rebases invalidate line-based matching

### Proposed Changes

**4a. Commit-aware incremental diffing**

On `synchronize` events, GitHub provides `before`/`after` SHAs. Use them:

```python
def get_incremental_diff(self, pr, before_sha, after_sha):
    compare = pr.base.repo.compare(before_sha, after_sha)
    diff_parts = []
    for f in compare.files:
        if f.patch:
            diff_parts.append(f"diff --git a/{f.filename} b/{f.filename}\n{f.patch}")
    return "\n".join(diff_parts)
```

When `since_sha` is available, send both diffs to agents: "Focus your review on the NEW changes (incremental diff). The full PR diff is provided for context only."

**4b. Embed commit SHA in inline comments**

```
<!-- ai-reviewer-id: {hash} commit: {head_sha} -->
```

On subsequent reviews, parse the commit SHA. If current HEAD is a descendant (no force-push), only review new commits. If force-pushed, do full review but use hash-based matching.

**4c. Force-push detection**

```python
def is_force_push(self, repo, before_sha, after_sha):
    try:
        return repo.compare(before_sha, after_sha).status == "diverged"
    except Exception:
        return True
```

On force-push: full review, rely exclusively on `finding_hash` (not line-based) since line numbers will have shifted.

**4d. Webhook integration**

Extend `PREvent` in `webhook.py` with `before_sha` and `after_sha` fields. Pass through to `review_pr_async()`.

### Files to Modify

- `src/ai_reviewer/github/client.py` -- `get_incremental_diff()`, enhanced `compute_review_delta()`
- `src/ai_reviewer/github/webhook.py` -- extend `PREvent` dataclass
- `src/ai_reviewer/review.py` -- `review_pr_with_cursor_agent()` accept `since_sha`
- `src/ai_reviewer/cli.py` -- accept `--since-sha` parameter

---

## 5. Deduplication & Comment Management

### Current State (post Phase 1)

- Single hash: `SHA256(file_path:line_start:normalized_title)[:12]` -- breaks on renames, line shifts, title variations
- ~~Inline comment cap hard-coded `[:10]`~~ -> Now config-driven via `apply_comment_limits()` (`client.py:46-69`), uses `config.output.max_total_findings` (default 50) and `config.output.max_findings_per_file` (default 10)
- No cross-file dedup: same pattern in 5 files = 5 separate comments

### Proposed Changes

**5a. Multi-tier hash for stability**

```python
@property
def finding_hash_primary(self) -> str:
    """Strict: file + line + title."""
    key = f"{self.file_path}:{self.line_start}:{self.title.lower().strip()}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]

@property
def finding_hash_fuzzy(self) -> str:
    """Fuzzy: file + category + title keywords (ignores line)."""
    words = sorted(set(re.findall(r'\b\w{4,}\b', self.title.lower())))
    key = f"{self.file_path}:{self.category.value}:{':'.join(words[:5])}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]
```

In `compute_review_delta()`: try primary hash -> fuzzy hash -> file+title tuple.

**5b. Cross-file deduplication**

```python
def dedup_cross_file(findings):
    groups = defaultdict(list)
    for f in findings:
        groups[(f.category, f.title.lower().strip())].append(f)
    result = []
    for group in groups.values():
        if len(group) <= 2:
            result.extend(group)
        else:
            group.sort(key=lambda f: f.priority_score, reverse=True)
            primary = group[0]
            others = [f.file_path for f in group[1:]]
            primary.description += f"\n\nAlso found in: {', '.join(others[:5])}"
            result.append(primary)
    return result
```

**5c. Enforce config caps**

Replace hard-coded `[:10]` in `post_inline_comments()` with config-driven limits:

```python
def apply_comment_limits(findings, max_total, max_per_file):
    findings.sort(key=lambda f: f.priority_score, reverse=True)
    per_file = defaultdict(int)
    result = []
    for f in findings:
        if len(result) >= max_total: break
        if per_file[f.file_path] >= max_per_file: continue
        result.append(f)
        per_file[f.file_path] += 1
    return result
```

### Files to Modify

- `src/ai_reviewer/models/findings.py` -- add `finding_hash_fuzzy`, keep `finding_hash` as primary
- `src/ai_reviewer/github/client.py` -- multi-tier matching in `compute_review_delta()`, config-driven limits
- `src/ai_reviewer/review.py` -- cross-file dedup in `aggregate_findings()`

---

## 6. Convergence & "Stop Reviewing" Logic

### Current State

- Every webhook event triggers a full review -- no concept of convergence
- Same number of agents/rounds regardless of review history
- `delta.all_issues_resolved` (`client.py:76-78`) exists but only used for final status message

### Proposed Changes

**6a. Convergence detection**

```python
def has_converged(delta):
    if delta.new_findings: return False
    if delta.fixed_findings: return False
    return True
```

**6b. Review count throttling**

```python
def should_skip_review(review_count, delta):
    if review_count <= 1: return False
    if has_converged(delta): return True
    if review_count >= 3:
        new_non_nit = [f for f in delta.new_findings if f.severity != Severity.NITPICK]
        if not new_non_nit: return True
    return False
```

**6c. LGTM fast path**

When `delta.all_issues_resolved == True` on a re-review, post "All previously identified issues addressed. Approving." with `APPROVE` action. Skip running agents unless `force-review` label is present.

**6d. Re-review debouncing**

Embed metadata: `<!-- ai-reviewer-meta: {"timestamp": "...", "review_count": 3} -->`. If last review < 2 min ago with no new commits, skip.

### Files to Modify

- `src/ai_reviewer/review.py` -- convergence check before running agents
- `src/ai_reviewer/github/client.py` -- parse review metadata, count previous reviews
- `src/ai_reviewer/cli.py` -- add `--force-review` flag

---

## 7. Large PR Handling (5k+ LOC)

### Current State

- Diff truncated at 50k chars (`review.py:144`)
- Only 5 files get contents, each 5k chars (`review.py:114-115`)
- No file prioritization -- first 5 files by iteration order
- No sampling or hotspot detection

### Proposed Changes

**7a. Hotspot-based file scoring**

```python
def score_file_hotspot(file):
    score = 0.0
    score += min(1.0, (file.additions + file.deletions) / 200) * 0.4
    if file.filename.endswith(('.py', '.ts', '.js', '.rs', '.go', '.java')):
        score += 0.3
    elif file.filename.endswith(('.yaml', '.yml', '.json', '.toml')):
        score += 0.1
    if file.status == "added":
        score += 0.2
    defs = len(re.findall(r'^\+.*\b(def |class |fn |func |function )', file.patch or '', re.M))
    score += min(0.1, defs * 0.02)
    return score
```

**7b. Budget-aware sampling**

```python
def select_review_scope(files, max_chars=50000):
    files_sorted = sorted(files, key=score_file_hotspot, reverse=True)
    selected, total_chars = [], 0
    for f in files_sorted:
        patch_len = len(f.patch or "")
        if total_chars + patch_len > max_chars:
            selected.append(f"# {f.filename}: +{f.additions} -{f.deletions} (truncated)")
            continue
        selected.append(f.patch)
        total_chars += patch_len
    return selected
```

**7c. Tiered review for >1000 LOC PRs**

1. **Architecture pass** (1 agent): File summary only. "Analyze overall design. Does it make architectural sense?"
2. **Hotspot detail pass** (2-3 agents): Full diff for top-priority files only.

### Files to Modify

- `src/ai_reviewer/review.py` -- `review_pr_with_cursor_agent()`, new `review_large_pr()`, `select_review_scope()`
- `src/ai_reviewer/github/client.py` -- `get_pr_diff()` to support selective file inclusion

---

## 8. Commit-to-Commit Consistency

### Current State

- Each review is independent -- no comparison across runs
- Quality score fluctuates: same issue might be "warning" one run, "suggestion" the next

### Proposed Changes

**8a. Structured metadata in inline comments**

```
<!-- ai-reviewer-meta: {"hash":"abc123","severity":"warning","confidence":0.85,
     "first_seen":"a1b2c3d","review_count":2} -->
```

**8b. Severity stabilization**

```python
def stabilize_severity(current, previous, review_count):
    ORDER = [Severity.CRITICAL, Severity.WARNING, Severity.SUGGESTION, Severity.NITPICK]
    cur_idx, prev_idx = ORDER.index(current), ORDER.index(previous)
    if review_count >= 2 and cur_idx > prev_idx:
        return previous  # Don't downgrade after 2+ consistent reviews
    if cur_idx < prev_idx:
        return current   # Always allow upgrade
    return current
```

**8c. Prevent re-opening resolved issues**

```python
def filter_reopened(new_findings, resolved_hashes):
    return [f for f in new_findings
            if f.finding_hash_fuzzy not in resolved_hashes or f.confidence > 0.9]
```

**8d. ReviewHistory persistence via PR comments**

```python
@dataclass
class ReviewHistory:
    review_count: int
    previous_hashes: set[str]
    resolved_hashes: set[str]
    last_severity_map: dict[str, Severity]
    last_quality_score: float | None
    last_review_sha: str | None
```

No external DB -- GitHub PR comments are the persistence layer.

### Files to Modify

- `src/ai_reviewer/github/client.py` -- parse metadata, build `ReviewHistory`
- `src/ai_reviewer/models/review.py` -- add `ReviewHistory` dataclass
- `src/ai_reviewer/review.py` -- apply stabilization, filter re-opens

---

## 9. Security-Aware Reviewing

### Current State (post Phase 1)

- ~~Brief security agent prompt~~ -> Now has full OWASP Top 10 coverage in `AGENT_CONFIGS[0]` (`review.py:40-81`)
- ~~No secret detection~~ -> `security/scanner.py` implements 10+ regex patterns + Shannon entropy analysis (threshold 4.5)
- ~~Critical security findings dropped by cross-review~~ -> Now bypass consensus in `apply_cross_review()` (`review.py:373-375`)
- Detailed `SecurityAgent` class (`agents/security.py:13-53`) still unused by main pipeline

### Completed (Phase 1)

- **9a/9b. Secret detection + entropy** -- Done. `src/ai_reviewer/security/scanner.py` with pattern + entropy scanning, config-driven excludes.
- **9d. Enhanced security prompt** -- Done. Full OWASP knowledge in `AGENT_CONFIGS[0]`.
- **9e. Critical security bypass** -- Done. CRITICAL+SECURITY findings always kept in cross-review.

### Remaining

**9a reference. Pre-screening secret detection (already implemented)**

New file `src/ai_reviewer/security/scanner.py`:

```python
SECRET_PATTERNS = [
    (r'(?i)(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*["\'][^"\']{8,}["\']', "Hardcoded secret"),
    (r'AKIA[0-9A-Z]{16}', "AWS Access Key"),
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub PAT"),
    (r'-----BEGIN (RSA |EC |DSA )?PRIVATE KEY-----', "Private Key"),
    (r'sk-[a-zA-Z0-9]{20,}', "API Secret Key"),
]

def scan_for_secrets(diff_lines):
    findings = []
    for i, line in enumerate(diff_lines):
        if not line.startswith('+'): continue
        for pattern, desc in SECRET_PATTERNS:
            if re.search(pattern, line):
                findings.append(ConsolidatedFinding(
                    severity=Severity.CRITICAL, category=Category.SECURITY,
                    title=f"Potential {desc} in diff", confidence=0.95, ...))
    return findings
```

**9b. Entropy analysis** for detecting base64-encoded or high-entropy strings using Shannon entropy (threshold 4.5).

**9c. Integration**: Run as synchronous pre-step before spawning agents. Results bypass aggregation -- always included. Allow `.ai-reviewer.yaml` to define `secret_scan_exclude` patterns.

**9d. Enhanced security agent prompt**: Port OWASP knowledge from `agents/security.py` into `AGENT_CONFIGS[0]`. Add language-specific unsafe API patterns per detected language.

**9e. Critical security findings bypass cross-review**

In `apply_cross_review()` (`review.py:292`):

```python
if finding.severity == Severity.CRITICAL and finding.category == Category.SECURITY:
    kept.append((finding, 1.0, 0))  # Always keep, rank first
    continue
```

### Files to Modify

- New: `src/ai_reviewer/security/scanner.py`
- `src/ai_reviewer/review.py` -- secret scanning pre-step, enhanced prompt, cross-review bypass
- `src/ai_reviewer/config.py` -- `secret_scan_exclude` patterns

---

## 10. Practical Implementation Strategy

### Phase 1: Quick Wins (1-2 weeks, localized changes) ✅ Completed


| #   | Item                                       | Section | Effort  | Files                     |
| --- | ------------------------------------------ | ------- | ------- | ------------------------- |
| 1   | Confidence-based filtering                 | 2a      | 1 day   | review.py                 |
| 2   | Adaptive agent count for small PRs         | 2b      | 1 day   | review.py                 |
| 3   | Enforce max_total/max_per_file from config | 5c      | 0.5 day | client.py                 |
| 4   | Secret detection pre-scan                  | 9a, 9b  | 2 days  | new scanner.py, review.py |
| 5   | Few-shot examples in prompt                | 1d      | 0.5 day | review.py                 |
| 6   | Enhanced security agent prompt             | 9d      | 0.5 day | review.py                 |
| 7   | Adaptive max findings per agent            | 2c      | 0.5 day | review.py                 |
| 8   | Critical security bypass cross-review      | 9e      | 0.5 day | review.py                 |


All Phase 1 items are **independent** and can be done in parallel.

**Status:** Done.

### Phase 2: Medium Effort (2-4 weeks, moderate refactoring) ✅ Completed


| #   | Item                                      | Section | Effort | Depends On | Implementation Location |
| --- | ----------------------------------------- | ------- | ------ | ---------- | ----------------------- |
| 1   | Load .ai-reviewer.yaml from target repo   | 1a      | 2 days | --         | `client.py:load_repo_config()`, `client.py:load_repo_conventions()`, `context.py:repo_config/conventions` |
| 2   | New quality scoring formula               | 3a, 3d  | 2 days | --         | `review.py:compute_quality_score()`, `models/review.py:ScoreBreakdown` |
| 3   | Multi-tier finding hash                   | 5a      | 2 days | --         | `findings.py:compute_fuzzy_hash()`, `client.py:compute_review_delta()` 3-tier matching |
| 4   | Cross-file deduplication                  | 5b      | 1 day  | --         | `review.py:dedup_cross_file()`, called from `aggregate_findings()` |
| 5   | Convergence detection                     | 6a, 6b  | 2 days | P2-3       | `client.py:has_converged()`, `client.py:should_skip_review()`, `cli.py:--force-review` |
| 6   | Severity stabilization across runs        | 8a, 8b  | 2 days | P2-3       | `client.py:stabilize_severity()`, applied in `compute_review_delta()` |
| 7   | PR size classification + adaptive prompts | 1c      | 1 day  | P2-1       | `review.py:classify_pr()`, size-aware instructions in `get_base_prompt()` |
| 8   | Language-specific prompt rules            | 1e      | 1 day  | --         | `review.py:_LANGUAGE_RULES`, `review.py:get_language_rules()` |

All Phase 2 items are **complete**. Test coverage: 259 tests (pytest), ruff clean, mypy clean.

**Status:** Done.


### Phase 3: Architectural (4-8 weeks, new subsystems)


| #   | Item                                    | Section | Effort | Depends On |
| --- | --------------------------------------- | ------- | ------ | ---------- |
| 1   | Commit-aware incremental diffing        | 4a-d    | 1 week | P2-3       |
| 2   | Large PR tiered review                  | 7a-c    | 1 week | --         |
| 3   | Embedding-based semantic similarity     | 2d      | 1 week | --         |
| 4   | Full ReviewHistory + re-open prevention | 8c, 8d  | 1 week | P2-3, P2-6 |
| 5   | Score damping with rolling average      | 3b      | 3 days | P2-2, P3-4 |
| 6   | Consolidate dual aggregation paths      | --      | 3 days | P2-2       |


### Magic Numbers to Extract to Config


| Value                              | Location      | Config Key                         |
| ---------------------------------- | ------------- | ---------------------------------- |
| `50000` (diff char limit)          | review.py:144 | `review.max_diff_chars`            |
| `5` (max files with contents)      | review.py:114 | `review.max_file_contents`         |
| `5000` (file content char limit)   | review.py:115 | `review.max_file_content_chars`    |
| `5` (max findings per agent)       | review.py:157 | `review.max_findings_per_agent`    |
| `10` (max inline comments)         | client.py:325 | `output.max_inline_comments`       |
| `0.85` (similarity threshold)      | review.py:424 | Already in config, wire it up      |
| `3` (fix detection line tolerance) | client.py:587 | `delta.line_tolerance`             |
| `5` (clustering line tolerance)    | review.py:442 | `aggregator.line_tolerance`        |
| `15000` (cross-review diff max)    | review.py:202 | `review.cross_review_diff_chars`   |
| `20` (cross-review max findings)   | review.py:200 | `review.cross_review_max_findings` |


### Risks & Mitigations


| Risk                                           | Mitigation                                                            |
| ---------------------------------------------- | --------------------------------------------------------------------- |
| Secret detection false positives               | `secret_scan_exclude` patterns in config. Start high-confidence only. |
| Embedding model ~500MB + cold start            | Opt-in via config. Lazy-load. TF-IDF as lighter alternative.          |
| Incremental diffing misses cross-commit issues | Always include full PR diff as context, focus on incremental.         |
| Convergence suppresses legitimate findings     | `--force-review` flag / PR label. Only suppress on 3rd+ review.       |
| Score damping hides regressions                | Cap at 0.3. If drop > 0.2, skip damping.                              |
| Tiered review doubles API cost                 | Only for >1000 LOC. Architecture pass = 1 agent, short prompt.        |


---

## Verification Plan

1. **Unit tests**: New functions (confidence filtering, hotspot scoring, secret scanning, convergence)
2. **Integration**: `ai-reviewer review-pr --dry-run` against test PRs of varying sizes
3. **Regression**: Existing pytest suite (100 tests) must pass
4. **Dogfooding**: `ai-review.yaml` workflow reviews itself -- verify improved behavior
5. **Stability**: Review same PR 3x; verify quality score variance < 5%

