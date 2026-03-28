# Architecture Documentation

> Comprehensive technical reference for the AI Code Reviewer system.
> For a quick overview, see the [README](../README.md).

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Review Pipeline](#2-review-pipeline)
3. [Multi-Agent Consensus](#3-multi-agent-consensus)
4. [Scoring System](#4-scoring-system)
5. [Incremental Review (Delta Tracking)](#5-incremental-review-delta-tracking)
6. [Convergence and "Stop Reviewing" Logic](#6-convergence-and-stop-reviewing-logic)
7. [Documentation Review](#7-documentation-review)
8. [Prompt Engineering](#8-prompt-engineering)
9. [Security](#9-security)

---

## 1. System Overview

AI Code Reviewer orchestrates multiple LLM agents — each with a specialized focus area — to produce consensus-based code reviews. All LLM access goes through the Cursor Background Agent API; there are no direct provider SDK dependencies.

```mermaid
flowchart LR
    PR["PR Event / CLI"]
    PR --> Pipeline["review_pr_with_cursor_agent()"]
    Pipeline --> A1["Security Agent"]
    Pipeline --> A2["Performance Agent"]
    Pipeline --> A3["Quality Agent"]
    A1 --> Agg["Aggregation + Cross-Review"]
    A2 --> Agg
    A3 --> Agg
    Agg --> Delta["Delta Tracking"]
    Delta --> Conv{"Converged?"}
    Conv -- yes --> Skip["Skip / LGTM"]
    Conv -- no --> Post["Format + Post to GitHub"]
    Post --> DocReview["Documentation Review\n(rule-based, no LLM)"]
    DocReview --> DocComment["Post/update doc-bot\nPR comment"]
```

### Module Map

| Module | Responsibility |
|--------|----------------|
| `cli.py` | Click CLI: `review-pr`, `config validate/show`, `serve` (uvicorn webhook) |
| `config.py` | YAML/env `Config` dataclasses, `load_config`, `validate_config` |
| `review.py` | Main pipeline: agent configs, prompts, aggregation, cross-review, `review_pr_with_cursor_agent` |
| `models/context.py` | `ReviewContext` dataclass for PR/repo metadata and repo config hooks |
| `models/findings.py` | `Severity`, `Category`, `ReviewFinding`, `ConsolidatedFinding`, `compute_fuzzy_hash` |
| `models/review.py` | `ReviewHistory`, `ScoreBreakdown`, `AgentReview`, `ConsolidatedReview` |
| `agents/cursor_client.py` | `CursorClient` / `CursorConfig`: HTTP client for Cursor Background Agent API |
| `agents/base.py` | `ReviewAgent` ABC (alternate agent path, not used by main Cursor flow) |
| `agents/security.py` | `SecurityAgent` subclass of `ReviewAgent` |
| `agents/performance.py` | `PerformanceAgent` subclass |
| `agents/patterns.py` | `PatternsAgent` subclass |
| `orchestrator/orchestrator.py` | Generic parallel `AgentOrchestrator` (asyncio tasks) |
| `orchestrator/aggregator.py` | `ReviewAggregator` clustering/merge (alternate path; production uses `aggregate_findings` in `review.py`) |
| `security/scanner.py` | Regex + Shannon entropy secret scanner on unified diffs |
| `github/client.py` | `GitHubClient`, delta/convergence, inline comments, metadata, thread resolution, doc-bot comment upsert, repo path probing |
| `github/formatter.py` | `GitHubFormatter`: markdown bodies, compact/delta layouts, review actions, JSON export |
| `github/webhook.py` | FastAPI app, HMAC verification, PR + `/ai-review` comment handlers |
| `docs/__init__.py` | Package init for documentation review module |
| `docs/analyzer.py` | `DocAnalyzer`, `DocSuggestion`, `is_architecture_impacting`, `format_doc_comment` — rule-based doc review |

---

## 2. Review Pipeline

### Entry Points

| Entry | Function | Path |
|-------|----------|------|
| **CLI** | `review_pr` → `review_pr_async()` | `cli.py` |
| **Webhook** | `handle_pr_event` → `default_review_handler` | `github/webhook.py` |
| **Serve** | `cli serve` starts uvicorn with the webhook app | `cli.py` |

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Trigger as Webhook / CLI
    participant Review as review_pr_with_cursor_agent()
    participant Secret as scan_for_secrets()
    participant Agent as run_single_agent() × N
    participant Agg as aggregate_findings()
    participant XR as run_cross_review_round()
    participant Delta as compute_review_delta()
    participant Conv as should_skip_review()
    participant Fmt as GitHubFormatter
    participant GH as GitHubClient

    Trigger->>Review: repo, pr_number, config
    Review->>GH: get_pull_request, get_pr_diff, get_changed_files
    Review->>GH: load_repo_config, load_repo_conventions
    Review->>Secret: scan diff for secrets (pre-agent)
    Review->>Agent: asyncio.gather(run_single_agent × N)
    Agent-->>Review: raw findings per agent
    Review->>Agg: cluster, dedup, score, confidence filter
    Agg-->>Review: ConsolidatedFinding[]
    Review->>XR: cross-review validation (if agents > 2)
    XR-->>Review: filtered + ranked findings
    Note over Review: Prepend secret findings (bypass aggregation)
    Review-->>Trigger: ConsolidatedReview

    Trigger->>Delta: compute_review_delta(pr, findings)
    Delta-->>Trigger: ReviewDelta (new, fixed, open)
    Trigger->>Conv: should_skip_review(count, delta)
    Conv-->>Trigger: skip? (unless --force-review)
    Trigger->>Fmt: format_review_compact / format_review_delta
    Trigger->>GH: post_review, resolve_fixed_comments, post_inline_comments

    Note over Trigger,GH: Documentation review (rule-based, after main review)
    Trigger->>GH: probe_repo_paths(convention files + arch dirs)
    GH-->>Trigger: existing_repo_paths
    Trigger->>Trigger: DocAnalyzer.run()
    Trigger->>GH: post_or_update_doc_comment(body, marker)
```

### Key Functions

- **`review_pr_with_cursor_agent()`** (`review.py`): Core orchestration. Fetches PR data, builds context, spawns agents in parallel, aggregates, cross-reviews, prepends secret findings, returns `ConsolidatedReview`.
- **`run_single_agent()`** (`review.py`): Sends a prompt to one Cursor Background Agent, parses JSON response into findings.
- **`aggregate_findings()`** (`review.py`): Clusters raw findings by similarity, computes consensus scores, applies confidence filtering and cross-file dedup.
- **`default_review_handler()`** (`webhook.py`): Webhook's async handler — includes pre-agent skip checks, LGTM fast path, metadata embedding, and the full post flow.

---

## 3. Multi-Agent Consensus

### Agent Spawning

Production uses **parallel `asyncio.gather`** of `run_single_agent()`, each configured from `AGENT_CONFIGS`:

| Agent | Focus | Prompt Specialization |
|-------|-------|-----------------------|
| `security-agent` | Security | OWASP Top 10, injection, auth, crypto, secrets |
| `performance-agent` | Performance + Correctness | Algorithmic complexity, resource leaks, concurrency |
| `quality-agent` | Maintainability | SOLID, DRY, KISS, YAGNI, API design, error handling |

Agent count is adaptive: `_effective_agent_count()` scales 1–3 agents based on PR size. Cross-review is auto-skipped when ≤ 2 agents run.

### Aggregation Pipeline

```mermaid
flowchart TD
    Raw["Raw findings from N agents"]
    Raw --> Cluster["_cluster_raw_findings()"]
    Cluster --> |"same file, category,\noverlapping lines ±5,\ntitle+desc similarity ≥ 0.85"| Merged["ConsolidatedFinding[]"]
    Merged --> Dedup["dedup_cross_file()"]
    Dedup --> |"3+ findings with same\n(category, title) across files\n→ collapse to 1"| Filtered["Confidence filter"]
    Filtered --> |"per-severity thresholds:\ncritical ≥ 0.5, warning ≥ 0.6,\nsuggestion ≥ 0.7, nitpick ≥ 0.8"| XR{"Cross-review?"}
    XR -- "agents > 2" --> CrossReview["run_cross_review_round()"]
    CrossReview --> Apply["apply_cross_review()"]
    Apply --> |"drop if valid fraction\n< 2/3 agreement\n(CRITICAL+SECURITY always kept)"| Final["Final findings"]
    XR -- "agents ≤ 2" --> Final
```

**Clustering** (`_cluster_raw_findings`): Groups findings that share the same file, category, overlapping line ranges (±5 lines), and combined title+description similarity ≥ 0.85 (character-level `SequenceMatcher`). Each cluster becomes one `ConsolidatedFinding` with `consensus_score = unique_agents_in_cluster / total_agents`.

**Cross-file dedup** (`dedup_cross_file`): When 3+ findings share the same `(category, title)` across different files, they collapse into a single finding with an "Also found in: ..." annotation.

**Cross-review validation** (`run_cross_review_round` → `apply_cross_review`): A second-pass LLM call where agents validate each other's findings. Findings with < 2/3 validation agreement are dropped — except `CRITICAL` severity + `SECURITY` category findings, which always bypass this filter.

---

## 4. Scoring System

### `compute_quality_score()`

Located in `review.py`. Returns a `float` between 0.0 and 0.95 along with a `ScoreBreakdown`.

#### When findings exist

```
severity_weights = {CRITICAL: 0.20, WARNING: 0.06, SUGGESTION: 0.02, NITPICK: 0.005}

severity_penalty = Σ (weight[f.severity] × f.confidence)  for each finding f
density_penalty  = min(0.15, (len(findings) / max(total_lines/100, 1)) × 0.03)
consensus_factor = 0.8 + mean(f.consensus_score) × 0.2
agent_factor     = min(1.0, agent_count / 3)

raw_score = max(0, 1 - severity_penalty - density_penalty)
final     = min(0.95, round(raw_score × consensus_factor × agent_factor, 2))
```

#### Clean review (no findings)

```
raw_score    = 0.85
agent_bonus  = max(0, min(0.10, (agent_count - 1) × 0.05))
agent_factor = (raw_score + agent_bonus) / raw_score
final        = min(0.95, raw_score × agent_factor)
```

### `ScoreBreakdown`

```python
@dataclass
class ScoreBreakdown:
    severity_penalty: float
    density_penalty: float
    consensus_factor: float
    agent_factor: float
    raw_score: float
```

Displayed in the review footer as a collapsed `<details>` section so reviewers can understand how the score was derived.

---

## 5. Incremental Review (Delta Tracking)

### `ReviewDelta`

```python
@dataclass
class ReviewDelta:
    new_findings: list[ConsolidatedFinding]       # Not seen before
    fixed_findings: list[PreviousComment]          # Previously reported, now resolved
    open_findings: list[ConsolidatedFinding]       # Still present from prior review
    previous_comments: list[PreviousComment]       # All prior AI review comments
    suppressed_findings: list[ConsolidatedFinding] # Low-severity on fix-zone lines

    @property
    def all_issues_resolved(self) -> bool:
        return len(self.open_findings) == 0 and len(self.new_findings) == 0
```

### Three-Tier Matching in `compute_review_delta()`

Each current finding is matched against previous inline comments using a three-tier cascade:

```mermaid
flowchart TD
    F["Current finding"]
    F --> T1{"Strict hash match?"}
    T1 -- "finding_hash == comment.finding_hash" --> Matched["→ open_findings"]
    T1 -- miss --> T2{"Fuzzy hash match?"}
    T2 -- "finding_hash_fuzzy == comment.finding_hash_fuzzy" --> Matched
    T2 -- miss --> T3{"Title + line match?"}
    T3 -- "(file, line, normalized_title)" --> Matched
    T3 -- miss --> New["→ new_findings"]
```

| Tier | Hash Key | Stable Across |
|------|----------|---------------|
| **Strict** | `SHA256(file_path:line_start:normalized_title)[:12]` | Same file, line, title |
| **Fuzzy** | `SHA256(file_path:sorted_keywords_4+_chars)[:12]` | Line shifts, title rewording |
| **Legacy** | `(file_path, line, normalized_title)` tuple | Fallback for pre-hash comments |

Unmatched previous comments become `fixed_findings`.

### Severity Stabilization

`stabilize_severity(current, previous, review_count)` prevents severity flip-flopping:

- **Upgrades** (more severe) are always allowed.
- **Downgrades** are blocked after 2+ reviews at the higher severity.
- Applied during `compute_review_delta()` when a finding matches a previous comment.

### Finding ID Embedding

Each inline comment includes an HTML comment for future matching:

```
<!-- ai-reviewer-id: {finding.finding_hash} -->
```

Parsed by `_parse_review_comment()` on subsequent reviews to build the `PreviousComment.finding_hash` field.

### Review Metadata

Top-level review comments embed structured metadata:

```
<!-- ai-reviewer-meta: {"commit_sha": "abc123", "review_count": 2, "timestamp": "...", "findings_hash": "a1b2c3d4e5f6"} -->
```

Used by `should_skip_before_agents()` for same-SHA detection and findings-hash comparison, and by `check_lgtm_fast_path()` for the LGTM candidate check.

---

## 6. Convergence and "Stop Reviewing" Logic

The convergence system prevents redundant reviews when findings have stabilized and suppresses low-value noise on recently-fixed code.

### Decision Flowchart

```mermaid
flowchart TD
    Start["PR event received"]
    Start --> Force{"force-review flag\nor label?"}
    Force -- yes --> RunAgents["Run agents"]
    Force -- no --> PreAgent["should_skip_before_agents()"]
    PreAgent --> SameSHA{"Same SHA as\nlast review?"}
    SameSHA -- yes --> SkipAlready["Skip: ALREADY_REVIEWED"]
    SameSHA -- no --> HashCheck{"findings_hash set and\ndiff doesn't touch\nfiles with findings?"}
    HashCheck -- yes --> SkipUnchanged["Skip: FINDINGS_UNCHANGED"]
    HashCheck -- no --> LGTM{"LGTM fast path?\n(review_count ≥ 2,\nall issues resolved)"}
    LGTM -- yes --> Recheck["1-agent re-check"]
    Recheck --> Clean{"Re-check clean?"}
    Clean -- yes --> PostLGTM["Post COMMENT (LGTM),\nresolve fixed threads"]
    Clean -- no --> RunAgents["Run full agent set"]
    LGTM -- no --> RunAgents
    RunAgents --> Delta["compute_review_delta()"]
    Delta --> FixZones["Build fix zones from\nfixed_findings"]
    FixZones --> Suppress["Suppress SUGGESTION/NITPICK\non fix-zone lines"]
    Suppress --> EstCount["estimate_review_count()"]
    EstCount --> ShouldSkip{"should_skip_review()?"}
    ShouldSkip -- skip --> Silent["Skip posting (log only)"]
    ShouldSkip -- post --> Format["Format + post review"]
```

### Pre-Agent Skip (`should_skip_before_agents`)

Runs before any LLM agents are spawned. Returns a `SkipReason` or `None`:

| Check | `SkipReason` | Condition |
|-------|-------------|-----------|
| Same commit | `ALREADY_REVIEWED` | `meta.commit_sha == current_sha` |
| Unchanged findings | `FINDINGS_UNCHANGED` | `meta.findings_hash` is set and the diff only touches files with no previous findings |

`force_review` overrides both checks.

### LGTM Fast Path

`check_lgtm_fast_path(self, pr, meta)` computes a lightweight delta with an empty findings list. Returns the delta when `review_count >= 2` and `all_issues_resolved`, otherwise `None`.

When a candidate is found, callers run a **1-agent re-check** (no cross-review). If the re-check finds zero findings, a `COMMENT` review is posted with the LGTM delta and fixed threads are resolved. If the re-check finds issues, its result is **discarded** and the full agent pipeline runs — the re-check is never reused as the main review.

### Graduated Suppression on Fix-Zone Lines

After `compute_review_delta()` determines which previous findings were fixed, it builds **fix zones**: a mapping of `file_path → set[int]` covering each fixed finding's line ± 3 lines of tolerance.

New findings that land in a fix zone are suppressed if their severity is `SUGGESTION` or `NITPICK`. `WARNING` and `CRITICAL` findings on fix-zone lines are always posted. Suppressed findings are stored in `ReviewDelta.suppressed_findings` and mentioned in the review body (e.g., "2 suggestions suppressed on recently-fixed code").

```mermaid
flowchart TD
    NewFinding["New finding on line N"]
    NewFinding --> InFixZone{"Line N in a\nfix zone?"}
    InFixZone -- no --> PostAll["Post normally\n(all severities)"]
    InFixZone -- yes --> SevCheck{"Severity ≥ WARNING?"}
    SevCheck -- yes --> PostIt["Post it"]
    SevCheck -- no --> Suppress["Suppress\n(SUGGESTION/NITPICK on fix code)"]
```

### Post-Agent Skip (`should_skip_review`)

Runs after agents complete and delta is computed:

| Condition | Action |
|-----------|--------|
| First review (`count == 1`) | Always post |
| Converged (`new == 0` and `fixed == 0`) on 2nd+ review | Skip |
| 3rd+ review where all new findings are `NITPICK` | Skip |

### Functions

| Function | Signature | Logic |
|----------|-----------|-------|
| `has_converged(delta)` | `(ReviewDelta) → bool` | `True` when `new_findings == 0` and `fixed_findings == 0` |
| `should_skip_review(count, delta)` | `(int, ReviewDelta) → bool` | See table above |
| `estimate_review_count(delta)` | `(ReviewDelta) → int` | 1 if no previous comments; else `max(2, len(previous_comments) // 3 + 1)` |
| `should_skip_before_agents(meta, sha, force, diff_files, previous_comments)` | `(...) → SkipReason \| None` | Same-SHA or findings-unchanged check |
| `check_lgtm_fast_path(self, pr, meta)` | `(PullRequest, ReviewMeta) → ReviewDelta \| None` | `review_count ≥ 2` and `all_issues_resolved` |

### Overrides

- **CLI**: `--force-review` flag bypasses all convergence checks.
- **GitHub**: `force-review` label on the PR triggers a full review regardless of convergence state.

---

## 7. Documentation Review

A rule-based check that runs after the main AI review. It detects architecture-impacting changes and flags missing documentation updates. No LLM calls are consumed — the check is fast, free, and deterministic.

### Two-Tier Design

```mermaid
flowchart TD
    Start["DocAnalyzer.run()"]
    HasConfig{"repo has .ai-reviewer.yaml\nwith documentation section?"}

    subgraph tier1 [Tier 1: Zero-Config]
        ArchCheck["check_architecture_folder()"]
        ConvCheck["check_convention_files()"]
    end

    subgraph tier2 [Tier 2: Configured]
        MappingCheck["check_source_to_docs_mapping()"]
    end

    Start --> HasConfig
    HasConfig -->|"No"| ArchCheck
    HasConfig -->|"Yes"| ArchCheck
    ArchCheck --> ConvCheck
    ConvCheck --> MappingCheck
    MappingCheck --> Dedup["Deduplicate by target file,\nsort high-priority first"]
```

**Tier 1 (zero-config)** runs on every repo, including those without `.ai-reviewer.yaml`:

| Check | What it does |
|-------|-------------|
| `check_architecture_folder()` | Probes for `architecture/`, `docs/`, or `doc/` directories. If none exist, emits a high-priority suggestion. |
| `check_convention_files()` | Probes for `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `.cursor/rules/README.md`. If any exist and the PR is architecture-impacting but doesn't modify them, emits a suggestion per stale file. |

**Tier 2 (configured)** adds `check_source_to_docs_mapping()` when the repo has a `documentation.source_to_docs_mapping` section in `.ai-reviewer.yaml`. Each changed file is matched against glob patterns; unupdated doc targets produce suggestions.

### Architecture-Impact Heuristics

`is_architecture_impacting()` returns `True` when any changed file matches:

| Heuristic | Examples |
|-----------|---------|
| New or deleted top-level directory | Adding `newpkg/`, removing `legacy/` |
| Project manifest files | `pyproject.toml`, `package.json`, `Cargo.toml`, `go.mod`, `Gemfile`, `build.gradle`, `pom.xml`, `CMakeLists.txt` |
| CI/workflow files | `.github/workflows/*`, `.gitlab-ci.yml`, `Jenkinsfile` |
| Entry-point files | Any file named `main.*`, `cli.*`, `app.*`, `index.*`, `server.*` |
| Infrastructure files | `Dockerfile*`, `docker-compose*`, `*.tf`, `cloudbuild.yaml` |

If none match, the convention file check produces zero suggestions (silent on routine PRs).

### Comment Deduplication

Doc suggestions are posted as a separate issue comment from the AI code review. An HTML marker (`<!-- AI-CODE-REVIEWER-DOC-BOT -->`) is embedded in the comment body. On subsequent pushes:

- `find_doc_bot_comment()` searches existing issue comments for the marker.
- `post_or_update_doc_comment()` updates the existing comment in-place if found, or creates a new one.

This prevents duplicate comments across commits within the same PR.

### Configuration

**Operator-level** (`config.yaml`):

```yaml
doc_review:
  enabled: true
  architecture_paths: ["architecture/", "docs/", "doc/"]
  convention_files: ["AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", ".cursor/rules/README.md"]
  comment_marker: "<!-- AI-CODE-REVIEWER-DOC-BOT -->"
```

**Repo-level** (`.ai-reviewer.yaml`):

```yaml
documentation:
  enabled: true
  source_to_docs_mapping:
    "src/ai_reviewer/agents/**":
      - .ai/rules/agents.md
    "src/ai_reviewer/config.py":
      - config.example.yaml
      - README.md
```

Setting `documentation.enabled: false` in the repo config skips both tiers entirely.

**CLI**: `--doc-check` / `--no-doc-check` overrides the config-level `enabled` flag for a single run.

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `is_architecture_impacting()` | `docs/analyzer.py` | Heuristic detection of architecture-impacting changes |
| `DocAnalyzer.run()` | `docs/analyzer.py` | Orchestrates all checks, deduplicates, sorts by priority |
| `format_doc_comment()` | `docs/analyzer.py` | Renders suggestions as markdown with the dedup marker |
| `probe_repo_paths()` | `github/client.py` | Checks which convention files / dirs exist in the repo |
| `find_doc_bot_comment()` | `github/client.py` | Finds existing doc-bot comment by marker |
| `post_or_update_doc_comment()` | `github/client.py` | Creates or updates the doc-bot PR comment |
| `_run_doc_review()` | `cli.py` | Wires the analyzer into the `review-pr` CLI flow |

---

## 8. Prompt Engineering

### PR Classification

`classify_pr(changed_paths, additions, deletions)` returns `(pr_type, size)`:

| Type | Detection |
|------|-----------|
| `docs` | All files are `.md`, `.rst`, `.txt`, or under `docs/` |
| `ci` | All files under `.github/`, `.circleci/`, etc. |
| `code` | Everything else |

| Size | Threshold (additions + deletions) |
|------|-----------------------------------|
| `trivial` | < 50 lines |
| `small` | 50–199 lines |
| `medium` | 200–999 lines |
| `large` | ≥ 1000 lines |

Size-adaptive instructions are injected into the prompt:
- **trivial/small**: "Be extra precise — only flag genuine issues. Do not pad with low-value suggestions."
- **large**: "Focus on architectural concerns and high-severity issues first. Ignore minor style."

### Language-Specific Rules

`_LANGUAGE_RULES` dict provides per-language guidance injected via `get_language_rules()`:

| Language | Key Rules |
|----------|-----------|
| **Python** | Mutable default args, bare `except`, missing type hints, f-string injection in logging, `subprocess shell=True` |
| **Rust** | `.unwrap()` in non-test code, `unsafe` without `// SAFETY:`, unnecessary `.clone()`, unbounded allocations |
| **JavaScript** | Prototype pollution, `==` vs `===`, unhandled Promise rejections, `eval()`/`innerHTML` |
| **TypeScript** | `any` type escapes, missing error boundaries, `@ts-ignore` without justification |
| **Go** | Unchecked errors, SQL string concatenation, goroutine leaks, missing `defer` |

### Repo-Aware Prompts

- **`.ai-reviewer.yaml`**: `load_repo_config()` fetches the config from the target repo. `custom_rules` are injected as a "Repository-Specific Rules" section. `ignore` patterns filter files from the diff before agents see them.
- **Convention files**: `load_repo_conventions()` best-effort loads `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `.cursor/rules/README.md` (capped at 3k chars total) and injects them as a "Repository Conventions" section.
- **PR metadata**: Title, description, base/head branches, changed file list, and detected languages are included in every prompt.

### Few-Shot Examples

`get_output_format()` includes good and bad finding examples to calibrate agent output quality:

- **Good**: Specific file, line, severity, actionable title and description.
- **Bad**: Vague "consider adding more tests" style findings explicitly marked as what NOT to produce.

---

## 9. Security

### Secret Detection Pre-Scan

`scan_for_secrets()` in `security/scanner.py` runs synchronously before agents are spawned. It scans added lines in the unified diff using two methods:

1. **Regex patterns** (`SECRET_PATTERNS`): 10+ compiled patterns covering AWS keys, GitHub tokens (PAT, OAuth, App, Fine-Grained), private keys, Slack tokens, generic API keys/secrets, database connection strings, and JWT tokens.

2. **Shannon entropy analysis**: Strings ≥ 20 characters with entropy ≥ 4.5 bits/char are flagged as potential high-entropy secrets (base64-encoded keys, random tokens).

Secret findings are created as `ConsolidatedFinding` with `severity=CRITICAL` and `category=SECURITY`. They **bypass aggregation and cross-review** entirely — they are prepended directly to the final findings list.

The `.ai-reviewer.yaml` config supports `secret_scan_exclude` patterns to suppress false positives on known-safe paths.

### Critical Security Bypass in Cross-Review

In `apply_cross_review()`, findings with `severity == CRITICAL` and `category == SECURITY` are unconditionally kept regardless of cross-review validation scores. This prevents legitimate security findings from being filtered out by the consensus mechanism.

### Config Validation

`validate_config()` checks:
- Cursor API key is present and non-empty.
- GitHub token is present.
- At least one agent is configured.
- `min_agents_required` ≤ number of configured agents.

### Webhook Security

`verify_signature()` validates incoming webhook payloads using HMAC SHA-256 with the configured `GITHUB_WEBHOOK_SECRET`. The request body is read once and reused for both signature verification and JSON parsing.
