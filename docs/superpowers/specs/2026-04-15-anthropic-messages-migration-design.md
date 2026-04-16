# Migration to Anthropic Messages API — Design Spec

**Date:** 2026-04-15
**Status:** Draft — awaiting user approval
**Goal:** Replace Cursor Background Agent API with the Anthropic Messages API as the backing LLM layer, without regressing (and ideally improving) review quality.

---

## 1. Motivation

Today the project uses Cursor's Background Agent API (`api.cursor.com/v0/agents`) for all LLM calls. Two problems:

1. **Architectural mismatch.** Cursor's API is a repo-scoped, async, polling-based agent runner. Most of our flows want a chat-completion shape. `CursorClient.complete()` in `src/ai_reviewer/agents/cursor_client.py:303` actually raises `NotImplementedError`, so the `ReviewAgent.review()` path in `agents/base.py` has never worked; only the `run_review_agent()` path in `review.py` is live.
2. **Vendor fit.** The user wants to consolidate on Anthropic's official API for control over models (Opus 4.6, Sonnet 4.6), prompt caching, extended thinking, and structured outputs — none of which are first-class through Cursor's wrapper.

The migration target is the Messages API at `https://api.anthropic.com/v1/messages`, via the official `anthropic` Python SDK.

## 2. Non-goals

- **No change** to orchestration, aggregation, convergence, cross-review, secret scanning, doc-analyzer, or finding/review data models. These operate on `AgentReview` objects and are LLM-agnostic.
- **No change** to the GitHub integration surface (`GitHubClient`, webhook flow, formatter, PR comment shape).
- **No multi-provider abstraction.** We fully commit to Anthropic; the `ReviewAgent` base speaks to an `AnthropicClient`, not an abstract `LLMClient`. A shim abstraction is reintroducible later if a second provider is ever needed.
- **No local repo cloning** introduced in this migration. The codebase currently does zero cloning; we keep it that way. Repo exploration is done via the GitHub Contents API when needed.

## 3. Quality posture (the "best reviewer every time" target)

To match-and-beat Cursor's background agents, every review layers:

1. **Rich baseline prompt context** (cached) — conventions, manifests, changed files, neighbors.
2. **Tool use** — Claude can call `read_file`, `glob`, `grep` on demand.
3. **Extended thinking** on reasoning-heavy agents.
4. **Model mix** — Opus 4.6 on reasoning-heavy agents, Sonnet 4.6 on fast/broad agents.
5. **Structured output** via `output_config.format = json_schema`.
6. **Prompt caching** on shared system prompts across the multi-agent run.

Details for each layer are in §5.

## 4. Architecture

### 4.1 Component map (after migration)

```
ai_reviewer/
├── agents/
│   ├── anthropic_client.py   [NEW]  Messages API wrapper: tool loop, caching, structured output, thinking
│   ├── base.py               [MOD]  ReviewAgent.review() rewritten on the Anthropic client
│   ├── security.py           [MOD]  MODEL + THINKING_BUDGET class-vars updated
│   ├── patterns.py           [MOD]  same
│   ├── performance.py        [MOD]  same
│   └── __init__.py           [MOD]  exports
├── context/
│   └── neighbors.py          [NEW]  import-graph + sibling-files heuristics
├── tools/
│   └── repo_tools.py         [NEW]  read_file / glob / grep tool implementations backed by GitHubClient
├── config.py                 [MOD]  CursorApiConfig → AnthropicApiConfig; per-agent thinking toggles
├── review.py                 [MOD]  Replace CursorClient calls; build richer prompts; no run_review_agent
├── cli.py                    [MOD]  Config wiring
├── github/
│   └── webhook.py            [MOD]  Config wiring
└── models/                   [unchanged]
    ├── findings.py
    ├── review.py
    └── context.py

DELETED: src/ai_reviewer/agents/cursor_client.py
```

### 4.2 Request flow (per agent)

```
review.py
  └─▶ build_prompt_context()           ◀── GitHubClient.get_contents() for conventions,
                                             manifests, changed files, neighbors
  └─▶ AnthropicClient.run_review(…)
        ├─ first request: system (cached) + user (diff + files) + tools + thinking
        ├─ while response.stop_reason == "tool_use":
        │    ├─ dispatch tool_use blocks to repo_tools.execute(tool_name, input)
        │    └─ send tool_result blocks back
        └─ parse JSON content against output_config schema ─▶ AgentReview
```

### 4.3 Tool-use backing

Tools are backed by `GitHubClient.get_contents()` (already in the codebase). No cloning. Specifically:

- `read_file(path)` → `repo.get_contents(path, ref=pr.head.sha)` → decoded text.
- `glob(pattern)` → one-time fetch of recursive tree (`GET /repos/:o/:r/git/trees/:sha?recursive=1`), cached per review, filtered client-side with `fnmatch`.
- `grep(pattern, path_glob)` → `glob(path_glob)` → fetch matching files (lazy, with LRU cache) → run `re.finditer` in-process, return matches with line numbers.

Token- and quota-safety bounds:
- Per-tool per-agent max calls: configurable, default 20.
- Per-review GitHub API budget: configurable, default 200 requests.
- Per-file size cap fetched: 512 KB; larger files return a truncation notice.

Rationale for Contents API (not clone): webhook-primary deployment, Cloud Run tmpfs concerns, zero new infra, existing auth reuse, and GitHub's 5k/hr PAT and 15k/hr App rate limits are comfortably above our 200 req/review budget.

### 4.4 Workspace lifecycle

There is no local workspace. Per-review transient state (cached tree listing, tool-call counters, per-file content cache) lives in a new `ReviewSession` dataclass in `src/ai_reviewer/session.py`. This is **not** a model — it is a runtime object passed to the `AnthropicClient` and `ToolRegistry` constructors. The `models/` package is untouched, consistent with §2. `ReviewSession` is instantiated at the top of `review_pr()` and discarded when the function returns.

## 5. Detail: quality layers

### 5.1 Baseline prompt context

Built once per review, shared by all agents via prompt caching:

**System prompt blocks (marked `cache_control: ephemeral`):**
1. Agent role + behavioral rules (already in `base.py`).
2. Strict output JSON schema (see §5.5).
3. Project conventions bundle:
   - `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`
   - `.ai/rules/*.md`
   - `.cursor/rules/README.md` (if present — kept for backwards compat during transition)
4. Repo map:
   - Top-level directory listing (one hop)
   - Build manifests: `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`, `requirements.txt` (whichever exist)

**User message:**
1. PR metadata: title, body, author, linked issues, commit messages.
2. The unified diff.
3. Full contents of every changed file at `pr.head.sha`.
4. Heuristic neighbor files (§5.2).

When the combined context exceeds a configurable token budget (default 150K), we truncate in this priority order (lowest first): neighbors → conventions → changed-file contents. The diff is **never** truncated.

### 5.2 Neighbor-file heuristics (`context/neighbors.py`)

Goal: give the model the smallest set of non-changed files that materially improve its understanding.

Selection for each changed file:
1. **Sibling files** in the same directory (up to 5, preferring files with similar extensions).
2. **One-hop import graph:**
   - Files imported by the changed file (outbound edges).
   - Files that import the changed file (inbound edges — computed by grepping for the changed file's module path across the repo via `glob` + `grep`).

Language-specific import parsing:
- Python: `ast.parse` → `Import`, `ImportFrom` nodes.
- TypeScript/JavaScript: regex over `import … from "…"` and `require("…")` — no AST to keep dependency footprint small.
- Go, Rust, Java: regex over `import …` forms.
- Unknown languages: siblings only.

Per-file token cap: 8K. Total neighbors token cap: 40K. Truncated with a clear marker.

### 5.3 Tool use

Covered in detail in §4.3 (backing, bounds, language parsers). Per-agent enablement via `allow_tool_use` (default true); disabled agents are sent an empty `tools` parameter and their `tool_registry` argument is `None`. Tool descriptions passed to Claude emphasize: (a) prefer reading only files you specifically need, (b) batch queries via `grep` over reading many files, (c) cite file paths + line numbers in findings.

### 5.4 Extended thinking

Enabled via the `thinking` parameter on the Messages API.

| Agent | Model | Thinking | Budget |
|---|---|---|---|
| security-reviewer | `claude-opus-4-6` | enabled | 8192 |
| patterns-reviewer | `claude-opus-4-6` | enabled | 8192 |
| logic-reviewer | `claude-opus-4-6` | enabled | 8192 |
| performance-reviewer | `claude-sonnet-4-6` | disabled | — |
| style-reviewer | `claude-sonnet-4-6` | disabled | — |

`max_tokens` is set to `thinking.budget_tokens + expected_output_tokens + margin`, e.g., 16384 for thinking-enabled agents, 8192 otherwise.

`claude-mythos-preview` is **not** used. Confirmed unavailable outside Anthropic's Project Glasswing consortium (invitation-only, ~40 approved orgs), with no plan for general availability. The security agent stays on `claude-opus-4-6`.

### 5.5 Model mix

Diversity (previously achieved via Cursor's Claude+GPT mix) is now achieved via:
- **Model size:** Opus 4.6 vs. Sonnet 4.6.
- **System prompts:** distinct per agent (already in place).
- **Temperature:** 0.2 on reasoning-heavy agents, 0.3 on broad agents (keeps current defaults).
- **Focus areas:** distinct per agent (already in place).

### 5.6 Structured output

Every agent request sets:

```python
output_config = {
    "format": {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "required": ["findings", "summary"],
            "properties": {
                "findings": {
                    "type": "array",
                    "items": { "$ref": "#/$defs/Finding" }
                },
                "summary": { "type": "string" }
            },
            "$defs": {
                "Finding": {
                    "type": "object",
                    "required": ["file_path", "line_start", "severity", "category", "title", "description", "confidence"],
                    "properties": {
                        "file_path":    { "type": "string" },
                        "line_start":   { "type": "integer", "minimum": 1 },
                        "line_end":     { "type": ["integer", "null"], "minimum": 1 },
                        "severity":     { "enum": ["critical", "warning", "suggestion", "nitpick"] },
                        "category":     { "enum": ["security", "performance", "logic", "style", "architecture", "testing", "documentation"] },
                        "title":        { "type": "string" },
                        "description":  { "type": "string" },
                        "suggested_fix":{ "type": ["string", "null"] },
                        "confidence":   { "type": "number", "minimum": 0.0, "maximum": 1.0 }
                    }
                }
            }
        }
    }
}
```

This eliminates the regex-based JSON extraction in `_parse_json_response()` and removes markdown-fence failure modes.

### 5.7 Prompt caching

- `system` is constructed as an array of `TextBlockParam`s. The shared conventions + schema blocks are marked `cache_control: { type: "ephemeral", ttl: "5m" }`.
- 5-minute TTL is sufficient because all agents for one review run within a few minutes in parallel.
- Per-request cache-read/write token metrics are logged for tuning.

## 6. Configuration changes

### 6.1 New `config.py` shape

```python
@dataclass
class AnthropicApiConfig:
    api_key: str                                        # ${ANTHROPIC_API_KEY}
    base_url: str = "https://api.anthropic.com"
    timeout_seconds: int = 300
    max_retries: int = 3
    default_model: str = "claude-opus-4-6"
    enable_prompt_caching: bool = True
    max_combined_context_tokens: int = 150_000          # soft cap for §5.1 truncation
    per_file_max_bytes: int = 512 * 1024                # §4.3 tool-read cap
    per_review_github_request_budget: int = 200         # §4.3 quota counter

@dataclass
class AgentConfig:
    # existing fields preserved
    thinking_enabled: bool = False
    thinking_budget_tokens: int = 8192
    allow_tool_use: bool = True        # Claude may call repo tools
    max_tool_calls: int = 20
```

`Config.cursor: CursorApiConfig` becomes `Config.anthropic: AnthropicApiConfig`.

### 6.2 `config.example.yaml` / `.ai-reviewer.yaml`

```yaml
anthropic:
  api_key: ${ANTHROPIC_API_KEY}
  default_model: claude-opus-4-6
  timeout_seconds: 300
  enable_prompt_caching: true

agents:
  - name: security-reviewer
    model: claude-opus-4-6
    focus_areas: [security, authentication, data_validation]
    thinking_enabled: true
    thinking_budget_tokens: 8192
    allow_tool_use: true
    max_tool_calls: 20
    temperature: 0.2

  - name: performance-reviewer
    model: claude-sonnet-4-6
    focus_areas: [performance, complexity, resource_management]
    thinking_enabled: false
    allow_tool_use: true
    max_tool_calls: 10
    temperature: 0.3

  - name: patterns-reviewer
    model: claude-opus-4-6
    focus_areas: [consistency, patterns, architecture, breaking_changes]
    thinking_enabled: true
    thinking_budget_tokens: 8192
    allow_tool_use: true
    max_tool_calls: 30                # explores more
    temperature: 0.2
```

### 6.3 Environment / secrets

- New: `ANTHROPIC_API_KEY` (secret).
- Removed: `CURSOR_API_KEY` — deleted after cutover.
- `.github/workflows/ai-review.yaml` updates the env name.
- Cloud Run secret mount and `deploy-gcp.sh` updated.

## 7. Code contracts

### 7.1 `AnthropicClient` (new)

```python
class AnthropicClient:
    def __init__(self, config: AnthropicApiConfig) -> None: ...

    async def close(self) -> None: ...

    async def run_review(
        self,
        model: str,
        system_blocks: list[dict],         # TextBlockParam list w/ cache_control
        user_blocks: list[dict],           # TextBlockParam list: PR meta, diff, files, neighbors
        output_schema: dict,               # JSON schema for findings
        tool_registry: ToolRegistry | None,
        thinking_budget: int | None,
        max_tokens: int,
        temperature: float,
    ) -> AnthropicReviewResult: ...
```

`AnthropicReviewResult` carries: parsed findings JSON, raw final message, usage counts (input, output, cache_creation, cache_read), tool-call trace.

### 7.2 `ToolRegistry` (new)

```python
class ToolRegistry:
    def tool_specs(self) -> list[dict]: ...             # Messages API `tools` parameter
    async def execute(self, name: str, input: dict) -> str: ...
```

Implements `read_file`, `glob`, `grep` backed by `GitHubClient` + per-review cache + quota counters.

### 7.3 `ReviewAgent.review()` (rewritten)

Replaces the current `complete_json` call with a single `anthropic_client.run_review(...)`. Agent classes change only their `MODEL`, `THINKING_ENABLED`, `THINKING_BUDGET` class vars. No behavioral change visible to the orchestrator.

### 7.4 `review.py` changes

- Delete all three `async with CursorClient(...)` blocks and the `run_review_agent()` / `run_single_agent()` helpers that wrap the Cursor-specific flow.
- Replace with a single `async with AnthropicClient(anthropic_config) as client:` around the multi-agent orchestration.
- Build the shared `system_blocks` and `user_blocks` once via `context/` helpers and pass them to each agent's `review()` call.
- Cross-review round is preserved — it already uses `AgentReview` objects; only the underlying client changes.

## 8. Operational / deployment

- **Dockerfile:** no new binaries required (no git, no ripgrep — we're not cloning).
- **Cloud Run:** no sizing change required.
- **Secrets:** add `ANTHROPIC_API_KEY` to Secret Manager; delete `CURSOR_API_KEY` after cutover.
- **Observability:** log `input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, tool call count, and extended-thinking token usage per agent per review. Surface in existing Prometheus metrics path.
- **Backoff:** the official `anthropic` SDK handles 429 and 529 with retries and jitter out-of-the-box; configure `max_retries=3`.
- **Timeouts:** per-request 300s (accommodates thinking); global per-review wall clock stays at `orchestrator.timeout_seconds` default 120s per agent (thinking agents may need 300s — update default).

## 9. Testing strategy

1. **Unit tests (new):**
   - `anthropic_client.py` with mocked SDK: happy path, tool-use loop, structured-output parse, thinking handling, retry behavior.
   - `tools/repo_tools.py` with mocked `GitHubClient`: read/glob/grep, quota enforcement, size cap.
   - `context/neighbors.py` with fixture repos: Python AST path, TS regex path, unknown-language fallback.

2. **Unit tests (updated):**
   - `tests/test_agents.py`: replace Cursor mocks with Anthropic mocks; assert per-agent model/thinking selection.
   - `tests/test_convergence.py`: same.
   - `tests/test_cli.py`: updated config loading; env-var name change.

3. **Integration test:**
   - One end-to-end test (marked `@pytest.mark.integration`, off by default) that runs a real review against a pinned small public PR. Requires `ANTHROPIC_API_KEY` env; CI job gates on secret availability.

4. **Quality regression test (manual, documented):**
   - Run the new reviewer against 5 pinned internal PRs that were reviewed under the Cursor setup. Compare findings counts, severity distribution, and true-positive rate. Record results in `docs/` as a migration sign-off artifact.

## 10. Cutover plan

Phased so that each phase is independently mergeable and reversible.

- **Phase 1 — Client + infra.** Add `anthropic` SDK, add `AnthropicClient` + `ToolRegistry` + `context/neighbors.py`, unit tests. No wiring yet. Cursor path remains live.
- **Phase 2 — Agent rewrite.** Rewrite `base.py`, update subclasses. Tests updated.
- **Phase 3 — Config + wiring.** `AnthropicApiConfig`, YAML updates, update `cli.py`, `webhook.py`, `review.py` to use the new client. Delete `cursor_client.py` and `CursorApiConfig`. Docs updated.
- **Phase 4 — Deploy + observe.** Secret rotation, Cloud Run env update, run against 5 pinned PRs, sign off.

## 11. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Tool-use loop never terminates | low | Max tool-call budget; loop cap = 30 rounds; hard timeout per agent |
| GitHub rate-limit exhaustion | low | Per-review request budget (200); per-file size cap; tree cache |
| Structured-output rejection on edge cases | low | Retry once with `stop_reason: "max_tokens"` escalation; fall back to legacy JSON-in-text parse |
| Cost spike from thinking + tools | medium | Per-agent thinking toggle in YAML; per-agent tool-call cap; log cache-hit rate |
| Extended thinking tokens > max_tokens error | low | Validate at config load: `max_tokens > thinking_budget + 4096` |
| Prompt caching miss due to non-deterministic block ordering | low | Build system blocks in deterministic order; test with usage assertions |

## 12. Open items (to resolve during implementation)

- Exact token budgets per layer — will tune after first real-repo measurement.
- Whether to keep `include_codebase_context: true` as a YAML option or drop the field (tool use supersedes it). Current plan: drop it and document the migration mapping in `CHANGELOG`.
- Concrete list of neighbor-heuristic languages; current plan covers Python, TS/JS, Go, Rust, Java — confirm before implementation.

## 13. Blast radius summary

- **~20 files touched.** 4 new, 1 deleted, ~15 modified.
- **~900 lines of new code**, ~500 modified, ~350 deleted, ~200 lines of test updates.
- **Untouched:** orchestrator, aggregator, models, docs analyzer, most of github client, secret scanner, convergence/cross-review.

## 14. Acceptance criteria

- All existing unit tests pass with Anthropic mocks.
- Integration test against a real PR produces valid JSON findings.
- Quality regression pass (§9.4): aggregated finding set is ≥ parity with the Cursor baseline on 5 pinned PRs (no more than 10% reduction in true positives; false-positive rate no worse).
- No references to Cursor remain in code, config, docs, or CI (except historical notes in CHANGELOG).
- Cloud Run deploy succeeds; a webhook-triggered review on a test repo completes end-to-end and posts the expected PR comment.
