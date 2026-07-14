# AI Code Reviewer – What Changed & What’s Next

## 2026-07-14 - Large-PR reliability (sharded review, shared caching, circuit breaker fix)

### Added
- **Sharded map-reduce review for large PRs**: gated on `additions + deletions > 1000` or
  `changed_files_count > 20`. The PR is split into up to 8 directory-coherent shards
  (~600 changed lines each, a directory group is never split across shards). Each agent
  reviews every shard in a fresh conversation with a reduced tool budget (6 calls) plus a
  deterministic PR-map block (`build_pr_map_block`) for whole-PR context. Per-shard findings
  are concatenated per agent, then one cross-shard pass per agent looks for issues that span
  shards (stale callers of a signature changed elsewhere, moved definitions, etc.). A single
  shard failure is recorded as a coverage-gap note in the summary; the agent as a whole only
  fails if every shard fails. Small/medium PRs are unaffected and keep the existing single-
  conversation path.

### Fixed
- **Cross-agent prompt cache sharing**: the per-agent role prompt moved from system block[0]
  to the end of the user turn (`ReviewAgent._build_user_blocks`). The cacheable prefix
  (`[system][shared user]`) is now byte-identical across agents, so the first agent's cache
  write is reused by the rest instead of each agent cache-writing its own ~80k-token prefix.
  The cache breakpoint sits on the last shared user block, with the role block appended after it.
- **Circuit breaker accuracy**: now measures the last request's true context (input +
  cache_read + cache_creation, from the API's own per-response usage) instead of cumulative
  uncached `input_tokens`, which excludes cache reads and never tripped once caching engaged.
- **Tool result size cap**: tool results fed back to the model are capped at
  `max_tool_result_bytes` (16KB default) with a "narrow your search" marker so an oversized
  match doesn't blow out context; the 512KB `per_file_max_bytes` cap on the GitHub fetch
  itself is unchanged.

---

## 2026-07-14 - Sonnet 5 reliability + recall fixes

### Migrated
- **Models**: `claude-sonnet-4-6` → `claude-sonnet-5` for review agents; style agent stays on
  `claude-haiku-4-5`.
- **Thinking**: `claude-sonnet-5` supports only adaptive thinking. security-reviewer and
  logic-reviewer now run with `thinking_enabled: true`, `max_tokens: 32000`, and
  `output_config.effort: medium` so thinking doesn't starve the findings JSON. patterns,
  performance, and style stay thinking-off.
- **Default agent order**: `DEFAULT_AGENT_ORDER` reordered to security-reviewer,
  logic-reviewer, patterns-reviewer, performance-reviewer, style-reviewer, so `--agents 3`
  with no repo config runs security + logic + patterns.

### Fixed
- **Tool-loop salvage**: the final tool round now drops tool access instead of letting the
  agent exit empty behind a "tool loop cap" marker, which used to discard the whole review.
- **Confidence floors**: lowered per-severity floors (critical 0.3 / warning 0.4 / suggestion
  0.5 / nitpick 0.6) restore recall lost to Sonnet 5's more conservative reporting, but only
  when cross-review will run (≥ 3 effective agents). When cross-review is skipped (1-2
  agents), floors fall back to the conservative set (0.5 / 0.6 / 0.7 / 0.8) since the
  cross-review precision gate isn't there to catch false positives.
- **Streaming + retries**: all Messages API calls now stream (`messages.stream` +
  `get_final_message`) to stop dropped/corrupted responses on long generations, and dropped
  API connections are retried with a widened webhook timeout.

---

## 2026-04 — Anthropic Messages API migration

### Migrated
- **LLM backend**: Cursor Background Agent API → Anthropic Messages API (official SDK).
- **Models**: `claude-4.5-opus-high-thinking` / `gpt-5.2` → `claude-sonnet-4-6` (security,
  performance, patterns, logic, auth); `claude-haiku-4-5-20251001` (style, doc generation).
  Extended thinking disabled on all agents — see `docs/optimization.md` for rationale.
- **Repo context**: Cursor's background-agent exploration replaced by Claude tool use
  (`read_file` / `glob` / `grep`) backed by the GitHub Contents API. No local cloning.
- **Quality stack**: Prompt caching on system blocks, JSON-schema structured output.
- **Config**: `cursor:` block replaced by `anthropic:`. New per-agent knobs:
  `thinking_enabled`, `thinking_budget_tokens`, `allow_tool_use`, `max_tool_calls`.

### Secrets
- Rename `CURSOR_API_KEY` → `ANTHROPIC_API_KEY` in GitHub Actions, Cloud Run, and local `.env`.

### Spec & plan
- Design: [`docs/superpowers/specs/2026-04-15-anthropic-messages-migration-design.md`](superpowers/specs/2026-04-15-anthropic-messages-migration-design.md)
- Implementation plan: [`docs/superpowers/plans/2026-04-16-anthropic-messages-migration.md`](superpowers/plans/2026-04-16-anthropic-messages-migration.md)

---

## Summary of changes (this round)

### P0 (already merged or in branch)
- **Inline-first body:** When posting inline comments, the PR-level review body is compact (e.g. "🔴 1 critical, 🟡 2 warnings. See inline comments."). No duplicate of each finding in the top-level comment.
- **Webhook body read once:** Read `request.body()` once; verify signature then `json.loads(body)` to avoid double-read on ASGI.
- **cursor_client:** Fixed `complete_json()` kwargs; use `get_running_loop()` instead of deprecated `get_event_loop().time()`.
- **Concise prompts + docs/CI-aware:** PR-type detection (docs / ci / code), context-aware instructions, and concise output rules (one sentence per finding, no nitpicks for docs/CI).

### Review standard (Google-style)
- **Base prompt:** "Review standard" block – favor approving when the CL improves code health; no perfectionism; use severity `nitpick` and prefix "Nit: " for optional points; comment on the code not the author; be courteous; explain why.
- **What to look for:** Short ordered checklist – Design → Functionality → Complexity → Tests → Naming, comments, style, consistency, documentation.
- **Output format:** Severity semantics (critical/warning = required; suggestion = consider; nitpick = optional, title prefix "Nit: "); summary should "include one positive if something is done well"; description "explain why it matters when helpful."

### Design principles (SOLID, DRY, KISS, etc.)
- **Base prompt:** New "Design principles" block – SOLID, DRY, KISS, YAGNI, Composition over Inheritance, Law of Demeter, Convention over Configuration. Instruction: only flag violations that meaningfully hurt maintainability or clarity; use "Nit:" for minor style.
- **Quality agent:** Explicit focus on SOLID, DRY, KISS, YAGNI, Composition over Inheritance, Law of Demeter, plus API design, error handling, maintainability, tests, documentation.

### Docs
- **review.py** module docstring and **models/findings.Severity** docstring document the review standard, design principles, and severity semantics (single source of truth in code).

---

## What else to improve (prioritized)

### High impact (do next)

| # | Improvement | Why |
|---|-------------|-----|
| 1 | **Cap or disable nitpicks** | Config option e.g. `max_nitpicks: 0` (or 1–2) per review so trivial nits don’t clutter. |
| 2 | **Single agent for small PRs** | When e.g. &lt;3 files and &lt;150 lines changed, use 1 agent to reduce duplicate findings and "going in circles." |
| 3 | **Apply config caps** | Enforce `max_total_findings` and `max_findings_per_file` from config when formatting/posting; trim by severity/consensus when over cap. |
| 4 | **LGTM-with-comments** | When only nitpicks/suggestions (no critical/warning), use COMMENT instead of REQUEST_CHANGES so the author isn’t blocked. |
| 5 | **Unify review path** | Either use config-driven agents + orchestrator + aggregator for PR flow, or remove/decommission the unused base-agent path and document that only Background Agent flow is supported. |

### Medium impact

| # | Improvement | Why |
|---|-------------|-----|
| 6 | **Config for PR flow** | Drive agent list and limits from `config.yaml` instead of hard-coded `AGENT_CONFIGS` and CLI `--agents`. |
| 7 | **Stable finding IDs** | Use hash(file_path, line, title) for finding IDs instead of order-dependent index. |
| 8 | **Inline comment line** | Use `line_end` when available for GitHub inline comments so they attach to the right line. |
| 9 | **Repository `.ai-reviewer.yaml`** | Load and merge repo-root config for ignore patterns, custom prompts, policy (or document "not implemented"). |
| 10 | **Retries for Anthropic API** | Use tenacity for Messages API calls with backoff (SDK handles 429/529 natively). |
| 11 | **Broad view first (prompt)** | Add one line: "First consider: does this change make sense? If not, say why and suggest an alternative." |

### Lower priority

| # | Improvement | Why |
|---|-------------|-----|
| 12 | **Consolidate aggregation** | Use a single aggregation implementation (e.g. only `review.aggregate_findings` or only `ReviewAggregator`) and reuse everywhere. |
| 13 | **Validate required env vars** | When config expands `${ANTHROPIC_API_KEY}` to empty, fail or warn instead of silent empty string. |
| 14 | **Version single source** | One source of truth for version (e.g. from pyproject or `importlib.metadata`). |
| 15 | **Magic numbers → config** | Diff/file size limits, line tolerance, max inline comments (10) as config or named constants. |
| 16 | **Tests** | More tests for formatter compact/delta, config validation, webhook signature, and full review flow with mocked Cursor/GitHub. |
| 17 | **README vs CLI** | Add `ai-reviewer review --diff/--commit` or update README to match actual commands. |
| 18 | **Remove or use sentence-transformers** | Dependency is unused; remove or implement embedding-based clustering behind config. |

---

## Quick reference: what’s in the prompt today

- **Review standard:** Favor approve when code health improves; Nit: for optional; comment on code not author; explain why.
- **What to look for:** Design → Functionality → Complexity → Tests → Naming, comments, style, consistency, docs.
- **Design principles:** SOLID, DRY, KISS, YAGNI, Composition over Inheritance, Law of Demeter, Convention over Configuration; only flag meaningful violations.
- **PR-type:** docs-only or CI-only instructions when applicable.
- **Output:** Severity semantics; one sentence per finding; summary with optional positive; max 5 findings per agent; only changed lines.
