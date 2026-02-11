# AI Code Reviewer – What Changed & What’s Next

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
- **REVIEW-STANDARD-ENHANCEMENTS.md:** Documents Google mapping, severity semantics, design principles table, and what’s implemented.

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
| 10 | **Retries for Cursor API** | Use tenacity for create_agent / get_agent with backoff. |
| 11 | **Broad view first (prompt)** | Add one line: "First consider: does this change make sense? If not, say why and suggest an alternative." |

### Lower priority

| # | Improvement | Why |
|---|-------------|-----|
| 12 | **Consolidate aggregation** | Use a single aggregation implementation (e.g. only `review.aggregate_findings` or only `ReviewAggregator`) and reuse everywhere. |
| 13 | **Validate required env vars** | When config expands `${CURSOR_API_KEY}` to empty, fail or warn instead of silent empty string. |
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
