# Doc-Update Pipeline Redesign — Design Spec

**Date:** 2026-06-18
**Status:** Approved (working reference — not committed by request)
**Component:** `ai-code-reviewer` → `src/ai_reviewer/docs/` + `src/ai_reviewer/github/client.py`

## Problem

The current doc auto-update bot (`ai-reviewer update-docs`) produces shallow, often
cosmetic doc updates. Concrete failure: core PR #2792 (`emit op-events after the op-log
persists`) was a behavioral/ordering change (events buffered and flushed only after the
op-log durably persists; dropped on replay). The bot's resulting PR #2794 changed a
**single word** in `architecture/auto-follow.html` — a function rename
(`emit_auto_follow_set_if_enabled` → `build_auto_follow_set_if_enabled`) — and missed the
entire substance of the change.

Root causes (in `src/ai_reviewer/docs/analyzer.py`):
1. **Blind diff truncation** — `_MAX_DIFF_CHARS = 4000` head-chops the diff. PR #2792's diff
   is 90,467 chars; the behavioral hunk (`for event in pending_events { notify(event) }`)
   lives at char ~5,880, **past the cutoff**. The model never saw the real change.
2. **Minimal-patch HTML prompt** — `_DOC_DRAFT_SYSTEM_HTML` instructs "output ONLY the
   specific text that needs changing... leave everything else untouched", biasing toward the
   smallest literal edit (a rename) and structurally unable to *add* new explanation.
3. **No semantic understanding** — docs are matched by path heuristics + per-file
   "is this textually contradicted" prompting. There's no notion of "this PR introduces a
   new invariant the doc should describe."

## Goals

- The doc update must **reflect the real (behavioral/architectural) change**, not just renames.
- Support **two modes**: fixing an already-documented thing, and documenting a **new feature**
  that may have no doc home yet.
- New-feature reach includes **creating brand-new pages**, wired into the site's nav/search.
- Cost posture: **read the full PR once**, reuse a structured summary downstream (no blind
  per-doc diff re-sends).
- When the bot can't confidently capture the change, it must **flag (say so), not ship a
  cosmetic guess**.

## Non-goals

- Auto-merging doc PRs (unchanged — humans still review/merge).
- Rewriting whole untouched sections for stylistic reasons.
- Replacing the rule-based `DocAnalyzer` entirely (it remains a fallback/safety net).

## Approach (chosen): Staged pipeline — Understand → Route → Apply → Verify

```
run_doc_update()  [updater.py — rewritten orchestrator]
  1. UNDERSTAND  (understanding.py, new)
       summarize_pr_changes(title, body, commits, FULL diff) -> ChangeSummary
       - ONE call, capable model (Sonnet), full PR. "read once".
       - Giant-PR fallback: diff > cap -> map-reduce (per-file summarize, then merge).
  2. ROUTE  (router.py, new)
       route_changes(ChangeSummary, doc_config, doc_index) -> [DocAction]
       - doc_index = cheap parse of existing pages (titles + <h2> headings + nav.js NAV[])
       - deterministic where possible (source_to_docs_mapping + per-change doc_impact);
         one small LLM call only for ambiguous best-fit / "is there a home" decisions.
  3. APPLY  (analyzer.py, extended; concurrent w/ semaphore)
       apply_action(DocAction, ChangeSummary, current_doc) -> DocDraft
       - reuses ChangeSummary; does NOT re-send the giant diff per doc.
       - three edit modes, one applier.
  4. VERIFY  (verify.py, new; concurrent)
       verify_draft(change, before, after) -> Verdict{reflects_change, confidence}
       - low confidence -> drop draft, add to flagged[].
  -> create_doc_update_pr(...)  [github/client.py — extended: new files + nav wiring]
       PR body = Updated (with what_changed) + Flagged-for-human (with reason).
```

Rejected alternatives:
- **B — summary + per-doc single prompt (no router):** can't *decide to create a new page*
  (only ever sees existing docs one at a time). Fails the new-feature requirement.
- **C — agentic doc-writer:** non-deterministic, unbounded cost, hard to test; wrong for an
  unattended on-every-merge job.

## Data contracts (frozen dataclasses, validated like existing `DocSuggestion`/`DocDraft`)

```
Change:           kind ("new_feature"|"fix"|"rename"|"removal"|"behavior_change"),
                  title, what_changed, why, symbols[], files[], doc_impact
ChangeSummary:    pr_intent, changes: [Change]
DocAction:        change, action ("update_section"|"add_section"|"create_page"),
                  target_path, anchor|None, best_fit_reason
DocDraft (ext):   action, target_path, before_content, updated_content,
                  aux_edits: [FileEdit], error|flagged_reason
Verdict:          reflects_change: bool, confidence, notes
```

`Change.what_changed` + `doc_impact` are the heart: for #2792 they carry "events buffered,
flushed only after op-log persists, dropped on replay" — not "function renamed".

## Three edit modes (Apply), one applier interface

| Mode | Mechanism |
|---|---|
| `update_section` | FIND/REPLACE patch — **reuse** `_apply_html_patches` + whitespace-tolerant, ambiguity-refusing matcher (Fran's, already battle-tested). |
| `add_section` | Insert a new `<div class="card gX"><h2>…</h2>…</div>` at a resolved anchor; pick the **next** class in the `ga→gb→gc→gd→ga` cycle; don't disturb siblings. |
| `create_page` | Clone a top-level sibling page (e.g. `auto-follow.html`) as skeleton; model fills card bodies only; emit `nav.js` `NAV[]` aux edit + best-effort `index.html` link. |

All three are pure functions `(action, change, current_content) -> DocDraft` — unit-testable
without network.

## Site conventions the generator MUST obey (from reading core/architecture/)

Page skeleton (every page):
```html
<!DOCTYPE html><html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{Title} — Calimero Core Architecture</title>           <!-- always this suffix -->
<link rel="stylesheet" href="styles.css"></head>
<body><div class="main"><div class="content">
  <div class="breadcrumb"><a href="index.html">Home</a><span class="sep">/</span><span>{Page}</span></div>
  <h1>{Title} <em>{accent}</em></h1>
  <p class="page-subtitle">{one-liner}</p>
  <div class="card ga"><h2>…</h2>…</div>   <!-- content = sequence of cards, gX cycles mod 4 -->
</div></div>
<script src="nav.js"></script></body></html>
```

Fixed vocabulary (never invent markup): `.main>.content`, `.breadcrumb`, `.card`+`ga/gb/gc/gd`,
grids `g2/g3/g4`, `.stat`(`.v`/`.l`), `.step`(`.n`+h4), `.hero-card`, `<pre class="code">`,
`.tag`, `.legend`. Colors **only** via CSS vars (`--blue #58a6ff`, `--green #3fb950`,
`--violet`, `--cyan`, `--pink`, `--orange`, `--amber`, `--lime/accent #a5ff11`, `--red`).
Dark/light + TOC + sidebar + search + breadcrumb enhancement are all auto-built by `nav.js`;
pages just need clean `<h2>/<h3>` headings (≥3 → TOC appears).

### Wiring (corrected from nav.js)

| Edit | Required? | Detail |
|---|---|---|
| `nav.js` `NAV[]` entry | **Mandatory for new pages** | `{ label, href, dot }` under right `{section:…}` group (`For Builders`/`For Operators`/`Architecture Deep-Dive`/`Crate Internals`). This alone makes a page appear in sidebar **and** `⌘K` search (`buildIndex()` iterates `NAV`). |
| `index.html` link | Best-effort | `.hero-card` in Crate Index grid or cross-link in Key Concepts. Never blocks. |
| `dot` color | Mandatory | From palette, matching subsystem (governance/context → green `#10b981`). |

**Orphan guard:** a new page ships **only if the `nav.js` `NAV[]` insertion succeeds**.
Otherwise downgrade to `add_section` on best-fit page, or flag. Never commit an unlinked page.

## Error handling — every failure resolves to ship | flag | skip

| Stage | Failure | Outcome |
|---|---|---|
| Understand | API error (after 1 retry) | **Skip** run, log. No fallback to truncated-diff path. |
| Understand | "no doc-relevant changes" | **Skip** cleanly (valid, common). |
| Route | Ambiguous best-fit | Prefer `add_section` on top page; none above threshold & new-page disabled → **flag**. |
| Apply `update_section` | FIND not found / ambiguous | `_apply_html_patches` → None → **flag** (no partial write). |
| Apply `create_page` | `nav.js` anchor miss | **Downgrade** to `add_section`; else **flag**. Never orphan. |
| Verify | `reflects_change=false` or conf < threshold | **Flag** — drop draft, list in PR body with note. |
| Any | one doc fails | others proceed (per-doc isolation; `asyncio.gather` + try/except). |

PR body = **Updated** (files + per-file `what_changed`) + **Flagged for human review**
(stale docs the bot couldn't confidently fix, with reason). If everything flags → no code PR,
but post a comment on the source PR listing flagged docs (staleness never silently dropped).

Preserved guardrails: `has_open_doc_update_pr` dedupe, whitespace-tolerant ambiguity-refusing
matcher, `NO_UPDATE_NEEDED` sentinel, byte-safe writes.

## Config (`.ai-reviewer.yaml` → `doc_generation`), safe defaults

```yaml
doc_generation:
  understanding_model: claude-sonnet-4-6
  apply_model: claude-haiku-4-5-20251001
  verify_model: claude-haiku-4-5-20251001
  max_understanding_diff_chars: 250000      # full-PR read; map-reduce beyond
  allow_new_pages: true
  allow_new_sections: true
  verify_confidence_threshold: medium       # below -> flag, don't ship
  static_docs_dirs: [architecture/]
```
Remove `_MAX_DIFF_CHARS = 4000` per-doc chop — stage 3 consumes `ChangeSummary`, not raw diff.

## Testing

- **Understand:** schema-valid `ChangeSummary`; `kind` classification (rename-only vs
  added-behavior); map-reduce fallback above cap.
- **Route:** table-driven mapping → DocAction; new feature → `create_page`/`add_section`;
  ambiguous → top page.
- **Apply:** `update_section` (existing matcher tests); `add_section` (valid card at anchor,
  next cycle class, siblings intact); `create_page` (canonical head/title-suffix/breadcrumb/
  `nav.js` include; only established classes; produces `NAV[]` aux edit).
- **Verify:** `reflects_change=false` → flagged, never successful.
- **Wiring:** `NAV[]` insert under correct section, valid JS, length+1; orphan guard
  downgrade; HTML validity of generated output.
- **Regression #2792→#2794 (golden, deterministic/fake client):** feed real #2792 diff +
  real `auto-follow.html`; assert `ChangeSummary` has a `behavior_change` mentioning
  emit-after-persist/deferred-flush/drop-on-replay; assert draft **adds** Propagation content
  about the new invariant AND preserves the rest of the page outside the touched span.
- **Cost guard:** full diff sent to model **exactly once** regardless of N docs.
- **Integration (opt-in, live API, existing `integration.yaml`):** e2e run opening a PR with
  both an `update_section` and a `create_page`, gated behind the API key.
