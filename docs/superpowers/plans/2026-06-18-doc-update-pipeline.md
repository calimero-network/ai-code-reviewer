# Doc-Update Pipeline Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the doc-bot's blind-truncation, minimal-patch doc updater with a staged Understand → Route → Apply → Verify pipeline that reflects the *real* behavioral change in a PR, can add new sections and create new pages, and flags (rather than ships) low-confidence updates.

**Architecture:** One capable-model pass reads the full PR into a structured `ChangeSummary` (read once). A deterministic-plus-LLM router maps each change to an `update_section` / `add_section` / `create_page` action against an index of existing docs. An applier produces full-file writes (reusing the existing whitespace-tolerant FIND/REPLACE matcher for surgical edits, cloning a sibling page for new pages, splicing `nav.js`/`index.html`). A verify pass drops any draft that doesn't reflect its change. The orchestrator opens a PR listing what was updated and what was flagged for humans.

**Tech Stack:** Python 3.11, `anthropic` SDK (only via `AnthropicClient`), `PyGithub` (only via `GitHubClient`), `pytest` + `pytest-asyncio`, `ruff`.

## Global Constraints

- **I1 — single SDK importer:** never `import anthropic` outside `src/ai_reviewer/agents/anthropic_client.py`. All model calls go through `AnthropicClient`. (ruff TID251 enforces this.)
- **GitHub only via `GitHubClient`** (`src/ai_reviewer/github/client.py`).
- **No silent failures:** every failure resolves to ship | flag | skip; a flagged/failed doc is never committed as if it succeeded.
- **Never commit an orphan page:** a new page ships only if its `nav.js` `NAV[]` entry is inserted.
- **Reuse, don't reinvent:** the surgical-edit path reuses `_apply_html_patches` / `_whitespace_insensitive_span` from `docs/analyzer.py`. The commit path reuses `GitHubClient.create_doc_update_pr`'s create/update logic.
- **Test pattern:** patch `ai_reviewer.agents.anthropic_client.AnthropicClient`; mock `run_completion` as `AsyncMock`. Stages must open the client via `async with AnthropicClient(cfg) as client:`.
- **Models:** understanding default `claude-sonnet-4-6`; apply/verify default `claude-haiku-4-5-20251001`.
- All new dataclasses `from __future__ import annotations`, frozen where they are value objects.
- Run `ruff format` + `ruff check` before each commit.

---

## File Structure

- `src/ai_reviewer/docs/models.py` — **new** — shared dataclasses (`Change`, `ChangeSummary`, `DocAction`, `Verdict`, `FileWrite`), extended `DocDraft`, and `extract_json`.
- `src/ai_reviewer/docs/understanding.py` — **new** — stage 1: `summarize_pr_changes`.
- `src/ai_reviewer/docs/router.py` — **new** — stage 2: `build_doc_index`, `route_changes`.
- `src/ai_reviewer/docs/apply.py` — **new** — stage 3 edits: `apply_update_section`, `apply_add_section`, pure helpers `next_card_class`, `insert_section`.
- `src/ai_reviewer/docs/page_builder.py` — **new** — stage 3 new pages: `build_new_page`, `insert_nav_entry`, `insert_index_link`.
- `src/ai_reviewer/docs/verify.py` — **new** — stage 4: `verify_draft`.
- `src/ai_reviewer/docs/updater.py` — **rewrite** — orchestrate the 4 stages, build PR body, ship/flag/skip.
- `src/ai_reviewer/docs/analyzer.py` — **keep** — `_apply_html_patches`, `_whitespace_insensitive_span`, `is_architecture_impacting` reused; legacy `DocAnalyzer` stays as fallback.
- `src/ai_reviewer/config.py:128-148` — **modify** — extend `DocGenerationSettings` + loader.
- `src/ai_reviewer/github/client.py:1582` — **modify** — accept `list[FileWrite]`.
- Tests mirror each module under `tests/`.

---

### Task 1: Extend `DocGenerationSettings` config

**Files:**
- Modify: `src/ai_reviewer/config.py:128-148`
- Modify: the `doc_generation` loader block in `config.py` (find with `grep -n "doc_generation" src/ai_reviewer/config.py`)
- Test: `tests/test_config_doc_generation.py` (create)

**Interfaces:**
- Produces: `DocGenerationSettings` fields `understanding_model: str`, `apply_model: str`, `verify_model: str`, `max_understanding_diff_chars: int`, `allow_new_pages: bool`, `allow_new_sections: bool`, `verify_confidence_threshold: str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_doc_generation.py
from __future__ import annotations

from ai_reviewer.config import DocGenerationSettings


def test_new_defaults():
    s = DocGenerationSettings()
    assert s.understanding_model == "claude-sonnet-4-6"
    assert s.apply_model == "claude-haiku-4-5-20251001"
    assert s.verify_model == "claude-haiku-4-5-20251001"
    assert s.max_understanding_diff_chars == 250_000
    assert s.allow_new_pages is True
    assert s.allow_new_sections is True
    assert s.verify_confidence_threshold == "medium"


def test_apply_model_falls_back_to_legacy_model_field():
    # `model` stays as the legacy/back-compat apply model default.
    s = DocGenerationSettings()
    assert s.apply_model == s.model
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config_doc_generation.py -v`
Expected: FAIL with `AttributeError: 'DocGenerationSettings' object has no attribute 'understanding_model'`

- [ ] **Step 3: Add the fields**

In `src/ai_reviewer/config.py`, inside `@dataclass class DocGenerationSettings`, after `pr_draft: bool = True`:

```python
    # Stage models for the Understand → Route → Apply → Verify pipeline.
    understanding_model: str = "claude-sonnet-4-6"
    apply_model: str = "claude-haiku-4-5-20251001"
    verify_model: str = "claude-haiku-4-5-20251001"
    # Full-PR read budget for stage 1; above this, map-reduce per file.
    max_understanding_diff_chars: int = 250_000
    # Capability gates.
    allow_new_pages: bool = True
    allow_new_sections: bool = True
    # Below this verifier confidence, flag instead of ship: "low" | "medium" | "high".
    verify_confidence_threshold: str = "medium"
```

- [ ] **Step 4: Map the fields in the loader**

Find the loader (`grep -n "DocGenerationSettings(" src/ai_reviewer/config.py`). In the block that builds `DocGenerationSettings(...)` from the raw `doc_generation` dict, add (matching the existing `dg.get(...)` style):

```python
        understanding_model=dg.get("understanding_model", "claude-sonnet-4-6"),
        apply_model=dg.get("apply_model", "claude-haiku-4-5-20251001"),
        verify_model=dg.get("verify_model", "claude-haiku-4-5-20251001"),
        max_understanding_diff_chars=int(dg.get("max_understanding_diff_chars", 250_000)),
        allow_new_pages=bool(dg.get("allow_new_pages", True)),
        allow_new_sections=bool(dg.get("allow_new_sections", True)),
        verify_confidence_threshold=dg.get("verify_confidence_threshold", "medium"),
```

If the loader builds the dataclass with no explicit kwargs (relying on defaults), this step is a no-op — verify by reading the block. Add a test for the loader only if it explicitly maps fields.

- [ ] **Step 5: Run tests + commit**

Run: `pytest tests/test_config_doc_generation.py -v` → PASS
Run: `ruff format src/ai_reviewer/config.py tests/test_config_doc_generation.py && ruff check src/ai_reviewer/config.py`

```bash
git add src/ai_reviewer/config.py tests/test_config_doc_generation.py
git commit -m "feat(docs): add pipeline config fields to DocGenerationSettings"
```

---

### Task 2: Shared data contracts + JSON helper (`docs/models.py`)

**Files:**
- Create: `src/ai_reviewer/docs/models.py`
- Test: `tests/test_docs_models.py`

**Interfaces:**
- Produces:
  - `Change(kind: str, title: str, what_changed: str, why: str, symbols: list[str], files: list[str], doc_impact: str)` — frozen.
  - `ChangeSummary(pr_intent: str, changes: list[Change])`.
  - `DocAction(change: Change, action: str, target_path: str, anchor: str | None, best_fit_reason: str)` — `action ∈ {"update_section","add_section","create_page"}`.
  - `Verdict(reflects_change: bool, confidence: str, notes: str)` — `confidence ∈ {"low","medium","high"}`.
  - `FileWrite(path: str, content: str)` — frozen.
  - `DocDraft(action: str, target_path: str, updated_content: str, before_content: str | None = None, aux_edits: list[FileWrite] = [], change: Change | None = None, error: str | None = None, flagged_reason: str | None = None)`.
  - `extract_json(text: str) -> dict` — tolerant parser; raises `ValueError` on failure.
  - `CONFIDENCE_RANK: dict[str, int]` and `meets_threshold(confidence, threshold) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs_models.py
from __future__ import annotations

import pytest

from ai_reviewer.docs.models import (
    Change,
    ChangeSummary,
    DocAction,
    DocDraft,
    FileWrite,
    Verdict,
    extract_json,
    meets_threshold,
)


def test_change_is_frozen():
    c = Change("fix", "t", "what", "why", ["sym"], ["f.rs"], "impact")
    with pytest.raises(AttributeError):
        c.title = "x"  # type: ignore[misc]


def test_change_summary_holds_changes():
    c = Change("fix", "t", "w", "y", [], [], "i")
    cs = ChangeSummary(pr_intent="intent", changes=[c])
    assert cs.changes[0] is c


def test_docdraft_defaults():
    d = DocDraft(action="add_section", target_path="architecture/x.html", updated_content="<html>")
    assert d.aux_edits == []
    assert d.error is None and d.flagged_reason is None


def test_filewrite_frozen():
    fw = FileWrite(path="nav.js", content="x")
    with pytest.raises(AttributeError):
        fw.path = "y"  # type: ignore[misc]


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_prose_around_object():
    assert extract_json('Here is the result:\n{"a": [1,2]}\nDone.') == {"a": [1, 2]}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("not json at all")


def test_meets_threshold():
    assert meets_threshold("high", "medium")
    assert meets_threshold("medium", "medium")
    assert not meets_threshold("low", "medium")


def test_docaction_fields():
    c = Change("new_feature", "t", "w", "y", [], [], "i")
    a = DocAction(change=c, action="create_page", target_path="architecture/x.html",
                  anchor=None, best_fit_reason="no existing home")
    assert a.action == "create_page"


def test_verdict_fields():
    v = Verdict(reflects_change=False, confidence="low", notes="missed the invariant")
    assert v.reflects_change is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_reviewer.docs.models'`

- [ ] **Step 3: Implement `docs/models.py`**

```python
# src/ai_reviewer/docs/models.py
"""Shared data contracts for the doc-update pipeline (Understand → Route → Apply → Verify)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

CONFIDENCE_RANK: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def meets_threshold(confidence: str, threshold: str) -> bool:
    """True if *confidence* is at least *threshold* on the low<medium<high scale."""
    return CONFIDENCE_RANK.get(confidence, 0) >= CONFIDENCE_RANK.get(threshold, 1)


@dataclass(frozen=True)
class Change:
    kind: str  # "new_feature" | "fix" | "rename" | "removal" | "behavior_change"
    title: str
    what_changed: str
    why: str
    symbols: list[str]
    files: list[str]
    doc_impact: str


@dataclass
class ChangeSummary:
    pr_intent: str
    changes: list[Change]


@dataclass
class DocAction:
    change: Change
    action: str  # "update_section" | "add_section" | "create_page"
    target_path: str
    anchor: str | None = None
    best_fit_reason: str = ""


@dataclass(frozen=True)
class FileWrite:
    path: str
    content: str


@dataclass
class DocDraft:
    """A concrete doc edit ready to commit (or flagged/failed)."""

    action: str
    target_path: str
    updated_content: str
    before_content: str | None = None
    aux_edits: list[FileWrite] = field(default_factory=list)
    change: Change | None = None
    error: str | None = None
    flagged_reason: str | None = None


@dataclass
class Verdict:
    reflects_change: bool
    confidence: str  # "low" | "medium" | "high"
    notes: str


def extract_json(text: str) -> dict:
    """Parse a JSON object from a model response, tolerating fences/prose.

    Raises ValueError if no JSON object can be parsed.
    """
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    obj = re.search(r"\{[\s\S]*\}", s)
    if obj:
        try:
            return json.loads(obj.group(0))
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not parse JSON object: {exc}") from exc
    raise ValueError("no JSON object found in response")
```

- [ ] **Step 4: Run tests + ruff**

Run: `pytest tests/test_docs_models.py -v` → PASS
Run: `ruff format src/ai_reviewer/docs/models.py tests/test_docs_models.py && ruff check src/ai_reviewer/docs/models.py`

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/docs/models.py tests/test_docs_models.py
git commit -m "feat(docs): add pipeline data contracts and JSON helper"
```

---

### Task 3: Understand stage (`docs/understanding.py`)

**Files:**
- Create: `src/ai_reviewer/docs/understanding.py`
- Test: `tests/test_docs_understanding.py`

**Interfaces:**
- Consumes: `AnthropicApiConfig`, `ChangeSummary`, `Change`, `extract_json` (Task 2).
- Produces: `async def summarize_pr_changes(*, pr_title: str, pr_body: str, commit_messages: list[str], diff: str, anthropic_cfg, model: str, max_diff_chars: int = 250_000) -> ChangeSummary`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs_understanding.py
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.understanding import summarize_pr_changes

_SUMMARY_JSON = json.dumps(
    {
        "pr_intent": "Emit op-events only after the op-log persists.",
        "changes": [
            {
                "kind": "behavior_change",
                "title": "Defer op-event emission until after op-log append",
                "what_changed": "Events are buffered and flushed only after the op-log entry "
                "is durably appended; dropped on replay of an already-logged op.",
                "why": "Avoid double-firing on re-gossip/DAG replay.",
                "symbols": ["build_auto_follow_set_if_enabled"],
                "files": ["crates/governance-store/src/lib.rs"],
                "doc_impact": "Propagation section must state emit-after-persist + drop-on-replay.",
            }
        ],
    }
)


@pytest.mark.asyncio
async def test_summarize_returns_parsed_changes():
    cfg = AnthropicApiConfig(api_key="sk-test")
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=_SUMMARY_JSON)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        cs = await summarize_pr_changes(
            pr_title="fix(governance-store): emit op-events after the op-log persists",
            pr_body="Closes #2770",
            commit_messages=["emit after persist"],
            diff="diff --git a/x b/x\n+stuff",
            anthropic_cfg=cfg,
            model="claude-sonnet-4-6",
        )

    assert cs.pr_intent.startswith("Emit op-events")
    assert len(cs.changes) == 1
    assert cs.changes[0].kind == "behavior_change"
    assert "emit-after-persist" in cs.changes[0].doc_impact


@pytest.mark.asyncio
async def test_full_diff_sent_once_under_cap():
    """Cost guard: a diff under the cap is summarized in exactly one model call."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=_SUMMARY_JSON)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        await summarize_pr_changes(
            pr_title="t", pr_body="", commit_messages=[], diff="small diff",
            anthropic_cfg=cfg, model="m", max_diff_chars=10_000,
        )
    assert inst.run_completion.call_count == 1


@pytest.mark.asyncio
async def test_map_reduce_over_cap_summarizes_per_file_then_merges():
    """A diff over the cap triggers per-file summarize + one merge call."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    big_file_a = "diff --git a/a.rs b/a.rs\n" + ("+x\n" * 200)
    big_file_b = "diff --git a/b.rs b/b.rs\n" + ("+y\n" * 200)
    diff = big_file_a + big_file_b
    per_file = json.dumps({"changes": [{"kind": "fix", "title": "t", "what_changed": "w",
                                        "why": "y", "symbols": [], "files": [], "doc_impact": "i"}]})
    merged = _SUMMARY_JSON
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(side_effect=[per_file, per_file, merged])
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        cs = await summarize_pr_changes(
            pr_title="t", pr_body="", commit_messages=[], diff=diff,
            anthropic_cfg=cfg, model="m", max_diff_chars=100,
        )
    # 2 per-file calls + 1 merge call
    assert inst.run_completion.call_count == 3
    assert cs.changes[0].kind == "behavior_change"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_understanding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_reviewer.docs.understanding'`

- [ ] **Step 3: Implement `docs/understanding.py`**

```python
# src/ai_reviewer/docs/understanding.py
"""Stage 1 — read the full PR once into a structured ChangeSummary."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ai_reviewer.docs.models import Change, ChangeSummary, extract_json

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a software historian. Given a pull request (title, body, commit messages, and a
unified diff), describe the SUBSTANTIVE changes — what behavior/architecture changed and why —
NOT a line-by-line restatement. A function rename that reflects a behavioral change must be
reported as the behavioral change, with the rename mentioned as a detail.

Return ONLY a JSON object, no prose, with this exact shape:
{
  "pr_intent": "one or two sentences",
  "changes": [
    {
      "kind": "new_feature | fix | rename | removal | behavior_change",
      "title": "short imperative title",
      "what_changed": "the substance, in prose",
      "why": "rationale if known, else empty string",
      "symbols": ["affected function/type names"],
      "files": ["paths"],
      "doc_impact": "what documentation should now say (or empty if none)"
    }
  ]
}
If nothing is documentation-relevant, return {"pr_intent": "...", "changes": []}."""

_MERGE_SYSTEM = """\
You are merging several partial change-lists from one pull request into a single coherent
summary. Deduplicate, group related edits, and drop noise. Return ONLY the same JSON object
shape: {"pr_intent": "...", "changes": [...]} as previously specified."""


def _change_from_dict(d: dict) -> Change:
    return Change(
        kind=str(d.get("kind", "fix")),
        title=str(d.get("title", "")),
        what_changed=str(d.get("what_changed", "")),
        why=str(d.get("why", "")),
        symbols=[str(s) for s in d.get("symbols", [])],
        files=[str(f) for f in d.get("files", [])],
        doc_impact=str(d.get("doc_impact", "")),
    )


def _summary_from_dict(d: dict) -> ChangeSummary:
    return ChangeSummary(
        pr_intent=str(d.get("pr_intent", "")),
        changes=[_change_from_dict(c) for c in d.get("changes", [])],
    )


def _user_prompt(pr_title: str, pr_body: str, commit_messages: list[str], diff: str) -> str:
    commits = "\n".join(f"- {m}" for m in commit_messages)
    return (
        f"## PR Title\n{pr_title}\n\n"
        f"## PR Body\n{pr_body}\n\n"
        f"## Commit Messages\n{commits}\n\n"
        f"## Diff\n{diff}\n"
    )


def _split_diff_by_file(diff: str) -> list[str]:
    """Split a unified diff into per-file chunks on `diff --git` boundaries."""
    parts = diff.split("\ndiff --git ")
    if len(parts) == 1:
        return [diff]
    chunks = [parts[0]] if parts[0].startswith("diff --git ") else []
    chunks.extend("diff --git " + p for p in parts[1:])
    return [c for c in chunks if c.strip()]


async def summarize_pr_changes(
    *,
    pr_title: str,
    pr_body: str,
    commit_messages: list[str],
    diff: str,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
    max_diff_chars: int = 250_000,
) -> ChangeSummary:
    """Produce a ChangeSummary from the full PR. Map-reduces when the diff exceeds the cap."""
    from ai_reviewer.agents.anthropic_client import AnthropicClient  # local: avoid circular

    async with AnthropicClient(anthropic_cfg) as client:
        if len(diff) <= max_diff_chars:
            raw = await client.run_completion(
                model=model,
                system=_SYSTEM,
                user=_user_prompt(pr_title, pr_body, commit_messages, diff),
                max_tokens=4096,
            )
            return _summary_from_dict(extract_json(raw))

        # Map-reduce: summarize each file chunk (truncated to the cap), then merge.
        logger.info("Diff %d chars exceeds cap %d — map-reduce", len(diff), max_diff_chars)
        partials: list[dict] = []
        for chunk in _split_diff_by_file(diff):
            raw = await client.run_completion(
                model=model,
                system=_SYSTEM,
                user=_user_prompt(pr_title, pr_body, commit_messages, chunk[:max_diff_chars]),
                max_tokens=2048,
            )
            try:
                partials.append(extract_json(raw))
            except ValueError:
                logger.warning("Skipping unparseable partial summary for a diff chunk")
        merged_input = (
            f"## PR Title\n{pr_title}\n\n## Partial change-lists (JSON)\n"
            + "\n".join(str(p) for p in partials)
        )
        raw = await client.run_completion(
            model=model, system=_MERGE_SYSTEM, user=merged_input, max_tokens=4096
        )
        return _summary_from_dict(extract_json(raw))
```

- [ ] **Step 4: Run tests + ruff**

Run: `pytest tests/test_docs_understanding.py -v` → PASS
Run: `ruff format src/ai_reviewer/docs/understanding.py tests/test_docs_understanding.py && ruff check src/ai_reviewer/docs/understanding.py`

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/docs/understanding.py tests/test_docs_understanding.py
git commit -m "feat(docs): add stage-1 PR understanding (read-once ChangeSummary)"
```

---

### Task 4: Route stage (`docs/router.py`)

**Files:**
- Create: `src/ai_reviewer/docs/router.py`
- Test: `tests/test_docs_router.py`

**Interfaces:**
- Consumes: `ChangeSummary`, `Change`, `DocAction` (Task 2).
- Produces:
  - `build_doc_index(existing_paths: list[str]) -> list[str]` — filters to the doc/static dirs (pure; deterministic).
  - `async def route_changes(*, summary: ChangeSummary, source_to_docs_mapping: dict[str, list[str]], changed_paths: list[str], doc_index: list[str], allow_new_pages: bool, allow_new_sections: bool, anthropic_cfg, model: str) -> list[DocAction]`.
- Routing rules (deterministic first):
  1. For each change, if any `changed_paths` matches a `source_to_docs_mapping` glob whose target is in `doc_index` → `update_section` on that target.
  2. Else ask the model to pick best-fit page from `doc_index`, or propose a new page; honor `allow_new_*` gates. New-page proposals when `allow_new_pages` is False downgrade to `add_section` on best-fit; if no best-fit, the change is dropped from routing (later surfaced as a flag by the orchestrator).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs_router.py
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.models import Change, ChangeSummary
from ai_reviewer.docs.router import build_doc_index, route_changes


def _summary(kind: str, files: list[str]) -> ChangeSummary:
    return ChangeSummary(
        pr_intent="x",
        changes=[Change(kind, "t", "w", "y", [], files, "impact")],
    )


def test_build_doc_index_filters_to_doc_dirs():
    idx = build_doc_index(["architecture/auto-follow.html", "src/lib.rs", "README.md"])
    assert "architecture/auto-follow.html" in idx
    assert "src/lib.rs" not in idx


@pytest.mark.asyncio
async def test_mapping_hit_routes_update_section_no_model_call():
    cfg = AnthropicApiConfig(api_key="sk-test")
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        actions = await route_changes(
            summary=_summary("fix", ["crates/governance-store/src/lib.rs"]),
            source_to_docs_mapping={"crates/governance-store/**": ["architecture/auto-follow.html"]},
            changed_paths=["crates/governance-store/src/lib.rs"],
            doc_index=["architecture/auto-follow.html"],
            allow_new_pages=True,
            allow_new_sections=True,
            anthropic_cfg=cfg,
            model="m",
        )
    assert len(actions) == 1
    assert actions[0].action == "update_section"
    assert actions[0].target_path == "architecture/auto-follow.html"
    inst.run_completion.assert_not_called()


@pytest.mark.asyncio
async def test_new_feature_routes_create_page_when_allowed():
    cfg = AnthropicApiConfig(api_key="sk-test")
    decision = json.dumps({"action": "create_page", "target_path": "architecture/widgets.html",
                           "anchor": None, "best_fit_reason": "no existing widget page"})
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=decision)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        actions = await route_changes(
            summary=_summary("new_feature", ["crates/widgets/src/lib.rs"]),
            source_to_docs_mapping={},
            changed_paths=["crates/widgets/src/lib.rs"],
            doc_index=["architecture/auto-follow.html"],
            allow_new_pages=True,
            allow_new_sections=True,
            anthropic_cfg=cfg,
            model="m",
        )
    assert actions[0].action == "create_page"
    assert actions[0].target_path == "architecture/widgets.html"


@pytest.mark.asyncio
async def test_create_page_downgrades_to_add_section_when_pages_disabled():
    cfg = AnthropicApiConfig(api_key="sk-test")
    decision = json.dumps({"action": "create_page", "target_path": "architecture/widgets.html",
                           "anchor": None, "best_fit_reason": "x", "best_fit_existing":
                           "architecture/auto-follow.html"})
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=decision)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        actions = await route_changes(
            summary=_summary("new_feature", ["crates/widgets/src/lib.rs"]),
            source_to_docs_mapping={},
            changed_paths=["crates/widgets/src/lib.rs"],
            doc_index=["architecture/auto-follow.html"],
            allow_new_pages=False,
            allow_new_sections=True,
            anthropic_cfg=cfg,
            model="m",
        )
    assert actions[0].action == "add_section"
    assert actions[0].target_path == "architecture/auto-follow.html"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_reviewer.docs.router'`

- [ ] **Step 3: Implement `docs/router.py`**

```python
# src/ai_reviewer/docs/router.py
"""Stage 2 — map each change to a doc action (update_section | add_section | create_page)."""

from __future__ import annotations

import fnmatch
import logging
from typing import TYPE_CHECKING

from ai_reviewer.docs.models import Change, ChangeSummary, DocAction, extract_json

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig

logger = logging.getLogger(__name__)

_DOC_DIR_PREFIXES = ("architecture/", "docs/", "docs-static/", "doc/")

_ROUTE_SYSTEM = """\
You route a code change to the single best documentation action.
You are given the change and a list of existing doc pages (with titles inferred from paths).
Decide ONE of:
- "update_section": an existing page already documents this; we'll edit it. (give target_path from the list)
- "add_section": an existing page is the right home but lacks a section for this. (give target_path from the list)
- "create_page": nothing fits; a new page is warranted. (give a new target_path like "architecture/<slug>.html")
Always also include "best_fit_existing": the closest existing page path from the list (or "").
Return ONLY JSON: {"action": "...", "target_path": "...", "anchor": null,
"best_fit_reason": "...", "best_fit_existing": "..."}"""


def build_doc_index(existing_paths: list[str]) -> list[str]:
    """Existing doc pages only (under known doc dirs)."""
    return [p for p in existing_paths if p.startswith(_DOC_DIR_PREFIXES) and p.endswith(".html")]


def _mapping_target(
    change: Change,
    mapping: dict[str, list[str]],
    changed_paths: list[str],
    doc_index: list[str],
) -> str | None:
    for glob_pattern, targets in mapping.items():
        if any(fnmatch.fnmatch(p, glob_pattern) for p in changed_paths):
            for t in targets:
                if t in doc_index:
                    return t
    return None


async def route_changes(
    *,
    summary: ChangeSummary,
    source_to_docs_mapping: dict[str, list[str]],
    changed_paths: list[str],
    doc_index: list[str],
    allow_new_pages: bool,
    allow_new_sections: bool,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
) -> list[DocAction]:
    from ai_reviewer.agents.anthropic_client import AnthropicClient  # local: avoid circular

    actions: list[DocAction] = []
    needs_model: list[Change] = []

    for change in summary.changes:
        target = _mapping_target(change, source_to_docs_mapping, changed_paths, doc_index)
        if target is not None:
            actions.append(
                DocAction(change=change, action="update_section", target_path=target,
                          best_fit_reason="source_to_docs_mapping")
            )
        else:
            needs_model.append(change)

    if not needs_model:
        return actions

    index_listing = "\n".join(f"- {p}" for p in doc_index) or "(no existing doc pages)"
    async with AnthropicClient(anthropic_cfg) as client:
        for change in needs_model:
            user = (
                f"## Change\nkind: {change.kind}\ntitle: {change.title}\n"
                f"what_changed: {change.what_changed}\ndoc_impact: {change.doc_impact}\n\n"
                f"## Existing doc pages\n{index_listing}\n"
            )
            try:
                raw = await client.run_completion(
                    model=model, system=_ROUTE_SYSTEM, user=user, max_tokens=512
                )
                d = extract_json(raw)
            except Exception as exc:  # noqa: BLE001 — extract_json ValueError or any client error
                logger.warning("Routing failed for %r: %s", change.title, exc)
                continue

            action = str(d.get("action", "add_section"))
            target_path = str(d.get("target_path", ""))
            best_fit = str(d.get("best_fit_existing", "")) or (doc_index[0] if doc_index else "")

            if action == "create_page" and not allow_new_pages:
                if allow_new_sections and best_fit:
                    action, target_path = "add_section", best_fit
                else:
                    logger.info("Dropping change %r: new pages disabled, no best-fit", change.title)
                    continue
            if action == "add_section" and not allow_new_sections:
                action = "update_section"
            if action in ("update_section", "add_section") and not target_path:
                target_path = best_fit
                if not target_path:
                    continue

            actions.append(
                DocAction(change=change, action=action, target_path=target_path,
                          anchor=d.get("anchor"), best_fit_reason=str(d.get("best_fit_reason", "")))
            )
    return actions
```

- [ ] **Step 4: Run tests + ruff**

Run: `pytest tests/test_docs_router.py -v` → PASS
Run: `ruff format src/ai_reviewer/docs/router.py tests/test_docs_router.py && ruff check src/ai_reviewer/docs/router.py`

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/docs/router.py tests/test_docs_router.py
git commit -m "feat(docs): add stage-2 change router (update/add/create routing)"
```

---

### Task 5: Apply — `update_section` + `add_section` (`docs/apply.py`)

**Files:**
- Create: `src/ai_reviewer/docs/apply.py`
- Test: `tests/test_docs_apply.py`

**Interfaces:**
- Consumes: `DocAction`, `Change`, `DocDraft` (Task 2); `_apply_html_patches` (`docs/analyzer.py`).
- Produces:
  - `next_card_class(html: str) -> str` — returns next of `ga→gb→gc→gd→ga` after the last `card gX` in *html* (default `ga`).
  - `insert_section(html: str, section_html: str) -> str | None` — inserts before the closing of the last top-level `</div>\n</div>` content wrapper (anchor: last `\n</div>\n</div>\n<script src="nav.js">`). Returns None if the anchor isn't found.
  - `async def apply_update_section(action, current_content, change, anthropic_cfg, model) -> DocDraft`.
  - `async def apply_add_section(action, current_content, change, anthropic_cfg, model) -> DocDraft`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs_apply.py
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.apply import (
    apply_add_section,
    apply_update_section,
    insert_section,
    next_card_class,
)
from ai_reviewer.docs.models import Change, DocAction

_PAGE = (
    '<body>\n<div class="main">\n<div class="content">\n'
    '<div class="card ga"><h2>One</h2></div>\n'
    '<div class="card gb"><h2>Two</h2></div>\n'
    "</div>\n</div>\n"
    '<script src="nav.js"></script>\n</body>'
)


def test_next_card_class_cycles():
    assert next_card_class('<div class="card gb">') == "gc"
    assert next_card_class('<div class="card gd">') == "ga"
    assert next_card_class("<div>no cards</div>") == "ga"


def test_insert_section_places_before_content_close():
    out = insert_section(_PAGE, '<div class="card gc"><h2>Three</h2></div>')
    assert out is not None
    assert "Three" in out
    # Inserted inside .content, before its close and the nav script.
    assert out.index("Three") < out.index('<script src="nav.js">')
    assert out.index("Two") < out.index("Three")


def test_insert_section_anchor_missing_returns_none():
    assert insert_section("<html>no wrapper</html>", "<div>x</div>") is None


@pytest.mark.asyncio
async def test_apply_update_section_uses_find_replace():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("rename", "t", "w", "y", ["build_x"], [], "i")
    action = DocAction(change=change, action="update_section", target_path="architecture/x.html")
    current = "<p>old emit_x text</p>"
    patch_resp = "<<<FIND\n<p>old emit_x text</p>\nFIND>>>\n<<<REPLACE\n<p>new build_x text</p>\nREPLACE>>>"
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=patch_resp)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_update_section(action, current, change, cfg, "m")
    assert draft.error is None
    assert draft.updated_content == "<p>new build_x text</p>"
    assert draft.before_content == current


@pytest.mark.asyncio
async def test_apply_update_section_bad_patch_flags():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("fix", "t", "w", "y", [], [], "i")
    action = DocAction(change=change, action="update_section", target_path="architecture/x.html")
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value="<<<FIND\nNOT PRESENT\nFIND>>>\n<<<REPLACE\nx\nREPLACE>>>")
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_update_section(action, "<p>actual</p>", change, cfg, "m")
    assert draft.updated_content == ""
    assert draft.error is not None


@pytest.mark.asyncio
async def test_apply_add_section_inserts_card():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("new_feature", "Widgets", "adds widgets", "y", [], [], "document widgets")
    action = DocAction(change=change, action="add_section", target_path="architecture/x.html")
    section = '<div class="card gc"><h2>Widgets</h2><p>new</p></div>'
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=section)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_add_section(action, _PAGE, change, cfg, "m")
    assert draft.error is None
    assert "Widgets" in draft.updated_content
    assert draft.updated_content.index("Widgets") < draft.updated_content.index('<script src="nav.js">')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_apply.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_reviewer.docs.apply'`

- [ ] **Step 3: Implement `docs/apply.py`**

```python
# src/ai_reviewer/docs/apply.py
"""Stage 3 (existing pages) — surgical update_section and additive add_section edits."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ai_reviewer.docs.analyzer import _apply_html_patches
from ai_reviewer.docs.models import Change, DocAction, DocDraft

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig

_CARD_CYCLE = ["ga", "gb", "gc", "gd"]

# Insert a new section just before the .content/.main close that precedes the nav script.
_CONTENT_CLOSE_RE = re.compile(r"(\n</div>\s*\n</div>\s*\n<script src=\"nav\.js\")")

_UPDATE_SYSTEM = """\
You are updating an existing HTML documentation page after a code change.
Output ONLY FIND/REPLACE blocks — do not return the whole file. Make the page reflect the
change described, including ADDING a sentence/bullet where the change introduces something new.
Format (repeatable):
<<<FIND
exact text copied verbatim from the page (enough to be unique)
FIND>>>
<<<REPLACE
replacement text
REPLACE>>>
If no change is needed, output exactly: NO_UPDATE_NEEDED"""

_ADD_SECTION_SYSTEM = """\
You are adding ONE new section to an existing HTML documentation page.
Output ONLY a single HTML block of the form:
<div class="card {card_class}"><h2>Title</h2> ...content... </div>
Use ONLY these constructs: <h2>/<h3>, <p>, <ul>/<ol>/<li>, <code>, <pre class="code">,
<strong>, <em>. Do NOT invent CSS classes. No commentary before or after the block."""


def next_card_class(html: str) -> str:
    matches = re.findall(r'class="card (g[abcd])"', html)
    if not matches:
        return "ga"
    last = matches[-1]
    return _CARD_CYCLE[(_CARD_CYCLE.index(last) + 1) % len(_CARD_CYCLE)]


def insert_section(html: str, section_html: str) -> str | None:
    """Insert *section_html* just before the .content wrapper closes. None if no anchor."""
    m = _CONTENT_CLOSE_RE.search(html)
    if not m:
        return None
    insert_at = m.start(1)
    return html[:insert_at] + "\n" + section_html + html[insert_at:]


async def apply_update_section(
    action: DocAction,
    current_content: str,
    change: Change,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
) -> DocDraft:
    from ai_reviewer.agents.anthropic_client import AnthropicClient

    user = (
        f"## Page: {action.target_path}\n\n{current_content}\n\n"
        f"## Change to reflect\ntitle: {change.title}\nwhat_changed: {change.what_changed}\n"
        f"doc_impact: {change.doc_impact}\nsymbols: {', '.join(change.symbols)}\n"
    )
    async with AnthropicClient(anthropic_cfg) as client:
        try:
            raw = (await client.run_completion(
                model=model, system=_UPDATE_SYSTEM, user=user, max_tokens=8192
            )).strip()
        except Exception as exc:  # noqa: BLE001
            return DocDraft(action="update_section", target_path=action.target_path,
                            updated_content="", change=change, error=str(exc))

    patched = _apply_html_patches(current_content, raw)
    if patched is None:
        return DocDraft(
            action="update_section", target_path=action.target_path, updated_content="",
            change=change, before_content=current_content,
            error="could not apply HTML patches (FIND not found or malformed)",
        )
    return DocDraft(action="update_section", target_path=action.target_path,
                    updated_content=patched, before_content=current_content, change=change)


async def apply_add_section(
    action: DocAction,
    current_content: str,
    change: Change,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
) -> DocDraft:
    from ai_reviewer.agents.anthropic_client import AnthropicClient

    card_class = next_card_class(current_content)
    system = _ADD_SECTION_SYSTEM.replace("{card_class}", card_class)
    user = (
        f"## Page: {action.target_path}\n\n{current_content}\n\n"
        f"## New thing to document\ntitle: {change.title}\nwhat_changed: {change.what_changed}\n"
        f"why: {change.why}\ndoc_impact: {change.doc_impact}\n"
    )
    async with AnthropicClient(anthropic_cfg) as client:
        try:
            section = (await client.run_completion(
                model=model, system=system, user=user, max_tokens=4096
            )).strip()
        except Exception as exc:  # noqa: BLE001
            return DocDraft(action="add_section", target_path=action.target_path,
                            updated_content="", change=change, error=str(exc))

    merged = insert_section(current_content, section)
    if merged is None:
        return DocDraft(
            action="add_section", target_path=action.target_path, updated_content="",
            change=change, before_content=current_content,
            error="could not locate content-wrapper anchor for section insertion",
        )
    return DocDraft(action="add_section", target_path=action.target_path,
                    updated_content=merged, before_content=current_content, change=change)
```

- [ ] **Step 4: Run tests + ruff**

Run: `pytest tests/test_docs_apply.py -v` → PASS
Run: `ruff format src/ai_reviewer/docs/apply.py tests/test_docs_apply.py && ruff check src/ai_reviewer/docs/apply.py`

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/docs/apply.py tests/test_docs_apply.py
git commit -m "feat(docs): add stage-3 update_section + add_section appliers"
```

---

### Task 6: Apply — `create_page` + nav/index wiring (`docs/page_builder.py`)

**Files:**
- Create: `src/ai_reviewer/docs/page_builder.py`
- Test: `tests/test_docs_page_builder.py`

**Interfaces:**
- Consumes: `DocAction`, `Change`, `DocDraft`, `FileWrite` (Task 2).
- Produces:
  - `insert_nav_entry(nav_js: str, label: str, href: str, dot: str, section: str) -> str | None` — inserts `{ label: '...', href: '...', dot: '...' },` immediately after the matching `{ section: '<section>' }` line in the `NAV` array. None if the section line isn't found.
  - `insert_index_link(index_html: str, href: str, title: str, blurb: str) -> str | None` — best-effort; inserts a `.hero-card` into the first `<div class="g3">` after `Crate Index`. None if not found.
  - `async def build_new_page(*, action, change, sibling_html, anthropic_cfg, model) -> str` — returns full page HTML (clone sibling skeleton, model fills body).
  - `async def apply_create_page(*, action, sibling_html, nav_js, index_html, change, section_group, dot, anthropic_cfg, model, allow_new_sections, best_fit_for_downgrade, best_fit_html) -> DocDraft` — builds the page, wires nav (mandatory) + index (best-effort); **orphan guard**: if `insert_nav_entry` fails, downgrade to `add_section` on `best_fit_for_downgrade` (using `apply_add_section`) or flag.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs_page_builder.py
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.models import Change, DocAction
from ai_reviewer.docs.page_builder import (
    apply_create_page,
    insert_index_link,
    insert_nav_entry,
)

_NAV = (
    "  const NAV = [\n"
    "    { label: 'Home', href: 'index.html', dot: '#f59e0b' },\n"
    "    { section: 'Architecture Deep-Dive' },\n"
    "    { label: 'Auto-Follow', href: 'auto-follow.html', dot: '#10b981' },\n"
    "  ];\n"
)

_SIBLING = (
    "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
    "<title>Auto-Follow — Calimero Core Architecture</title>\n"
    "<link rel=\"stylesheet\" href=\"styles.css\"></head>\n<body>\n"
    '<div class="main"><div class="content">\n'
    '<div class="breadcrumb"><a href="index.html">Home</a><span class="sep">/</span><span>Auto-Follow</span></div>\n'
    "<h1>Auto-Follow</h1>\n"
    "</div></div>\n<script src=\"nav.js\"></script>\n</body></html>"
)


def test_insert_nav_entry_after_section():
    out = insert_nav_entry(_NAV, "Widgets", "widgets.html", "#10b981", "Architecture Deep-Dive")
    assert out is not None
    assert "widgets.html" in out
    # Entry sits right after the section marker.
    assert out.index("Architecture Deep-Dive") < out.index("widgets.html")
    assert out.index("widgets.html") < out.index("Auto-Follow")
    # Still a single NAV array close.
    assert out.count("];") == 1


def test_insert_nav_entry_missing_section_returns_none():
    assert insert_nav_entry(_NAV, "X", "x.html", "#fff", "Nonexistent Section") is None


def test_insert_index_link_missing_returns_none():
    assert insert_index_link("<html>no crate index</html>", "x.html", "X", "b") is None


@pytest.mark.asyncio
async def test_create_page_wires_nav_and_emits_page_filewrite():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("new_feature", "Widgets", "adds widgets", "y", [], [], "doc widgets")
    action = DocAction(change=change, action="create_page", target_path="architecture/widgets.html")
    page_body = (
        "<!DOCTYPE html>\n<html lang=\"en\"><head>"
        "<title>Widgets — Calimero Core Architecture</title>"
        '<link rel="stylesheet" href="styles.css"></head><body>'
        '<div class="main"><div class="content"><h1>Widgets</h1>'
        '<div class="card ga"><h2>Overview</h2></div>'
        '</div></div><script src="nav.js"></script></body></html>'
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=page_body)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_create_page(
            action=action, sibling_html=_SIBLING, nav_js=_NAV, index_html="<html></html>",
            change=change, section_group="Architecture Deep-Dive", dot="#10b981",
            anthropic_cfg=cfg, model="m", allow_new_sections=True,
            best_fit_for_downgrade="architecture/auto-follow.html", best_fit_html=_SIBLING,
        )
    assert draft.error is None
    assert draft.action == "create_page"
    assert draft.target_path == "architecture/widgets.html"
    # Page content + a nav.js aux edit are present; nav.js edit registered the page.
    aux_paths = {fw.path for fw in draft.aux_edits}
    assert any(p.endswith("nav.js") for p in aux_paths)
    nav_edit = next(fw for fw in draft.aux_edits if fw.path.endswith("nav.js"))
    assert "widgets.html" in nav_edit.content


@pytest.mark.asyncio
async def test_create_page_orphan_guard_downgrades_when_nav_anchor_missing():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change("new_feature", "Widgets", "adds widgets", "y", [], [], "doc widgets")
    action = DocAction(change=change, action="create_page", target_path="architecture/widgets.html")
    section_block = '<div class="card gb"><h2>Widgets</h2><p>new</p></div>'
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        # First call = page body (unused after downgrade), then add_section block.
        inst.run_completion = AsyncMock(side_effect=["<html>page</html>", section_block])
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_create_page(
            action=action, sibling_html=_SIBLING, nav_js="const NAV = [];",  # no section anchors
            index_html="<html></html>", change=change,
            section_group="Nonexistent Section", dot="#10b981",
            anthropic_cfg=cfg, model="m", allow_new_sections=True,
            best_fit_for_downgrade="architecture/auto-follow.html", best_fit_html=_SIBLING,
        )
    # Downgraded to an add_section on the best-fit page; no orphan page emitted.
    assert draft.action == "add_section"
    assert draft.target_path == "architecture/auto-follow.html"
    assert all(not fw.path.endswith("widgets.html") for fw in draft.aux_edits)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_page_builder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_reviewer.docs.page_builder'`

- [ ] **Step 3: Implement `docs/page_builder.py`**

```python
# src/ai_reviewer/docs/page_builder.py
"""Stage 3 (new pages) — build a new page and wire it into nav.js (+ best-effort index.html)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ai_reviewer.docs.apply import apply_add_section
from ai_reviewer.docs.models import Change, DocAction, DocDraft, FileWrite

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig

_NEW_PAGE_SYSTEM = """\
You are writing a new HTML documentation page for the Calimero Core architecture site.
You are given a SIBLING page to copy the exact skeleton from (doctype, <head>, <title> suffix
"— Calimero Core Architecture", stylesheet link, .main/.content wrappers, breadcrumb, and the
trailing <script src="nav.js"></script>). Produce a COMPLETE page that:
- keeps that skeleton EXACTLY (only change the <title>, breadcrumb label, and <h1>),
- expresses the content as a sequence of <div class="card ga|gb|gc|gd"> blocks (cycle the class),
- uses ONLY these constructs inside cards: <h2>/<h3>, <p>, <ul>/<ol>/<li>, <code>,
  <pre class="code">, <strong>, <em>. Do NOT invent CSS classes or inline styles.
Output ONLY the page HTML, nothing else."""


def insert_nav_entry(nav_js: str, label: str, href: str, dot: str, section: str) -> str | None:
    """Insert a NAV entry right after the `{ section: '<section>' }` marker. None if absent."""
    pattern = re.compile(r"(\{\s*section:\s*'" + re.escape(section) + r"'\s*\},?[ \t]*\n)")
    m = pattern.search(nav_js)
    if not m:
        return None
    indent_m = re.match(r"([ \t]*)", m.group(1))
    indent = indent_m.group(1) if indent_m else "    "
    entry = f"{indent}{{ label: '{label}', href: '{href}', dot: '{dot}' }},\n"
    return nav_js[: m.end(1)] + entry + nav_js[m.end(1) :]


def insert_index_link(index_html: str, href: str, title: str, blurb: str) -> str | None:
    """Best-effort: add a hero-card to the first g3 grid after 'Crate Index'. None if not found."""
    idx = index_html.find("Crate Index")
    if idx == -1:
        return None
    grid = index_html.find('<div class="g3">', idx)
    if grid == -1:
        return None
    insert_at = grid + len('<div class="g3">')
    card = (
        f'\n  <a href="{href}" class="hero-card">'
        f'<span class="card-icon">&#9670;</span><h3>{title}</h3><p>{blurb}</p></a>'
    )
    return index_html[:insert_at] + card + index_html[insert_at:]


async def build_new_page(
    *,
    action: DocAction,
    change: Change,
    sibling_html: str,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
) -> str:
    from ai_reviewer.agents.anthropic_client import AnthropicClient

    user = (
        f"## Sibling page (copy the skeleton)\n{sibling_html}\n\n"
        f"## New page path\n{action.target_path}\n\n"
        f"## Feature to document\ntitle: {change.title}\nwhat_changed: {change.what_changed}\n"
        f"why: {change.why}\ndoc_impact: {change.doc_impact}\n"
    )
    async with AnthropicClient(anthropic_cfg) as client:
        return (await client.run_completion(
            model=model, system=_NEW_PAGE_SYSTEM, user=user, max_tokens=8192
        )).strip()


async def apply_create_page(
    *,
    action: DocAction,
    sibling_html: str,
    nav_js: str,
    index_html: str,
    change: Change,
    section_group: str,
    dot: str,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
    allow_new_sections: bool,
    best_fit_for_downgrade: str,
    best_fit_html: str,
) -> DocDraft:
    """Build a new page and register it. Orphan guard: must wire nav.js or we downgrade."""
    label = change.title
    href = action.target_path.split("/")[-1]

    # Orphan guard FIRST: if we can't register the page in nav, never create it.
    new_nav = insert_nav_entry(nav_js, label, href, dot, section_group)
    if new_nav is None:
        if allow_new_sections and best_fit_for_downgrade and best_fit_html:
            downgrade = DocAction(change=change, action="add_section",
                                  target_path=best_fit_for_downgrade)
            return await apply_add_section(downgrade, best_fit_html, change, anthropic_cfg, model)
        return DocDraft(action="create_page", target_path=action.target_path, updated_content="",
                        change=change, error="orphan guard: nav.js section anchor not found")

    page_html = await build_new_page(
        action=action, change=change, sibling_html=sibling_html,
        anthropic_cfg=anthropic_cfg, model=model,
    )
    aux = [FileWrite(path="architecture/nav.js" if "/" in action.target_path else "nav.js",
                     content=new_nav)]
    new_index = insert_index_link(index_html, href, change.title, change.what_changed[:120])
    if new_index is not None:
        aux.append(FileWrite(
            path="architecture/index.html" if "/" in action.target_path else "index.html",
            content=new_index,
        ))
    return DocDraft(action="create_page", target_path=action.target_path,
                    updated_content=page_html, change=change, aux_edits=aux)
```

> **Note on aux paths:** the `nav.js`/`index.html` paths must be sibling to `action.target_path`. The simple heuristic above assumes top-level `architecture/`. In Task 9 the orchestrator passes the real directory; refine `apply_create_page` to take a `doc_dir` param if a repo uses a different layout. For the default `architecture/` site this is correct.

- [ ] **Step 4: Run tests + ruff**

Run: `pytest tests/test_docs_page_builder.py -v` → PASS
Run: `ruff format src/ai_reviewer/docs/page_builder.py tests/test_docs_page_builder.py && ruff check src/ai_reviewer/docs/page_builder.py`

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/docs/page_builder.py tests/test_docs_page_builder.py
git commit -m "feat(docs): add stage-3 new-page builder + nav/index wiring with orphan guard"
```

---

### Task 7: Verify stage (`docs/verify.py`)

**Files:**
- Create: `src/ai_reviewer/docs/verify.py`
- Test: `tests/test_docs_verify.py`

**Interfaces:**
- Consumes: `DocDraft`, `Change`, `Verdict`, `extract_json`, `meets_threshold` (Task 2).
- Produces: `async def verify_draft(*, draft: DocDraft, anthropic_cfg, model: str, threshold: str) -> DocDraft` — returns the draft unchanged if it passes; otherwise returns a copy with `flagged_reason` set and `updated_content=""` (so it is never committed). Drafts that already carry `error` pass through untouched.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs_verify.py
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.models import Change, DocDraft
from ai_reviewer.docs.verify import verify_draft


def _draft() -> DocDraft:
    c = Change("behavior_change", "t", "events now flushed after persist", "y", [], [], "i")
    return DocDraft(action="update_section", target_path="architecture/x.html",
                    updated_content="<p>events flushed after persist</p>",
                    before_content="<p>old</p>", change=c)


@pytest.mark.asyncio
async def test_passing_verdict_keeps_draft():
    cfg = AnthropicApiConfig(api_key="sk-test")
    verdict = json.dumps({"reflects_change": True, "confidence": "high", "notes": "ok"})
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=verdict)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        out = await verify_draft(draft=_draft(), anthropic_cfg=cfg, model="m", threshold="medium")
    assert out.flagged_reason is None
    assert out.updated_content


@pytest.mark.asyncio
async def test_failing_verdict_flags_and_clears_content():
    cfg = AnthropicApiConfig(api_key="sk-test")
    verdict = json.dumps({"reflects_change": False, "confidence": "low",
                          "notes": "only renamed; missed the invariant"})
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=verdict)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        out = await verify_draft(draft=_draft(), anthropic_cfg=cfg, model="m", threshold="medium")
    assert out.updated_content == ""
    assert out.flagged_reason is not None
    assert "missed the invariant" in out.flagged_reason


@pytest.mark.asyncio
async def test_low_confidence_below_threshold_flags():
    cfg = AnthropicApiConfig(api_key="sk-test")
    verdict = json.dumps({"reflects_change": True, "confidence": "low", "notes": "unsure"})
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=verdict)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        out = await verify_draft(draft=_draft(), anthropic_cfg=cfg, model="m", threshold="medium")
    assert out.flagged_reason is not None


@pytest.mark.asyncio
async def test_errored_draft_passes_through_untouched():
    cfg = AnthropicApiConfig(api_key="sk-test")
    c = Change("fix", "t", "w", "y", [], [], "i")
    errored = DocDraft(action="update_section", target_path="x.html", updated_content="",
                       change=c, error="bad patch")
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        out = await verify_draft(draft=errored, anthropic_cfg=cfg, model="m", threshold="medium")
    assert out is errored
    inst.run_completion.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_verify.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ai_reviewer.docs.verify'`

- [ ] **Step 3: Implement `docs/verify.py`**

```python
# src/ai_reviewer/docs/verify.py
"""Stage 4 — confidence gate: flag (don't ship) drafts that don't reflect their change."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from ai_reviewer.docs.models import DocDraft, extract_json, meets_threshold

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig

logger = logging.getLogger(__name__)

_VERIFY_SYSTEM = """\
You check whether an edited documentation page now reflects a specific code change.
You are given the change, the BEFORE content (or a note that the page is new), and the AFTER
content. Judge ONLY whether the AFTER faithfully conveys the change's substance — not style.
Return ONLY JSON: {"reflects_change": true|false, "confidence": "low|medium|high",
"notes": "one sentence; if false, say what is missing"}"""


async def verify_draft(
    *,
    draft: DocDraft,
    anthropic_cfg: AnthropicApiConfig,
    model: str,
    threshold: str,
) -> DocDraft:
    from ai_reviewer.agents.anthropic_client import AnthropicClient

    if draft.error or not draft.updated_content:
        return draft  # already failed upstream; nothing to verify

    change = draft.change
    user = (
        f"## Change\ntitle: {getattr(change, 'title', '')}\n"
        f"what_changed: {getattr(change, 'what_changed', '')}\n"
        f"doc_impact: {getattr(change, 'doc_impact', '')}\n\n"
        f"## BEFORE\n{draft.before_content or '(new page — no prior content)'}\n\n"
        f"## AFTER\n{draft.updated_content}\n"
    )
    async with AnthropicClient(anthropic_cfg) as client:
        try:
            raw = await client.run_completion(
                model=model, system=_VERIFY_SYSTEM, user=user, max_tokens=512
            )
            v = extract_json(raw)
        except Exception as exc:  # noqa: BLE001 — extract_json ValueError or any client error
            logger.warning("Verify failed for %s: %s — flagging", draft.target_path, exc)
            return replace(draft, updated_content="",
                           flagged_reason=f"verification error: {exc}")

    reflects = bool(v.get("reflects_change", False))
    confidence = str(v.get("confidence", "low"))
    notes = str(v.get("notes", ""))
    if reflects and meets_threshold(confidence, threshold):
        return draft
    return replace(
        draft,
        updated_content="",
        flagged_reason=f"low-confidence doc update ({confidence}): {notes}",
    )
```

- [ ] **Step 4: Run tests + ruff**

Run: `pytest tests/test_docs_verify.py -v` → PASS
Run: `ruff format src/ai_reviewer/docs/verify.py tests/test_docs_verify.py && ruff check src/ai_reviewer/docs/verify.py`

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/docs/verify.py tests/test_docs_verify.py
git commit -m "feat(docs): add stage-4 verify gate (flag, do not ship, on low confidence)"
```

---

### Task 8: `FileWrite` support in `create_doc_update_pr`

**Files:**
- Modify: `src/ai_reviewer/github/client.py:1582-1685`
- Test: `tests/test_github_filewrite.py`

**Interfaces:**
- Consumes: `FileWrite` (Task 2).
- Produces: `create_doc_update_pr(..., file_writes: list[FileWrite], ...)` — replaces `updates: list[DocDraft]`. Commits every `FileWrite` (create if absent, update if present). Aborts if nothing committed (unchanged behavior).

First: `grep -rn "create_doc_update_pr(" src/ tests/` to find all callers. The only production caller is `docs/updater.py` (rewritten in Task 9). Update any existing tests that call it with `updates=`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_github_filewrite.py
from __future__ import annotations

from unittest.mock import MagicMock

from ai_reviewer.config import GitHubConfig
from ai_reviewer.docs.models import FileWrite
from ai_reviewer.github.client import GitHubClient


def _client_with_repo(repo):
    cfg = GitHubConfig(token="t")
    c = GitHubClient(cfg)
    c._gh = MagicMock()
    c._gh.get_repo.return_value = repo
    return c


def test_commits_new_and_existing_files():
    repo = MagicMock()
    repo.create_git_ref.return_value = None

    # First file exists (update), second doesn't (create).
    def get_contents(path, ref):
        if path == "architecture/auto-follow.html":
            m = MagicMock()
            m.sha = "abc"
            return m
        raise Exception("404")

    repo.get_contents.side_effect = get_contents
    pr = MagicMock()
    pr.number = 7
    pr.html_url = "https://example/pr/7"
    repo.create_pull.return_value = pr

    client = _client_with_repo(repo)
    url = client.create_doc_update_pr(
        repo_name="o/r", base_branch="master", base_sha="deadbeefcafe",
        file_writes=[
            FileWrite(path="architecture/auto-follow.html", content="<html>updated</html>"),
            FileWrite(path="architecture/widgets.html", content="<html>new</html>"),
        ],
        pr_title="docs: auto", pr_body="body", pr_number=42,
    )
    assert url == "https://example/pr/7"
    repo.update_file.assert_called_once()
    repo.create_file.assert_called_once()


def test_aborts_when_nothing_committed():
    import pytest

    repo = MagicMock()
    repo.create_git_ref.return_value = None
    repo.get_contents.side_effect = Exception("404")
    repo.create_file.side_effect = Exception("write blocked")
    client = _client_with_repo(repo)
    with pytest.raises(RuntimeError, match="No files were successfully committed"):
        client.create_doc_update_pr(
            repo_name="o/r", base_branch="master", base_sha="deadbeefcafe",
            file_writes=[FileWrite(path="x.html", content="<html>")],
            pr_title="t", pr_body="b",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_github_filewrite.py -v`
Expected: FAIL with `TypeError` (unexpected keyword `file_writes`) or signature mismatch.

- [ ] **Step 3: Change the signature + commit loop**

In `src/ai_reviewer/github/client.py`, add the import near the top (with other `ai_reviewer` imports):

```python
from ai_reviewer.docs.models import FileWrite
```

Replace the `updates: list[DocDraft]` parameter with `file_writes: list[FileWrite]`, and replace the commit loop body (lines ~1620-1658) so it iterates `file_writes`:

```python
        committed: list[str] = []
        for fw in file_writes:
            if not fw.content:
                logger.warning("Skipping %s: empty content", fw.path)
                continue
            try:
                existing_sha: str | None = None
                try:
                    existing = repo.get_contents(fw.path, ref=branch_name)
                    if not isinstance(existing, list):
                        existing_sha = existing.sha
                except Exception as inner_e:
                    _raise_if_forbidden(inner_e)
                    # file doesn't exist yet — create it

                commit_msg = f"docs: auto-update {fw.path}"
                if existing_sha:
                    repo.update_file(fw.path, commit_msg, fw.content, existing_sha, branch=branch_name)
                else:
                    repo.create_file(fw.path, commit_msg, fw.content, branch=branch_name)
                committed.append(fw.path)
                logger.info("Committed %s to %s", fw.path, branch_name)
            except Exception as e:
                _raise_if_forbidden(e)
                logger.warning("Could not commit %s: %s", fw.path, e)
```

Update the docstring to say it commits `FileWrite` objects. Leave branch creation, PR creation, assignee, and labels logic unchanged.

- [ ] **Step 4: Run tests + ruff**

Run: `pytest tests/test_github_filewrite.py -v` → PASS
Run: `pytest tests/ -k "doc or github" -q` (catch any caller still passing `updates=`)
Run: `ruff format src/ai_reviewer/github/client.py tests/test_github_filewrite.py && ruff check src/ai_reviewer/github/client.py`

- [ ] **Step 5: Commit**

```bash
git add src/ai_reviewer/github/client.py tests/test_github_filewrite.py
git commit -m "refactor(github): commit list[FileWrite] in create_doc_update_pr"
```

---

### Task 9: Orchestrator rewrite (`docs/updater.py`)

**Files:**
- Rewrite: `src/ai_reviewer/docs/updater.py`
- Test: `tests/test_docs_updater.py` (rewrite/extend existing)

**Interfaces:**
- Consumes: every stage above + `GitHubClient` + `DocGenerationSettings` + `AnthropicApiConfig`.
- Produces: `async def run_doc_update(*, repo, pr_number, gh, anthropic_cfg, doc_generation, base=None, dry_run=False) -> DocUpdateResult` (same call shape as today). `DocUpdateResult` gains `flagged: list[DocDraft]`.
- Flow: dedupe guard → fetch PR/diff/commits → `summarize_pr_changes` → `build_doc_index` (from `get_html_files_in_dirs` + changed targets) → `route_changes` → per-action apply (pipeline: each routed action applied, then verified) → split into `successful` / `flagged` → build PR body → `create_doc_update_pr(file_writes=...)` (drafts + aux edits flattened). If nothing successful but some flagged → post a comment listing flagged docs, open no PR.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_docs_updater.py
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig, DocGenerationSettings
from ai_reviewer.docs.models import Change, ChangeSummary, DocAction, DocDraft
from ai_reviewer.docs.updater import run_doc_update


def _gh_for(diff: str, html_files: list[str]):
    gh = MagicMock()
    pr = MagicMock()
    pr.base.ref = "master"
    pr.merge_commit_sha = "deadbeefcafe1234"
    pr.head.sha = "headsha"
    pr.title = "fix: x"
    pr.body = "body"
    pr.html_url = "https://example/pr/1"
    pr.user.login = "alice"
    pr.get_files.return_value = [MagicMock(filename="crates/gov/src/lib.rs", status="modified")]
    pr.get_commits.return_value = [MagicMock(commit=MagicMock(message="m"))]
    gh.get_pull_request.return_value = pr
    gh.has_open_doc_update_pr.return_value = False
    gh.load_repo_config.return_value = {"doc_generation": {"enabled": True},
                                        "documentation": {"static_docs_dirs": ["architecture/"]}}
    gh.get_pr_diff.return_value = diff
    gh.get_html_files_in_dirs.return_value = html_files
    gh.create_doc_update_pr.return_value = "https://example/pr/2"
    return gh, pr


@pytest.mark.asyncio
async def test_successful_update_opens_pr():
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)

    change = Change("behavior_change", "t", "w", "y", [], ["crates/gov/src/lib.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[change])
    action = DocAction(change=change, action="update_section",
                       target_path="architecture/auto-follow.html")
    good_draft = DocDraft(action="update_section", target_path="architecture/auto-follow.html",
                          updated_content="<html>updated</html>", change=change)

    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[action])),
        patch("ai_reviewer.docs.updater.apply_update_section", AsyncMock(return_value=good_draft)),
        patch("ai_reviewer.docs.updater.verify_draft", AsyncMock(return_value=good_draft)),
    ):
        gh.get_file_contents.return_value = MagicMock(decoded_content=b"<html>old</html>")
        result = await run_doc_update(repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg,
                                      doc_generation=dg)

    assert result.pr_url == "https://example/pr/2"
    assert len(result.successful) == 1
    assert gh.create_doc_update_pr.called


@pytest.mark.asyncio
async def test_all_flagged_posts_comment_no_pr():
    gh, pr = _gh_for("diff", ["architecture/auto-follow.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)

    change = Change("behavior_change", "t", "w", "y", [], ["crates/gov/src/lib.rs"], "i")
    summary = ChangeSummary(pr_intent="i", changes=[change])
    action = DocAction(change=change, action="update_section",
                       target_path="architecture/auto-follow.html")
    flagged = DocDraft(action="update_section", target_path="architecture/auto-follow.html",
                       updated_content="", change=change,
                       flagged_reason="low-confidence (low): missed the invariant")

    with (
        patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=summary)),
        patch("ai_reviewer.docs.updater.route_changes", AsyncMock(return_value=[action])),
        patch("ai_reviewer.docs.updater.apply_update_section", AsyncMock(return_value=flagged)),
        patch("ai_reviewer.docs.updater.verify_draft", AsyncMock(return_value=flagged)),
    ):
        gh.get_file_contents.return_value = MagicMock(decoded_content=b"<html>old</html>")
        result = await run_doc_update(repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg,
                                      doc_generation=dg)

    assert result.pr_url is None
    assert len(result.flagged) == 1
    assert not gh.create_doc_update_pr.called
    assert gh.post_or_update_doc_comment.called


@pytest.mark.asyncio
async def test_no_changes_skips():
    gh, _ = _gh_for("diff", ["architecture/auto-follow.html"])
    cfg = AnthropicApiConfig(api_key="sk-test")
    dg = DocGenerationSettings(enabled=True)
    empty = ChangeSummary(pr_intent="nothing doc-relevant", changes=[])
    with patch("ai_reviewer.docs.updater.summarize_pr_changes", AsyncMock(return_value=empty)):
        result = await run_doc_update(repo="o/r", pr_number=1, gh=gh, anthropic_cfg=cfg,
                                      doc_generation=dg)
    assert result.skipped
    assert not gh.create_doc_update_pr.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_updater.py -v`
Expected: FAIL (old `run_doc_update` has no stage imports to patch / no `flagged` field).

- [ ] **Step 3: Rewrite `docs/updater.py`**

```python
# src/ai_reviewer/docs/updater.py
"""Doc-update orchestration: Understand -> Route -> Apply -> Verify -> open PR (or flag)."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ai_reviewer.docs.apply import apply_add_section, apply_update_section
from ai_reviewer.docs.models import DocDraft, FileWrite
from ai_reviewer.docs.page_builder import apply_create_page
from ai_reviewer.docs.router import build_doc_index, route_changes
from ai_reviewer.docs.understanding import summarize_pr_changes
from ai_reviewer.docs.verify import verify_draft

if TYPE_CHECKING:
    from ai_reviewer.config import AnthropicApiConfig, DocGenerationSettings
    from ai_reviewer.github.client import GitHubClient

logger = logging.getLogger(__name__)

_DOC_COMMENT_MARKER = "<!-- AI-CODE-REVIEWER-DOC-BOT -->"
_DEFAULT_SECTION_GROUP = "Architecture Deep-Dive"
_DEFAULT_DOT = "#10b981"


@dataclass
class DocUpdateResult:
    successful: list[DocDraft] = field(default_factory=list)
    failed: list[DocDraft] = field(default_factory=list)
    flagged: list[DocDraft] = field(default_factory=list)
    pr_url: str | None = None
    skipped: bool = False
    skip_reason: str | None = None


def _read_file(gh: GitHubClient, repo: str, path: str, ref: str) -> str | None:
    try:
        raw = gh.get_file_contents(repo, path, ref)
        if isinstance(raw, list):
            return None
        return raw.decoded_content.decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read %s@%s: %s", path, ref, exc)
        return None


def _build_pr_body(pr_number: int, pr_html_url: str, successful, flagged) -> str:
    updated = "\n".join(
        f"- `{d.target_path}` — {getattr(d.change, 'what_changed', '') or 'updated'}"
        for d in successful
    )
    body = (
        f"## Automatic Documentation Update\n\n"
        f"Opened automatically after [PR #{pr_number}]({pr_html_url}) merged.\n\n"
        f"### Updated\n\n{updated}\n"
    )
    if flagged:
        flags = "\n".join(f"- `{d.target_path}` — {d.flagged_reason}" for d in flagged)
        body += (
            "\n### Flagged for human review\n\n"
            "These docs look stale but the bot could not confidently update them:\n\n"
            f"{flags}\n"
        )
    body += "\n---\n*Generated by `ai-reviewer update-docs`. Nothing was auto-merged.*"
    return body


def _flagged_comment(pr_number: int, flagged) -> str:
    flags = "\n".join(f"- `{d.target_path}` — {d.flagged_reason}" for d in flagged)
    return (
        f"{_DOC_COMMENT_MARKER}\n\n## Documentation Review\n\n"
        f"PR #{pr_number} likely makes these docs stale, but the bot could not confidently "
        f"update them. Please review:\n\n{flags}\n"
    )


async def _apply_one(action, gh, repo, ref, doc_dir, anthropic_cfg, dg) -> DocDraft:
    current = _read_file(gh, repo, action.target_path, ref) or ""
    if action.action == "update_section":
        return await apply_update_section(action, current, action.change,
                                          anthropic_cfg, dg.apply_model)
    if action.action == "add_section":
        return await apply_add_section(action, current, action.change,
                                       anthropic_cfg, dg.apply_model)
    # create_page: gather sibling/nav/index
    nav_js = _read_file(gh, repo, f"{doc_dir}nav.js", ref) or ""
    index_html = _read_file(gh, repo, f"{doc_dir}index.html", ref) or ""
    siblings = [p for p in gh.get_html_files_in_dirs(repo, ref, [doc_dir])
                if not p.endswith(("index.html",))]
    sibling_html = _read_file(gh, repo, siblings[0], ref) if siblings else ""
    best_fit = siblings[0] if siblings else ""
    best_fit_html = sibling_html or ""
    return await apply_create_page(
        action=action, sibling_html=sibling_html or "", nav_js=nav_js, index_html=index_html,
        change=action.change, section_group=_DEFAULT_SECTION_GROUP, dot=_DEFAULT_DOT,
        anthropic_cfg=anthropic_cfg, model=dg.apply_model,
        allow_new_sections=dg.allow_new_sections,
        best_fit_for_downgrade=best_fit, best_fit_html=best_fit_html,
    )


async def run_doc_update(
    *,
    repo: str,
    pr_number: int,
    gh: GitHubClient,
    anthropic_cfg: AnthropicApiConfig,
    doc_generation: DocGenerationSettings,
    base: str | None = None,
    dry_run: bool = False,
) -> DocUpdateResult:
    pr = gh.get_pull_request(repo, pr_number)
    base_branch = base or pr.base.ref

    if not dry_run and gh.has_open_doc_update_pr(repo, base_branch):
        return DocUpdateResult(skipped=True,
                               skip_reason=f"open doc-update PR already exists for {base_branch}")

    ref = pr.merge_commit_sha or pr.head.sha
    repo_config = gh.load_repo_config(repo, ref) or {}
    repo_docgen = repo_config.get("doc_generation", {})
    if repo_docgen.get("enabled") is False:
        return DocUpdateResult(skipped=True, skip_reason="doc_generation disabled in repo config")
    if not (repo_docgen.get("enabled") or doc_generation.enabled):
        return DocUpdateResult(skipped=True, skip_reason="doc_generation not enabled")

    doc_config = repo_config.get("documentation") or {}
    static_dirs = (repo_docgen.get("static_docs_dirs") or doc_config.get("static_docs_dirs")
                   or doc_generation.static_docs_dirs)
    doc_dir = static_dirs[0] if static_dirs else "architecture/"
    mapping = doc_config.get("source_to_docs_mapping", {})

    pr_files = list(pr.get_files())
    changed_paths = [f.filename for f in pr_files]
    commit_messages = [c.commit.message for c in pr.get_commits()]
    diff = gh.get_pr_diff(pr)

    summary = await summarize_pr_changes(
        pr_title=pr.title, pr_body=pr.body or "", commit_messages=commit_messages,
        diff=diff, anthropic_cfg=anthropic_cfg, model=doc_generation.understanding_model,
        max_diff_chars=doc_generation.max_understanding_diff_chars,
    )
    if not summary.changes:
        return DocUpdateResult(skipped=True, skip_reason="no doc-relevant changes detected")

    html_files = gh.get_html_files_in_dirs(repo, ref, static_dirs)
    doc_index = build_doc_index(html_files)

    actions = await route_changes(
        summary=summary, source_to_docs_mapping=mapping, changed_paths=changed_paths,
        doc_index=doc_index, allow_new_pages=doc_generation.allow_new_pages,
        allow_new_sections=doc_generation.allow_new_sections,
        anthropic_cfg=anthropic_cfg, model=doc_generation.understanding_model,
    )
    if not actions:
        return DocUpdateResult(skipped=True, skip_reason="no documentation targets routed")

    async def _pipeline(action) -> DocDraft:
        draft = await _apply_one(action, gh, repo, ref, doc_dir, anthropic_cfg, doc_generation)
        return await verify_draft(draft=draft, anthropic_cfg=anthropic_cfg,
                                  model=doc_generation.verify_model,
                                  threshold=doc_generation.verify_confidence_threshold)

    drafts = await asyncio.gather(*[_pipeline(a) for a in actions])

    successful = [d for d in drafts if d.updated_content and not d.error and not d.flagged_reason]
    flagged = [d for d in drafts if d.flagged_reason]
    failed = [d for d in drafts if d.error and not d.flagged_reason]

    if dry_run:
        return DocUpdateResult(successful=successful, failed=failed, flagged=flagged)

    if not successful:
        if flagged:
            with __import__("contextlib").suppress(Exception):
                gh.post_or_update_doc_comment(pr, _flagged_comment(pr_number, flagged),
                                              _DOC_COMMENT_MARKER)
        return DocUpdateResult(failed=failed, flagged=flagged, skipped=True,
                               skip_reason="no confident doc updates; flagged for humans")

    file_writes: list[FileWrite] = []
    for d in successful:
        file_writes.append(FileWrite(path=d.target_path, content=d.updated_content))
        file_writes.extend(d.aux_edits)

    pr_url = gh.create_doc_update_pr(
        repo_name=repo, base_branch=base_branch, base_sha=ref, file_writes=file_writes,
        pr_title=f"docs: auto-update for PR #{pr_number} — {pr.title}",
        pr_body=_build_pr_body(pr_number, pr.html_url, successful, flagged),
        assignee=pr.user.login if pr.user else None, labels=doc_generation.pr_labels,
        draft=doc_generation.pr_draft, pr_number=pr_number,
    )
    return DocUpdateResult(successful=successful, failed=failed, flagged=flagged, pr_url=pr_url)
```

> Replace the `__import__("contextlib")` line with a top-of-file `import contextlib` and `contextlib.suppress(Exception)` — inlined here only to keep the block self-contained; clean it up when implementing.

- [ ] **Step 4: Run tests + ruff**

Run: `pytest tests/test_docs_updater.py -v` → PASS
Run: `ruff format src/ai_reviewer/docs/updater.py tests/test_docs_updater.py && ruff check src/ai_reviewer/docs/updater.py`

- [ ] **Step 5: Verify CLI still wires in + commit**

Run: `grep -n "run_doc_update" src/ai_reviewer/cli.py` — confirm the call site still matches the keyword signature (it does; signature is unchanged except the new `flagged` result field). If the CLI prints `result.successful`, optionally also print `result.flagged`.
Run: `pytest tests/ -k doc -q` → all green

```bash
git add src/ai_reviewer/docs/updater.py tests/test_docs_updater.py
git commit -m "feat(docs): rewrite orchestrator as Understand/Route/Apply/Verify pipeline"
```

---

### Task 10: Regression golden test (#2792 → #2794)

**Files:**
- Create: `tests/fixtures/pr2792.diff` (paste the real unified diff of calimero-network/core#2792)
- Create: `tests/fixtures/auto-follow.html` (copy the real page as it was BEFORE #2794)
- Create: `tests/test_regression_pr2792.py`

**Interfaces:**
- Consumes: `summarize_pr_changes`, `apply_update_section` / `apply_add_section`, `verify_draft`.

This test locks in that the exact failure (a bare rename) cannot recur. It uses a deterministic fake client whose responses encode the *correct* behavior, and asserts the pipeline carries the behavioral substance through.

- [ ] **Step 1: Create fixtures**

Save the real diff:
```bash
gh pr diff 2792 --repo calimero-network/core > tests/fixtures/pr2792.diff
git -C /Users/beast/Developer/Calimero/core show ffbf8b3d:architecture/auto-follow.html > tests/fixtures/auto-follow.html  # pre-#2794 state
```
(If the pre-#2794 blob hash differs, use `git -C <core> show <auto-follow-before-2794>:architecture/auto-follow.html`.)

- [ ] **Step 2: Write the failing test**

```python
# tests/test_regression_pr2792.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.apply import apply_update_section
from ai_reviewer.docs.models import DocAction
from ai_reviewer.docs.understanding import summarize_pr_changes
from ai_reviewer.docs.verify import verify_draft

FIX = Path(__file__).parent / "fixtures"


@pytest.mark.asyncio
async def test_pr2792_summary_captures_behavior_not_just_rename():
    diff = (FIX / "pr2792.diff").read_text()
    cfg = AnthropicApiConfig(api_key="sk-test")
    realistic = json.dumps({
        "pr_intent": "Emit op-events only after the op-log durably persists.",
        "changes": [{
            "kind": "behavior_change",
            "title": "Defer op-event emission until after op-log append",
            "what_changed": "Events are buffered and flushed only after the op-log entry is "
            "durably appended, and dropped on replay of an already-logged op; "
            "build_auto_follow_set_if_enabled now returns the event instead of emitting it.",
            "why": "Avoid double-firing on network re-gossip / DAG replay.",
            "symbols": ["build_auto_follow_set_if_enabled"],
            "files": ["crates/governance-store/src/lib.rs"],
            "doc_impact": "Propagation must state emit-after-persist + drop-on-replay.",
        }],
    })
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=realistic)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        summary = await summarize_pr_changes(
            pr_title="fix(governance-store): emit op-events after the op-log persists",
            pr_body="Closes #2770", commit_messages=["emit after persist"], diff=diff,
            anthropic_cfg=cfg, model="claude-sonnet-4-6", max_diff_chars=500_000,
        )
    # The summary must carry the behavioral substance, not just a rename.
    assert any(c.kind == "behavior_change" for c in summary.changes)
    blob = " ".join(c.what_changed.lower() for c in summary.changes)
    assert "after" in blob and "persist" in blob
    assert "drop" in blob or "replay" in blob
    # Cost guard: read the full diff once.
    assert inst.run_completion.call_count == 1


@pytest.mark.asyncio
async def test_pr2792_update_adds_invariant_and_preserves_page():
    page = (FIX / "auto-follow.html").read_text()
    cfg = AnthropicApiConfig(api_key="sk-test")
    from ai_reviewer.docs.models import Change

    change = Change(
        "behavior_change", "Defer op-event emission",
        "Events flushed only after the op-log persists; dropped on replay.",
        "avoid double-fire", ["build_auto_follow_set_if_enabled"],
        ["crates/governance-store/src/lib.rs"],
        "Propagation must state emit-after-persist + drop-on-replay.",
    )
    action = DocAction(change=change, action="update_section",
                       target_path="architecture/auto-follow.html")
    # Patch that (a) applies the rename AND (b) adds a sentence about the invariant.
    find_anchor = "an <code>OpEvent</code> is broadcast and the handler emits the corresponding join op."
    assert find_anchor in page  # guard: the fixture is the pre-#2794 page
    patch_resp = (
        "<<<FIND\n" + find_anchor + "\nFIND>>>\n"
        "<<<REPLACE\n" + find_anchor +
        " Events are buffered during apply and flushed via op_events::notify only after the "
        "op-log entry is durably appended; on a re-received (already-logged) op they are dropped "
        "rather than re-emitted, so re-gossip / DAG replay no longer double-fires.\nREPLACE>>>"
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=patch_resp)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_update_section(action, page, change, cfg, "m")

    assert draft.error is None
    # New invariant present...
    assert "flushed via op_events::notify only after the op-log" in draft.updated_content
    assert "drop" in draft.updated_content.lower()
    # ...and the rest of the page preserved (the whole document minus the touched span).
    assert "<h2>Rate Limit" in draft.updated_content
    assert "TEE Fleet Integration" in draft.updated_content
    # NOT a bare-rename-only change: the added clause is longer than a single word.
    assert len(draft.updated_content) > len(page) + 50


@pytest.mark.asyncio
async def test_pr2792_bare_rename_would_be_flagged():
    """A cosmetic rename-only edit that doesn't convey the invariant must be flagged."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    from ai_reviewer.docs.models import Change, DocDraft

    change = Change("behavior_change", "Defer emission",
                    "Events flushed only after op-log persists; dropped on replay.",
                    "", [], [], "state emit-after-persist + drop-on-replay")
    bare = DocDraft(action="update_section", target_path="architecture/auto-follow.html",
                    updated_content="<p>build_auto_follow_set_if_enabled synthesises ...</p>",
                    before_content="<p>emit_auto_follow_set_if_enabled synthesises ...</p>",
                    change=change)
    verdict = json.dumps({"reflects_change": False, "confidence": "high",
                          "notes": "only the function name changed; the emit-after-persist "
                          "invariant is not described"})
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=verdict)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        out = await verify_draft(draft=bare, anthropic_cfg=cfg, model="m", threshold="medium")
    assert out.updated_content == ""
    assert out.flagged_reason is not None
```

- [ ] **Step 3: Run to verify it fails, then passes**

Run: `pytest tests/test_regression_pr2792.py -v`
Expected first: FAIL if the fixture anchor text differs — adjust `find_anchor` to the exact sentence in the fixture page (the `assert find_anchor in page` guard pinpoints this). Once fixtures match, all three tests PASS.

- [ ] **Step 4: Full suite + ruff**

Run: `pytest tests/ -q` → all green
Run: `ruff format tests/test_regression_pr2792.py && ruff check tests/`

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/pr2792.diff tests/fixtures/auto-follow.html tests/test_regression_pr2792.py
git commit -m "test(docs): regression locking #2792->#2794 (capture behavior, not bare rename)"
```

---

## Self-Review

**Spec coverage:**
- Read-once ChangeSummary → Task 3 (+ cost-guard test in Tasks 3 & 10). ✓
- Route to update/add/create + allow-flags + downgrade → Task 4. ✓
- update_section reusing whitespace-tolerant matcher → Task 5. ✓
- add_section insertion + card-class cycle → Task 5. ✓
- create_page from sibling template + nav.js mandatory wiring + index best-effort + orphan guard → Task 6. ✓
- Site conventions (title suffix, breadcrumb, card classes, vocabulary) → encoded in Task 5/6 system prompts. ✓
- Verify gate "flag don't ship" → Task 7; surfaced in PR body / comment → Task 9. ✓
- Config knobs (models, diff cap, allow flags, threshold) → Task 1. ✓
- Remove 4000-char per-doc chop → achieved by Task 9 (apply consumes ChangeSummary, never raw diff). ✓
- New-file commit path → Task 8 (`FileWrite`). ✓
- Error model ship|flag|skip; all-flagged → comment, no PR → Task 9. ✓
- #2792/#2794 regression → Task 10. ✓
- Preserved guardrails (dedupe, matcher ambiguity-refusal, sentinel) → dedupe in Task 9; matcher/sentinel reused unchanged from `analyzer.py`. ✓

**Gaps / follow-ups (not blockers):**
- `dot`/`section_group` selection for new pages is currently a constant (`#10b981` / "Architecture Deep-Dive"); a later enhancement can have the router choose these. Logged, not silent.
- `_apply_one` reads the best-fit sibling as the first non-index HTML file; good enough for the `architecture/` site. If a repo has crate subpages mixed in, prefer a top-level sibling — refine if needed.

**Type consistency:** `DocDraft`, `Change`, `ChangeSummary`, `DocAction`, `Verdict`, `FileWrite` are defined once in Task 2 and imported everywhere; `apply_*` signatures in Task 5/6 match the calls in Task 9's `_apply_one`; `create_doc_update_pr(file_writes=...)` in Task 8 matches the call in Task 9. ✓

**Placeholder scan:** the one inlined `__import__("contextlib")` is flagged in-line with the fix (use a top-level `import contextlib`). No TBD/TODO. ✓
