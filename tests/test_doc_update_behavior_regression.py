"""Regression lock for calimero-network/core #2792 -> #2794.

#2792 deferred op-event emission until after the op-log persists (and dropped
re-emits on replay) — a real behavioral change. The doc bot's PR #2794 changed
only the function name in auto-follow.html and missed the new invariant. These
tests lock in that the pipeline understands the behavior change (not just the
rename), that an update *adds* the invariant rather than producing a bare rename,
and that a bare rename is flagged.

The Anthropic client is mocked, so small representative inputs are sufficient —
the full ~90 KB PR diff and the real architecture page are intentionally NOT
vendored as fixtures (they would add bytes, not test coverage).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.apply import apply_update_section
from ai_reviewer.docs.models import Change, DocAction, DocDraft
from ai_reviewer.docs.understanding import summarize_pr_changes
from ai_reviewer.docs.verify import verify_draft

# A small representative slice of the #2792 diff. Content is irrelevant to the
# assertions (the client is mocked); it only needs to be a diff string under the
# stage-1 cap so the cost-guard "one call" assertion holds.
_DIFF = (
    "diff --git a/crates/governance-store/src/lib.rs b/crates/governance-store/src/lib.rs\n"
    "-pub(crate) fn emit_auto_follow_set_if_enabled(...) -> EyreResult<()> {\n"
    "+pub(crate) fn build_auto_follow_set_if_enabled(...) -> EyreResult<Option<OpEvent>> {\n"
    "+    for event in pending_events { crate::op_events::notify(event); }  // after op-log persists\n"
)

# Compact stand-in for architecture/auto-follow.html: the Propagation paragraph
# (the FIND anchor) plus two sections that must survive the targeted edit untouched.
_PAGE = (
    '<div class="card gb"><h2>Design</h2><h3>Propagation</h3><p>'
    "When a relevant op is applied to local state, an "
    "<code>OpEvent</code> is broadcast and the handler emits the corresponding join op."
    " Every emitted op is itself a DAG op.</p></div>\n"
    '<div class="card ga"><h2>Rate Limit</h2><p>token bucket</p></div>\n'
    '<div class="card gb"><h2>TEE Fleet Integration</h2><p>fleet-join</p></div>'
)


@pytest.mark.asyncio
async def test_summary_captures_behavior_not_just_rename():
    cfg = AnthropicApiConfig(api_key="sk-test")
    realistic = json.dumps(
        {
            "pr_intent": "Emit op-events only after the op-log durably persists.",
            "changes": [
                {
                    "kind": "behavior_change",
                    "title": "Defer op-event emission until after op-log append",
                    "what_changed": "Events are buffered and flushed only after the op-log entry "
                    "durably persists; dropped on replay of an already-logged op; "
                    "build_auto_follow_set_if_enabled now returns the event instead of emitting it.",
                    "why": "Avoid double-firing on network re-gossip / DAG replay.",
                    "symbols": ["build_auto_follow_set_if_enabled"],
                    "files": ["crates/governance-store/src/lib.rs"],
                    "doc_impact": "Propagation must state emit-after-persist + drop-on-replay.",
                }
            ],
        }
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=realistic)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        summary = await summarize_pr_changes(
            pr_title="fix(governance-store): emit op-events after the op-log persists",
            pr_body="Closes #2770",
            commit_messages=["emit after persist"],
            diff=_DIFF,
            anthropic_cfg=cfg,
            model="claude-sonnet-4-6",
            max_diff_chars=500_000,
        )
    assert any(c.kind == "behavior_change" for c in summary.changes)
    blob = " ".join(c.what_changed.lower() for c in summary.changes)
    assert "after" in blob and "persist" in blob
    assert "drop" in blob or "replay" in blob
    # Cost guard: the diff is read in exactly one call.
    assert inst.run_completion.call_count == 1


@pytest.mark.asyncio
async def test_update_adds_invariant_and_preserves_page():
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change(
        "behavior_change",
        "Defer op-event emission",
        "Events flushed only after the op-log persists; dropped on replay.",
        "avoid double-fire",
        ["build_auto_follow_set_if_enabled"],
        ["crates/governance-store/src/lib.rs"],
        "Propagation must state emit-after-persist + drop-on-replay.",
    )
    action = DocAction(
        change=change, action="update_section", target_path="architecture/auto-follow.html"
    )
    # Single-line substring of the Propagation paragraph.
    find_anchor = (
        "<code>OpEvent</code> is broadcast and the handler emits the corresponding join op."
    )
    assert find_anchor in _PAGE  # guard
    patch_resp = (
        "<<<FIND\n" + find_anchor + "\nFIND>>>\n"
        "<<<REPLACE\n"
        + find_anchor
        + " Events are buffered during apply and flushed via op_events::notify only after the "
        "op-log entry is durably appended; on a re-received (already-logged) op they are dropped "
        "rather than re-emitted, so re-gossip / DAG replay no longer double-fires.\nREPLACE>>>"
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=patch_resp)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        draft = await apply_update_section(action, _PAGE, change, cfg, "m")

    assert draft.error is None
    assert "flushed via op_events::notify only after the op-log" in draft.updated_content
    assert "drop" in draft.updated_content.lower()
    # Rest of the page preserved (untouched sections survive the targeted edit).
    assert "<h2>Rate Limit" in draft.updated_content
    assert "TEE Fleet Integration" in draft.updated_content
    # Not a bare rename: the added clause is substantial.
    assert len(draft.updated_content) > len(_PAGE) + 50


@pytest.mark.asyncio
async def test_bare_rename_is_flagged():
    """A cosmetic rename-only edit that doesn't convey the invariant must be flagged."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    change = Change(
        "behavior_change",
        "Defer emission",
        "Events flushed only after op-log persists; dropped on replay.",
        "",
        [],
        [],
        "state emit-after-persist + drop-on-replay",
    )
    bare = DocDraft(
        action="update_section",
        target_path="architecture/auto-follow.html",
        updated_content="<p>build_auto_follow_set_if_enabled synthesises ...</p>",
        before_content="<p>emit_auto_follow_set_if_enabled synthesises ...</p>",
        change=change,
    )
    verdict = json.dumps(
        {
            "reflects_change": False,
            "confidence": "high",
            "notes": "only the function name changed; the emit-after-persist "
            "invariant is not described",
        }
    )
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=verdict)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        out = await verify_draft(draft=bare, anthropic_cfg=cfg, model="m", threshold="medium")
    assert out.updated_content == ""
    assert out.flagged_reason is not None
