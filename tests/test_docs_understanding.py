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
            pr_title="t",
            pr_body="",
            commit_messages=[],
            diff="small diff",
            anthropic_cfg=cfg,
            model="m",
            max_diff_chars=10_000,
        )
    assert inst.run_completion.call_count == 1


@pytest.mark.asyncio
async def test_map_reduce_over_cap_summarizes_per_file_then_merges():
    """A diff over the cap triggers per-file summarize + one merge call."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    big_file_a = "diff --git a/a.rs b/a.rs\n" + ("+x\n" * 200)
    big_file_b = "diff --git a/b.rs b/b.rs\n" + ("+y\n" * 200)
    diff = big_file_a + big_file_b
    per_file = json.dumps(
        {
            "changes": [
                {
                    "kind": "fix",
                    "title": "t",
                    "what_changed": "w",
                    "why": "y",
                    "symbols": [],
                    "files": [],
                    "doc_impact": "i",
                }
            ]
        }
    )
    merged = _SUMMARY_JSON
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(side_effect=[per_file, per_file, merged])
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

        cs = await summarize_pr_changes(
            pr_title="t",
            pr_body="",
            commit_messages=[],
            diff=diff,
            anthropic_cfg=cfg,
            model="m",
            max_diff_chars=100,
        )
    # 2 per-file calls + 1 merge call
    assert inst.run_completion.call_count == 3
    assert cs.changes[0].kind == "behavior_change"
