from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from ai_reviewer.config import AnthropicApiConfig
from ai_reviewer.docs.understanding import (
    _MAX_SUMMARY_TOKENS,
    _MIN_SUMMARY_TOKENS,
    _summary_max_tokens,
    summarize_pr_changes,
)

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


def test_summary_max_tokens_scales_with_diff_size():
    """The output cap scales with the diff: small PRs sit at the floor, large PRs
    get headroom, and everything clamps to [_MIN, _MAX]."""
    # Tiny PR → floor (it's a ceiling, so the PR still bills only its real output).
    assert _summary_max_tokens(0) == _MIN_SUMMARY_TOKENS
    assert _summary_max_tokens(1_000) == _MIN_SUMMARY_TOKENS
    # PR #2790's real diff (133,147 chars) truncated at the old hard 4096 cap;
    # the scaled budget must now comfortably clear it.
    assert _summary_max_tokens(133_147) > 4096
    # Monotonic: a bigger diff never gets a smaller budget.
    assert _summary_max_tokens(60_000) <= _summary_max_tokens(120_000)
    # A very large diff (e.g. the merge path over a >250K-char PR like #2821's
    # 488K) clamps to the ceiling, staying well within Sonnet's 64K output limit.
    assert _summary_max_tokens(488_577) == _MAX_SUMMARY_TOKENS
    assert _summary_max_tokens(10_000_000) == _MAX_SUMMARY_TOKENS


@pytest.mark.asyncio
async def test_summary_max_tokens_passed_through_to_model_call():
    """A large under-cap diff sends a scaled (not the legacy 4096) max_tokens."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    diff = "diff --git a/x b/x\n" + ("+line\n" * 30_000)  # ~180K chars
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value=_SUMMARY_JSON)
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        await summarize_pr_changes(
            pr_title="t",
            pr_body="",
            commit_messages=[],
            diff=diff,
            anthropic_cfg=cfg,
            model="m",
            max_diff_chars=250_000,
        )
    assert inst.run_completion.call_args.kwargs["max_tokens"] == _summary_max_tokens(len(diff))
    assert inst.run_completion.call_args.kwargs["max_tokens"] > 4096


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


@pytest.mark.asyncio
async def test_map_reduce_skips_unparseable_partial():
    """An unparseable per-file partial is skipped, not aborted; merge still runs."""
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
        inst.run_completion = AsyncMock(side_effect=["not json at all", per_file, merged])
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
    assert (
        inst.run_completion.call_count == 3
    )  # both files attempted (1st unparseable, skipped) + merge
    assert cs.changes[0].kind == "behavior_change"


@pytest.mark.asyncio
async def test_map_reduce_all_unparseable_returns_empty_no_merge_call():
    """If every per-file partial is unparseable, return an empty summary WITHOUT a merge call."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    big_a = "diff --git a/a.rs b/a.rs\n" + ("+x\n" * 200)
    big_b = "diff --git a/b.rs b/b.rs\n" + ("+y\n" * 200)
    diff = big_a + big_b
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(side_effect=["not json", "also not json"])
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
    assert cs.changes == []
    # Two per-file calls only — the merge call is skipped (no parseable partials).
    assert inst.run_completion.call_count == 2


@pytest.mark.asyncio
async def test_unparseable_summary_returns_empty_not_crash():
    """A non-JSON stage-1 response yields an empty summary, not an unhandled error."""
    cfg = AnthropicApiConfig(api_key="sk-test")
    with patch("ai_reviewer.agents.anthropic_client.AnthropicClient") as MockClient:
        inst = AsyncMock()
        inst.run_completion = AsyncMock(return_value="sorry, I can't produce JSON")
        MockClient.return_value.__aenter__ = AsyncMock(return_value=inst)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        cs = await summarize_pr_changes(
            pr_title="t",
            pr_body="",
            commit_messages=[],
            diff="small diff",
            anthropic_cfg=cfg,
            model="m",
            max_diff_chars=10_000,
        )
    assert cs.changes == []
    assert cs.pr_intent == "t"  # falls back to the PR title
